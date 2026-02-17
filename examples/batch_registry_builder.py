#!/usr/bin/env python3
"""
Batch Registry Builder — 6-stage pipeline for building PCO registries from scratch.

Adapted from the Artiswa batch pipeline pattern (manifest-driven sequential
orchestrator), this script takes a template definition (geometric primitives,
YCB objects, or custom objects) and walks through six stages:

    Stage 1: ORGANIZE   — Validate objects, create output directories
    Stage 2: HASH       — Generate synthetic images + compute composite hashes
    Stage 3: ENRICH     — Build full 4-layer SKBs from ground-truth seeds
    Stage 4: SIGN       — Compute Soulmarks (SHA-256 / HMAC-SHA256)
    Stage 5: REGISTER   — Populate CodexRegistry with hash-indexed SKBs
    Stage 6: VALIDATE   — Run codex-lab-kit 4-phase experiment protocol

Usage
-----
    # Standard evaluation (5 geometric primitives)
    python examples/batch_registry_builder.py --template standard_evaluation

    # YCB subset (20 benchmark objects)
    python examples/batch_registry_builder.py --template ycb_20

    # Dry run (no files written)
    python examples/batch_registry_builder.py --template standard_evaluation --dry-run

    # Resume from a specific stage
    python examples/batch_registry_builder.py --template ycb_20 --start-stage sign

Copyright (c) 2026 Metavolve Labs, Inc. — Robotics R&D Division
Patent Pending: U.S. Provisional Application No. 63/983,304 + 63/984,299
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the SDK is importable when run from the examples/ directory.
_SDK_ROOT = Path(__file__).resolve().parent.parent
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))

from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry
from gcp_robotics.soulmark import SoulmarkVerifier
from gcp_robotics.schema.models import (
    SpatialKinematicBlueprint,
    Identifiers,
    TripleHash,
    Timestamps,
    ProvenanceLayer,
    SemanticTopologyLayer,
    AffordanceLayer,
    SafetyLayer,
    ObjectClassification,
    ActionPrimitive,
    SpeedProfile,
    PerceptualSignature,
    PhysicalProperties,
    MaterialGraph,
    XYZ,
    ActionSchema,
    ActionParameters,
    ForceProfile,
    TargetPoseRelative,
    HazardClass,
    OperatingConstraints,
    Telemetry,
    PipelineStatus,
    ControlLoop,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_registry_builder")

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

GEOMETRIC_PRIMITIVES: Dict[str, dict] = {
    "cube_red_50mm": {
        "name": "Red Cube (50 mm)",
        "mass_kg": 0.125,
        "dimensions_m": [0.05, 0.05, 0.05],
        "material": "plastic",
        "category": "block",
        "graspable": True,
        "color_rgb": (200, 40, 40),
        "shape": "cube",
        "action_primitive": "PICK_VERTICAL",
        "hazard_class": "NONE",
        "notes": "Stable, easy to grasp. Default pick-and-place baseline.",
    },
    "sphere_blue_60mm": {
        "name": "Blue Sphere (60 mm)",
        "mass_kg": 0.113,
        "dimensions_m": [0.06, 0.06, 0.06],
        "material": "plastic",
        "category": "block",
        "graspable": True,
        "color_rgb": (40, 80, 200),
        "shape": "sphere",
        "action_primitive": "GRASP_PARALLEL_JAW",
        "hazard_class": "NONE",
        "notes": "Rolling risk. Requires adaptive grip.",
    },
    "cylinder_green_40x80mm": {
        "name": "Green Cylinder (40 x 80 mm)",
        "mass_kg": 0.100,
        "dimensions_m": [0.04, 0.04, 0.08],
        "material": "plastic",
        "category": "block",
        "graspable": True,
        "color_rgb": (40, 180, 60),
        "shape": "cylinder",
        "action_primitive": "PICK_VERTICAL",
        "hazard_class": "NONE",
        "notes": "Orientation matters for cylinder grasp.",
    },
    "pyramid_yellow_50mm": {
        "name": "Yellow Pyramid (50 mm base)",
        "mass_kg": 0.042,
        "dimensions_m": [0.05, 0.05, 0.05],
        "material": "plastic",
        "category": "block",
        "graspable": True,
        "color_rgb": (220, 200, 40),
        "shape": "pyramid",
        "action_primitive": "PICK_ANGLED",
        "hazard_class": "SHARP_EDGES",
        "notes": "Sharp apex. Angled approach required.",
    },
    "l_bracket_white_80x60mm": {
        "name": "White L-Bracket (80 x 60 mm)",
        "mass_kg": 0.095,
        "dimensions_m": [0.08, 0.06, 0.04],
        "material": "metal",
        "category": "tool",
        "graspable": True,
        "color_rgb": (230, 230, 230),
        "shape": "l_bracket",
        "action_primitive": "GRASP_PARALLEL_JAW",
        "hazard_class": "SHARP_EDGES",
        "notes": "Asymmetric centre of mass. Requires offset grip.",
    },
}

TEMPLATES: Dict[str, Dict[str, dict]] = {
    "standard_evaluation": GEOMETRIC_PRIMITIVES,
    # ycb_20 is loaded dynamically from YCBLoader
}

# Mapping helpers
_CATEGORY_TO_CLASSIFICATION: Dict[str, ObjectClassification] = {
    "block": ObjectClassification.RIGID_BODY_GRASPABLE,
    "tool": ObjectClassification.TOOL,
    "food_container": ObjectClassification.RIGID_BODY_GRASPABLE,
    "food_box": ObjectClassification.RIGID_BODY_GRASPABLE,
    "condiment": ObjectClassification.LIQUID_CONTAINER,
    "fruit_model": ObjectClassification.RIGID_BODY_GRASPABLE,
    "kitchen_tool": ObjectClassification.RIGID_BODY_GRASPABLE,
    "cleaning": ObjectClassification.LIQUID_CONTAINER,
    "tableware": ObjectClassification.RIGID_BODY_GRASPABLE,
    "stationery": ObjectClassification.RIGID_BODY_GRASPABLE,
}

_MATERIAL_TO_FRICTION: Dict[str, float] = {
    "plastic": 0.4,
    "metal": 0.3,
    "cardboard": 0.5,
    "ceramic": 0.35,
    "wood": 0.45,
    "cardboard_metal": 0.4,
    "plastic_metal": 0.35,
    "metal_plastic": 0.35,
}

_MATERIAL_TO_HAZARD: Dict[str, HazardClass] = {
    "metal": HazardClass.SHARP_EDGES,
    "metal_plastic": HazardClass.SHARP_EDGES,
    "plastic_metal": HazardClass.SHARP_EDGES,
    "plastic": HazardClass.NONE,
    "cardboard": HazardClass.NONE,
    "cardboard_metal": HazardClass.NONE,
    "wood": HazardClass.NONE,
    "ceramic": HazardClass.NONE,
}

_MATERIAL_TO_COLOUR: Dict[str, tuple] = {
    "plastic": (100, 180, 240),
    "metal": (180, 180, 190),
    "cardboard": (210, 170, 110),
    "ceramic": (230, 230, 220),
    "wood": (180, 140, 90),
    "cardboard_metal": (195, 175, 150),
    "plastic_metal": (140, 180, 215),
    "metal_plastic": (140, 180, 215),
}

STAGE_NAMES = ["organize", "hash", "enrich", "sign", "register", "validate"]


def _strip_0x(hex_str: str) -> str:
    """Strip ``0x`` prefix from hex hash strings."""
    return hex_str.lower().removeprefix("0x")


# ---------------------------------------------------------------------------
# Synthetic image generation for geometric primitives
# ---------------------------------------------------------------------------


def generate_primitive_images(
    object_id: str,
    obj_def: dict,
    output_dir: Path,
    count: int = 3,
) -> List[Path]:
    """Generate synthetic images for a geometric primitive.

    Draws the appropriate shape (cube, sphere, cylinder, pyramid, L-bracket)
    using the object's colour and dimensions.
    """
    from PIL import Image, ImageDraw, ImageFont

    dims = obj_def["dimensions_m"]
    shape = obj_def.get("shape", "cube")
    base_colour = obj_def.get("color_rgb", _MATERIAL_TO_COLOUR.get(obj_def["material"], (160, 160, 160)))

    # Map real-world dimensions to pixels (~1000 px/m)
    px_w = max(int(dims[0] * 1000), 40)
    px_h = max(int(dims[2] * 1000), 40)

    obj_dir = output_dir / object_id
    obj_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for i in range(count):
        jitter = (i * 11) % 30 - 15
        fill = tuple(max(0, min(255, c + jitter)) for c in base_colour)

        canvas_w = max(px_w + 80, 120)
        canvas_h = max(px_h + 80, 120)
        img = Image.new("RGB", (canvas_w, canvas_h), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        cx = canvas_w // 2
        cy = canvas_h // 2

        if shape == "sphere":
            r = min(px_w, px_h) // 2
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=fill,
                outline=(40, 40, 40),
            )
        elif shape == "cylinder":
            # Top-down: ellipse (wide) + rectangle body
            ew = px_w // 2
            eh = px_w // 4  # foreshortened top
            draw.rectangle(
                [cx - ew, cy - px_h // 2 + eh, cx + ew, cy + px_h // 2],
                fill=fill,
                outline=(40, 40, 40),
            )
            draw.ellipse(
                [cx - ew, cy - px_h // 2, cx + ew, cy - px_h // 2 + eh * 2],
                fill=tuple(min(255, c + 30) for c in fill),
                outline=(40, 40, 40),
            )
        elif shape == "pyramid":
            # Triangle
            half_w = px_w // 2
            points = [
                (cx, cy - px_h // 2),         # apex
                (cx - half_w, cy + px_h // 2), # bottom-left
                (cx + half_w, cy + px_h // 2), # bottom-right
            ]
            draw.polygon(points, fill=fill, outline=(40, 40, 40))
        elif shape == "l_bracket":
            # L-shape: vertical bar + horizontal bar
            bar_w = px_w // 3
            # Vertical bar (left side)
            draw.rectangle(
                [cx - px_w // 2, cy - px_h // 2,
                 cx - px_w // 2 + bar_w, cy + px_h // 2],
                fill=fill,
                outline=(40, 40, 40),
            )
            # Horizontal bar (bottom)
            draw.rectangle(
                [cx - px_w // 2, cy + px_h // 2 - bar_w,
                 cx + px_w // 2, cy + px_h // 2],
                fill=fill,
                outline=(40, 40, 40),
            )
        else:
            # Default: cube (rectangle)
            draw.rectangle(
                [cx - px_w // 2, cy - px_h // 2,
                 cx + px_w // 2, cy + px_h // 2],
                fill=fill,
                outline=(40, 40, 40),
            )

        # Label
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10
            )
        except (IOError, OSError):
            font = ImageFont.load_default()

        draw.text((4, 4), object_id, fill=(20, 20, 20), font=font)

        out_path = obj_dir / f"test_{i:04d}.png"
        img.save(out_path)
        saved.append(out_path)

    return saved


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def stage_organize(
    objects: Dict[str, dict],
    output_dir: Path,
    dry_run: bool = False,
) -> Dict[str, dict]:
    """Stage 1: ORGANIZE — Validate objects, create directory structure.

    Returns a manifest dict keyed by object_id.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: ORGANIZE — %d objects", len(objects))
    logger.info("=" * 60)

    manifest: Dict[str, dict] = {}
    for oid, obj_def in objects.items():
        entry = {
            "object_id": oid,
            "name": obj_def["name"],
            "definition": obj_def,
            "status": "organized",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if not dry_run:
            obj_dir = output_dir / oid
            obj_dir.mkdir(parents=True, exist_ok=True)
            entry["output_dir"] = str(obj_dir)

        manifest[oid] = entry
        logger.info("  [ORGANIZE] %s — %s", oid, obj_def["name"])

    return manifest


def stage_hash(
    manifest: Dict[str, dict],
    output_dir: Path,
    hasher: CompositeHasher,
    images_per_object: int = 3,
    dry_run: bool = False,
) -> Dict[str, dict]:
    """Stage 2: HASH — Generate images and compute composite perceptual hashes."""
    logger.info("=" * 60)
    logger.info("STAGE 2: HASH — Computing perceptual hashes")
    logger.info("=" * 60)

    for oid, entry in manifest.items():
        obj_def = entry["definition"]
        t0 = time.perf_counter()

        if dry_run:
            entry["images"] = []
            entry["hashes"] = {"dhash": "DRY_RUN", "phash": "DRY_RUN"}
            entry["status"] = "hashed"
            logger.info("  [HASH] %s — dry run", oid)
            continue

        # Generate synthetic images
        images = generate_primitive_images(
            oid, obj_def, output_dir, count=images_per_object
        )
        entry["images"] = [str(p) for p in images]

        # Hash the first image as the canonical representative
        canonical_img = str(images[0])
        composite = hasher.compute_composite(canonical_img)
        content_sha = hasher.content_hash_sha256(canonical_img)

        entry["hashes"] = {
            "dhash_hex": composite.dhash_hex,
            "phash_hex": composite.phash_hex,
            "ppp_256bit": composite.ppp_256bit,
            "color_hash": composite.color_hash,
            "content_sha256": content_sha,
        }

        # Also hash additional images for multi-view registration
        entry["additional_hashes"] = []
        for img_path in images[1:]:
            c = hasher.compute_composite(str(img_path))
            entry["additional_hashes"].append({
                "image": str(img_path),
                "dhash_hex": c.dhash_hex,
            })

        elapsed = (time.perf_counter() - t0) * 1000
        entry["hash_time_ms"] = round(elapsed, 2)
        entry["status"] = "hashed"
        logger.info(
            "  [HASH] %s — dHash=%s (%.1f ms, %d images)",
            oid, composite.dhash_hex[:10] + "...", elapsed, len(images),
        )

    return manifest


def stage_enrich(
    manifest: Dict[str, dict],
    workspace_zone: str = "evaluation_tabletop",
    end_effector: str = "franka_hand",
    dry_run: bool = False,
) -> Dict[str, dict]:
    """Stage 3: ENRICH — Build full 4-layer SKBs from ground-truth seeds."""
    logger.info("=" * 60)
    logger.info("STAGE 3: ENRICH — Building Spatial Kinematic Blueprints")
    logger.info("=" * 60)

    for oid, entry in manifest.items():
        obj_def = entry["definition"]
        t0 = time.perf_counter()

        if dry_run:
            entry["skb"] = {"dry_run": True}
            entry["status"] = "enriched"
            logger.info("  [ENRICH] %s — dry run", oid)
            continue

        hashes = entry["hashes"]
        dims = obj_def["dimensions_m"]
        mass_kg = obj_def["mass_kg"]
        material = obj_def["material"]
        category = obj_def.get("category", "block")

        # Determine action primitive
        primitive_str = obj_def.get("action_primitive", "PICK_VERTICAL")
        primitive = ActionPrimitive(primitive_str)

        # Determine hazard class
        hazard_str = obj_def.get("hazard_class", "NONE")
        try:
            hazard = HazardClass(hazard_str)
        except ValueError:
            hazard = _MATERIAL_TO_HAZARD.get(material, HazardClass.NONE)

        # Classification
        classification = _CATEGORY_TO_CLASSIFICATION.get(
            category, ObjectClassification.RIGID_BODY_GRASPABLE
        )

        # Grip force: clamp to [10N, 50N]
        grip_force = round(max(10.0, min(50.0, mass_kg * 40.0)), 1)

        # Friction
        friction = _MATERIAL_TO_FRICTION.get(material, 0.4)

        now = datetime.now(timezone.utc)
        skb_uuid = str(_uuid.uuid4())

        # Build full Pydantic SKB
        skb = SpatialKinematicBlueprint(
            schema_version="2.0-GCP-ROBOTICS",
            agent="Batch_Registry_Builder",
            timestamps=Timestamps(
                state_encountered=now,
                schema_generated=now,
            ),
            identifiers=Identifiers(
                uuid=skb_uuid,
                artifact_id=f"geo-{oid}",
                codex_id=f"GCP-GEO-{oid.upper()}",
            ),
            triple_hash=TripleHash(
                content_hash_sha256=hashes["content_sha256"],
                ppp_256bit=_strip_0x(hashes["ppp_256bit"]),
                soulmark_sha256="pending",  # Computed in Stage 4
                fast_path_dhash_64bit=_strip_0x(hashes["dhash_hex"]),
            ),
            layer_1_provenance=ProvenanceLayer(
                manufacturer="Standard Evaluation Template",
                model_no=oid,
                serial=f"geo-{oid}-{skb_uuid[:8]}",
                workspace_zone=workspace_zone,
                source_dataset="geometric_primitives_v1",
                enrichment_timestamp=now,
                enrichment_model_version="batch-registry-builder-1.0",
            ),
            layer_2_semantic_topology=SemanticTopologyLayer(
                object_classification=classification,
                physical_properties=PhysicalProperties(
                    estimated_mass_kg=mass_kg,
                    friction_coefficient=friction,
                    dimensions=XYZ(x=dims[0], y=dims[1], z=dims[2]),
                ),
                material_graph=MaterialGraph(
                    primary_material=material,
                    component_count=1,
                ),
                perceptual_signature=PerceptualSignature(
                    dhash_hex=_strip_0x(hashes["dhash_hex"]),
                    phash_hex=_strip_0x(hashes["phash_hex"]),
                    color_histogram_hash=hashes.get("color_hash", ""),
                    variance_tolerance_pct=5.0,
                    optical_flow_stable=True,
                ),
            ),
            layer_3_affordance=AffordanceLayer(
                action_schema=ActionSchema(
                    primitive_id=primitive,
                    parameters=ActionParameters(
                        target_pose_relative=TargetPoseRelative(
                            x=0.0, y=0.0, z=-0.05,
                            qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                        ),
                        force_profile=ForceProfile(
                            max_grip_newtons=grip_force,
                            lateral_torque_limit_nm=round(mass_kg * 5.0, 1),
                        ),
                        speed_profile=SpeedProfile.FINE,
                    ),
                    failure_modes=[
                        "Object slipped during grasp",
                        f"Orientation error > 15 deg ({obj_def.get('notes', '')})",
                    ],
                ),
            ),
            layer_4_safety=SafetyLayer(
                hazard_class=hazard,
                ppe_required=["safety_glasses"] if hazard != HazardClass.NONE else [],
                operating_constraints=OperatingConstraints(
                    max_speed_mm_s=250.0,
                    max_force_n=grip_force,
                ),
            ),
            telemetry=Telemetry(
                target_controller="franka_panda_eval_cell",
                transport_layer="ROS2_Humble",
                end_effector=end_effector,
                pipeline_status=PipelineStatus.FAST_PATH_ACTIVE,
            ),
            control_loop=ControlLoop(
                vision_hashed=True,
                ram_cache_updated=True,
            ),
        )

        entry["skb"] = skb.model_dump(mode="json", by_alias=True)
        entry["skb_uuid"] = skb_uuid

        elapsed = (time.perf_counter() - t0) * 1000
        entry["enrich_time_ms"] = round(elapsed, 2)
        entry["status"] = "enriched"
        logger.info(
            "  [ENRICH] %s — %s, %s, grip=%.0fN (%.1f ms)",
            oid, classification.value, primitive.value, grip_force, elapsed,
        )

    return manifest


def stage_sign(
    manifest: Dict[str, dict],
    fleet_secret: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, dict]:
    """Stage 4: SIGN — Compute Soulmarks for each SKB."""
    logger.info("=" * 60)
    logger.info("STAGE 4: SIGN — Computing Soulmarks")
    logger.info("=" * 60)

    verifier = SoulmarkVerifier(fleet_secret=fleet_secret)

    for oid, entry in manifest.items():
        if dry_run:
            entry["soulmark"] = "DRY_RUN"
            entry["status"] = "signed"
            logger.info("  [SIGN] %s — dry run", oid)
            continue

        skb_data = entry["skb"]

        # Compute Soulmark over safety-critical fields
        signed = verifier.sign(skb_data)
        soulmark = signed.soulmark

        # Update the SKB's triple_hash with the real soulmark
        skb_data["triple_hash"]["soulmark_sha256"] = soulmark

        entry["soulmark"] = soulmark
        entry["soulmark_method"] = "HMAC-SHA256" if fleet_secret else "SHA-256"
        entry["status"] = "signed"

        logger.info(
            "  [SIGN] %s — %s (%s)",
            oid, soulmark[:16] + "...", entry["soulmark_method"],
        )

    return manifest


def stage_register(
    manifest: Dict[str, dict],
    registry: CodexRegistry,
    hasher: CompositeHasher,
    dry_run: bool = False,
) -> Dict[str, dict]:
    """Stage 5: REGISTER — Populate CodexRegistry with hash-indexed SKBs."""
    logger.info("=" * 60)
    logger.info("STAGE 5: REGISTER — Populating CodexRegistry")
    logger.info("=" * 60)

    total_registered = 0

    for oid, entry in manifest.items():
        if dry_run:
            entry["registered_count"] = 0
            entry["status"] = "registered"
            logger.info("  [REGISTER] %s — dry run", oid)
            continue

        hashes = entry["hashes"]
        skb_data = entry["skb"]

        # Register canonical image hash
        registry.register(
            dhash_hex=hashes["dhash_hex"],
            skb_data=skb_data,
            phash_hex=hashes.get("phash_hex"),
            ppp_256=hashes.get("ppp_256bit"),
        )
        total_registered += 1

        # Register additional image hashes (same SKB, different viewpoints)
        for ah in entry.get("additional_hashes", []):
            registry.register(
                dhash_hex=ah["dhash_hex"],
                skb_data=skb_data,
            )
            total_registered += 1

        entry["registered_count"] = 1 + len(entry.get("additional_hashes", []))
        entry["status"] = "registered"

        # Verify round-trip lookup
        result = registry.lookup(hashes["dhash_hex"])
        entry["lookup_verified"] = result.match_type == "EXACT"
        entry["lookup_time_ms"] = result.lookup_time_ms

        logger.info(
            "  [REGISTER] %s — %d entries, lookup: %s (%.3f ms)",
            oid, entry["registered_count"], result.match_type, result.lookup_time_ms,
        )

    logger.info("  Total registry entries: %d", total_registered)
    return manifest


def stage_validate(
    manifest: Dict[str, dict],
    output_dir: Path,
    dry_run: bool = False,
) -> Dict[str, dict]:
    """Stage 6: VALIDATE — Run codex-lab-kit 4-phase experiment protocol."""
    logger.info("=" * 60)
    logger.info("STAGE 6: VALIDATE — Generating experiment protocol")
    logger.info("=" * 60)

    if dry_run:
        logger.info("  [VALIDATE] dry run — skipping")
        return manifest

    try:
        from codex_lab_kit import ExperimentProtocol
    except ImportError:
        logger.warning(
            "  [VALIDATE] codex-lab-kit not installed. "
            "Install with: pip install codex-lab-kit"
        )
        return manifest

    object_ids = list(manifest.keys())

    protocol = ExperimentProtocol(
        lab_name="Batch Registry Builder",
        lab_contact="research@iaeternum.ai",
        robot_model="Franka Emika Panda",
        gripper_model="Franka Hand (parallel jaw)",
        camera_model="Intel RealSense D435",
        workspace_zone="evaluation_tabletop",
        object_ids=object_ids,
    )
    protocol_data = protocol.generate_full_protocol()

    # Write protocol
    protocol_path = output_dir / "experiment_protocol.json"
    protocol_path.write_text(json.dumps(protocol_data, indent=2, default=str))

    total_trials = protocol_data.get("summary", {}).get("total_trials", 0)

    num_phases = protocol_data.get("summary", {}).get("total_phases", 0)
    logger.info(
        "  [VALIDATE] Protocol generated: %d phases, %d total trials",
        num_phases,
        total_trials,
    )
    logger.info("  [VALIDATE] Saved to %s", protocol_path)

    return manifest


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    manifest: Dict[str, dict],
    output_dir: Path,
    template_name: str,
    start_time: float,
) -> dict:
    """Generate a batch report summarizing the pipeline run."""
    elapsed = time.perf_counter() - start_time

    report = {
        "batch_report": {
            "template": template_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_objects": len(manifest),
            "total_elapsed_s": round(elapsed, 2),
            "stages_completed": 6,
        },
        "objects": {},
    }

    completed = 0
    for oid, entry in manifest.items():
        summary = {
            "name": entry["name"],
            "status": entry.get("status", "unknown"),
            "soulmark": entry.get("soulmark", "n/a")[:16] + "..." if entry.get("soulmark") else "n/a",
            "registered_count": entry.get("registered_count", 0),
            "lookup_verified": entry.get("lookup_verified", False),
            "lookup_time_ms": entry.get("lookup_time_ms", 0),
            "hash_time_ms": entry.get("hash_time_ms", 0),
            "enrich_time_ms": entry.get("enrich_time_ms", 0),
        }
        report["objects"][oid] = summary
        if entry.get("status") == "registered":
            completed += 1

    report["batch_report"]["completed"] = completed
    report["batch_report"]["success_rate"] = (
        round(completed / len(manifest) * 100, 1) if manifest else 0
    )

    # Write report
    report_path = output_dir / f"batch_report_{int(time.time())}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Batch report saved to %s", report_path)

    return report


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def load_template(template_name: str) -> Dict[str, dict]:
    """Load a template by name."""
    if template_name == "ycb_20":
        from gcp_robotics.datasets.ycb_loader import YCB_OBJECTS
        return dict(YCB_OBJECTS)
    elif template_name in TEMPLATES:
        return dict(TEMPLATES[template_name])
    else:
        raise ValueError(
            f"Unknown template: {template_name!r}. "
            f"Available: {list(TEMPLATES.keys()) + ['ycb_20']}"
        )


def run_pipeline(
    template_name: str = "standard_evaluation",
    output_dir: str = None,
    start_stage: str = "organize",
    images_per_object: int = 3,
    fleet_secret: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Run the 6-stage batch registry builder pipeline.

    Parameters
    ----------
    template_name : str
        Template to use: "standard_evaluation" (geometric primitives) or "ycb_20".
    output_dir : str, optional
        Output directory. Defaults to ``data/{template_name}/``.
    start_stage : str
        Resume from this stage (organize, hash, enrich, sign, register, validate).
    images_per_object : int
        Number of synthetic images to generate per object.
    fleet_secret : str, optional
        HMAC secret for Soulmark signing.
    dry_run : bool
        Preview pipeline without writing files.
    """
    start_time = time.perf_counter()
    start_idx = STAGE_NAMES.index(start_stage) if start_stage in STAGE_NAMES else 0

    # Resolve output directory
    if output_dir is None:
        output_dir = str(_SDK_ROOT / "data" / template_name)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BATCH REGISTRY BUILDER")
    logger.info("  Template: %s", template_name)
    logger.info("  Output:   %s", out)
    logger.info("  Start:    Stage %d (%s)", start_idx + 1, start_stage)
    logger.info("  Dry run:  %s", dry_run)
    logger.info("=" * 60)

    # Load template
    objects = load_template(template_name)
    logger.info("Loaded %d objects from template %r", len(objects), template_name)

    # Initialize shared resources
    hasher = CompositeHasher()
    registry = CodexRegistry()

    # --- Stage 1: ORGANIZE ---
    manifest: Dict[str, dict] = {}
    if start_idx <= 0:
        manifest = stage_organize(objects, out, dry_run=dry_run)

    # --- Stage 2: HASH ---
    if start_idx <= 1:
        manifest = stage_hash(
            manifest, out, hasher,
            images_per_object=images_per_object,
            dry_run=dry_run,
        )

    # --- Stage 3: ENRICH ---
    if start_idx <= 2:
        manifest = stage_enrich(manifest, dry_run=dry_run)

    # --- Stage 4: SIGN ---
    if start_idx <= 3:
        manifest = stage_sign(manifest, fleet_secret=fleet_secret, dry_run=dry_run)

    # --- Stage 5: REGISTER ---
    if start_idx <= 4:
        manifest = stage_register(manifest, registry, hasher, dry_run=dry_run)

    # --- Stage 6: VALIDATE ---
    if start_idx <= 5:
        manifest = stage_validate(manifest, out, dry_run=dry_run)

    # --- Save registry ---
    if not dry_run:
        registry_path = out / f"{template_name}_registry.json"
        registry.save(str(registry_path))
        logger.info("Registry saved to %s", registry_path)

        # Save individual SKB JSON-LD files
        skbs_dir = out / "skbs"
        skbs_dir.mkdir(parents=True, exist_ok=True)
        for oid, entry in manifest.items():
            if "skb" in entry and not isinstance(entry["skb"].get("dry_run"), bool):
                skb_path = skbs_dir / f"{oid}_skb.json"
                skb_path.write_text(json.dumps(entry["skb"], indent=2, default=str))

    # --- Report ---
    report = generate_report(manifest, out, template_name, start_time)

    # --- Summary ---
    elapsed = time.perf_counter() - start_time
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Template:     {template_name}")
    print(f"  Objects:      {len(manifest)}")
    print(f"  Success rate: {report['batch_report']['success_rate']}%")
    print(f"  Total time:   {elapsed:.2f}s")
    print(f"  Output:       {out}")
    if not dry_run:
        print(f"  Registry:     {out / f'{template_name}_registry.json'}")
    print("=" * 60)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build a PCO registry from a template definition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Templates:
  standard_evaluation   5 geometric primitives (cube, sphere, cylinder, pyramid, L-bracket)
  ycb_20                20 YCB benchmark objects with measured ground truth

Examples:
  %(prog)s --template standard_evaluation
  %(prog)s --template ycb_20 --images 5
  %(prog)s --template standard_evaluation --dry-run
  %(prog)s --template ycb_20 --start-stage enrich
""",
    )
    parser.add_argument(
        "--template", "-t",
        default="standard_evaluation",
        help="Template name (default: standard_evaluation)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: data/<template>/)",
    )
    parser.add_argument(
        "--start-stage", "-s",
        default="organize",
        choices=STAGE_NAMES,
        help="Resume from this stage (default: organize)",
    )
    parser.add_argument(
        "--images", "-i",
        type=int,
        default=3,
        help="Images per object (default: 3)",
    )
    parser.add_argument(
        "--fleet-secret",
        default=None,
        help="HMAC secret for Soulmark signing (default: plain SHA-256)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview pipeline without writing files",
    )

    args = parser.parse_args()

    run_pipeline(
        template_name=args.template,
        output_dir=args.output,
        start_stage=args.start_stage,
        images_per_object=args.images,
        fleet_secret=args.fleet_secret,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

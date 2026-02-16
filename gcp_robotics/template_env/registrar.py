"""
Template Environment Registrar — pre-hash every object in a workspace.

Implements the PPA (Pre-hashed Perceptual Atlas) concept from the Golden Codex
Protocol: before any robot begins operating in a workspace, **every known object
is hashed and registered** so that the Fast Path can resolve any encounter via
O(1) lookup with zero environment-specific training.

Workflow
--------
1.  Instantiate ``TemplateEnvironmentRegistrar`` for a named workspace zone.
2.  Call ``register_full_environment()`` with a ``YCBLoader`` (or register
    individual objects with ``register_object_from_image`` / ``register_ycb_object``).
3.  Retrieve the populated ``CodexRegistry`` via ``get_registry()`` and hand it
    to the Fast Path runtime.

Copyright (c) 2026 Metavolve Labs -- Robotics R&D Division
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry
from gcp_robotics.datasets.ycb_loader import YCBLoader
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
    ValidationSignature,
    OnFailureAction,
    ExecutionAct,
    HazardClass,
    OperatingConstraints,
    Telemetry,
    PipelineStatus,
    ControlLoop,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — default builders for each SKB layer
# ---------------------------------------------------------------------------

_CATEGORY_TO_CLASSIFICATION: Dict[str, ObjectClassification] = {
    "food_container": ObjectClassification.RIGID_BODY_GRASPABLE,
    "food_box": ObjectClassification.RIGID_BODY_GRASPABLE,
    "condiment": ObjectClassification.LIQUID_CONTAINER,
    "fruit_model": ObjectClassification.RIGID_BODY_GRASPABLE,
    "kitchen_tool": ObjectClassification.RIGID_BODY_GRASPABLE,
    "cleaning": ObjectClassification.LIQUID_CONTAINER,
    "tableware": ObjectClassification.RIGID_BODY_GRASPABLE,
    "tool": ObjectClassification.TOOL,
    "block": ObjectClassification.RIGID_BODY_GRASPABLE,
    "stationery": ObjectClassification.RIGID_BODY_GRASPABLE,
}

_CATEGORY_TO_PRIMITIVE: Dict[str, ActionPrimitive] = {
    "food_container": ActionPrimitive.PICK_VERTICAL,
    "food_box": ActionPrimitive.PICK_VERTICAL,
    "condiment": ActionPrimitive.PICK_VERTICAL,
    "fruit_model": ActionPrimitive.GRASP_PARALLEL_JAW,
    "kitchen_tool": ActionPrimitive.PICK_VERTICAL,
    "cleaning": ActionPrimitive.PICK_VERTICAL,
    "tableware": ActionPrimitive.PICK_VERTICAL,
    "tool": ActionPrimitive.GRASP_PARALLEL_JAW,
    "block": ActionPrimitive.PICK_VERTICAL,
    "stationery": ActionPrimitive.GRASP_PARALLEL_JAW,
}

_MATERIAL_TO_HAZARD: Dict[str, HazardClass] = {
    "metal": HazardClass.SHARP_EDGES,
    "metal_plastic": HazardClass.SHARP_EDGES,
    "plastic_metal": HazardClass.SHARP_EDGES,
    "ceramic": HazardClass.NONE,
    "plastic": HazardClass.NONE,
    "cardboard": HazardClass.NONE,
    "cardboard_metal": HazardClass.NONE,
    "wood": HazardClass.NONE,
}


def _strip_0x(hex_str: str) -> str:
    """Strip the ``0x`` prefix from a hex hash string and lowercase it."""
    return hex_str.lower().removeprefix("0x")


def _build_skb(
    composite_hash,
    content_sha256: str,
    skb_seed: dict,
    workspace_zone: str,
    end_effector: str,
    manufacturer: str,
    model_no: str,
    serial: str,
    object_id: str,
) -> SpatialKinematicBlueprint:
    """Construct a fully-populated SpatialKinematicBlueprint from seed data."""

    now = datetime.now(timezone.utc)
    skb_uuid = str(_uuid.uuid4())
    codex_id = f"GCP-{object_id}-{skb_uuid[:8]}"

    # --- Extract seed data ------------------------------------------------
    l2_seed = skb_seed.get("layer_2_semantic_topology", {})
    phys_seed = l2_seed.get("physical_properties", {})
    category = l2_seed.get("category", "food_container")
    material = phys_seed.get("material", "plastic")

    # --- Perceptual signature (Layer 2 sub-model) -------------------------
    dhash_16 = _strip_0x(composite_hash.dhash_hex)
    phash_16 = _strip_0x(composite_hash.phash_hex)

    perceptual_sig = PerceptualSignature(
        dhash_hex=dhash_16,
        phash_hex=phash_16,
        color_histogram_hash=composite_hash.color_hash,
        variance_tolerance_pct=5.0,
        optical_flow_stable=True,
    )

    # --- Physical properties (Layer 2 sub-model) --------------------------
    bbox = phys_seed.get("bounding_box_m", {})
    physical_props = PhysicalProperties(
        estimated_mass_kg=phys_seed.get("mass_kg"),
        friction_coefficient=phys_seed.get("friction_coefficient"),
        dimensions=XYZ(
            x=bbox.get("x", 0.0),
            y=bbox.get("y", 0.0),
            z=bbox.get("z", 0.0),
        ),
    )

    material_graph = MaterialGraph(
        primary_material=material,
        component_count=1,
    )

    classification = _CATEGORY_TO_CLASSIFICATION.get(
        category, ObjectClassification.RIGID_BODY_GRASPABLE
    )

    # --- Action primitive (Layer 3 sub-model) -----------------------------
    primitive = _CATEGORY_TO_PRIMITIVE.get(
        category, ActionPrimitive.PICK_VERTICAL
    )

    mass_kg = phys_seed.get("mass_kg", 0.5)
    grip_force = max(10.0, min(50.0, mass_kg * 40.0))

    action_schema = ActionSchema(
        primitive_id=primitive,
        parameters=ActionParameters(
            target_pose_relative=TargetPoseRelative(
                x=0.0, y=0.0, z=-0.05, qx=0.0, qy=0.0, qz=0.0, qw=1.0
            ),
            force_profile=ForceProfile(
                max_grip_newtons=round(grip_force, 1),
                lateral_torque_limit_nm=10.0,
                compliance_stiffness=XYZ(x=500.0, y=500.0, z=1000.0),
            ),
            speed_profile=SpeedProfile.FINE,
        ),
        validation_signature=ValidationSignature(
            success_tolerance_bits=6,
            on_failure=OnFailureAction.RETRY,
        ),
        execution_narrative=[
            ExecutionAct(
                time="T+0.0s",
                state="APPROACH",
                description=f"Approach {object_id} in zone {workspace_zone}.",
            ),
            ExecutionAct(
                time="T+1.0s",
                state="CONTACT",
                description=f"Initiate {primitive.value} with {end_effector}.",
            ),
            ExecutionAct(
                time="T+2.5s",
                state="EXECUTE",
                description=f"Execute {primitive.value} on {object_id}.",
            ),
            ExecutionAct(
                time="T+4.0s",
                state="VERIFY",
                description="Verify post-action state via perceptual hash.",
            ),
        ],
        failure_modes=[
            "Object slipped during grasp",
            "Unexpected collision with neighbour",
            "Post-action hash mismatch",
        ],
    )

    # --- Hazard (Layer 4) -------------------------------------------------
    hazard = _MATERIAL_TO_HAZARD.get(material, HazardClass.NONE)

    # --- Triple hash block ------------------------------------------------
    # Build a temporary canonical payload for soulmark computation
    ppp_256 = _strip_0x(composite_hash.ppp_256bit)
    soulmark_input = f"{content_sha256}:{ppp_256}:{dhash_16}"
    soulmark = hashlib.sha256(soulmark_input.encode("utf-8")).hexdigest()

    triple_hash = TripleHash(
        content_hash_sha256=content_sha256,
        ppp_256bit=ppp_256,
        soulmark_sha256=soulmark,
        fast_path_dhash_64bit=dhash_16,
    )

    # --- Assemble SKB -----------------------------------------------------
    skb = SpatialKinematicBlueprint(
        schema_version="2.0-GCP-ROBOTICS",
        agent="Template_Environment_Registrar",
        timestamps=Timestamps(
            state_encountered=now,
            schema_generated=now,
        ),
        identifiers=Identifiers(
            uuid=skb_uuid,
            artifact_id=f"ycb-{object_id}",
            codex_id=codex_id,
        ),
        triple_hash=triple_hash,
        layer_1_provenance=ProvenanceLayer(
            manufacturer=manufacturer,
            model_no=model_no,
            serial=serial,
            workspace_zone=workspace_zone,
            source_dataset=skb_seed.get("source", "unknown"),
            enrichment_timestamp=now,
            enrichment_model_version="template-env-registrar-1.0",
        ),
        layer_2_semantic_topology=SemanticTopologyLayer(
            object_classification=classification,
            physical_properties=physical_props,
            material_graph=material_graph,
            perceptual_signature=perceptual_sig,
        ),
        layer_3_affordance=AffordanceLayer(
            action_schema=action_schema,
        ),
        layer_4_safety=SafetyLayer(
            hazard_class=hazard,
            ppe_required=["safety_glasses"] if hazard != HazardClass.NONE else [],
            regulatory_certifications=["ISO 10218-1"],
            operating_constraints=OperatingConstraints(
                max_speed_mm_s=250.0,
                max_force_n=grip_force,
            ),
        ),
        telemetry=Telemetry(
            target_controller="template_env_controller",
            end_effector=end_effector,
            pipeline_status=PipelineStatus.FAST_PATH_ACTIVE,
        ),
        control_loop=ControlLoop(
            vision_hashed=True,
            ram_cache_updated=True,
        ),
    )

    return skb


# ---------------------------------------------------------------------------
# TemplateEnvironmentRegistrar
# ---------------------------------------------------------------------------


class TemplateEnvironmentRegistrar:
    """Pre-hash every object in a workspace for zero-shot Fast Path operation.

    Parameters
    ----------
    environment_name : str
        Human-readable name for this environment (e.g. ``"Lab Cell A"``).
    workspace_zone : str
        Logical zone identifier passed through to every generated SKB.
    registry : CodexRegistry, optional
        An existing registry instance.  If ``None``, a new one is created.
    hasher : CompositeHasher, optional
        An existing hasher instance.  If ``None``, a new one is created.
    """

    def __init__(
        self,
        environment_name: str,
        workspace_zone: str,
        registry: CodexRegistry = None,
        hasher: CompositeHasher = None,
    ) -> None:
        self.environment_name = environment_name
        self.workspace_zone = workspace_zone
        self.registry = registry or CodexRegistry()
        self.hasher = hasher or CompositeHasher()

        # Internal manifest data
        self._registered_objects: Dict[str, List[dict]] = {}

        logger.info(
            "TemplateEnvironmentRegistrar initialised: name=%r, zone=%r",
            environment_name,
            workspace_zone,
        )

    # -- Single image registration -----------------------------------------

    def register_object_from_image(
        self,
        image_path: str,
        object_id: str,
        skb_seed: dict,
        end_effector: str = "robotiq_2f85",
        manufacturer: str = "Unknown",
        model_no: str = "Unknown",
        serial: str = "Unknown",
    ) -> dict:
        """Hash an image, build a full SKB, and register it in the Fast Path.

        Parameters
        ----------
        image_path : str
            Filesystem path to the object image.
        object_id : str
            Unique identifier for the object (e.g. ``"001_chips_can"``).
        skb_seed : dict
            Partial SKB dictionary from ``YCBLoader.build_skb_seed()`` or
            equivalent, containing ground-truth physical properties.
        end_effector : str
            End-effector descriptor for the robot.
        manufacturer, model_no, serial : str
            Provenance fields for Layer 1.

        Returns
        -------
        dict
            The complete SKB serialised as a dictionary.
        """
        # Compute composite perceptual hash
        composite = self.hasher.compute_composite(image_path)

        # Compute content SHA-256
        content_sha = self.hasher.content_hash_sha256(image_path)

        # Build full SKB
        skb = _build_skb(
            composite_hash=composite,
            content_sha256=content_sha,
            skb_seed=skb_seed,
            workspace_zone=self.workspace_zone,
            end_effector=end_effector,
            manufacturer=manufacturer,
            model_no=model_no,
            serial=serial,
            object_id=object_id,
        )

        # Serialise to dict for registry storage
        skb_dict = skb.model_dump(mode="json", by_alias=True)

        # Register in the CodexRegistry
        self.registry.register(
            dhash_hex=composite.dhash_hex,
            skb_data=skb_dict,
            phash_hex=composite.phash_hex,
            ppp_256=composite.ppp_256bit,
        )

        # Track in internal manifest
        if object_id not in self._registered_objects:
            self._registered_objects[object_id] = []
        self._registered_objects[object_id].append(skb_dict)

        logger.info(
            "Registered %s from %s (dHash=%s)",
            object_id,
            image_path,
            composite.dhash_hex,
        )
        return skb_dict

    # -- YCB convenience ---------------------------------------------------

    def register_ycb_object(
        self,
        loader: YCBLoader,
        object_id: str,
        end_effector: str = "robotiq_2f85",
    ) -> List[dict]:
        """Generate test images for a YCB object and register each one.

        Parameters
        ----------
        loader : YCBLoader
            Loader instance used to generate test images and build SKB seeds.
        object_id : str
            YCB object identifier (e.g. ``"001_chips_can"``).
        end_effector : str
            End-effector descriptor.

        Returns
        -------
        list[dict]
            List of registered SKB dictionaries (one per test image).
        """
        images = loader.generate_test_images(object_id, count=3)
        skb_seed = loader.build_skb_seed(object_id)
        gt = loader.get_ground_truth(object_id)

        registered: List[dict] = []
        for img_path in images:
            skb_dict = self.register_object_from_image(
                image_path=str(img_path),
                object_id=object_id,
                skb_seed=skb_seed,
                end_effector=end_effector,
                manufacturer="YCB_Dataset",
                model_no=object_id,
                serial=f"ycb-{object_id}-{img_path.stem}",
            )
            registered.append(skb_dict)

        logger.info(
            "Registered YCB object %s: %d images", object_id, len(registered)
        )
        return registered

    # -- Full environment registration -------------------------------------

    def register_full_environment(
        self,
        loader: YCBLoader,
        object_ids: List[str] = None,
        images_per_object: int = 3,
        end_effector: str = "robotiq_2f85",
    ) -> dict:
        """Register multiple YCB objects into the environment template.

        Parameters
        ----------
        loader : YCBLoader
            Loader instance for image generation and ground-truth data.
        object_ids : list[str], optional
            Object identifiers to register.  If ``None``, all 20 YCB objects
            are registered.
        images_per_object : int
            Number of synthetic test images per object (default 3).
        end_effector : str
            End-effector descriptor.

        Returns
        -------
        dict
            Manifest with environment metadata and registration statistics.
        """
        if object_ids is None:
            object_ids = list(loader.get_all_objects().keys())

        total_registered = 0
        object_summaries: Dict[str, dict] = {}

        for oid in object_ids:
            images = loader.generate_test_images(oid, count=images_per_object)
            skb_seed = loader.build_skb_seed(oid)

            obj_skbs: List[dict] = []
            for img_path in images:
                skb_dict = self.register_object_from_image(
                    image_path=str(img_path),
                    object_id=oid,
                    skb_seed=skb_seed,
                    end_effector=end_effector,
                    manufacturer="YCB_Dataset",
                    model_no=oid,
                    serial=f"ycb-{oid}-{img_path.stem}",
                )
                obj_skbs.append(skb_dict)
                total_registered += 1

            object_summaries[oid] = {
                "images_registered": len(obj_skbs),
                "ground_truth": loader.get_ground_truth(oid),
            }

        manifest = {
            "environment_name": self.environment_name,
            "workspace_zone": self.workspace_zone,
            "total_objects": len(object_ids),
            "total_images_registered": total_registered,
            "images_per_object": images_per_object,
            "objects": object_summaries,
            "registry_stats": self.registry.stats(),
        }

        logger.info(
            "Full environment registered: %d objects, %d total images",
            len(object_ids),
            total_registered,
        )
        return manifest

    # -- Export ------------------------------------------------------------

    def export_manifest(self, filepath: str) -> None:
        """Save the environment manifest to a JSON file.

        Parameters
        ----------
        filepath : str
            Destination path for the manifest JSON.
        """
        manifest = {
            "environment_name": self.environment_name,
            "workspace_zone": self.workspace_zone,
            "registered_objects": {
                oid: len(skbs) for oid, skbs in self._registered_objects.items()
            },
            "total_images": sum(
                len(skbs) for skbs in self._registered_objects.values()
            ),
            "registry_stats": self.registry.stats(),
        }

        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, default=str))
        logger.info("Environment manifest exported to %s", filepath)

    # -- Accessor ----------------------------------------------------------

    def get_registry(self) -> CodexRegistry:
        """Return the underlying ``CodexRegistry`` for Fast Path use."""
        return self.registry

"""
GCP-Robotics SKB API — Slow Path as a Service.

Upload an object image, get back a complete, signed Spatial Kinematic Blueprint.
Eliminates days of manual URDF/grasp planning with a single API call.

Endpoints
---------
POST /v1/skb/generate
    Upload an image → get a full 4-layer SKB (LLM-generated, schema-validated, signed).

POST /v1/skb/generate/mock
    Same interface but uses keyword heuristics instead of LLM. Free, no API key needed.

POST /v1/hash
    Upload an image → get composite perceptual hashes (dHash, pHash, PPP-256, color).

POST /v1/registry/lookup
    Submit a dHash → get the matching SKB from a loaded registry.

GET /v1/health
    Health check.

Usage
-----
    # Start the server
    uvicorn gcp_robotics.api:app --host 0.0.0.0 --port 8000

    # Or run directly
    python -m gcp_robotics.api

    # Generate an SKB from an image (mock mode, no API key needed)
    curl -X POST http://localhost:8000/v1/skb/generate/mock \\
      -F "image=@my_object.png" \\
      -F "workspace_zone=lab_bench" \\
      -F "end_effector=franka_hand"

Copyright (c) 2026 Metavolve Labs, Inc. — Robotics R&D Division
Patent Pending: U.S. Provisional Application No. 63/983,304 + 63/984,299
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import time
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gcp_robotics.api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GCP-Robotics SKB API",
    description=(
        "Upload an object image, get a robot-ready Spatial Kinematic Blueprint. "
        "Eliminates days of manual URDF/grasp planning with a single API call."
    ),
    version="2.0.1",
    contact={
        "name": "Metavolve Labs — Robotics R&D",
        "email": "research@iaeternum.ai",
        "url": "https://iaeternum.ai/robotics",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared resources (initialized once)
_hasher = CompositeHasher()
_verifier = SoulmarkVerifier(fleet_secret=os.environ.get("GCP_FLEET_SECRET"))
_registry = CodexRegistry()

# Load a pre-built registry if available
_REGISTRY_PATH = os.environ.get("GCP_REGISTRY_PATH")
if _REGISTRY_PATH and Path(_REGISTRY_PATH).exists():
    _registry.load(_REGISTRY_PATH)
    logger.info("Loaded registry from %s", _REGISTRY_PATH)


# ---------------------------------------------------------------------------
# Helper: strip 0x prefix
# ---------------------------------------------------------------------------

def _strip_0x(hex_str: str) -> str:
    return hex_str.lower().removeprefix("0x")


# ---------------------------------------------------------------------------
# Helper: image to temp file
# ---------------------------------------------------------------------------

async def _save_upload(upload: UploadFile) -> str:
    """Save an uploaded file to a temp path and return the path."""
    content = await upload.read()
    suffix = Path(upload.filename or "image.png").suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Helper: LLM response → full SKB
# ---------------------------------------------------------------------------

def _build_full_skb(
    llm_response: dict,
    composite_hash,
    content_sha256: str,
    workspace_zone: str,
    end_effector: str,
) -> dict:
    """Assemble a complete 4-layer SKB from LLM output + computed hashes."""
    now = datetime.now(timezone.utc)
    skb_uuid = str(_uuid.uuid4())

    dhash_16 = _strip_0x(composite_hash.dhash_hex)
    phash_16 = _strip_0x(composite_hash.phash_hex)
    ppp_256 = _strip_0x(composite_hash.ppp_256bit)

    # Soulmark from hashes
    soulmark_input = f"{content_sha256}:{ppp_256}:{dhash_16}"
    soulmark = hashlib.sha256(soulmark_input.encode("utf-8")).hexdigest()

    # Extract LLM fields with safe defaults
    primitive_str = llm_response.get("primitive_id", "PICK_VERTICAL")
    hazard_str = llm_response.get("hazard_class", "NONE")
    speed_str = llm_response.get("speed_profile", "FINE")
    confidence = llm_response.get("confidence_score", 0.5)
    reasoning = llm_response.get("reasoning_trace", "")

    force_profile = llm_response.get("force_profile", {})
    grip_force = force_profile.get("max_grip_newtons", 20.0)
    torque = force_profile.get("lateral_torque_limit_nm", 10.0)

    target_pose = llm_response.get("target_pose_relative", {})
    execution_narrative = llm_response.get("execution_narrative", [])
    failure_modes = llm_response.get("failure_modes", [])

    # Physical properties from LLM (estimated)
    phys = llm_response.get("physical_properties", {})
    mass_kg = phys.get("estimated_mass_kg", 0.3)
    material = phys.get("material", "plastic")
    dims = phys.get("dimensions", {})

    skb = SpatialKinematicBlueprint(
        schema_version="2.0-GCP-ROBOTICS",
        agent="SKB_API_Oracle",
        timestamps=Timestamps(
            state_encountered=now,
            schema_generated=now,
            promoted_to_fast_path=now,
        ),
        identifiers=Identifiers(
            uuid=skb_uuid,
            artifact_id=f"api-{skb_uuid[:8]}",
            codex_id=f"GCP-API-{skb_uuid[:8].upper()}",
        ),
        triple_hash=TripleHash(
            content_hash_sha256=content_sha256,
            ppp_256bit=ppp_256,
            soulmark_sha256=soulmark,
            fast_path_dhash_64bit=dhash_16,
        ),
        layer_1_provenance=ProvenanceLayer(
            manufacturer="Unknown (API submission)",
            model_no="unknown",
            serial=f"api-{skb_uuid[:8]}",
            workspace_zone=workspace_zone,
            source_dataset="user_upload",
            enrichment_timestamp=now,
            enrichment_model_version="skb-api-oracle-1.0",
        ),
        layer_2_semantic_topology=SemanticTopologyLayer(
            object_classification=ObjectClassification.RIGID_BODY_GRASPABLE,
            physical_properties=PhysicalProperties(
                estimated_mass_kg=mass_kg,
                friction_coefficient=0.4,
                dimensions=XYZ(
                    x=dims.get("x", 0.05),
                    y=dims.get("y", 0.05),
                    z=dims.get("z", 0.05),
                ),
            ),
            material_graph=MaterialGraph(
                primary_material=material,
                component_count=1,
            ),
            perceptual_signature=PerceptualSignature(
                dhash_hex=dhash_16,
                phash_hex=phash_16,
                color_histogram_hash=composite_hash.color_hash,
                variance_tolerance_pct=5.0,
                optical_flow_stable=True,
            ),
        ),
        layer_3_affordance=AffordanceLayer(
            action_schema=ActionSchema(
                primitive_id=ActionPrimitive(primitive_str),
                parameters=ActionParameters(
                    target_pose_relative=TargetPoseRelative(
                        x=target_pose.get("x", 0.0),
                        y=target_pose.get("y", 0.0),
                        z=target_pose.get("z", -0.05),
                        qx=target_pose.get("qx", 0.0),
                        qy=target_pose.get("qy", 0.0),
                        qz=target_pose.get("qz", 0.0),
                        qw=target_pose.get("qw", 1.0),
                    ),
                    force_profile=ForceProfile(
                        max_grip_newtons=grip_force,
                        lateral_torque_limit_nm=torque,
                    ),
                    speed_profile=SpeedProfile(speed_str),
                ),
                failure_modes=failure_modes,
            ),
        ),
        layer_4_safety=SafetyLayer(
            hazard_class=HazardClass(hazard_str),
            ppe_required=["safety_glasses"] if hazard_str != "NONE" else [],
            operating_constraints=OperatingConstraints(
                max_speed_mm_s=250.0,
                max_force_n=grip_force,
            ),
        ),
        telemetry=Telemetry(
            target_controller="api_client",
            end_effector=end_effector,
            pipeline_status=PipelineStatus.FAST_PATH_ACTIVE,
        ),
        control_loop=ControlLoop(
            vision_hashed=True,
            vlm_inference_completed=True,
            ram_cache_updated=False,
        ),
    )

    skb_dict = skb.model_dump(mode="json", by_alias=True)

    # Attach oracle metadata (not part of the Pydantic schema)
    skb_dict["_oracle_metadata"] = {
        "confidence_score": confidence,
        "reasoning_trace": reasoning,
        "generation_method": "llm" if "mock" not in str(llm_response.get("_generator", "")) else "mock",
    }

    return skb_dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "service": "gcp-robotics-skb-api",
        "version": "2.0.1",
        "registry_entries": _registry.stats().get("total_entries", 0),
    }


@app.post("/v1/hash")
async def compute_hash(image: UploadFile = File(...)):
    """Compute composite perceptual hashes for an uploaded image.

    Returns dHash (64-bit), pHash (64-bit), PPP-256 (256-bit),
    color hash, and content SHA-256.
    """
    tmp_path = await _save_upload(image)
    try:
        t0 = time.perf_counter()
        composite = _hasher.compute_composite(tmp_path)
        content_sha = _hasher.content_hash_sha256(tmp_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "dhash_64bit": composite.dhash_hex,
            "phash_64bit": composite.phash_hex,
            "ppp_256bit": composite.ppp_256bit,
            "color_hash": composite.color_hash,
            "content_sha256": content_sha,
            "computation_time_ms": round(elapsed_ms, 2),
        }
    finally:
        os.unlink(tmp_path)


@app.post("/v1/skb/generate/mock")
async def generate_skb_mock(
    image: UploadFile = File(...),
    workspace_zone: str = Form("evaluation_tabletop"),
    end_effector: str = Form("franka_hand"),
    failure_context: str = Form("New object detected — no registry entry found."),
):
    """Generate an SKB using mock heuristics (no LLM API key required).

    This endpoint demonstrates the full pipeline: hash → analyze → build SKB → sign.
    Uses keyword heuristics instead of an LLM for the analysis step.
    """
    from gcp_robotics.slow_path.prompts import MockSlowPathGenerator

    tmp_path = await _save_upload(image)
    try:
        t0 = time.perf_counter()

        # Stage 1: Hash
        composite = _hasher.compute_composite(tmp_path)
        content_sha = _hasher.content_hash_sha256(tmp_path)
        hash_ms = (time.perf_counter() - t0) * 1000

        # Stage 2: Oracle (mock)
        t1 = time.perf_counter()
        generator = MockSlowPathGenerator()
        llm_response = generator.generate_and_validate(
            trigger_event="HASH_MISS_API_REQUEST",
            workspace_zone=workspace_zone,
            end_effector=end_effector,
            failure_context=failure_context,
            image_path=tmp_path,
        )
        oracle_ms = (time.perf_counter() - t1) * 1000

        # Stage 3: Build full SKB
        t2 = time.perf_counter()
        skb_dict = _build_full_skb(
            llm_response, composite, content_sha,
            workspace_zone, end_effector,
        )
        build_ms = (time.perf_counter() - t2) * 1000

        # Stage 4: Sign
        t3 = time.perf_counter()
        signed = _verifier.sign(skb_dict)
        skb_dict["triple_hash"]["soulmark_sha256"] = signed.soulmark
        sign_ms = (time.perf_counter() - t3) * 1000

        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "skb": skb_dict,
            "soulmark": signed.soulmark,
            "timing": {
                "hash_ms": round(hash_ms, 2),
                "oracle_ms": round(oracle_ms, 2),
                "build_ms": round(build_ms, 2),
                "sign_ms": round(sign_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "method": "mock_heuristic",
        }
    finally:
        os.unlink(tmp_path)


@app.post("/v1/skb/generate")
async def generate_skb_llm(
    image: UploadFile = File(...),
    workspace_zone: str = Form("evaluation_tabletop"),
    end_effector: str = Form("franka_hand"),
    failure_context: str = Form("New object detected — no registry entry found."),
):
    """Generate an SKB using the LLM oracle (requires ANTHROPIC_API_KEY).

    Full pipeline: hash → Claude vision analysis → build 4-layer SKB → Soulmark sign.
    This is the production endpoint for generating robot-ready blueprints from photos.
    """
    from gcp_robotics.slow_path.prompts import SlowPathGenerator

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "ANTHROPIC_API_KEY not set. Use /v1/skb/generate/mock for "
                "API-key-free testing, or set the environment variable."
            ),
        )

    tmp_path = await _save_upload(image)
    try:
        t0 = time.perf_counter()

        # Stage 1: Hash
        composite = _hasher.compute_composite(tmp_path)
        content_sha = _hasher.content_hash_sha256(tmp_path)
        hash_ms = (time.perf_counter() - t0) * 1000

        # Stage 2: Oracle (LLM)
        t1 = time.perf_counter()
        generator = SlowPathGenerator(api_key=api_key)
        llm_response = generator.generate_and_validate(
            trigger_event="HASH_MISS_API_REQUEST",
            workspace_zone=workspace_zone,
            end_effector=end_effector,
            failure_context=failure_context,
            image_path=tmp_path,
        )
        oracle_ms = (time.perf_counter() - t1) * 1000

        # Stage 3: Build full SKB
        t2 = time.perf_counter()
        skb_dict = _build_full_skb(
            llm_response, composite, content_sha,
            workspace_zone, end_effector,
        )
        build_ms = (time.perf_counter() - t2) * 1000

        # Stage 4: Sign
        t3 = time.perf_counter()
        signed = _verifier.sign(skb_dict)
        skb_dict["triple_hash"]["soulmark_sha256"] = signed.soulmark
        sign_ms = (time.perf_counter() - t3) * 1000

        total_ms = (time.perf_counter() - t0) * 1000

        return {
            "skb": skb_dict,
            "soulmark": signed.soulmark,
            "timing": {
                "hash_ms": round(hash_ms, 2),
                "oracle_ms": round(oracle_ms, 2),
                "build_ms": round(build_ms, 2),
                "sign_ms": round(sign_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "method": "llm_claude",
        }
    finally:
        os.unlink(tmp_path)


@app.post("/v1/registry/lookup")
async def registry_lookup(dhash_hex: str = Form(...)):
    """Look up an SKB in the loaded registry by dHash.

    Returns the matching SKB if found (EXACT or FUZZY match),
    or a MISS result with suggestions.
    """
    t0 = time.perf_counter()
    result = _registry.lookup(dhash_hex)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "match_type": result.match_type,
        "skb_data": result.skb_data,
        "hamming_distance": result.hamming_distance,
        "lookup_time_ms": round(elapsed_ms, 4),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting GCP-Robotics SKB API on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)

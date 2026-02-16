"""
Golden Codex Protocol 2.0-GCP-ROBOTICS -- Pydantic v2 Schema Models
====================================================================
Four-Layer Ontology for Spatial Kinematic Blueprints (SKBs),
aligned with the patent FIG 2 reference architecture.

Layers:
    200  Provenance
    210  Semantic Topology / NEST
    220  Affordance
    230  Safety

Copyright (c) 2026 Metavolve Labs -- Robotics R&D Division
"""

from __future__ import annotations

import hashlib
import json
import uuid as _uuid
from datetime import date, datetime
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python < 3.11."""
        pass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations (StrEnum for clean JSON serialisation)
# ---------------------------------------------------------------------------

class ObjectClassification(StrEnum):
    RIGID_BODY_GRASPABLE = "Rigid_Body_Graspable"
    DEFORMABLE = "Deformable"
    ARTICULATED = "Articulated"
    LIQUID_CONTAINER = "Liquid_Container"
    FRAGILE = "Fragile"
    TOOL = "Tool"
    FASTENER = "Fastener"


class ActionPrimitive(StrEnum):
    PICK_VERTICAL = "PICK_VERTICAL"
    PICK_ANGLED = "PICK_ANGLED"
    GRASP_PARALLEL_JAW = "GRASP_PARALLEL_JAW"
    GRASP_SUCTION = "GRASP_SUCTION"
    PUSH = "PUSH"
    SLIDE = "SLIDE"
    ROTATE_IN_HAND = "ROTATE_IN_HAND"
    SHAKE_BIN_HORIZONTAL = "SHAKE_BIN_HORIZONTAL"
    INSERT_PRESS_FIT = "INSERT_PRESS_FIT"
    PLACE_PRECISE = "PLACE_PRECISE"
    SCREW_DRIVE = "SCREW_DRIVE"
    FLIP = "FLIP"
    HANDOVER = "HANDOVER"


class SpeedProfile(StrEnum):
    RAPID = "RAPID"
    FINE = "FINE"
    RAPID_OSCILLATION = "RAPID_OSCILLATION"
    GUARDED = "GUARDED"


class OnFailureAction(StrEnum):
    RETRY = "RETRY"
    TRIGGER_SLOW_PATH = "TRIGGER_SLOW_PATH"
    ABORT = "ABORT"
    ESCALATE_HUMAN = "ESCALATE_HUMAN"


class HazardClass(StrEnum):
    NONE = "NONE"
    SHARP_EDGES = "SHARP_EDGES"
    HOT_SURFACE = "HOT_SURFACE"
    CHEMICAL = "CHEMICAL"
    ELECTRICAL = "ELECTRICAL"
    HEAVY_LOAD = "HEAVY_LOAD"
    PINCH_POINT = "PINCH_POINT"


class TriggerEvent(StrEnum):
    HASH_MISS_OOD_STATE = "HASH_MISS_OOD_STATE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    EXECUTION_TIMEOUT = "EXECUTION_TIMEOUT"
    TORQUE_LIMIT_EXCEEDED = "TORQUE_LIMIT_EXCEEDED"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"


class AmendmentVisibility(StrEnum):
    FLEET_WIDE = "fleet_wide"
    LOCAL = "local"
    SUPERVISOR_ONLY = "supervisor_only"


class TrustTag(StrEnum):
    VERIFIED_INSTITUTION = "verified_institution"
    VERIFIED_HUMAN = "verified_human"
    VERIFIED_AMENDED = "verified_amended"
    INFERRED_MODEL = "inferred_model"
    INFERRED_CROSS_REFERENCE = "inferred_cross_reference"


class PipelineStatus(StrEnum):
    REGISTERED = "REGISTERED"
    FAST_PATH_ACTIVE = "FAST_PATH_ACTIVE"
    SLOW_PATH_GENERATING = "SLOW_PATH_GENERATING"
    PROMOTED_TO_FAST_PATH = "PROMOTED_TO_FAST_PATH"
    DEPRECATED = "DEPRECATED"


# ---------------------------------------------------------------------------
# Shared / reusable sub-models
# ---------------------------------------------------------------------------

class XYZ(BaseModel):
    """Three-component vector (metres or unitless depending on context)."""
    model_config = {"frozen": False, "populate_by_name": True}

    x: float = Field(0.0, description="X component")
    y: float = Field(0.0, description="Y component")
    z: float = Field(0.0, description="Z component")


class BoundingBox2D(BaseModel):
    """Axis-aligned 2-D bounding box in pixel coordinates."""
    x_min: int = Field(..., description="Left edge (pixels)")
    y_min: int = Field(..., description="Top edge (pixels)")
    x_max: int = Field(..., description="Right edge (pixels)")
    y_max: int = Field(..., description="Bottom edge (pixels)")


# ---------------------------------------------------------------------------
# Layer 1 -- Provenance (FIG 2 ref 200)
# ---------------------------------------------------------------------------

class ProvenanceLayer(BaseModel):
    """Layer 1: Provenance -- origin, calibration, and environmental context."""
    model_config = {"json_schema_extra": {"$comment": "FIG 2 ref 200"}}

    manufacturer: str = Field(..., description="Robot / sensor manufacturer name")
    model_no: str = Field(..., description="Manufacturer model number")
    serial: str = Field(..., description="Unique serial identifier of the unit")
    calibration_date: Optional[datetime] = Field(
        None, description="Date of last successful calibration"
    )
    maintenance_schedule: Optional[str] = Field(
        None, description="Cron-style or human-readable maintenance cadence"
    )
    workspace_zone: Optional[str] = Field(
        None, description="Logical zone identifier within the facility"
    )
    lighting_condition: Optional[str] = Field(
        None,
        description="Ambient lighting descriptor (e.g. 'overhead_LED_5000K')",
    )
    source_dataset: Optional[str] = Field(
        None, description="Dataset or data-feed that produced the observation"
    )
    enrichment_timestamp: Optional[datetime] = Field(
        None, description="When the enrichment pipeline last touched this record"
    )
    enrichment_model_version: Optional[str] = Field(
        None,
        description="Version string of the enrichment model (e.g. 'vlm-4.1-turbo')",
    )


# ---------------------------------------------------------------------------
# Layer 2 -- Semantic Topology / NEST (FIG 2 ref 210)
# ---------------------------------------------------------------------------

class PhysicalProperties(BaseModel):
    """Estimated physical characteristics of the observed object."""
    estimated_mass_kg: Optional[float] = Field(
        None, ge=0.0, description="Estimated mass in kilograms"
    )
    center_of_mass_offset: Optional[XYZ] = Field(
        None, description="CoM offset from geometric centroid (metres)"
    )
    friction_coefficient: Optional[float] = Field(
        None, ge=0.0, description="Static friction coefficient estimate"
    )
    dimensions: Optional[XYZ] = Field(
        None,
        description="Bounding-box dimensions (width, height, depth) in metres",
    )


class MaterialGraph(BaseModel):
    """Simplified material composition descriptor."""
    primary_material: Optional[str] = Field(
        None, description="Dominant material (e.g. 'ABS_plastic', 'aluminium')"
    )
    component_count: Optional[int] = Field(
        None, ge=1, description="Number of distinct sub-components"
    )


class KinematicJoint(BaseModel):
    """Single kinematic joint descriptor for articulated objects."""
    joint_name: str = Field(..., description="Human-readable joint identifier")
    joint_type: str = Field(
        ..., description="Joint type (revolute, prismatic, spherical, fixed)"
    )
    axis: Optional[XYZ] = Field(None, description="Joint axis direction vector")
    limits_deg: Optional[list[float]] = Field(
        None,
        min_length=2,
        max_length=2,
        description="[min, max] angular limits in degrees",
    )


class PerceptualSignature(BaseModel):
    """Multi-hash perceptual fingerprint for O(1) fast-path lookup."""
    dhash_hex: str = Field(
        ...,
        min_length=16,
        max_length=16,
        description="Difference-hash, 64-bit, hex-encoded (16 hex chars)",
    )
    phash_hex: str = Field(
        ...,
        min_length=16,
        max_length=16,
        description="Perceptual hash, 64-bit, hex-encoded (16 hex chars)",
    )
    color_histogram_hash: Optional[str] = Field(
        None, description="Hash of the quantised colour histogram"
    )
    keyframe_timestamp_ms: Optional[int] = Field(
        None, ge=0, description="Capture timestamp in milliseconds since epoch"
    )
    variance_tolerance_pct: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description="Acceptable Hamming-distance variance as a percentage",
    )
    region_of_interest: Optional[BoundingBox2D] = Field(
        None, description="Bounding box of the object within the frame"
    )
    optical_flow_stable: bool = Field(
        True,
        description="Whether the object is optically stable between frames",
    )


class SemanticTopologyLayer(BaseModel):
    """Layer 2: Semantic Topology / NEST -- classification, physics, perception."""
    model_config = {"json_schema_extra": {"$comment": "FIG 2 ref 210"}}

    object_classification: ObjectClassification = Field(
        ..., description="High-level object taxonomy class"
    )
    physical_properties: Optional[PhysicalProperties] = Field(
        None, description="Estimated physical properties"
    )
    material_graph: Optional[MaterialGraph] = Field(
        None, description="Material composition summary"
    )
    kinematic_joints: Optional[list[KinematicJoint]] = Field(
        None,
        description="Joint descriptors (populated when classification is Articulated)",
    )
    perceptual_signature: PerceptualSignature = Field(
        ..., description="Multi-hash perceptual fingerprint"
    )
    related_skbs: list[str] = Field(
        default_factory=list,
        description="UUIDs of related Spatial Kinematic Blueprints",
    )


# ---------------------------------------------------------------------------
# Layer 3 -- Affordance (FIG 2 ref 220)
# ---------------------------------------------------------------------------

class TargetPoseRelative(BaseModel):
    """SE(3) pose expressed as position + unit quaternion, relative to object frame."""
    x: float = Field(0.0, description="Translation X (metres)")
    y: float = Field(0.0, description="Translation Y (metres)")
    z: float = Field(0.0, description="Translation Z (metres)")
    qx: float = Field(0.0, description="Quaternion X")
    qy: float = Field(0.0, description="Quaternion Y")
    qz: float = Field(0.0, description="Quaternion Z")
    qw: float = Field(1.0, description="Quaternion W (scalar)")


class ForceProfile(BaseModel):
    """Force and compliance envelope for the action primitive."""
    max_grip_newtons: float = Field(
        ..., ge=0.0, description="Maximum grip force (N)"
    )
    lateral_torque_limit_nm: float = Field(
        ..., ge=0.0, description="Lateral torque limit (Nm)"
    )
    compliance_stiffness: Optional[XYZ] = Field(
        None,
        description="Cartesian compliance stiffness [kx, ky, kz] (N/m)",
    )


class ActionParameters(BaseModel):
    """Parameters governing how an action primitive is executed."""
    target_pose_relative: TargetPoseRelative = Field(
        default_factory=TargetPoseRelative,
        description="Relative SE(3) target pose for the end-effector",
    )
    force_profile: Optional[ForceProfile] = Field(
        None, description="Force and compliance constraints"
    )
    speed_profile: SpeedProfile = Field(
        SpeedProfile.FINE, description="Motion speed profile"
    )


class ValidationSignature(BaseModel):
    """Post-action perceptual validation gate."""
    expected_post_action_dhash: Optional[str] = Field(
        None,
        min_length=16,
        max_length=16,
        description="Expected dHash of the scene after action completion",
    )
    success_tolerance_bits: int = Field(
        6,
        ge=0,
        description="Maximum Hamming distance (bits) to accept as success",
    )
    on_failure: OnFailureAction = Field(
        OnFailureAction.RETRY,
        description="Recovery strategy when post-action validation fails",
    )


class ExecutionAct(BaseModel):
    """Single act within the execution narrative."""
    time: str = Field(..., description="Relative timestamp or label (e.g. 'T+0.0s')")
    state: str = Field(..., description="State name (e.g. 'APPROACH', 'CONTACT')")
    description: str = Field(..., description="Human-readable description of the act")


class ActionSchema(BaseModel):
    """Complete action primitive specification."""
    primitive_id: ActionPrimitive = Field(
        ..., description="Canonical action-primitive identifier"
    )
    parameters: ActionParameters = Field(
        default_factory=ActionParameters,
        description="Execution parameters for this primitive",
    )
    validation_signature: Optional[ValidationSignature] = Field(
        None, description="Post-action perceptual validation"
    )
    execution_narrative: list[ExecutionAct] = Field(
        default_factory=list,
        description="Ordered list of acts describing the execution sequence",
    )
    failure_modes: list[str] = Field(
        default_factory=list,
        description="Known failure modes for this action primitive",
    )


class AffordanceLayer(BaseModel):
    """Layer 3: Affordance -- what can be done with this object and how."""
    model_config = {"json_schema_extra": {"$comment": "FIG 2 ref 220"}}

    action_schema: ActionSchema = Field(
        ..., description="Primary action primitive specification"
    )


# ---------------------------------------------------------------------------
# Layer 4 -- Safety (FIG 2 ref 230)
# ---------------------------------------------------------------------------

class OperatingConstraints(BaseModel):
    """Dynamic operating-envelope constraints."""
    max_speed_mm_s: Optional[float] = Field(
        None, ge=0.0, description="Maximum TCP speed (mm/s)"
    )
    max_force_n: Optional[float] = Field(
        None, ge=0.0, description="Maximum applicable force (N)"
    )
    restricted_zones: list[str] = Field(
        default_factory=list,
        description="Zone identifiers where this action is prohibited",
    )


class EnterpriseLicense(BaseModel):
    """Enterprise-level licensing and IP constraints."""
    holder: str = Field(..., description="Licence holder entity name")
    facility: Optional[str] = Field(
        None, description="Facility or site the licence is bound to"
    )
    proprietary_workflow: bool = Field(
        False,
        description="Whether the workflow contains proprietary trade-secret logic",
    )


class ExportRestrictions(BaseModel):
    """Data-governance and export-control flags."""
    commercial_use: bool = Field(
        True, description="Whether this SKB may be used commercially"
    )
    cross_fleet_sync_enabled: bool = Field(
        False,
        description="Allow synchronisation of this SKB across robot fleets",
    )
    export_to_open_source: bool = Field(
        False,
        description="Whether this SKB may be published under an open-source licence",
    )


class SafetyLayer(BaseModel):
    """Layer 4: Safety -- hazard classification, PPE, regulatory, and licensing."""
    model_config = {"json_schema_extra": {"$comment": "FIG 2 ref 230"}}

    hazard_class: HazardClass = Field(
        HazardClass.NONE, description="Primary hazard classification"
    )
    ppe_required: list[str] = Field(
        default_factory=list,
        description="Required personal protective equipment (e.g. 'safety_glasses')",
    )
    regulatory_certifications: list[str] = Field(
        default_factory=list,
        description="Applicable certifications (e.g. 'ISO 10218-1', 'CE')",
    )
    operating_constraints: Optional[OperatingConstraints] = Field(
        None, description="Dynamic operating-envelope constraints"
    )
    enterprise_license: Optional[EnterpriseLicense] = Field(
        None, description="Enterprise licensing metadata"
    )
    export_restrictions: Optional[ExportRestrictions] = Field(
        None, description="Data-governance export-control flags"
    )


# ---------------------------------------------------------------------------
# Top-level sub-structures
# ---------------------------------------------------------------------------

class Timestamps(BaseModel):
    """Key lifecycle timestamps for the SKB."""
    state_encountered: Optional[datetime] = Field(
        None, description="When the physical state was first observed"
    )
    schema_generated: Optional[datetime] = Field(
        None, description="When this SKB was generated"
    )
    promoted_to_fast_path: Optional[datetime] = Field(
        None, description="When this SKB was promoted into the fast-path cache"
    )


class Identifiers(BaseModel):
    """Canonical identifiers for cross-system referencing."""
    uuid: str = Field(
        default_factory=lambda: str(_uuid.uuid4()),
        description="RFC 4122 UUID for this SKB instance",
    )
    artifact_id: str = Field(
        ..., description="Opaque artifact identifier within the data pipeline"
    )
    codex_id: str = Field(
        ..., description="Golden Codex registry identifier"
    )


class TripleHash(BaseModel):
    """Multi-resolution hash block for integrity and fast-path lookup."""
    content_hash_sha256: str = Field(
        ..., description="SHA-256 of the canonical content payload"
    )
    ppp_256bit: str = Field(
        ...,
        description="256-bit pHash for global registry deduplication",
    )
    soulmark_sha256: str = Field(
        ...,
        description="SHA-256 soulmark -- content-addressed identity of this SKB",
    )
    fast_path_dhash_64bit: str = Field(
        ...,
        min_length=16,
        max_length=16,
        description="64-bit dHash for onboard O(1) fast-path lookup (16 hex chars)",
    )


class RecoveryMetadata(BaseModel):
    """Populated when this SKB was generated via the slow-path recovery pipeline."""
    generated_by_slow_path: bool = Field(
        True,
        description="Whether this SKB was produced by the slow-path pipeline",
    )
    trigger_event: TriggerEvent = Field(
        ..., description="Event that triggered slow-path generation"
    )
    reasoning_trace: str = Field(
        ..., description="Free-text reasoning trace from the generative supervisor"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the generated SKB (0.0 - 1.0)",
    )
    llm_model_version: Optional[str] = Field(
        None,
        description="Version of the LLM / VLM used for slow-path generation",
    )


class Amendment(BaseModel):
    """A single amendment / annotation appended to the SKB over its lifetime."""
    author: str = Field(..., description="Identifier of the amending agent or person")
    message: str = Field(..., description="Amendment description")
    visibility: AmendmentVisibility = Field(
        AmendmentVisibility.FLEET_WIDE,
        description="Visibility scope of this amendment",
    )
    date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the amendment",
    )
    trust_tag: TrustTag = Field(
        TrustTag.INFERRED_MODEL,
        description="Provenance trust level of this amendment",
    )


class Telemetry(BaseModel):
    """Runtime telemetry and pipeline bookkeeping."""
    target_controller: str = Field(
        ..., description="Target robot controller identifier (e.g. 'ur5e_cell_03')"
    )
    transport_layer: str = Field(
        "ROS2_Humble",
        description="Middleware transport layer (e.g. 'ROS2_Humble', 'MQTT')",
    )
    end_effector: str = Field(
        ..., description="End-effector descriptor (e.g. 'robotiq_2f85')"
    )
    slow_path_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Wall-clock latency of the last slow-path generation (ms)",
    )
    fast_path_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Wall-clock latency of the last fast-path lookup (ms)",
    )
    pipeline_status: PipelineStatus = Field(
        PipelineStatus.REGISTERED,
        description="Current lifecycle status of this SKB in the pipeline",
    )


class ControlLoop(BaseModel):
    """Boolean checkpoints tracking which pipeline stages have completed."""
    vision_hashed: bool = Field(
        False, description="Whether the vision frame has been hashed"
    )
    vlm_inference_completed: bool = Field(
        False,
        description="Whether VLM inference has completed for this observation",
    )
    digital_twin_simulated: bool = Field(
        False,
        description="Whether a digital-twin simulation was run for validation",
    )
    ram_cache_updated: bool = Field(
        False,
        description="Whether the fast-path RAM cache has been updated with this SKB",
    )


# ---------------------------------------------------------------------------
# Default JSON-LD @context
# ---------------------------------------------------------------------------

_DEFAULT_JSON_LD_CONTEXT: dict[str, Any] = {
    "gcp": "https://schema.metavolve.io/gcp/2.0/",
    "ros2": "https://docs.ros.org/en/humble/",
    "schema": "https://schema.org/",
}


# ---------------------------------------------------------------------------
# Top-level model: SpatialKinematicBlueprint
# ---------------------------------------------------------------------------

class SpatialKinematicBlueprint(BaseModel):
    """
    Spatial Kinematic Blueprint (SKB) -- the canonical artefact of the
    Golden Codex Protocol 2.0-GCP-ROBOTICS schema.

    An SKB encodes the full four-layer ontology (Provenance, Semantic
    Topology, Affordance, Safety) for a single observed-object /
    action-primitive pair, together with lifecycle metadata, perceptual
    hashes, and telemetry required for fast-path / slow-path operation.
    """

    model_config = {
        "json_schema_extra": {
            "title": "SpatialKinematicBlueprint",
            "description": (
                "Golden Codex Protocol 2.0-GCP-ROBOTICS -- "
                "Four-Layer Ontology for Robotic Manipulation"
            ),
        },
        "populate_by_name": True,
    }

    # -- Header --------------------------------------------------------
    json_ld_context: dict[str, Any] = Field(
        default_factory=lambda: dict(_DEFAULT_JSON_LD_CONTEXT),
        alias="@context",
        description="JSON-LD @context namespace map",
    )
    schema_version: Literal["2.0-GCP-ROBOTICS"] = Field(
        "2.0-GCP-ROBOTICS",
        description="Protocol schema version identifier",
    )
    agent: str = Field(
        "Fast_Path_Registry",
        description=(
            "System that generated this SKB "
            "(e.g. 'System_2_Generative_Supervisor', 'Fast_Path_Registry')"
        ),
    )

    # -- Timestamps & Identifiers --------------------------------------
    timestamps: Timestamps = Field(
        default_factory=Timestamps,
        description="Lifecycle timestamps",
    )
    identifiers: Identifiers = Field(
        ..., description="Canonical cross-system identifiers"
    )
    triple_hash: TripleHash = Field(
        ..., description="Multi-resolution hash block"
    )

    # -- Four-layer ontology -------------------------------------------
    layer_1_provenance: ProvenanceLayer = Field(
        ..., description="Layer 1 (ref 200): Provenance"
    )
    layer_2_semantic_topology: SemanticTopologyLayer = Field(
        ..., description="Layer 2 (ref 210): Semantic Topology / NEST"
    )
    layer_3_affordance: AffordanceLayer = Field(
        ..., description="Layer 3 (ref 220): Affordance"
    )
    layer_4_safety: SafetyLayer = Field(
        ..., description="Layer 4 (ref 230): Safety"
    )

    # -- Recovery & Amendments -----------------------------------------
    recovery_metadata: Optional[RecoveryMetadata] = Field(
        None,
        description="Slow-path recovery metadata (populated only for slow-path SKBs)",
    )
    amendments: list[Amendment] = Field(
        default_factory=list,
        description="Ordered list of amendments applied to this SKB",
    )

    # -- Telemetry & Control Loop --------------------------------------
    telemetry: Telemetry = Field(
        ..., description="Runtime telemetry and pipeline bookkeeping"
    )
    control_loop: ControlLoop = Field(
        default_factory=ControlLoop,
        description="Pipeline stage completion checkpoints",
    )

    # -----------------------------------------------------------------
    # Methods
    # -----------------------------------------------------------------

    def _canonical_json(self) -> str:
        """Return the canonical JSON representation (sorted keys, minimal whitespace)."""
        payload = self.model_dump(mode="json", by_alias=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def compute_soulmark(self) -> str:
        """
        Compute the SHA-256 *soulmark* of this SKB.

        The soulmark is the SHA-256 hex-digest of the canonical JSON
        serialisation (sorted keys, minimal whitespace).  It serves as
        the content-addressed identity of the blueprint.
        """
        return hashlib.sha256(self._canonical_json().encode("utf-8")).hexdigest()

    def to_json_ld(self) -> dict[str, Any]:
        """
        Serialise this SKB as a JSON-LD-compatible dictionary.

        The ``@context`` key is included at the top level so that the
        output is directly consumable by JSON-LD processors.
        """
        data = self.model_dump(mode="json", by_alias=True)
        # Ensure @context is present at top level
        if "@context" not in data and "json_ld_context" in data:
            data["@context"] = data.pop("json_ld_context")
        return data

    @classmethod
    def from_json_ld(cls, data: dict[str, Any]) -> "SpatialKinematicBlueprint":
        """
        Deserialise a JSON-LD dictionary into a ``SpatialKinematicBlueprint``.

        Handles the ``@context`` key transparently so callers do not
        need to pre-process the input.
        """
        payload = dict(data)
        # Map @context -> json_ld_context if the alias isn't already resolved
        if "@context" in payload and "json_ld_context" not in payload:
            payload["json_ld_context"] = payload.pop("@context")
        return cls.model_validate(payload)

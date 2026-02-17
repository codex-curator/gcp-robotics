"""
Vertex AI Fine-Tuning Data Generator for GCP-Robotics SKB Schema
=================================================================

Converts existing Spatial Kinematic Blueprint (SKB) JSON files paired with
object images into JSONL training format suitable for fine-tuning Gemini
(via Vertex AI) or Anthropic models to generate SKBs from object photographs.

The pipeline:
  1. Scans data directories for SKB + image pairs
  2. Extracts the target model response (the fields the LLM should learn)
  3. Encodes images as base64 inline data
  4. Formats each example in Vertex AI (Gemini) or Anthropic fine-tuning JSONL
  5. Optionally augments images (brightness, contrast, rotation, crop)
  6. Generates synthetic negative examples for robustness
  7. Splits into train/validation sets and writes JSONL files

Usage:
    python gcp_robotics/vertex_tuning.py \\
        --data-dirs data/standard_evaluation data/ycb_20 \\
        --output data/training/skb_tuning_v1.jsonl \\
        --domain general_household \\
        --augment

Copyright (c) 2026 Metavolve Labs -- Robotics R&D Division
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports -- degrade gracefully if PIL is not installed
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ImageEnhance, ImageFilter
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    logger.warning("Pillow not installed. Image augmentation will be disabled.")


# ---------------------------------------------------------------------------
# 1. TASK DOMAIN SYSTEM PROMPTS
# ---------------------------------------------------------------------------

TASK_DOMAINS: dict[str, dict[str, Any]] = {
    "warehouse_picking": {
        "system_prompt": (
            "You are a warehouse robotic perception oracle operating under the "
            "Golden Codex Protocol 2.0-GCP-ROBOTICS. Given an image of an object "
            "on a conveyor belt or in a bin, generate a Spatial Kinematic Blueprint "
            "(SKB) action schema for grasping and placing it. Focus on: high "
            "throughput, bin clutter handling, cardboard/plastic materials. Prefer "
            "PICK_VERTICAL and GRASP_PARALLEL_JAW primitives for standard picks. "
            "Use PUSH or SLIDE when objects are wedged or overlapping.\n\n"
            "AVAILABLE ACTION PRIMITIVES:\n"
            "  PICK_VERTICAL, GRASP_PARALLEL_JAW, PUSH, SLIDE\n\n"
            "SPEED PROFILES:\n"
            "  RAPID, FINE, RAPID_OSCILLATION, GUARDED\n\n"
            "HAZARD CLASSES:\n"
            "  NONE, HEAVY_OBJECT, SHARP_EDGE\n\n"
            "OUTPUT FORMAT:\n"
            "  You MUST output ONLY valid JSON matching the SKB action schema. "
            "No conversational text, no markdown fences -- just raw JSON.\n\n"
            "SAFETY CONSTRAINTS:\n"
            "  - Never exceed 50N grip force without explicit authorization.\n"
            "  - Always set the hazard_class field.\n"
            "  - Always provide a reasoning_trace explaining your decision.\n\n"
            "REQUIRED OUTPUT FIELDS:\n"
            "  primitive_id, reasoning_trace, confidence_score, hazard_class, "
            "speed_profile, force_profile, target_pose_relative, physical_properties, "
            "execution_narrative, failure_modes"
        ),
        "action_primitives": ["PICK_VERTICAL", "GRASP_PARALLEL_JAW", "PUSH", "SLIDE"],
        "typical_forces": {"min_grip": 5.0, "max_grip": 40.0},
        "hazard_classes": ["NONE", "HEAVY_LOAD", "SHARP_EDGES"],
    },
    "kitchen_food": {
        "system_prompt": (
            "You are a kitchen robotics oracle operating under the Golden Codex "
            "Protocol 2.0-GCP-ROBOTICS. Objects may be deformable, fragile, or "
            "temperature-sensitive. Generate a Spatial Kinematic Blueprint (SKB) "
            "action schema for safe manipulation. Consider material compliance, "
            "surface moisture, and container contents when setting force profiles.\n\n"
            "AVAILABLE ACTION PRIMITIVES:\n"
            "  GRASP_PARALLEL_JAW, PICK_VERTICAL, PICK_ANGLED, GRASP_SUCTION\n\n"
            "SPEED PROFILES:\n"
            "  RAPID, FINE, RAPID_OSCILLATION, GUARDED\n\n"
            "HAZARD CLASSES:\n"
            "  NONE, SHARP_EDGES, HOT_SURFACE\n\n"
            "OUTPUT FORMAT:\n"
            "  You MUST output ONLY valid JSON matching the SKB action schema. "
            "No conversational text, no markdown fences -- just raw JSON.\n\n"
            "SAFETY CONSTRAINTS:\n"
            "  - Never exceed 15N grip force on food items.\n"
            "  - Always check for temperature hazards.\n"
            "  - Always provide a reasoning_trace explaining your decision.\n\n"
            "REQUIRED OUTPUT FIELDS:\n"
            "  primitive_id, reasoning_trace, confidence_score, hazard_class, "
            "speed_profile, force_profile, target_pose_relative, physical_properties, "
            "execution_narrative, failure_modes"
        ),
        "action_primitives": ["GRASP_PARALLEL_JAW", "PICK_VERTICAL", "PICK_ANGLED", "GRASP_SUCTION"],
        "typical_forces": {"min_grip": 2.0, "max_grip": 15.0},
        "hazard_classes": ["NONE", "SHARP_EDGES", "HOT_SURFACE"],
    },
    "assembly_line": {
        "system_prompt": (
            "You are a precision assembly oracle operating under the Golden Codex "
            "Protocol 2.0-GCP-ROBOTICS. Objects require exact placement, tight "
            "tolerances, and specific mating sequences. Generate a Spatial Kinematic "
            "Blueprint (SKB) action schema with precise pose targets and guarded "
            "force profiles for assembly operations.\n\n"
            "AVAILABLE ACTION PRIMITIVES:\n"
            "  INSERT_PRESS_FIT, ROTATE_IN_HAND, PICK_VERTICAL, PLACE_PRECISE\n\n"
            "SPEED PROFILES:\n"
            "  RAPID, FINE, RAPID_OSCILLATION, GUARDED\n\n"
            "HAZARD CLASSES:\n"
            "  NONE, SHARP_EDGES, ELECTRICAL\n\n"
            "OUTPUT FORMAT:\n"
            "  You MUST output ONLY valid JSON matching the SKB action schema. "
            "No conversational text, no markdown fences -- just raw JSON.\n\n"
            "SAFETY CONSTRAINTS:\n"
            "  - Use GUARDED speed for insertion operations.\n"
            "  - Never exceed 25N without explicit authorization.\n"
            "  - Always provide a reasoning_trace explaining your decision.\n\n"
            "REQUIRED OUTPUT FIELDS:\n"
            "  primitive_id, reasoning_trace, confidence_score, hazard_class, "
            "speed_profile, force_profile, target_pose_relative, physical_properties, "
            "execution_narrative, failure_modes"
        ),
        "action_primitives": ["INSERT_PRESS_FIT", "ROTATE_IN_HAND", "PICK_VERTICAL", "PLACE_PRECISE"],
        "typical_forces": {"min_grip": 3.0, "max_grip": 25.0},
        "hazard_classes": ["NONE", "SHARP_EDGES", "ELECTRICAL"],
    },
    "lab_medical": {
        "system_prompt": (
            "You are a laboratory robotics oracle operating under the Golden Codex "
            "Protocol 2.0-GCP-ROBOTICS. Handle specimens, reagents, and equipment "
            "with extreme care. PPE requirements must be specified. Generate a "
            "Spatial Kinematic Blueprint (SKB) action schema prioritising safety "
            "and contamination avoidance.\n\n"
            "AVAILABLE ACTION PRIMITIVES:\n"
            "  PICK_ANGLED, GRASP_PARALLEL_JAW, PICK_VERTICAL, PLACE_PRECISE\n\n"
            "SPEED PROFILES:\n"
            "  RAPID, FINE, RAPID_OSCILLATION, GUARDED\n\n"
            "HAZARD CLASSES:\n"
            "  NONE, CHEMICAL, SHARP_EDGES\n\n"
            "OUTPUT FORMAT:\n"
            "  You MUST output ONLY valid JSON matching the SKB action schema. "
            "No conversational text, no markdown fences -- just raw JSON.\n\n"
            "SAFETY CONSTRAINTS:\n"
            "  - Never exceed 10N grip force on lab specimens.\n"
            "  - Always specify PPE requirements in reasoning_trace.\n"
            "  - Always set hazard_class for chemical or biohazard items.\n"
            "  - Always provide a reasoning_trace explaining your decision.\n\n"
            "REQUIRED OUTPUT FIELDS:\n"
            "  primitive_id, reasoning_trace, confidence_score, hazard_class, "
            "speed_profile, force_profile, target_pose_relative, physical_properties, "
            "execution_narrative, failure_modes"
        ),
        "action_primitives": ["PICK_ANGLED", "GRASP_PARALLEL_JAW", "PICK_VERTICAL", "PLACE_PRECISE"],
        "typical_forces": {"min_grip": 1.0, "max_grip": 10.0},
        "hazard_classes": ["NONE", "CHEMICAL", "SHARP_EDGES"],
    },
    "general_household": {
        "system_prompt": (
            "You are a general-purpose household robotics oracle operating under "
            "the Golden Codex Protocol 2.0-GCP-ROBOTICS. Objects vary widely in "
            "size, shape, material, and fragility. Generate a Spatial Kinematic "
            "Blueprint (SKB) action schema that selects the safest and most "
            "efficient manipulation strategy for the observed object.\n\n"
            "AVAILABLE ACTION PRIMITIVES:\n"
            "  PICK_VERTICAL, GRASP_PARALLEL_JAW, PICK_ANGLED, PUSH, SLIDE, "
            "GRASP_SUCTION, ROTATE_IN_HAND, FLIP, HANDOVER\n\n"
            "SPEED PROFILES:\n"
            "  RAPID, FINE, RAPID_OSCILLATION, GUARDED\n\n"
            "HAZARD CLASSES:\n"
            "  NONE, SHARP_EDGES, HOT_SURFACE, CHEMICAL, ELECTRICAL, HEAVY_LOAD, "
            "PINCH_POINT\n\n"
            "OUTPUT FORMAT:\n"
            "  You MUST output ONLY valid JSON matching the SKB action schema. "
            "No conversational text, no markdown fences -- just raw JSON.\n\n"
            "SAFETY CONSTRAINTS:\n"
            "  - Never exceed 50N grip force without explicit authorization.\n"
            "  - Always set the hazard_class field.\n"
            "  - Always provide a reasoning_trace explaining your decision.\n\n"
            "REQUIRED OUTPUT FIELDS:\n"
            "  primitive_id, reasoning_trace, confidence_score, hazard_class, "
            "speed_profile, force_profile, target_pose_relative, physical_properties, "
            "execution_narrative, failure_modes"
        ),
        "action_primitives": [
            "PICK_VERTICAL", "GRASP_PARALLEL_JAW", "PICK_ANGLED",
            "PUSH", "SLIDE", "GRASP_SUCTION", "ROTATE_IN_HAND",
            "FLIP", "HANDOVER",
        ],
        "typical_forces": {"min_grip": 3.0, "max_grip": 30.0},
        "hazard_classes": ["NONE", "SHARP_EDGES", "HOT_SURFACE", "HEAVY_LOAD"],
    },
}

# Default trigger events and end effectors for training data generation
_DEFAULT_TRIGGER_EVENTS = [
    "HASH_MISS_OOD_STATE",
    "VALIDATION_FAILURE",
    "HUMAN_ESCALATION",
]
_DEFAULT_END_EFFECTORS = [
    "robotiq_2f85",
    "franka_hand",
    "schunk_egp_64",
]
_DEFAULT_WORKSPACE_ZONES = [
    "tabletop_zone_A",
    "bin_station_01",
    "conveyor_pickup",
    "evaluation_tabletop",
]
_DEFAULT_FAILURE_CONTEXTS = [
    "New object detected -- no matching hash in fast-path registry",
    "Hash miss: object appears at unexpected orientation",
    "First encounter with this object type -- generating initial SKB",
    "Object partially occluded; fast-path confidence below threshold",
    "Lighting change caused hash mismatch; regenerating SKB",
]


# ---------------------------------------------------------------------------
# 2. SKBToTrainingExample
# ---------------------------------------------------------------------------

class SKBToTrainingExample:
    """Convert existing SKB + image pairs into Vertex AI fine-tuning format.

    Each training example consists of:
      - A system instruction (domain-specific)
      - A user message with context text + base64-encoded image
      - A model response containing the target SKB action schema JSON

    Supports both Vertex AI (Gemini) and Anthropic fine-tuning JSONL formats.
    """

    def __init__(self, task_domain: str = "general_household") -> None:
        if task_domain not in TASK_DOMAINS:
            raise ValueError(
                f"Unknown task domain '{task_domain}'. "
                f"Available: {sorted(TASK_DOMAINS.keys())}"
            )
        self.task_domain = task_domain
        self.domain = TASK_DOMAINS[task_domain]

    # -- Core conversion ---------------------------------------------------

    def from_skb_file(
        self,
        skb_path: str,
        image_path: str,
        failure_context: str = "New object detected -- no matching hash in fast-path registry",
        trigger_event: str = "HASH_MISS_OOD_STATE",
        workspace_zone: str | None = None,
        end_effector: str | None = None,
    ) -> dict:
        """Convert one SKB + image into a Vertex AI (Gemini) training example.

        Parameters
        ----------
        skb_path : str
            Path to the SKB JSON file.
        image_path : str
            Path to the object image (PNG, JPG, etc.).
        failure_context : str
            Context string describing why slow-path was triggered.
        trigger_event : str
            The trigger event enum value.
        workspace_zone : str, optional
            Workspace zone; extracted from SKB if not provided.
        end_effector : str, optional
            End effector identifier; extracted from SKB if not provided.

        Returns
        -------
        dict
            A single training example in Vertex AI JSONL format.
        """
        skb_path = Path(skb_path)
        image_path = Path(image_path)

        if not skb_path.exists():
            raise FileNotFoundError(f"SKB file not found: {skb_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load SKB
        with open(skb_path, "r") as f:
            skb = json.load(f)

        # Extract workspace_zone and end_effector from SKB if not supplied
        if workspace_zone is None:
            workspace_zone = (
                skb.get("layer_1_provenance", {}).get("workspace_zone")
                or random.choice(_DEFAULT_WORKSPACE_ZONES)
            )
        if end_effector is None:
            end_effector = (
                skb.get("telemetry", {}).get("end_effector")
                or random.choice(_DEFAULT_END_EFFECTORS)
            )

        # Encode image
        image_b64, mime_type = self._encode_image(image_path)

        # Build target response
        target_response = self.extract_target_response(skb)

        # Build user prompt text
        user_text = self._build_user_prompt(
            trigger_event=trigger_event,
            workspace_zone=workspace_zone,
            end_effector=end_effector,
            failure_context=failure_context,
        )

        # Assemble Vertex AI (Gemini) format
        example = {
            "systemInstruction": {
                "parts": [{"text": self.domain["system_prompt"]}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_text},
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": image_b64,
                            }
                        },
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": json.dumps(
                                target_response, indent=2, sort_keys=False
                            )
                        }
                    ],
                },
            ],
        }

        return example

    def from_skb_file_anthropic(
        self,
        skb_path: str,
        image_path: str,
        failure_context: str = "New object detected -- no matching hash in fast-path registry",
        trigger_event: str = "HASH_MISS_OOD_STATE",
        workspace_zone: str | None = None,
        end_effector: str | None = None,
    ) -> dict:
        """Convert one SKB + image into Anthropic fine-tuning format.

        Returns
        -------
        dict
            A single training example in Anthropic Messages API format.
        """
        skb_path = Path(skb_path)
        image_path = Path(image_path)

        if not skb_path.exists():
            raise FileNotFoundError(f"SKB file not found: {skb_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(skb_path, "r") as f:
            skb = json.load(f)

        if workspace_zone is None:
            workspace_zone = (
                skb.get("layer_1_provenance", {}).get("workspace_zone")
                or random.choice(_DEFAULT_WORKSPACE_ZONES)
            )
        if end_effector is None:
            end_effector = (
                skb.get("telemetry", {}).get("end_effector")
                or random.choice(_DEFAULT_END_EFFECTORS)
            )

        image_b64, mime_type = self._encode_image(image_path)
        target_response = self.extract_target_response(skb)

        user_text = self._build_user_prompt(
            trigger_event=trigger_event,
            workspace_zone=workspace_zone,
            end_effector=end_effector,
            failure_context=failure_context,
        )

        # Anthropic fine-tuning format (Messages API style)
        example = {
            "system": self.domain["system_prompt"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_b64,
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        target_response, indent=2, sort_keys=False
                    ),
                },
            ],
        }

        return example

    def format_for_anthropic(self, vertex_example: dict) -> dict:
        """Convert a Vertex AI (Gemini) format example to Anthropic format.

        Parameters
        ----------
        vertex_example : dict
            A training example in Vertex AI format (as produced by from_skb_file).

        Returns
        -------
        dict
            The same example reformatted for Anthropic fine-tuning.
        """
        system_text = vertex_example["systemInstruction"]["parts"][0]["text"]
        contents = vertex_example["contents"]

        user_parts = contents[0]["parts"]
        model_parts = contents[1]["parts"]

        # Build Anthropic user content blocks
        anthropic_user_content = []
        for part in user_parts:
            if "text" in part:
                anthropic_user_content.append(
                    {"type": "text", "text": part["text"]}
                )
            elif "inlineData" in part:
                anthropic_user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part["inlineData"]["mimeType"],
                        "data": part["inlineData"]["data"],
                    },
                })

        # Model response
        model_text = model_parts[0]["text"]

        return {
            "system": system_text,
            "messages": [
                {"role": "user", "content": anthropic_user_content},
                {"role": "assistant", "content": model_text},
            ],
        }

    # -- Target extraction -------------------------------------------------

    def extract_target_response(self, skb: dict) -> dict:
        """Extract the fields the LLM should learn to output from a full SKB.

        These fields constitute the 'model response' in the training example.
        They correspond to the action schema, physical properties, safety
        classification, and recovery metadata that SlowPathGenerator produces.

        Parameters
        ----------
        skb : dict
            A full SKB dictionary (as loaded from a *_skb.json file).

        Returns
        -------
        dict
            The target response containing all fields the model should generate.
        """
        # Layer 3: Affordance -- action schema
        action_schema = skb.get("layer_3_affordance", {}).get("action_schema", {})
        params = action_schema.get("parameters", {})

        # Layer 2: Semantic Topology -- physical properties
        semantic = skb.get("layer_2_semantic_topology", {})
        physical_props = semantic.get("physical_properties")
        material_graph = semantic.get("material_graph")
        object_classification = semantic.get("object_classification", "Rigid_Body_Graspable")

        # Layer 4: Safety
        safety = skb.get("layer_4_safety", {})

        # Recovery metadata (may be None for non-slow-path SKBs)
        recovery = skb.get("recovery_metadata") or {}

        target = {
            "primitive_id": action_schema.get("primitive_id", "PICK_VERTICAL"),
            "force_profile": params.get("force_profile", {
                "max_grip_newtons": 10.0,
                "lateral_torque_limit_nm": 1.0,
                "compliance_stiffness": None,
            }),
            "target_pose_relative": params.get("target_pose_relative", {
                "x": 0.0, "y": 0.0, "z": -0.05,
                "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            }),
            "speed_profile": params.get("speed_profile", "FINE"),
            "hazard_class": safety.get("hazard_class", "NONE"),
            "physical_properties": physical_props or {
                "estimated_mass_kg": 0.1,
                "center_of_mass_offset": None,
                "friction_coefficient": 0.4,
                "dimensions": {"x": 0.05, "y": 0.05, "z": 0.05},
            },
            "object_classification": object_classification,
            "material_graph": material_graph or {
                "primary_material": "unknown",
                "component_count": 1,
            },
            "execution_narrative": action_schema.get("execution_narrative", [
                {
                    "time": "T+0.0s",
                    "state": "APPROACH",
                    "description": "Approach object from above.",
                },
                {
                    "time": "T+1.0s",
                    "state": "CONTACT",
                    "description": "Establish contact with object surface.",
                },
                {
                    "time": "T+2.0s",
                    "state": "EXECUTE",
                    "description": "Execute grasp action.",
                },
                {
                    "time": "T+3.0s",
                    "state": "VERIFY",
                    "description": "Verify post-action state via perceptual hash.",
                },
            ]),
            "failure_modes": action_schema.get("failure_modes", [
                "Object slipped during grasp",
                "Unexpected collision with adjacent object",
            ]),
            "confidence_score": recovery.get("confidence_score", 0.85),
            "reasoning_trace": recovery.get(
                "reasoning_trace",
                (
                    f"Object classified as {object_classification}. "
                    f"Selected {action_schema.get('primitive_id', 'PICK_VERTICAL')} "
                    f"with {params.get('speed_profile', 'FINE')} speed profile. "
                    f"Grip force set to "
                    f"{(params.get('force_profile') or {}).get('max_grip_newtons', 10.0)}N "
                    f"based on estimated mass and material properties. "
                    f"Hazard class: {safety.get('hazard_class', 'NONE')}."
                ),
            ),
        }

        # Include operating constraints if present
        operating_constraints = safety.get("operating_constraints")
        if operating_constraints:
            target["operating_constraints"] = operating_constraints

        # Include PPE if present
        ppe = safety.get("ppe_required", [])
        if ppe:
            target["ppe_required"] = ppe

        return target

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _encode_image(image_path: Path) -> tuple[str, str]:
        """Encode an image file to base64 and determine its MIME type.

        Returns
        -------
        tuple[str, str]
            (base64_data, mime_type)
        """
        suffix = image_path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_map.get(suffix, "image/png")

        image_bytes = image_path.read_bytes()
        b64_data = base64.standard_b64encode(image_bytes).decode("ascii")

        return b64_data, mime_type

    @staticmethod
    def _build_user_prompt(
        trigger_event: str,
        workspace_zone: str,
        end_effector: str,
        failure_context: str,
    ) -> str:
        """Build the user-facing text prompt for a training example."""
        return (
            f"TRIGGER EVENT: {trigger_event}\n"
            f"WORKSPACE ZONE: {workspace_zone}\n"
            f"END EFFECTOR: {end_effector}\n"
            f"FAILURE CONTEXT: {failure_context}\n\n"
            "Analyze the attached image of the object and generate a complete "
            "SKB action schema as raw JSON. Select the most appropriate action "
            "primitive, set safe force limits, choose the correct speed profile, "
            "estimate physical properties from the image, and provide a thorough "
            "reasoning trace explaining your decision."
        )


# ---------------------------------------------------------------------------
# 3. DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """Build Vertex AI fine-tuning datasets from existing SKBs and images.

    Scans data directories for SKB JSON files paired with object images,
    converts them to training examples, optionally augments the dataset,
    and writes train/validation JSONL splits.
    """

    def __init__(self, task_domain: str = "general_household") -> None:
        self.task_domain = task_domain
        self.converter = SKBToTrainingExample(task_domain=task_domain)
        self._rng = random.Random(42)

    # -- Directory scanning ------------------------------------------------

    def scan_directory(self, data_dir: str) -> list[dict]:
        """Find all SKB + image pairs in a data directory.

        Looks for the standard layout:
          - data_dir/skbs/<object_id>_skb.json
          - data_dir/<object_id>/*.png (or .jpg)

        Parameters
        ----------
        data_dir : str
            Root data directory to scan.

        Returns
        -------
        list[dict]
            List of dicts with keys: object_id, skb_path, image_paths, data_dir.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning("Data directory does not exist: %s", data_dir)
            return []

        skbs_dir = data_path / "skbs"
        if not skbs_dir.exists():
            logger.warning("No 'skbs' subdirectory in %s", data_dir)
            return []

        pairs = []
        for skb_file in sorted(skbs_dir.glob("*_skb.json")):
            # Derive object_id from filename: <object_id>_skb.json
            object_id = skb_file.stem.replace("_skb", "")

            # Find corresponding image directory
            obj_img_dir = data_path / object_id
            if not obj_img_dir.exists() or not obj_img_dir.is_dir():
                logger.debug(
                    "No image directory for %s (expected %s)", object_id, obj_img_dir
                )
                continue

            # Collect all image files
            image_paths = sorted(
                list(obj_img_dir.glob("*.png"))
                + list(obj_img_dir.glob("*.jpg"))
                + list(obj_img_dir.glob("*.jpeg"))
            )
            if not image_paths:
                logger.debug("No images found in %s", obj_img_dir)
                continue

            pairs.append({
                "object_id": object_id,
                "skb_path": str(skb_file),
                "image_paths": [str(p) for p in image_paths],
                "data_dir": data_dir,
            })

        logger.info(
            "Scanned %s: found %d object(s) with SKB + image pairs",
            data_dir,
            len(pairs),
        )
        return pairs

    # -- Dataset building --------------------------------------------------

    def build_dataset(
        self,
        data_dirs: list[str],
        output_path: str,
        augment: bool = True,
        split_ratio: float = 0.9,
        n_augments: int = 3,
        n_negatives: int = 20,
        anthropic_output_path: str | None = None,
    ) -> dict:
        """Build train/validation JSONL files for Vertex AI fine-tuning.

        Parameters
        ----------
        data_dirs : list[str]
            List of data directories to scan for SKB + image pairs.
        output_path : str
            Path for the training JSONL file. Validation file will be
            written alongside with a '_val' suffix.
        augment : bool
            Whether to apply image augmentation.
        split_ratio : float
            Fraction of examples for training (remainder for validation).
        n_augments : int
            Number of augmented versions per original example.
        n_negatives : int
            Number of synthetic negative examples to generate.
        anthropic_output_path : str, optional
            If provided, also write Anthropic-format JSONL to this path.

        Returns
        -------
        dict
            Statistics about the generated dataset.
        """
        # 1. Scan all directories
        all_pairs = []
        dir_counts = {}
        for data_dir in data_dirs:
            pairs = self.scan_directory(data_dir)
            dir_counts[data_dir] = len(pairs)
            all_pairs.extend(pairs)

        if not all_pairs:
            logger.error("No SKB + image pairs found in any data directory.")
            return {"error": "No data found", "directories_scanned": data_dirs}

        # 2. Convert to training examples
        examples = []
        skipped = 0
        for pair in all_pairs:
            for img_path in pair["image_paths"]:
                try:
                    example = self.converter.from_skb_file(
                        skb_path=pair["skb_path"],
                        image_path=img_path,
                        failure_context=self._rng.choice(_DEFAULT_FAILURE_CONTEXTS),
                        trigger_event=self._rng.choice(_DEFAULT_TRIGGER_EVENTS),
                    )
                    examples.append(example)
                except Exception as exc:
                    logger.warning(
                        "Failed to convert %s + %s: %s",
                        pair["skb_path"],
                        img_path,
                        exc,
                    )
                    skipped += 1

        logger.info(
            "Converted %d examples from %d objects (%d skipped)",
            len(examples),
            len(all_pairs),
            skipped,
        )

        # 3. Augment
        augmented = []
        if augment and _HAS_PIL:
            for ex in examples:
                augs = self.augment_example(ex, n_augments=n_augments)
                augmented.extend(augs)
            logger.info("Generated %d augmented examples", len(augmented))
        elif augment and not _HAS_PIL:
            logger.warning(
                "Augmentation requested but Pillow not installed. Skipping."
            )

        # 4. Generate synthetic negatives
        negatives = self.generate_synthetic_negatives(n=n_negatives)
        logger.info("Generated %d synthetic negative examples", len(negatives))

        # 5. Combine all examples
        all_examples = examples + augmented + negatives
        self._rng.shuffle(all_examples)

        # 6. Split
        split_idx = int(len(all_examples) * split_ratio)
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]

        # 7. Write JSONL files
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        val_output_path = output_path_obj.with_stem(
            output_path_obj.stem + "_val"
        )

        self._write_jsonl(train_examples, str(output_path_obj))
        self._write_jsonl(val_examples, str(val_output_path))

        # 8. Optionally write Anthropic format
        if anthropic_output_path:
            anthropic_train = [
                self.converter.format_for_anthropic(ex) for ex in train_examples
            ]
            anthropic_val = [
                self.converter.format_for_anthropic(ex) for ex in val_examples
            ]
            anthropic_path = Path(anthropic_output_path)
            anthropic_path.parent.mkdir(parents=True, exist_ok=True)
            anthropic_val_path = anthropic_path.with_stem(
                anthropic_path.stem + "_val"
            )
            self._write_jsonl(anthropic_train, str(anthropic_path))
            self._write_jsonl(anthropic_val, str(anthropic_val_path))

        # 9. Compute stats
        train_size = output_path_obj.stat().st_size
        val_size = val_output_path.stat().st_size

        stats = {
            "task_domain": self.task_domain,
            "directories_scanned": data_dirs,
            "objects_found": len(all_pairs),
            "objects_per_directory": dir_counts,
            "original_examples": len(examples),
            "augmented_examples": len(augmented),
            "negative_examples": len(negatives),
            "total_examples": len(all_examples),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "split_ratio": split_ratio,
            "skipped": skipped,
            "train_file": str(output_path_obj),
            "train_file_size_mb": round(train_size / (1024 * 1024), 2),
            "val_file": str(val_output_path),
            "val_file_size_mb": round(val_size / (1024 * 1024), 2),
            "total_size_mb": round((train_size + val_size) / (1024 * 1024), 2),
        }

        if anthropic_output_path:
            stats["anthropic_train_file"] = str(anthropic_output_path)
            stats["anthropic_val_file"] = str(
                Path(anthropic_output_path).with_stem(
                    Path(anthropic_output_path).stem + "_val"
                )
            )

        return stats

    # -- Augmentation ------------------------------------------------------

    def augment_example(
        self, example: dict, n_augments: int = 3
    ) -> list[dict]:
        """Create augmented versions of a training example.

        Applies random variations to the image (brightness, contrast,
        slight rotation, crop) while keeping the SKB target response
        identical -- the same object under different visual conditions
        should produce the same manipulation plan.

        Parameters
        ----------
        example : dict
            A Vertex AI format training example.
        n_augments : int
            Number of augmented versions to create.

        Returns
        -------
        list[dict]
            List of augmented training examples.
        """
        if not _HAS_PIL:
            return []

        augmented = []
        user_parts = example["contents"][0]["parts"]

        # Find the image part
        image_part = None
        for part in user_parts:
            if "inlineData" in part:
                image_part = part
                break

        if image_part is None:
            return []

        # Decode original image
        original_b64 = image_part["inlineData"]["data"]
        mime_type = image_part["inlineData"]["mimeType"]

        try:
            original_bytes = base64.standard_b64decode(original_b64)
            original_img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to decode image for augmentation: %s", exc)
            return []

        augmentation_ops = [
            ("brightness", lambda img: ImageEnhance.Brightness(img).enhance(
                self._rng.uniform(0.7, 1.3)
            )),
            ("contrast", lambda img: ImageEnhance.Contrast(img).enhance(
                self._rng.uniform(0.7, 1.4)
            )),
            ("rotation", lambda img: img.rotate(
                self._rng.uniform(-15, 15),
                fillcolor=(240, 240, 240),
                expand=False,
            )),
            ("crop", lambda img: self._random_crop(img)),
            ("blur", lambda img: img.filter(
                ImageFilter.GaussianBlur(radius=self._rng.uniform(0.5, 1.5))
            )),
            ("color_jitter", lambda img: ImageEnhance.Color(img).enhance(
                self._rng.uniform(0.8, 1.2)
            )),
        ]

        for i in range(n_augments):
            # Apply 1-3 random augmentation operations
            n_ops = self._rng.randint(1, 3)
            ops = self._rng.sample(augmentation_ops, min(n_ops, len(augmentation_ops)))

            aug_img = original_img.copy()
            for op_name, op_fn in ops:
                try:
                    aug_img = op_fn(aug_img)
                except Exception as exc:
                    logger.debug("Augmentation '%s' failed: %s", op_name, exc)

            # Encode augmented image
            buf = io.BytesIO()
            fmt = "PNG" if "png" in mime_type else "JPEG"
            aug_img.save(buf, format=fmt)
            aug_b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")

            # Build augmented example (deep copy, replace image data)
            aug_example = copy.deepcopy(example)
            for part in aug_example["contents"][0]["parts"]:
                if "inlineData" in part:
                    part["inlineData"]["data"] = aug_b64
                    break

            augmented.append(aug_example)

        return augmented

    def _random_crop(self, img: Image.Image) -> Image.Image:
        """Apply a random center-biased crop, then resize back to original."""
        w, h = img.size
        crop_fraction = self._rng.uniform(0.8, 0.95)
        new_w = int(w * crop_fraction)
        new_h = int(h * crop_fraction)
        left = self._rng.randint(0, w - new_w)
        top = self._rng.randint(0, h - new_h)
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.LANCZOS)

    # -- Synthetic negatives -----------------------------------------------

    def generate_synthetic_negatives(self, n: int = 20) -> list[dict]:
        """Generate failure-case training examples for robustness.

        These teach the model to output ABORT or low-confidence responses
        when presented with empty scenes, heavily occluded objects, or
        unrecognisable inputs.

        Parameters
        ----------
        n : int
            Number of negative examples to generate.

        Returns
        -------
        list[dict]
            Training examples with ABORT or low-confidence targets.
        """
        negatives = []
        negative_scenarios = [
            {
                "failure_context": "Empty scene -- no object detected in workspace",
                "target": {
                    "primitive_id": "PICK_VERTICAL",
                    "confidence_score": 0.05,
                    "reasoning_trace": (
                        "ABORT: No object detected in the scene. The workspace "
                        "appears empty or the camera feed may be obstructed. "
                        "Cannot generate a manipulation plan without a target object. "
                        "Recommend re-scanning or human verification."
                    ),
                    "hazard_class": "NONE",
                    "speed_profile": "GUARDED",
                    "force_profile": {
                        "max_grip_newtons": 0.0,
                        "lateral_torque_limit_nm": 0.0,
                        "compliance_stiffness": None,
                    },
                    "target_pose_relative": {
                        "x": 0.0, "y": 0.0, "z": 0.0,
                        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                    },
                    "physical_properties": None,
                    "object_classification": "Rigid_Body_Graspable",
                    "material_graph": None,
                    "execution_narrative": [],
                    "failure_modes": ["No object in scene", "Camera obstruction possible"],
                },
            },
            {
                "failure_context": "Object heavily occluded -- less than 20% visible",
                "target": {
                    "primitive_id": "PUSH",
                    "confidence_score": 0.15,
                    "reasoning_trace": (
                        "Object is heavily occluded with less than 20% visible. "
                        "Cannot reliably estimate physical properties or grasp points. "
                        "Recommending PUSH to reposition surrounding objects and reveal "
                        "the target, followed by re-scanning. Low confidence due to "
                        "insufficient visual information."
                    ),
                    "hazard_class": "NONE",
                    "speed_profile": "GUARDED",
                    "force_profile": {
                        "max_grip_newtons": 5.0,
                        "lateral_torque_limit_nm": 1.0,
                        "compliance_stiffness": None,
                    },
                    "target_pose_relative": {
                        "x": 0.05, "y": 0.0, "z": 0.0,
                        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                    },
                    "physical_properties": None,
                    "object_classification": "Rigid_Body_Graspable",
                    "material_graph": None,
                    "execution_narrative": [
                        {
                            "time": "T+0.0s",
                            "state": "APPROACH",
                            "description": "Slowly approach the occluded region.",
                        },
                        {
                            "time": "T+1.0s",
                            "state": "EXECUTE",
                            "description": "Push adjacent objects to reveal target.",
                        },
                        {
                            "time": "T+3.0s",
                            "state": "VERIFY",
                            "description": "Re-scan scene for improved visibility.",
                        },
                    ],
                    "failure_modes": [
                        "Target object still occluded after push",
                        "Pushed objects caused cascade",
                        "Target object damaged by contact",
                    ],
                },
            },
            {
                "failure_context": "Unknown object -- not in any training distribution",
                "target": {
                    "primitive_id": "PICK_VERTICAL",
                    "confidence_score": 0.25,
                    "reasoning_trace": (
                        "Object type is not recognized from any known training "
                        "distribution. Applying conservative default PICK_VERTICAL "
                        "with minimal force and guarded speed. Human verification "
                        "recommended before repeating this manipulation."
                    ),
                    "hazard_class": "NONE",
                    "speed_profile": "GUARDED",
                    "force_profile": {
                        "max_grip_newtons": 5.0,
                        "lateral_torque_limit_nm": 1.0,
                        "compliance_stiffness": None,
                    },
                    "target_pose_relative": {
                        "x": 0.0, "y": 0.0, "z": -0.03,
                        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                    },
                    "physical_properties": {
                        "estimated_mass_kg": 0.1,
                        "center_of_mass_offset": None,
                        "friction_coefficient": 0.3,
                        "dimensions": {"x": 0.05, "y": 0.05, "z": 0.05},
                    },
                    "object_classification": "Rigid_Body_Graspable",
                    "material_graph": {
                        "primary_material": "unknown",
                        "component_count": 1,
                    },
                    "execution_narrative": [
                        {
                            "time": "T+0.0s",
                            "state": "APPROACH",
                            "description": "Cautiously approach unknown object.",
                        },
                        {
                            "time": "T+1.5s",
                            "state": "CONTACT",
                            "description": "Establish gentle contact to probe object properties.",
                        },
                        {
                            "time": "T+3.0s",
                            "state": "EXECUTE",
                            "description": "Attempt minimal-force vertical pick.",
                        },
                        {
                            "time": "T+5.0s",
                            "state": "VERIFY",
                            "description": "Verify grasp and request human confirmation.",
                        },
                    ],
                    "failure_modes": [
                        "Object too heavy for estimated grip force",
                        "Object material incompatible with gripper",
                        "Unexpected deformability",
                        "Hidden hazard (sharp edge, chemical, electrical)",
                    ],
                },
            },
            {
                "failure_context": "Blurry image -- camera out of focus or in motion",
                "target": {
                    "primitive_id": "PICK_VERTICAL",
                    "confidence_score": 0.10,
                    "reasoning_trace": (
                        "ABORT: Image quality is too poor to reliably identify the "
                        "object or estimate its properties. The camera appears to be "
                        "out of focus or the image was captured during motion. "
                        "Recommend recapturing the image with a stable camera."
                    ),
                    "hazard_class": "NONE",
                    "speed_profile": "GUARDED",
                    "force_profile": {
                        "max_grip_newtons": 0.0,
                        "lateral_torque_limit_nm": 0.0,
                        "compliance_stiffness": None,
                    },
                    "target_pose_relative": {
                        "x": 0.0, "y": 0.0, "z": 0.0,
                        "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                    },
                    "physical_properties": None,
                    "object_classification": "Rigid_Body_Graspable",
                    "material_graph": None,
                    "execution_narrative": [],
                    "failure_modes": [
                        "Cannot identify object from blurry image",
                        "Risk of misidentification leading to unsafe action",
                    ],
                },
            },
        ]

        for i in range(n):
            scenario = negative_scenarios[i % len(negative_scenarios)]

            # Generate a small synthetic image (grey noise or blank)
            image_b64, mime_type = self._generate_negative_image(
                scenario_type=scenario["failure_context"]
            )

            user_text = SKBToTrainingExample._build_user_prompt(
                trigger_event=self._rng.choice(_DEFAULT_TRIGGER_EVENTS),
                workspace_zone=self._rng.choice(_DEFAULT_WORKSPACE_ZONES),
                end_effector=self._rng.choice(_DEFAULT_END_EFFECTORS),
                failure_context=scenario["failure_context"],
            )

            example = {
                "systemInstruction": {
                    "parts": [{"text": self.converter.domain["system_prompt"]}]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": user_text},
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": image_b64,
                                }
                            },
                        ],
                    },
                    {
                        "role": "model",
                        "parts": [
                            {
                                "text": json.dumps(
                                    scenario["target"], indent=2, sort_keys=False
                                )
                            }
                        ],
                    },
                ],
            }

            negatives.append(example)

        return negatives

    def _generate_negative_image(self, scenario_type: str) -> tuple[str, str]:
        """Generate a synthetic negative image (blank, noisy, or blurred).

        Returns (base64_data, mime_type).
        """
        mime_type = "image/png"
        width, height = 160, 120

        if _HAS_PIL:
            if "empty" in scenario_type.lower():
                # Plain grey background
                img = Image.new("RGB", (width, height), color=(200, 200, 200))
            elif "blurry" in scenario_type.lower():
                # Random noise, heavily blurred
                pixels = bytes(
                    self._rng.randint(100, 200)
                    for _ in range(width * height * 3)
                )
                img = Image.frombytes("RGB", (width, height), pixels)
                img = img.filter(ImageFilter.GaussianBlur(radius=8))
            elif "occluded" in scenario_type.lower():
                # Mostly dark with a small visible region
                img = Image.new("RGB", (width, height), color=(40, 40, 40))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                draw.rectangle(
                    [60, 40, 100, 80], fill=(160, 160, 160), outline=(80, 80, 80)
                )
            else:
                # Random noise
                pixels = bytes(
                    self._rng.randint(0, 255)
                    for _ in range(width * height * 3)
                )
                img = Image.frombytes("RGB", (width, height), pixels)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64_data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        else:
            # Fallback: minimal 1x1 grey PNG (base64)
            # This is a valid 1x1 grey PNG
            b64_data = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
                "2mN88P8/AwAJUgHRW5f2MQAAAABJRU5ErkJggg=="
            )

        return b64_data, mime_type

    # -- I/O helpers -------------------------------------------------------

    @staticmethod
    def _write_jsonl(examples: list[dict], path: str) -> None:
        """Write a list of dicts to a JSONL file (one JSON object per line)."""
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, separators=(",", ":")) + "\n")
        logger.info("Wrote %d examples to %s", len(examples), path)


# ---------------------------------------------------------------------------
# 4. CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Vertex AI / Anthropic fine-tuning datasets from "
            "GCP-Robotics SKB + image pairs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python gcp_robotics/vertex_tuning.py \\\n"
            "    --data-dirs data/standard_evaluation data/ycb_20 \\\n"
            "    --output data/training/skb_tuning_v1.jsonl \\\n"
            "    --domain general_household \\\n"
            "    --augment"
        ),
    )

    parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="Data directories to scan for SKB + image pairs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the training JSONL file.",
    )
    parser.add_argument(
        "--domain",
        default="general_household",
        choices=sorted(TASK_DOMAINS.keys()),
        help="Task domain for system prompt selection (default: general_household).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="Enable image augmentation to expand the dataset.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        default=False,
        help="Disable image augmentation.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        help="Train/validation split ratio (default: 0.9).",
    )
    parser.add_argument(
        "--n-augments",
        type=int,
        default=3,
        help="Number of augmented versions per example (default: 3).",
    )
    parser.add_argument(
        "--n-negatives",
        type=int,
        default=20,
        help="Number of synthetic negative examples (default: 20).",
    )
    parser.add_argument(
        "--anthropic-output",
        default=None,
        help="Optional path for Anthropic-format JSONL output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # Resolve --augment / --no-augment
    do_augment = args.augment and not args.no_augment

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"{'=' * 70}")
    print(f"  Vertex AI Fine-Tuning Data Generator")
    print(f"  GCP-Robotics SKB Schema v2.0")
    print(f"{'=' * 70}")
    print(f"  Domain:      {args.domain}")
    print(f"  Data dirs:   {args.data_dirs}")
    print(f"  Output:      {args.output}")
    print(f"  Augment:     {do_augment}")
    print(f"  Split:       {args.split}")
    print(f"  N-augments:  {args.n_augments}")
    print(f"  N-negatives: {args.n_negatives}")
    if args.anthropic_output:
        print(f"  Anthropic:   {args.anthropic_output}")
    print(f"{'=' * 70}\n")

    builder = DatasetBuilder(task_domain=args.domain)

    stats = builder.build_dataset(
        data_dirs=args.data_dirs,
        output_path=args.output,
        augment=do_augment,
        split_ratio=args.split,
        n_augments=args.n_augments,
        n_negatives=args.n_negatives,
        anthropic_output_path=args.anthropic_output,
    )

    # Print report
    print(f"\n{'=' * 70}")
    print(f"  DATASET GENERATION REPORT")
    print(f"{'=' * 70}")

    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        sys.exit(1)

    print(f"  Task domain:          {stats['task_domain']}")
    print(f"  Directories scanned:  {len(stats['directories_scanned'])}")
    for d, count in stats["objects_per_directory"].items():
        dir_name = Path(d).name
        print(f"    - {dir_name}: {count} objects")
    print(f"  Total objects:        {stats['objects_found']}")
    print(f"  Original examples:    {stats['original_examples']}")
    print(f"  Augmented examples:   {stats['augmented_examples']}")
    print(f"  Negative examples:    {stats['negative_examples']}")
    print(f"  Total examples:       {stats['total_examples']}")
    print(f"  {'---'}")
    print(f"  Train examples:       {stats['train_examples']}")
    print(f"  Val examples:         {stats['val_examples']}")
    print(f"  Split ratio:          {stats['split_ratio']}")
    print(f"  Skipped:              {stats['skipped']}")
    print(f"  {'---'}")
    print(f"  Train file:           {stats['train_file']}")
    print(f"  Train file size:      {stats['train_file_size_mb']} MB")
    print(f"  Val file:             {stats['val_file']}")
    print(f"  Val file size:        {stats['val_file_size_mb']} MB")
    print(f"  Total size:           {stats['total_size_mb']} MB")
    if "anthropic_train_file" in stats:
        print(f"  Anthropic train:      {stats['anthropic_train_file']}")
        print(f"  Anthropic val:        {stats['anthropic_val_file']}")
    print(f"{'=' * 70}")
    print(f"  Done.")


if __name__ == "__main__":
    main()

"""
Slow Path (System 2) -- LLM-based generative recovery for the Golden Codex Protocol.

When the Fast Path hash lookup fails (hash miss), this module invokes an LLM to
analyse the scene and generate a new Spatial Kinematic Blueprint (SKB) schema.
The generated SKB can then be validated, optionally re-prompted for corrections,
and promoted back into the Fast Path cache for future O(1) lookups.

Copyright (c) 2026 Metavolve Labs -- Robotics R&D Division
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed value sets (mirrored from schema.models enumerations)
# ---------------------------------------------------------------------------

ALLOWED_PRIMITIVES = frozenset({
    "PICK_VERTICAL",
    "PICK_ANGLED",
    "GRASP_PARALLEL_JAW",
    "GRASP_SUCTION",
    "PUSH",
    "SLIDE",
    "ROTATE_IN_HAND",
    "SHAKE_BIN_HORIZONTAL",
    "INSERT_PRESS_FIT",
    "PLACE_PRECISE",
    "SCREW_DRIVE",
    "FLIP",
    "HANDOVER",
})

ALLOWED_SPEED_PROFILES = frozenset({
    "RAPID",
    "FINE",
    "RAPID_OSCILLATION",
    "GUARDED",
})

ALLOWED_HAZARD_CLASSES = frozenset({
    "NONE",
    "SHARP_EDGES",
    "HOT_SURFACE",
    "CHEMICAL",
    "ELECTRICAL",
    "HEAVY_LOAD",
    "PINCH_POINT",
})

# Safety limits (defaults -- can be overridden via constraints)
DEFAULT_MAX_GRIP_FORCE_N = 50.0
DEFAULT_MAX_TORQUE_NM = 20.0

# ---------------------------------------------------------------------------
# 1. SYSTEM_PROMPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a Robotic Recovery Supervisor operating within the Golden Codex Protocol. "
    "You analyze visual scenes where the Fast Path hash lookup has failed, and generate "
    "structured Spatial Kinematic Blueprint (SKB) schemas to resolve the situation.\n"
    "\n"
    "AVAILABLE ACTION PRIMITIVES:\n"
    "  PICK_VERTICAL, PICK_ANGLED, GRASP_PARALLEL_JAW, GRASP_SUCTION, PUSH, SLIDE, "
    "ROTATE_IN_HAND, SHAKE_BIN_HORIZONTAL, INSERT_PRESS_FIT, PLACE_PRECISE, SCREW_DRIVE, "
    "FLIP, HANDOVER\n"
    "\n"
    "SPEED PROFILES:\n"
    "  RAPID, FINE, RAPID_OSCILLATION, GUARDED\n"
    "\n"
    "HAZARD CLASSES:\n"
    "  NONE, SHARP_EDGES, HOT_SURFACE, CHEMICAL, ELECTRICAL, HEAVY_LOAD, PINCH_POINT\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "  You MUST output ONLY valid JSON matching the SKB schema. No conversational text, "
    "no markdown fences, no explanations outside the JSON structure -- just raw JSON.\n"
    "\n"
    "SAFETY CONSTRAINTS:\n"
    "  - Never exceed 50N grip force without explicit authorization.\n"
    "  - Always set the hazard_class field.\n"
    "  - Always provide a reasoning_trace field explaining your decision.\n"
    "\n"
    "REQUIRED OUTPUT FIELDS:\n"
    "  Your JSON response MUST include at minimum:\n"
    "    - primitive_id: one of the allowed action primitives\n"
    "    - reasoning_trace: a string explaining why this action was selected\n"
    "    - confidence_score: a float between 0.0 and 1.0\n"
    "    - hazard_class: one of the allowed hazard classes\n"
    "    - speed_profile: one of the allowed speed profiles\n"
    "    - force_profile: object with max_grip_newtons and lateral_torque_limit_nm\n"
    "    - target_pose_relative: object with x, y, z, qx, qy, qz, qw\n"
    "    - execution_narrative: list of {time, state, description} objects\n"
    "    - failure_modes: list of strings describing potential failure modes\n"
)

# ---------------------------------------------------------------------------
# 2. ANALYSIS_PROMPT_TEMPLATE
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT_TEMPLATE: str = (
    "TRIGGER EVENT: {trigger_event}\n"
    "WORKSPACE ZONE: {workspace_zone}\n"
    "END EFFECTOR: {end_effector}\n"
    "AVAILABLE OBJECTS: {available_objects}\n"
    "FAILURE CONTEXT: {failure_context}\n"
    "PREVIOUS ACTION: {previous_action}\n"
    "CONSTRAINTS: {constraints}\n"
    "\n"
    "Analyze the situation above and generate a complete SKB action schema as raw JSON. "
    "Select the most appropriate action primitive, set safe force limits, choose the "
    "correct speed profile, and provide a thorough reasoning trace explaining your "
    "decision. Consider the failure context carefully and ensure your response addresses "
    "the root cause of the failure."
)

# ---------------------------------------------------------------------------
# 3. SlowPathGenerator -- real LLM-backed generator
# ---------------------------------------------------------------------------


class SlowPathGenerator:
    """Generates Spatial Kinematic Blueprints via the Anthropic LLM API.

    This is the System 2 (Slow Path) generative recovery mechanism. When
    the Fast Path hash lookup fails, this class builds a prompt from the
    scene context, calls the Anthropic Messages API, parses and validates
    the returned JSON, and produces a promotion-ready SKB entry.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

        if not self.api_key:
            logger.warning(
                "No Anthropic API key provided and ANTHROPIC_API_KEY env var is not set. "
                "API calls will fail until a key is configured."
            )

    # -- internal helpers --------------------------------------------------

    def _get_client(self):
        """Lazily import and instantiate the Anthropic client."""
        try:
            import anthropic  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for SlowPathGenerator. "
                "Install it with: pip install anthropic"
            ) from exc
        return anthropic.Anthropic(api_key=self.api_key)

    def _build_user_content(
        self,
        trigger_event: str,
        workspace_zone: str,
        end_effector: str,
        failure_context: str,
        image_path: str | None = None,
        available_objects: list | None = None,
        previous_action: str | None = None,
        constraints: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Build the user message content blocks (text + optional image)."""
        content_blocks: list[dict[str, Any]] = []

        # If an image is provided, include it as a base64 content block
        if image_path is not None:
            img_path = Path(image_path)
            if not img_path.exists():
                logger.warning("Image path does not exist: %s", image_path)
            else:
                image_bytes = img_path.read_bytes()
                b64_data = base64.standard_b64encode(image_bytes).decode("ascii")

                # Determine media type from suffix
                suffix = img_path.suffix.lower()
                media_type_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                media_type = media_type_map.get(suffix, "image/png")

                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
                logger.info(
                    "Attached image %s (%s, %d bytes)",
                    image_path, media_type, len(image_bytes),
                )

        # Build the text prompt
        prompt_text = ANALYSIS_PROMPT_TEMPLATE.format(
            trigger_event=trigger_event,
            workspace_zone=workspace_zone,
            end_effector=end_effector,
            available_objects=", ".join(available_objects) if available_objects else "N/A",
            failure_context=failure_context,
            previous_action=previous_action or "N/A",
            constraints=json.dumps(constraints) if constraints else "N/A",
        )
        content_blocks.append({"type": "text", "text": prompt_text})

        return content_blocks

    @staticmethod
    def _parse_json_response(raw_text: str) -> dict:
        """Parse JSON from the LLM response, stripping markdown fences if present."""
        text = raw_text.strip()

        # Strip markdown code fences if the model added them despite instructions
        if text.startswith("```"):
            # Remove opening fence (possibly with language tag)
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].rstrip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM response as JSON: %s", exc)
            raise ValueError(
                f"LLM response is not valid JSON: {exc}\nRaw response:\n{raw_text[:500]}"
            ) from exc

    # -- public API --------------------------------------------------------

    def generate_skb(
        self,
        trigger_event: str,
        workspace_zone: str,
        end_effector: str,
        failure_context: str,
        image_path: str | None = None,
        available_objects: list | None = None,
        previous_action: str | None = None,
        constraints: dict | None = None,
    ) -> dict:
        """Generate a Spatial Kinematic Blueprint via the Anthropic LLM API.

        Builds the prompt from templates, calls the API with tenacity retry
        (3 attempts, exponential backoff), and parses the JSON response.

        Parameters
        ----------
        trigger_event : str
            Why the Slow Path was triggered (e.g. ``"HASH_MISS_OOD_STATE"``).
        workspace_zone : str
            Logical zone identifier where the robot is operating.
        end_effector : str
            Gripper or tool currently attached.
        failure_context : str
            Human-readable description of what happened.
        image_path : str, optional
            Filesystem path to a scene image to include in the prompt.
        available_objects : list, optional
            Known objects in the workspace from the registry.
        previous_action : str, optional
            The action primitive that was attempted before the failure.
        constraints : dict, optional
            Additional constraints (max force, speed limits, etc.).

        Returns
        -------
        dict
            Parsed SKB action schema from the LLM.

        Raises
        ------
        ValueError
            If the response cannot be parsed as JSON or is missing required fields.
        ImportError
            If the ``anthropic`` SDK is not installed.
        """
        try:
            from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
        except ImportError:
            logger.warning(
                "tenacity package not installed; retries will not be available."
            )
            return self._call_api(
                trigger_event=trigger_event,
                workspace_zone=workspace_zone,
                end_effector=end_effector,
                failure_context=failure_context,
                image_path=image_path,
                available_objects=available_objects,
                previous_action=previous_action,
                constraints=constraints,
            )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call_with_retry():
            return self._call_api(
                trigger_event=trigger_event,
                workspace_zone=workspace_zone,
                end_effector=end_effector,
                failure_context=failure_context,
                image_path=image_path,
                available_objects=available_objects,
                previous_action=previous_action,
                constraints=constraints,
            )

        return _call_with_retry()

    def _call_api(
        self,
        trigger_event: str,
        workspace_zone: str,
        end_effector: str,
        failure_context: str,
        image_path: str | None = None,
        available_objects: list | None = None,
        previous_action: str | None = None,
        constraints: dict | None = None,
    ) -> dict:
        """Execute a single API call and parse the response."""
        client = self._get_client()
        user_content = self._build_user_content(
            trigger_event=trigger_event,
            workspace_zone=workspace_zone,
            end_effector=end_effector,
            failure_context=failure_context,
            image_path=image_path,
            available_objects=available_objects,
            previous_action=previous_action,
            constraints=constraints,
        )

        logger.info(
            "Calling Anthropic API (model=%s) for slow-path generation...",
            self.model,
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        raw_text = response.content[0].text
        logger.debug("Raw LLM response (%d chars): %s", len(raw_text), raw_text[:200])

        parsed = self._parse_json_response(raw_text)

        # Basic required-field check
        required_fields = {"primitive_id", "reasoning_trace", "confidence_score"}
        missing = required_fields - set(parsed.keys())
        if missing:
            raise ValueError(
                f"LLM response is missing required fields: {missing}"
            )

        return parsed

    def validate_skb_response(
        self, response_data: dict
    ) -> tuple[bool, list[str]]:
        """Validate the LLM-generated SKB action schema.

        Checks that all fields have valid values within allowed ranges and
        enforces safety limits.

        Parameters
        ----------
        response_data : dict
            The parsed JSON from the LLM.

        Returns
        -------
        tuple[bool, list[str]]
            ``(is_valid, errors)`` -- True if valid, with an empty error list.
        """
        errors: list[str] = []

        # primitive_id
        primitive_id = response_data.get("primitive_id")
        if primitive_id is None:
            errors.append("Missing required field: primitive_id")
        elif primitive_id not in ALLOWED_PRIMITIVES:
            errors.append(
                f"Invalid primitive_id '{primitive_id}'. "
                f"Must be one of: {sorted(ALLOWED_PRIMITIVES)}"
            )

        # confidence_score
        confidence = response_data.get("confidence_score")
        if confidence is None:
            errors.append("Missing required field: confidence_score")
        elif not isinstance(confidence, (int, float)):
            errors.append(
                f"confidence_score must be a number, got {type(confidence).__name__}"
            )
        elif not (0.0 <= confidence <= 1.0):
            errors.append(
                f"confidence_score must be between 0.0 and 1.0, got {confidence}"
            )

        # reasoning_trace
        if not response_data.get("reasoning_trace"):
            errors.append("Missing or empty required field: reasoning_trace")

        # hazard_class
        hazard_class = response_data.get("hazard_class")
        if hazard_class is None:
            errors.append("Missing required field: hazard_class")
        elif hazard_class not in ALLOWED_HAZARD_CLASSES:
            errors.append(
                f"Invalid hazard_class '{hazard_class}'. "
                f"Must be one of: {sorted(ALLOWED_HAZARD_CLASSES)}"
            )

        # speed_profile (optional but validated if present)
        speed_profile = response_data.get("speed_profile")
        if speed_profile is not None and speed_profile not in ALLOWED_SPEED_PROFILES:
            errors.append(
                f"Invalid speed_profile '{speed_profile}'. "
                f"Must be one of: {sorted(ALLOWED_SPEED_PROFILES)}"
            )

        # force_profile safety checks
        force_profile = response_data.get("force_profile", {})
        if isinstance(force_profile, dict):
            grip_force = force_profile.get("max_grip_newtons")
            if grip_force is not None and isinstance(grip_force, (int, float)):
                if grip_force > DEFAULT_MAX_GRIP_FORCE_N:
                    errors.append(
                        f"max_grip_newtons ({grip_force}N) exceeds safety limit "
                        f"of {DEFAULT_MAX_GRIP_FORCE_N}N without explicit authorization."
                    )

            torque = force_profile.get("lateral_torque_limit_nm")
            if torque is not None and isinstance(torque, (int, float)):
                if torque > DEFAULT_MAX_TORQUE_NM:
                    errors.append(
                        f"lateral_torque_limit_nm ({torque}Nm) exceeds safety limit "
                        f"of {DEFAULT_MAX_TORQUE_NM}Nm without explicit authorization."
                    )

        is_valid = len(errors) == 0
        if is_valid:
            logger.info("SKB validation passed.")
        else:
            logger.warning("SKB validation failed with %d error(s): %s", len(errors), errors)

        return is_valid, errors

    def generate_and_validate(self, **kwargs) -> dict:
        """Generate an SKB and validate it, re-prompting on validation failure.

        Calls :meth:`generate_skb`, then :meth:`validate_skb_response`. If
        validation fails, appends the error messages to the conversation and
        re-prompts the LLM up to 2 additional times.

        Parameters
        ----------
        **kwargs
            All keyword arguments are forwarded to :meth:`generate_skb`.

        Returns
        -------
        dict
            The validated SKB action schema.

        Raises
        ------
        ValueError
            If validation still fails after all retries.
        """
        max_validation_retries = 2
        last_errors: list[str] = []

        for attempt in range(1 + max_validation_retries):
            if attempt == 0:
                logger.info("Generating SKB (attempt %d)...", attempt + 1)
                skb = self.generate_skb(**kwargs)
            else:
                logger.info(
                    "Re-prompting LLM for corrected SKB (attempt %d, errors: %s)...",
                    attempt + 1, last_errors,
                )
                skb = self._reprompt_with_errors(last_errors, **kwargs)

            is_valid, errors = self.validate_skb_response(skb)
            if is_valid:
                logger.info("SKB generated and validated successfully on attempt %d.", attempt + 1)
                return skb

            last_errors = errors
            logger.warning(
                "Validation failed on attempt %d: %s", attempt + 1, errors
            )

        raise ValueError(
            f"SKB validation failed after {1 + max_validation_retries} attempts. "
            f"Last errors: {last_errors}"
        )

    def _reprompt_with_errors(self, errors: list[str], **kwargs) -> dict:
        """Re-prompt the LLM with the validation errors for correction."""
        client = self._get_client()
        user_content = self._build_user_content(
            trigger_event=kwargs["trigger_event"],
            workspace_zone=kwargs["workspace_zone"],
            end_effector=kwargs["end_effector"],
            failure_context=kwargs["failure_context"],
            image_path=kwargs.get("image_path"),
            available_objects=kwargs.get("available_objects"),
            previous_action=kwargs.get("previous_action"),
            constraints=kwargs.get("constraints"),
        )

        error_text = (
            "Your previous response had validation errors. Please correct them and "
            "output ONLY the corrected JSON.\n\nValidation errors:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

        messages = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": "I understand. Let me correct the errors and provide a valid SKB.",
            },
            {"role": "user", "content": error_text},
        ]

        logger.info("Re-prompting LLM with %d validation error(s)...", len(errors))
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        raw_text = response.content[0].text
        return self._parse_json_response(raw_text)

    def format_promotion_entry(
        self,
        trigger_hash: str,
        generated_skb: dict,
        generation_time_ms: float,
    ) -> dict:
        """Format the generated SKB for promotion to the Fast Path cache.

        Adds recovery metadata and pipeline status fields so the entry can
        be ingested directly by the Fast Path registry.

        Parameters
        ----------
        trigger_hash : str
            The perceptual hash that triggered the slow-path lookup.
        generated_skb : dict
            The validated SKB action schema from the LLM.
        generation_time_ms : float
            Wall-clock time taken for the slow-path generation in milliseconds.

        Returns
        -------
        dict
            A promotion-ready entry with recovery metadata.
        """
        now = datetime.now(timezone.utc)
        entry = {
            "uuid": str(_uuid.uuid4()),
            "trigger_hash": trigger_hash,
            "schema_version": "2.0-GCP-ROBOTICS",
            "agent": "System_2_Generative_Supervisor",
            "pipeline_status": "PROMOTED_TO_FAST_PATH",
            "generated_skb": generated_skb,
            "recovery_metadata": {
                "generated_by_slow_path": True,
                "trigger_event": generated_skb.get("trigger_event", "HASH_MISS_OOD_STATE"),
                "reasoning_trace": generated_skb.get("reasoning_trace", ""),
                "confidence_score": generated_skb.get("confidence_score", 0.0),
                "llm_model_version": self.model,
            },
            "timestamps": {
                "schema_generated": now.isoformat(),
                "promoted_to_fast_path": now.isoformat(),
            },
            "telemetry": {
                "slow_path_latency_ms": generation_time_ms,
            },
        }

        logger.info(
            "Formatted promotion entry: uuid=%s, trigger_hash=%s, latency=%.1fms",
            entry["uuid"], trigger_hash, generation_time_ms,
        )
        return entry


# ---------------------------------------------------------------------------
# 4. MockSlowPathGenerator -- for testing without API access
# ---------------------------------------------------------------------------


class MockSlowPathGenerator:
    """Mock Slow Path generator for testing without the Anthropic API.

    Produces plausible SKB responses using simple keyword heuristics on
    the ``failure_context`` string. No network calls are made.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mock-model-v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        logger.info("MockSlowPathGenerator initialised (no API calls will be made).")

    def generate_skb(
        self,
        trigger_event: str,
        workspace_zone: str,
        end_effector: str,
        failure_context: str,
        image_path: str | None = None,
        available_objects: list | None = None,
        previous_action: str | None = None,
        constraints: dict | None = None,
    ) -> dict:
        """Generate a mock SKB based on keyword heuristics.

        Examines ``failure_context`` for keywords and selects an appropriate
        action primitive. Returns a complete SKB dict with all required fields.
        """
        context_lower = failure_context.lower()

        # Heuristic selection
        if "entangle" in context_lower or "stuck" in context_lower:
            primitive = "SHAKE_BIN_HORIZONTAL"
            speed = "RAPID_OSCILLATION"
            grip_force = 25.0
            reasoning = (
                "Heuristic: failure context indicates entanglement or stuck parts. "
                "SHAKE_BIN_HORIZONTAL selected to dislodge entangled objects using "
                "rapid oscillation before reattempting pick."
            )
        elif "slip" in context_lower or "drop" in context_lower:
            primitive = "GRASP_PARALLEL_JAW"
            speed = "FINE"
            grip_force = 40.0
            reasoning = (
                "Heuristic: failure context indicates slippage or dropped object. "
                "GRASP_PARALLEL_JAW selected with higher grip force (40N) to ensure "
                "secure hold during subsequent manipulation."
            )
        elif "rotated" in context_lower or "flipped" in context_lower:
            if "flipped" in context_lower:
                primitive = "FLIP"
            else:
                primitive = "ROTATE_IN_HAND"
            speed = "FINE"
            grip_force = 30.0
            reasoning = (
                "Heuristic: failure context indicates unexpected object rotation. "
                f"{primitive} selected to reorient the object to the expected pose "
                "before continuing the task sequence."
            )
        elif "blocked" in context_lower or "obstacle" in context_lower:
            primitive = "PUSH"
            speed = "GUARDED"
            grip_force = 20.0
            reasoning = (
                "Heuristic: failure context indicates a blocked path or obstacle. "
                "PUSH selected with guarded speed to clear the obstruction before "
                "reattempting the primary task."
            )
        else:
            primitive = "PICK_VERTICAL"
            speed = "FINE"
            grip_force = 30.0
            reasoning = (
                "Heuristic: no specific failure pattern matched in context. "
                "Defaulting to PICK_VERTICAL with moderate grip force as a safe "
                "general-purpose recovery action."
            )

        skb = {
            "primitive_id": primitive,
            "reasoning_trace": reasoning,
            "confidence_score": 0.85,
            "trigger_event": trigger_event,
            "hazard_class": "NONE",
            "speed_profile": speed,
            "force_profile": {
                "max_grip_newtons": grip_force,
                "lateral_torque_limit_nm": 10.0,
                "compliance_stiffness": {"x": 500.0, "y": 500.0, "z": 1000.0},
            },
            "target_pose_relative": {
                "x": 0.0,
                "y": 0.0,
                "z": -0.05,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
            "execution_narrative": [
                {
                    "time": "T+0.0s",
                    "state": "APPROACH",
                    "description": f"Approach workspace zone {workspace_zone} with {end_effector}.",
                },
                {
                    "time": "T+1.0s",
                    "state": "CONTACT",
                    "description": f"Initiate {primitive} with {speed} speed profile.",
                },
                {
                    "time": "T+2.5s",
                    "state": "EXECUTE",
                    "description": f"Execute {primitive} action on target object.",
                },
                {
                    "time": "T+4.0s",
                    "state": "VERIFY",
                    "description": "Verify post-action state via perceptual hash comparison.",
                },
            ],
            "failure_modes": [
                "Object slipped during grasp",
                "Unexpected collision with neighbouring object",
                "End-effector failed to reach target pose",
                "Post-action hash mismatch (object not in expected state)",
            ],
            "workspace_zone": workspace_zone,
            "end_effector": end_effector,
            "available_objects": available_objects or [],
            "previous_action": previous_action,
        }

        logger.info(
            "Mock SKB generated: primitive=%s, confidence=%.2f, trigger=%s",
            primitive, skb["confidence_score"], trigger_event,
        )
        return skb

    def validate_skb_response(
        self, response_data: dict
    ) -> tuple[bool, list[str]]:
        """Validate using the same logic as SlowPathGenerator."""
        errors: list[str] = []

        primitive_id = response_data.get("primitive_id")
        if primitive_id is None:
            errors.append("Missing required field: primitive_id")
        elif primitive_id not in ALLOWED_PRIMITIVES:
            errors.append(
                f"Invalid primitive_id '{primitive_id}'. "
                f"Must be one of: {sorted(ALLOWED_PRIMITIVES)}"
            )

        confidence = response_data.get("confidence_score")
        if confidence is None:
            errors.append("Missing required field: confidence_score")
        elif not isinstance(confidence, (int, float)):
            errors.append(
                f"confidence_score must be a number, got {type(confidence).__name__}"
            )
        elif not (0.0 <= confidence <= 1.0):
            errors.append(
                f"confidence_score must be between 0.0 and 1.0, got {confidence}"
            )

        if not response_data.get("reasoning_trace"):
            errors.append("Missing or empty required field: reasoning_trace")

        hazard_class = response_data.get("hazard_class")
        if hazard_class is None:
            errors.append("Missing required field: hazard_class")
        elif hazard_class not in ALLOWED_HAZARD_CLASSES:
            errors.append(
                f"Invalid hazard_class '{hazard_class}'. "
                f"Must be one of: {sorted(ALLOWED_HAZARD_CLASSES)}"
            )

        speed_profile = response_data.get("speed_profile")
        if speed_profile is not None and speed_profile not in ALLOWED_SPEED_PROFILES:
            errors.append(
                f"Invalid speed_profile '{speed_profile}'. "
                f"Must be one of: {sorted(ALLOWED_SPEED_PROFILES)}"
            )

        force_profile = response_data.get("force_profile", {})
        if isinstance(force_profile, dict):
            grip_force = force_profile.get("max_grip_newtons")
            if grip_force is not None and isinstance(grip_force, (int, float)):
                if grip_force > DEFAULT_MAX_GRIP_FORCE_N:
                    errors.append(
                        f"max_grip_newtons ({grip_force}N) exceeds safety limit "
                        f"of {DEFAULT_MAX_GRIP_FORCE_N}N without explicit authorization."
                    )
            torque = force_profile.get("lateral_torque_limit_nm")
            if torque is not None and isinstance(torque, (int, float)):
                if torque > DEFAULT_MAX_TORQUE_NM:
                    errors.append(
                        f"lateral_torque_limit_nm ({torque}Nm) exceeds safety limit "
                        f"of {DEFAULT_MAX_TORQUE_NM}Nm without explicit authorization."
                    )

        is_valid = len(errors) == 0
        return is_valid, errors

    def generate_and_validate(self, **kwargs) -> dict:
        """Generate and validate a mock SKB (always passes on first try)."""
        skb = self.generate_skb(**kwargs)
        is_valid, errors = self.validate_skb_response(skb)
        if not is_valid:
            raise ValueError(
                f"Mock SKB validation unexpectedly failed: {errors}"
            )
        logger.info("Mock SKB generated and validated successfully.")
        return skb

    def format_promotion_entry(
        self,
        trigger_hash: str,
        generated_skb: dict,
        generation_time_ms: float,
    ) -> dict:
        """Format the mock SKB for promotion to Fast Path."""
        now = datetime.now(timezone.utc)
        entry = {
            "uuid": str(_uuid.uuid4()),
            "trigger_hash": trigger_hash,
            "schema_version": "2.0-GCP-ROBOTICS",
            "agent": "System_2_Generative_Supervisor",
            "pipeline_status": "PROMOTED_TO_FAST_PATH",
            "generated_skb": generated_skb,
            "recovery_metadata": {
                "generated_by_slow_path": True,
                "trigger_event": generated_skb.get("trigger_event", "HASH_MISS_OOD_STATE"),
                "reasoning_trace": generated_skb.get("reasoning_trace", ""),
                "confidence_score": generated_skb.get("confidence_score", 0.0),
                "llm_model_version": self.model,
            },
            "timestamps": {
                "schema_generated": now.isoformat(),
                "promoted_to_fast_path": now.isoformat(),
            },
            "telemetry": {
                "slow_path_latency_ms": generation_time_ms,
            },
        }
        logger.info(
            "Mock promotion entry formatted: uuid=%s, trigger_hash=%s",
            entry["uuid"], trigger_hash,
        )
        return entry

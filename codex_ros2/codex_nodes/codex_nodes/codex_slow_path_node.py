"""
CodexSlowPathNode — action server wrapping MockSlowPathGenerator /
SlowPathGenerator. Publishes SlowPathResult on completion and
auto-promotes to fast path.

Feedback stages: PROMPTING → CALLING_LLM → VALIDATING → PROMOTING
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from gcp_robotics.hash_engine.registry import CodexRegistry
from gcp_robotics.slow_path.prompts import MockSlowPathGenerator

logger = logging.getLogger(__name__)


@dataclass
class SlowPathResultMsg:
    """Python-side representation of SlowPathResult.msg."""
    stamp: float = 0.0
    frame_id: str = ""
    trigger_dhash_hex: str = ""
    primitive_id: str = ""
    confidence_score: float = 0.0
    generation_time_ms: float = 0.0
    promoted: bool = False
    skb_json: str = ""


@dataclass
class SlowPathFeedback:
    stage: str = ""
    progress: float = 0.0
    detail: str = ""


class CodexSlowPathNode:
    """Action server for LLM-based slow-path recovery.

    Parameters
    ----------
    node : Any
        A MockNode or rclpy.Node.
    registry : CodexRegistry
        Shared registry for auto-promotion.
    generator : optional
        SlowPathGenerator or MockSlowPathGenerator instance.
        Defaults to MockSlowPathGenerator.
    auto_promote : bool
        If True, automatically promote successful recoveries to fast path.
    """

    def __init__(
        self,
        node: Any,
        registry: CodexRegistry,
        generator: Any = None,
        auto_promote: bool = True,
    ) -> None:
        self._node = node
        self._registry = registry
        self._generator = generator or MockSlowPathGenerator()
        self._auto_promote = auto_promote
        self._log = node.get_logger()

        from .qos_profiles import (
            SKB_RESULT_QOS,
            SLOW_PATH_TRIGGER_QOS,
            TOPIC_SLOW_PATH_TRIGGER,
            TOPIC_SLOW_PATH_RESULT,
            ACTION_SLOW_PATH_RECOVER,
        )

        # Publisher for results
        self._result_pub = node.create_publisher(TOPIC_SLOW_PATH_RESULT, SKB_RESULT_QOS)

        # Subscribe to triggers for automatic recovery
        self._trigger_sub = node.create_subscription(
            TOPIC_SLOW_PATH_TRIGGER, self._on_trigger, SLOW_PATH_TRIGGER_QOS
        )

        # Action server
        self._action_server = node.create_action_server(
            ACTION_SLOW_PATH_RECOVER, self._execute_recovery
        )

        self._log.info("CodexSlowPathNode initialized (auto_promote=%s).", auto_promote)

    def _on_trigger(self, trigger_msg: Any) -> None:
        """Handle a SlowPathTrigger from the vision node."""
        self._log.info(
            "Slow path trigger received for hash %s",
            getattr(trigger_msg, "dhash_hex", "?"),
        )
        # Execute recovery synchronously in-line (for mock layer)
        result = self._run_recovery(
            dhash_hex=trigger_msg.dhash_hex,
            trigger_event=getattr(trigger_msg, "trigger_event", "hash_miss"),
            workspace_zone=getattr(trigger_msg, "workspace_zone", "default_zone"),
            end_effector=getattr(trigger_msg, "end_effector", "robotiq_2f85"),
            failure_context=getattr(trigger_msg, "failure_context", ""),
            image_path=getattr(trigger_msg, "image_path", None),
            previous_action=getattr(trigger_msg, "previous_action", None),
            available_objects=getattr(trigger_msg, "available_objects", []),
        )

        if result is not None:
            self._result_pub.publish(result)

    def _execute_recovery(
        self, goal_handle: Any, publish_feedback: Callable[[Any], None]
    ) -> dict:
        """Action server execute callback — runs recovery with feedback."""
        request = goal_handle.request

        result = self._run_recovery(
            dhash_hex=getattr(request, "dhash_hex", request.get("dhash_hex", "")),
            trigger_event=getattr(request, "trigger_event", request.get("trigger_event", "hash_miss")),
            workspace_zone=getattr(request, "workspace_zone", request.get("workspace_zone", "default_zone")),
            end_effector=getattr(request, "end_effector", request.get("end_effector", "robotiq_2f85")),
            failure_context=getattr(request, "failure_context", request.get("failure_context", "")),
            image_path=getattr(request, "image_path", request.get("image_path", None)),
            previous_action=getattr(request, "previous_action", request.get("previous_action", None)),
            available_objects=getattr(request, "available_objects", request.get("available_objects", [])),
            feedback_callback=publish_feedback,
        )

        if result is None:
            return {"success": False, "error_message": "Recovery failed"}

        # Publish result on topic as well
        self._result_pub.publish(result)

        return {
            "success": True,
            "primitive_id": result.primitive_id,
            "confidence_score": result.confidence_score,
            "generation_time_ms": result.generation_time_ms,
            "promoted": result.promoted,
            "skb_json": result.skb_json,
            "error_message": "",
        }

    def _run_recovery(
        self,
        dhash_hex: str,
        trigger_event: str,
        workspace_zone: str,
        end_effector: str,
        failure_context: str,
        image_path: Optional[str] = None,
        previous_action: Optional[str] = None,
        available_objects: Optional[list] = None,
        feedback_callback: Optional[Callable] = None,
    ) -> Optional[SlowPathResultMsg]:
        """Core recovery logic shared by trigger handler and action server."""

        def _feedback(stage: str, progress: float, detail: str = "") -> None:
            if feedback_callback:
                feedback_callback(SlowPathFeedback(
                    stage=stage, progress=progress, detail=detail
                ))

        start = time.monotonic()

        try:
            # Stage 1: Prompting
            _feedback("PROMPTING", 0.1, "Building LLM prompt")

            kwargs = dict(
                trigger_event=trigger_event,
                workspace_zone=workspace_zone,
                end_effector=end_effector,
                failure_context=failure_context,
            )
            if image_path:
                kwargs["image_path"] = image_path
            if previous_action:
                kwargs["previous_action"] = previous_action
            if available_objects:
                kwargs["available_objects"] = available_objects

            # Stage 2: Calling LLM
            _feedback("CALLING_LLM", 0.3, "Generating SKB via LLM")
            skb = self._generator.generate_and_validate(**kwargs)

            # Stage 3: Validating
            _feedback("VALIDATING", 0.7, "Validating generated SKB")

            elapsed_ms = (time.monotonic() - start) * 1000.0
            primitive_id = skb.get("primitive_id", "UNKNOWN")
            confidence = skb.get("confidence_score", 0.0)

            # Format promotion entry
            promotion_entry = self._generator.format_promotion_entry(
                trigger_hash=dhash_hex,
                generated_skb=skb,
                generation_time_ms=elapsed_ms,
            )

            # Stage 4: Auto-promote
            promoted = False
            if self._auto_promote:
                _feedback("PROMOTING", 0.9, "Promoting to fast path")
                try:
                    self._registry.promote_to_fast_path(dhash_hex, promotion_entry)
                    promoted = True
                    self._log.info("Auto-promoted hash %s to fast path.", dhash_hex[:10] + "...")
                except Exception:
                    self._log.exception("Auto-promotion failed for %s", dhash_hex)

            _feedback("PROMOTING", 1.0, "Complete")

            skb_json = json.dumps(skb)
            result = SlowPathResultMsg(
                stamp=time.time(),
                frame_id="slow_path",
                trigger_dhash_hex=dhash_hex,
                primitive_id=primitive_id,
                confidence_score=confidence,
                generation_time_ms=elapsed_ms,
                promoted=promoted,
                skb_json=skb_json,
            )

            self._log.info(
                "Recovery complete: %s (conf=%.2f, %.1fms, promoted=%s)",
                primitive_id, confidence, elapsed_ms, promoted,
            )
            return result

        except Exception:
            self._log.exception("Slow path recovery failed for %s", dhash_hex)
            return None

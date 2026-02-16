"""
CodexVisionNode — subscribes to camera images, hashes via CompositeHasher,
looks up in CodexRegistry, publishes GoldenCodexSignal (hit) or
SlowPathTrigger (miss).

Adapter pattern: no rclpy imports. Publishers/subscribers injected.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Protocol

from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry, LookupResult

logger = logging.getLogger(__name__)


@dataclass
class GoldenCodexSignalMsg:
    """Python-side representation of GoldenCodexSignal.msg."""
    stamp: float = 0.0
    frame_id: str = ""
    dhash_hex: str = ""
    match_type: str = ""
    pipeline_status: str = "NOMINAL"
    matched_hash: str = ""
    hamming_distance: int = -1
    lookup_time_ms: float = 0.0
    skb_json_ld: str = ""


@dataclass
class SlowPathTriggerMsg:
    """Python-side representation of SlowPathTrigger.msg."""
    stamp: float = 0.0
    frame_id: str = ""
    dhash_hex: str = ""
    trigger_event: str = "hash_miss"
    workspace_zone: str = ""
    end_effector: str = ""
    failure_context: str = ""
    image_path: str = ""
    previous_action: str = ""
    available_objects: list = field(default_factory=list)


class CodexVisionNode:
    """Processes images through the fast-path pipeline.

    Parameters
    ----------
    node : Any
        A MockNode or rclpy.Node providing pub/sub/service APIs.
    registry : CodexRegistry
        The hash registry for fast-path lookup.
    hasher : CompositeHasher
        Image hasher for dHash computation.
    workspace_zone : str
        Default workspace zone for slow-path triggers.
    end_effector : str
        Default end-effector for slow-path triggers.
    """

    def __init__(
        self,
        node: Any,
        registry: CodexRegistry,
        hasher: CompositeHasher,
        workspace_zone: str = "default_zone",
        end_effector: str = "robotiq_2f85",
        max_hamming_distance: int = 5,
    ) -> None:
        self._node = node
        self._registry = registry
        self._hasher = hasher
        self._workspace_zone = workspace_zone
        self._end_effector = end_effector
        self._max_hamming = max_hamming_distance
        self._log = node.get_logger()

        # Import QoS profiles
        from .qos_profiles import (
            SENSOR_IMAGE_QOS,
            SKB_RESULT_QOS,
            SLOW_PATH_TRIGGER_QOS,
            TOPIC_CAMERA_IMAGE,
            TOPIC_CODEX_SIGNAL,
            TOPIC_SLOW_PATH_TRIGGER,
        )

        # Publishers
        self._signal_pub = node.create_publisher(TOPIC_CODEX_SIGNAL, SKB_RESULT_QOS)
        self._trigger_pub = node.create_publisher(TOPIC_SLOW_PATH_TRIGGER, SLOW_PATH_TRIGGER_QOS)

        # Subscriber
        self._image_sub = node.create_subscription(
            TOPIC_CAMERA_IMAGE, self._on_image, SENSOR_IMAGE_QOS
        )

        self._log.info("CodexVisionNode initialized.")

    def _on_image(self, image_msg: Any) -> None:
        """Process an incoming image message."""
        self._log.debug("Received image on camera topic.")

        image_path = getattr(image_msg, "image_path", None)
        pil_image = getattr(image_msg, "pil_image", None)
        image_input = pil_image if pil_image is not None else image_path

        if image_input is None:
            self._log.warning("Image message has no image_path or pil_image.")
            return

        try:
            result = self._registry.lookup_by_image(
                image_input, self._hasher, self._max_hamming
            )
        except Exception:
            self._log.exception("Error during hash lookup.")
            return

        self._log.info(
            "Lookup: %s (hash=%s, hamming=%d, %.2fms)",
            result.match_type,
            result.query_hash[:10] + "...",
            result.hamming_distance,
            result.lookup_time_ms,
        )

        if result.match_type in ("EXACT", "FUZZY"):
            self._publish_signal(result, image_msg)
        else:
            self._publish_trigger(result, image_msg)

    def _publish_signal(self, result: LookupResult, image_msg: Any) -> None:
        skb_json_ld = ""
        if result.skb_data:
            try:
                skb_json_ld = json.dumps(result.skb_data)
            except (TypeError, ValueError):
                skb_json_ld = str(result.skb_data)

        msg = GoldenCodexSignalMsg(
            stamp=time.time(),
            frame_id=getattr(image_msg, "frame_id", "camera_link"),
            dhash_hex=result.query_hash,
            match_type=result.match_type,
            pipeline_status="NOMINAL",
            matched_hash=result.matched_hash or "",
            hamming_distance=result.hamming_distance,
            lookup_time_ms=result.lookup_time_ms,
            skb_json_ld=skb_json_ld,
        )
        self._signal_pub.publish(msg)
        self._log.info("Published GoldenCodexSignal (%s)", result.match_type)

    def _publish_trigger(self, result: LookupResult, image_msg: Any) -> None:
        msg = SlowPathTriggerMsg(
            stamp=time.time(),
            frame_id=getattr(image_msg, "frame_id", "camera_link"),
            dhash_hex=result.query_hash,
            trigger_event="hash_miss",
            workspace_zone=self._workspace_zone,
            end_effector=self._end_effector,
            failure_context=f"No match for dHash {result.query_hash}",
            image_path=getattr(image_msg, "image_path", ""),
            previous_action=getattr(image_msg, "previous_action", ""),
            available_objects=getattr(image_msg, "available_objects", []),
        )
        self._trigger_pub.publish(msg)
        self._log.info("Published SlowPathTrigger (MISS)")

    def process_image_directly(self, image_path_or_pil: Any, **kwargs: Any) -> LookupResult:
        """Convenience method for direct (non-pub/sub) image processing."""
        result = self._registry.lookup_by_image(
            image_path_or_pil, self._hasher, self._max_hamming
        )
        return result

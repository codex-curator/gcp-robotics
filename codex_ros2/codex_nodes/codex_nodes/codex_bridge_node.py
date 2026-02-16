"""
CodexBridgeNode — JSON-over-WebSocket bridge for non-ROS2 systems.

Subscribes to GoldenCodexSignal and SlowPathResult topics,
forwards them as JSON to connected WebSocket clients. Also accepts
inbound JSON commands (lookup, promote) and proxies them to services.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CodexBridgeNode:
    """Bridges ROS2 topics to JSON for external consumers.

    In mock mode, this simply collects messages and provides them
    via get_latest() / get_history(). A real deployment would add
    a WebSocket server layer on top.

    Parameters
    ----------
    node : Any
        A MockNode or rclpy.Node.
    """

    def __init__(self, node: Any) -> None:
        self._node = node
        self._log = node.get_logger()
        self._lock = threading.Lock()
        self._signal_history: List[dict] = []
        self._result_history: List[dict] = []
        self._max_history = 100

        from .qos_profiles import (
            SKB_RESULT_QOS,
            TOPIC_CODEX_SIGNAL,
            TOPIC_SLOW_PATH_RESULT,
        )

        node.create_subscription(
            TOPIC_CODEX_SIGNAL, self._on_signal, SKB_RESULT_QOS
        )
        node.create_subscription(
            TOPIC_SLOW_PATH_RESULT, self._on_slow_path_result, SKB_RESULT_QOS
        )

        self._log.info("CodexBridgeNode initialized.")

    def _msg_to_dict(self, msg: Any) -> dict:
        if hasattr(msg, "__dataclass_fields__"):
            return asdict(msg)
        if isinstance(msg, dict):
            return msg
        return {"raw": str(msg)}

    def _on_signal(self, msg: Any) -> None:
        data = self._msg_to_dict(msg)
        data["_topic"] = "/codex/signal"
        with self._lock:
            self._signal_history.append(data)
            if len(self._signal_history) > self._max_history:
                self._signal_history.pop(0)
        self._log.debug("Bridge forwarded signal: %s", data.get("match_type", "?"))

    def _on_slow_path_result(self, msg: Any) -> None:
        data = self._msg_to_dict(msg)
        data["_topic"] = "/codex/slow_path_result"
        with self._lock:
            self._result_history.append(data)
            if len(self._result_history) > self._max_history:
                self._result_history.pop(0)
        self._log.debug("Bridge forwarded slow_path_result.")

    def get_signal_history(self) -> List[dict]:
        with self._lock:
            return list(self._signal_history)

    def get_result_history(self) -> List[dict]:
        with self._lock:
            return list(self._result_history)

    def handle_command(self, command_json: str) -> str:
        """Process an inbound JSON command from a WebSocket client.

        Supported commands:
          {"cmd": "lookup", "dhash_hex": "...", "max_hamming_distance": 5}
          {"cmd": "promote", "trigger_hash": "...", "skb_json": "..."}
          {"cmd": "stats"}

        Returns JSON response string.
        """
        try:
            cmd = json.loads(command_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON: {exc}"})

        action = cmd.get("cmd", "")

        if action == "lookup":
            from .qos_profiles import SERVICE_LOOKUP
            from codex_ros2.codex_nodes.codex_nodes.codex_registry_node import (
                RegistryLookupRequest,
            )

            req = RegistryLookupRequest(
                dhash_hex=cmd.get("dhash_hex", ""),
                max_hamming_distance=cmd.get("max_hamming_distance", 5),
            )
            try:
                resp = self._node.call_service(SERVICE_LOOKUP, req)
                return json.dumps(self._msg_to_dict(resp))
            except Exception as exc:
                return json.dumps({"error": str(exc)})

        elif action == "promote":
            from .qos_profiles import SERVICE_PROMOTE
            from codex_ros2.codex_nodes.codex_nodes.codex_registry_node import (
                RegistryPromoteRequest,
            )

            req = RegistryPromoteRequest(
                trigger_hash=cmd.get("trigger_hash", ""),
                skb_json=cmd.get("skb_json", "{}"),
            )
            try:
                resp = self._node.call_service(SERVICE_PROMOTE, req)
                return json.dumps(self._msg_to_dict(resp))
            except Exception as exc:
                return json.dumps({"error": str(exc)})

        elif action == "stats":
            from .qos_profiles import TOPIC_STATS

            # Return latest from signal/result history
            return json.dumps({
                "signal_count": len(self._signal_history),
                "result_count": len(self._result_history),
            })

        else:
            return json.dumps({"error": f"Unknown command: {action}"})

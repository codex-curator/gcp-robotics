"""
CodexRegistryNode — service server wrapping CodexRegistry.lookup()
and promote_to_fast_path().

Also publishes periodic RegistryStats telemetry.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from gcp_robotics.hash_engine.registry import CodexRegistry

logger = logging.getLogger(__name__)


@dataclass
class RegistryLookupRequest:
    dhash_hex: str
    max_hamming_distance: int = 5


@dataclass
class RegistryLookupResponse:
    match_type: str = "MISS"
    query_hash: str = ""
    matched_hash: str = ""
    hamming_distance: int = -1
    lookup_time_ms: float = 0.0
    skb_json: str = ""


@dataclass
class RegistryPromoteRequest:
    trigger_hash: str
    skb_json: str


@dataclass
class RegistryPromoteResponse:
    success: bool = False
    message: str = ""


@dataclass
class RegistryStatsMsg:
    stamp: float = 0.0
    total_entries: int = 0
    total_lookups: int = 0
    exact_hits: int = 0
    fuzzy_hits: int = 0
    misses: int = 0
    promotion_count: int = 0
    secondary_phash_entries: int = 0
    secondary_ppp_entries: int = 0


class CodexRegistryNode:
    """Exposes CodexRegistry as ROS2 services + stats publisher.

    Parameters
    ----------
    node : Any
        A MockNode or rclpy.Node.
    registry : CodexRegistry
        The shared registry instance.
    stats_interval : float
        Seconds between stats publications (0 to disable).
    """

    def __init__(
        self,
        node: Any,
        registry: CodexRegistry,
        stats_interval: float = 5.0,
    ) -> None:
        self._node = node
        self._registry = registry
        self._log = node.get_logger()

        from .qos_profiles import (
            STATS_QOS,
            TOPIC_STATS,
            SERVICE_LOOKUP,
            SERVICE_PROMOTE,
        )

        # Services
        node.create_service(SERVICE_LOOKUP, self._handle_lookup)
        node.create_service(SERVICE_PROMOTE, self._handle_promote)

        # Stats publisher
        self._stats_pub = node.create_publisher(TOPIC_STATS, STATS_QOS)

        # Periodic stats timer
        self._stats_interval = stats_interval
        self._stats_stop = threading.Event()
        if stats_interval > 0:
            self._stats_thread = threading.Thread(
                target=self._stats_loop, daemon=True
            )
            self._stats_thread.start()

        self._log.info("CodexRegistryNode initialized.")

    def _handle_lookup(self, request: Any) -> RegistryLookupResponse:
        if isinstance(request, dict):
            dhash = request.get("dhash_hex", "")
            max_ham = request.get("max_hamming_distance", 5)
        else:
            dhash = getattr(request, "dhash_hex", "")
            max_ham = getattr(request, "max_hamming_distance", 5)

        result = self._registry.lookup(dhash, max_hamming_distance=max_ham)

        skb_json = ""
        if result.skb_data:
            try:
                skb_json = json.dumps(result.skb_data)
            except (TypeError, ValueError):
                skb_json = str(result.skb_data)

        resp = RegistryLookupResponse(
            match_type=result.match_type,
            query_hash=result.query_hash,
            matched_hash=result.matched_hash or "",
            hamming_distance=result.hamming_distance,
            lookup_time_ms=result.lookup_time_ms,
            skb_json=skb_json,
        )
        self._log.debug(
            "Lookup service: %s → %s (%.2fms)",
            dhash[:10] + "...",
            resp.match_type,
            resp.lookup_time_ms,
        )
        return resp

    def _handle_promote(self, request: Any) -> RegistryPromoteResponse:
        if isinstance(request, dict):
            trigger_hash = request.get("trigger_hash", "")
            skb_json = request.get("skb_json", "")
        else:
            trigger_hash = getattr(request, "trigger_hash", "")
            skb_json = getattr(request, "skb_json", "")

        try:
            skb_data = json.loads(skb_json) if isinstance(skb_json, str) else skb_json
            self._registry.promote_to_fast_path(trigger_hash, skb_data)
            self._log.info("Promoted hash %s to fast path.", trigger_hash[:10] + "...")
            return RegistryPromoteResponse(success=True, message="Promoted successfully.")
        except Exception as exc:
            self._log.error("Promotion failed: %s", exc)
            return RegistryPromoteResponse(success=False, message=str(exc))

    def _stats_loop(self) -> None:
        while not self._stats_stop.is_set():
            self.publish_stats()
            self._stats_stop.wait(self._stats_interval)

    def publish_stats(self) -> RegistryStatsMsg:
        stats = self._registry.stats()
        msg = RegistryStatsMsg(
            stamp=time.time(),
            total_entries=stats["total_entries"],
            total_lookups=stats["total_lookups"],
            exact_hits=stats["exact_hits"],
            fuzzy_hits=stats["fuzzy_hits"],
            misses=stats["misses"],
            promotion_count=stats["promotion_count"],
            secondary_phash_entries=stats["secondary_phash_entries"],
            secondary_ppp_entries=stats["secondary_ppp_entries"],
        )
        self._stats_pub.publish(msg)
        return msg

    def destroy(self) -> None:
        self._stats_stop.set()

"""
TelemetryLogger — structured event logging for GCP Robotics SDK.

Captures Hash Miss events, Slow Path invocations, Soulmark failures,
and cache/registry statistics for monitoring and diagnostics.

Events are logged to Python's standard logging system and accumulated
in memory for programmatic access via ``summary()``.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Telemetry event types."""

    CACHE_HIT = "cache_hit"
    REGISTRY_HIT = "registry_hit"
    HASH_MISS = "hash_miss"
    SLOW_PATH_INVOKED = "slow_path_invoked"
    SKB_GENERATED = "skb_generated"
    PROMOTED = "promoted"
    SOULMARK_FAILURE = "soulmark_failure"
    SOULMARK_VERIFIED = "soulmark_verified"
    GEOMETRIC_GATE_VETO = "geometric_gate_veto"
    CBF_INTERVENTION = "cbf_intervention"


@dataclass
class TelemetryEvent:
    """A single telemetry event."""

    event_type: EventType
    timestamp: float
    data: dict = field(default_factory=dict)


class TelemetryLogger:
    """Structured event logger with in-memory ring buffer.

    Parameters
    ----------
    enabled : bool
        If False, all logging is silently skipped (zero overhead).
    buffer_size : int
        Maximum number of events retained in memory (default: 10,000).
    """

    def __init__(self, enabled: bool = True, buffer_size: int = 10_000) -> None:
        self._enabled = enabled
        self._buffer: deque[TelemetryEvent] = deque(maxlen=buffer_size)
        self._counts: Counter = Counter()
        self._start_time = time.monotonic()

    def log(self, event_type: EventType, data: dict | None = None) -> None:
        """Record a telemetry event.

        Parameters
        ----------
        event_type : EventType
            The type of event.
        data : dict, optional
            Arbitrary metadata attached to the event.
        """
        if not self._enabled:
            return

        event = TelemetryEvent(
            event_type=event_type,
            timestamp=time.monotonic(),
            data=data or {},
        )

        self._buffer.append(event)
        self._counts[event_type.value] += 1

        # Structured log output
        if event_type == EventType.HASH_MISS:
            logger.info("HASH_MISS: %s", data)
        elif event_type == EventType.SOULMARK_FAILURE:
            logger.warning("SOULMARK_FAILURE: %s", data)
        elif event_type == EventType.CBF_INTERVENTION:
            logger.warning("CBF_INTERVENTION: %s", data)
        else:
            logger.debug("%s: %s", event_type.value, data)

    def summary(self) -> dict:
        """Return aggregate statistics."""
        elapsed = time.monotonic() - self._start_time
        return {
            "uptime_seconds": round(elapsed, 1),
            "total_events": sum(self._counts.values()),
            "event_counts": dict(self._counts),
            "buffer_size": len(self._buffer),
        }

    def recent(self, n: int = 50, event_type: EventType | None = None) -> list[dict]:
        """Return the most recent N events, optionally filtered by type.

        Parameters
        ----------
        n : int
            Number of events to return.
        event_type : EventType, optional
            Filter to a specific event type.

        Returns
        -------
        list[dict]
            Events as plain dicts (JSON-serializable).
        """
        events = list(self._buffer)
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return [
            {
                "type": e.event_type.value,
                "timestamp": e.timestamp - self._start_time,
                "data": e.data,
            }
            for e in events[-n:]
        ]

    @property
    def hash_miss_count(self) -> int:
        """Number of hash misses since startup."""
        return self._counts.get(EventType.HASH_MISS.value, 0)

    @property
    def hit_count(self) -> int:
        """Total cache + registry hits since startup."""
        return (
            self._counts.get(EventType.CACHE_HIT.value, 0)
            + self._counts.get(EventType.REGISTRY_HIT.value, 0)
        )

    def reset(self) -> None:
        """Clear all accumulated telemetry."""
        self._buffer.clear()
        self._counts.clear()
        self._start_time = time.monotonic()

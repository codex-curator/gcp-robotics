"""
FastPathCache — sub-millisecond RAM cache for hash-to-SKB mapping.

Provides O(1) exact-match lookup and bounded-size eviction for the
System 1 fast path.  Designed to sit in front of the CodexRegistry
to avoid even the dict lookup overhead on repeated frames.

Performance target: < 0.01 ms per get/put operation.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Operational statistics for the fast-path cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class FastPathCache:
    """Thread-safe LRU cache for hash-to-SKB mapping.

    Uses an ``OrderedDict`` for O(1) get/put with LRU eviction.
    Thread-safe via a lightweight lock (required for multi-threaded
    ROS2 callback groups).

    Parameters
    ----------
    max_entries : int
        Maximum number of hash-to-SKB entries before LRU eviction
        (default: 100,000).
    ttl_seconds : float
        Time-to-live for cache entries in seconds.  0 = no expiry
        (default: 0).
    """

    def __init__(self, max_entries: int = 100_000, ttl_seconds: float = 0) -> None:
        self._store: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, dhash_hex: str) -> Optional[dict]:
        """Retrieve an SKB by dHash.  Returns ``None`` on miss.

        Moves the accessed entry to the end (most recently used).
        """
        with self._lock:
            item = self._store.get(dhash_hex)
            if item is None:
                self._misses += 1
                return None

            skb_data, timestamp = item

            # Check TTL
            if self._ttl > 0 and (time.monotonic() - timestamp) > self._ttl:
                del self._store[dhash_hex]
                self._misses += 1
                return None

            # Move to end (LRU refresh)
            self._store.move_to_end(dhash_hex)
            self._hits += 1
            return skb_data

    def put(self, dhash_hex: str, skb_data: dict) -> None:
        """Insert or update a hash-to-SKB mapping.

        Evicts the least-recently-used entry if the cache is full.
        """
        with self._lock:
            if dhash_hex in self._store:
                self._store.move_to_end(dhash_hex)
                self._store[dhash_hex] = (skb_data, time.monotonic())
                return

            if len(self._store) >= self._max_entries:
                self._store.popitem(last=False)  # Evict LRU
                self._evictions += 1

            self._store[dhash_hex] = (skb_data, time.monotonic())

    def invalidate(self, dhash_hex: str) -> bool:
        """Remove a specific entry.  Returns True if it existed."""
        with self._lock:
            if dhash_hex in self._store:
                del self._store[dhash_hex]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    @property
    def stats(self) -> dict:
        """Return cache statistics as a plain dict."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "current_size": len(self._store),
                "max_size": self._max_entries,
                "hit_rate": self._hits / max(1, self._hits + self._misses),
            }

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, dhash_hex: str) -> bool:
        return dhash_hex in self._store

    def __repr__(self) -> str:
        return f"<FastPathCache entries={len(self._store)}/{self._max_entries}>"

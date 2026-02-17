"""Tests for the FastPathCache."""

import time
import pytest
from gcp_robotics.cache import FastPathCache


class TestFastPathCache:

    def test_put_and_get(self, cache, sample_skb):
        cache.put("0xABCD", sample_skb)
        assert cache.get("0xABCD") == sample_skb

    def test_miss_returns_none(self, cache):
        assert cache.get("0xNONEXISTENT") is None

    def test_lru_eviction(self):
        cache = FastPathCache(max_entries=3)
        cache.put("0x01", {"id": 1})
        cache.put("0x02", {"id": 2})
        cache.put("0x03", {"id": 3})

        # This should evict 0x01 (least recently used)
        cache.put("0x04", {"id": 4})

        assert cache.get("0x01") is None
        assert cache.get("0x02") == {"id": 2}
        assert cache.get("0x04") == {"id": 4}

    def test_lru_refresh_on_access(self):
        cache = FastPathCache(max_entries=3)
        cache.put("0x01", {"id": 1})
        cache.put("0x02", {"id": 2})
        cache.put("0x03", {"id": 3})

        # Access 0x01 to refresh it
        cache.get("0x01")

        # Now 0x02 is LRU and should be evicted
        cache.put("0x04", {"id": 4})
        assert cache.get("0x01") == {"id": 1}
        assert cache.get("0x02") is None

    def test_update_existing_key(self, cache, sample_skb):
        cache.put("0xABCD", {"old": True})
        cache.put("0xABCD", sample_skb)
        assert cache.get("0xABCD") == sample_skb

    def test_invalidate(self, cache, sample_skb):
        cache.put("0xABCD", sample_skb)
        assert cache.invalidate("0xABCD") is True
        assert cache.get("0xABCD") is None
        assert cache.invalidate("0xABCD") is False

    def test_clear(self, cache, sample_skb):
        cache.put("0x01", sample_skb)
        cache.put("0x02", sample_skb)
        cache.clear()
        assert len(cache) == 0

    def test_stats(self, cache, sample_skb):
        cache.put("0xABCD", sample_skb)
        cache.get("0xABCD")  # hit
        cache.get("0xMISS")  # miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["current_size"] == 1
        assert stats["hit_rate"] == 0.5

    def test_ttl_expiry(self):
        cache = FastPathCache(max_entries=100, ttl_seconds=0.1)
        cache.put("0xABCD", {"data": True})
        assert cache.get("0xABCD") is not None

        time.sleep(0.15)
        assert cache.get("0xABCD") is None

    def test_contains(self, cache, sample_skb):
        cache.put("0xABCD", sample_skb)
        assert "0xABCD" in cache
        assert "0xMISS" not in cache

    def test_performance_target(self, cache):
        """Verify sub-millisecond operation for 10K entries."""
        for i in range(10_000):
            cache.put(f"0x{i:08X}", {"id": i})

        start = time.perf_counter()
        for i in range(1_000):
            cache.get(f"0x{i:08X}")
        elapsed_ms = (time.perf_counter() - start) * 1000

        # 1000 lookups should complete in well under 10ms
        assert elapsed_ms < 10, f"1000 lookups took {elapsed_ms:.2f}ms (target: <10ms)"

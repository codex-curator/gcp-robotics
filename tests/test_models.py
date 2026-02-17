"""Tests for the SKB Pydantic schema validation."""

import pytest
from gcp_robotics.hash_engine.registry import CodexRegistry, LookupResult
from gcp_robotics.hash_engine.hasher import CompositeHasher, CompositeHash


class TestCompositeHasher:

    def test_hamming_distance_identical(self):
        assert CompositeHasher.hamming_distance("0xFFFF", "0xFFFF") == 0

    def test_hamming_distance_one_bit(self):
        assert CompositeHasher.hamming_distance("0xFFFE", "0xFFFF") == 1

    def test_hamming_distance_all_bits(self):
        # 0x0000 vs 0xFFFF = 16 bits different
        assert CompositeHasher.hamming_distance("0x0000", "0xFFFF") == 16

    def test_hamming_distance_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            CompositeHasher.hamming_distance("0xFF", "0xFFFF")


class TestCompositeHash:

    def test_as_dict(self, sample_composite_hash):
        d = sample_composite_hash.as_dict()
        assert "dhash_hex" in d
        assert "phash_hex" in d
        assert "ppp_256bit" in d
        assert "color_hash" in d

    def test_immutable(self, sample_composite_hash):
        with pytest.raises(AttributeError):
            sample_composite_hash.dhash_hex = "0xNEW"


class TestCodexRegistry:

    def test_register_and_lookup(self, registry, sample_skb):
        registry.register("0xABCD1234", sample_skb)
        result = registry.lookup("0xABCD1234")
        assert result.match_type == "EXACT"
        assert result.skb_data == sample_skb
        assert result.hamming_distance == 0

    def test_fuzzy_match(self, registry, sample_skb):
        registry.register("0xABCD1234ABCD1234", sample_skb)
        # One bit different
        result = registry.lookup("0xABCD1234ABCD1235", max_hamming_distance=5)
        assert result.match_type == "FUZZY"
        assert result.hamming_distance == 1

    def test_miss(self, registry):
        result = registry.lookup("0xDEADBEEF")
        assert result.match_type == "MISS"
        assert result.skb_data is None

    def test_promote_to_fast_path(self, registry, sample_skb):
        registry.promote_to_fast_path("0xNEW_HASH", sample_skb)
        assert "0xNEW_HASH" in registry
        result = registry.lookup("0xNEW_HASH")
        assert result.match_type == "EXACT"

    def test_stats(self, registry, sample_skb):
        registry.register("0xABCD", sample_skb)
        registry.lookup("0xABCD")
        registry.lookup("0xMISS")

        stats = registry.stats()
        assert stats["total_entries"] == 1
        assert stats["exact_hits"] == 1
        assert stats["misses"] == 1

    def test_save_and_load(self, registry, sample_skb, tmp_path):
        registry.register("0xABCD", sample_skb, phash_hex="0x1234")
        registry.promote_to_fast_path("0xPROMO", sample_skb)

        filepath = tmp_path / "test_registry.json"
        registry.save(filepath)

        new_registry = CodexRegistry()
        new_registry.load(filepath)

        assert len(new_registry) == 2
        result = new_registry.lookup("0xABCD")
        assert result.match_type == "EXACT"

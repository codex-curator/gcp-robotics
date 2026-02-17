"""Tests for the SoulmarkVerifier."""

import pytest
from gcp_robotics.soulmark import SoulmarkVerifier


class TestSoulmarkVerifier:

    def test_compute_soulmark_deterministic(self, verifier, sample_skb):
        s1 = verifier.compute_soulmark(sample_skb)
        s2 = verifier.compute_soulmark(sample_skb)
        assert s1 == s2
        assert len(s1) == 64  # SHA-256 hex

    def test_soulmark_changes_on_tamper(self, verifier, sample_skb):
        original = verifier.compute_soulmark(sample_skb)

        # Tamper with the grip force (the "Poisoned Blueprint" attack)
        tampered = {**sample_skb}
        tampered["affordance_layer"] = {**sample_skb["affordance_layer"]}
        tampered["affordance_layer"]["grip_force_n"] = 500.0  # Lethal force

        tampered_soulmark = verifier.compute_soulmark(tampered)
        assert original != tampered_soulmark

    def test_verify_valid(self, verifier, sample_skb):
        soulmark = verifier.compute_soulmark(sample_skb)
        result = verifier.verify(sample_skb, soulmark)
        assert result.valid is True
        assert result.reason == ""

    def test_verify_tampered(self, verifier, sample_skb):
        soulmark = verifier.compute_soulmark(sample_skb)

        # Tamper
        sample_skb["safety_layer"]["max_force_n"] = 999.0

        result = verifier.verify(sample_skb, soulmark)
        assert result.valid is False
        assert "tampering" in result.reason.lower()

    def test_sign_and_verify(self, verifier, sample_skb):
        signed = verifier.sign(sample_skb)
        assert signed.soulmark is not None

        result = verifier.verify(signed.skb_data, signed.soulmark)
        assert result.valid is True

    def test_hmac_with_fleet_secret(self, verifier_with_secret, sample_skb):
        soulmark = verifier_with_secret.compute_soulmark(sample_skb)

        # Plain SHA-256 should differ from HMAC
        plain = SoulmarkVerifier().compute_soulmark(sample_skb)
        assert soulmark != plain

    def test_hmac_verify(self, verifier_with_secret, sample_skb):
        soulmark = verifier_with_secret.compute_soulmark(sample_skb)
        result = verifier_with_secret.verify(sample_skb, soulmark)
        assert result.valid is True

    def test_verify_payload_unsigned(self, verifier, sample_skb):
        """Unsigned payloads (no embedded Soulmark) should pass."""
        result = verifier.verify_payload(sample_skb)
        assert result.valid is True

    def test_verify_payload_embedded(self, verifier, sample_skb):
        """Payload with correct embedded Soulmark should pass."""
        soulmark = verifier.compute_soulmark(sample_skb)
        sample_skb["provenance"]["soulmark"] = soulmark

        result = verifier.verify_payload(sample_skb)
        assert result.valid is True

    def test_verify_payload_tampered(self, verifier, sample_skb):
        """Payload with wrong embedded Soulmark should fail."""
        sample_skb["provenance"]["soulmark"] = "0" * 64  # Fake

        result = verifier.verify_payload(sample_skb)
        assert result.valid is False

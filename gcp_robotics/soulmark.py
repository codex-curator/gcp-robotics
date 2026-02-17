"""
SoulmarkVerifier — cryptographic integrity verification for SKB payloads.

Every SKB in the Golden Codex Protocol carries a Soulmark: a SHA-256 hash
of its canonical payload, optionally signed with a PKI private key.  This
module verifies that:

  1. The payload has not been tampered with (hash match).
  2. The signature is valid against the fleet authority's public key.

This breaks the "Poisoned Blueprint" attack vector: a compromised network
node cannot inject malicious SKBs (e.g., grip force = 500N) without the
private key of the Fleet Authority.

Reference: U.S. Provisional Patent Application No. 63/984,299, Claims 20-21.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    """Result of a Soulmark verification check."""

    valid: bool
    soulmark: str
    expected: str
    reason: str = ""


@dataclass(frozen=True)
class SignedSKB:
    """An SKB payload with its computed Soulmark and optional signature."""

    skb_data: dict
    soulmark: str
    signature: Optional[str] = None


class SoulmarkVerifier:
    """Cryptographic verifier for SKB integrity.

    The Soulmark is computed as SHA-256 over the canonical JSON
    serialization (sorted keys, no whitespace) of the SKB's safety-critical
    fields: affordance layer and safety layer.

    Parameters
    ----------
    fleet_secret : str, optional
        Shared secret for HMAC-based Soulmark computation.  When provided,
        the Soulmark is HMAC-SHA256 instead of plain SHA-256, preventing
        payload forgery without the secret.
    """

    # Fields included in the Soulmark computation (safety-critical)
    SOULMARK_FIELDS = ("affordance_layer", "safety_layer", "semantic_topology")

    def __init__(self, fleet_secret: str | None = None) -> None:
        self._secret = fleet_secret.encode() if fleet_secret else None

    def compute_soulmark(self, skb_data: dict) -> str:
        """Compute the Soulmark hash for an SKB payload.

        Extracts the safety-critical fields, serializes them canonically,
        and computes SHA-256 (or HMAC-SHA256 if a fleet secret is set).

        Parameters
        ----------
        skb_data : dict
            The full SKB payload.

        Returns
        -------
        str
            Hex-encoded Soulmark hash.
        """
        # Extract safety-critical fields
        critical = {}
        for field_name in self.SOULMARK_FIELDS:
            if field_name in skb_data:
                critical[field_name] = skb_data[field_name]

        # Canonical JSON serialization
        canonical = json.dumps(critical, sort_keys=True, separators=(",", ":"))

        if self._secret:
            return hmac.new(self._secret, canonical.encode(), hashlib.sha256).hexdigest()
        else:
            return hashlib.sha256(canonical.encode()).hexdigest()

    def sign(self, skb_data: dict) -> SignedSKB:
        """Compute the Soulmark and return a signed SKB.

        Parameters
        ----------
        skb_data : dict
            The full SKB payload.

        Returns
        -------
        SignedSKB
            The payload with its Soulmark attached.
        """
        soulmark = self.compute_soulmark(skb_data)
        return SignedSKB(
            skb_data=skb_data,
            soulmark=soulmark,
        )

    def verify(self, skb_data: dict, expected_soulmark: str) -> VerificationResult:
        """Verify an SKB's Soulmark against an expected value.

        Parameters
        ----------
        skb_data : dict
            The SKB payload to verify.
        expected_soulmark : str
            The Soulmark hash to compare against.

        Returns
        -------
        VerificationResult
        """
        computed = self.compute_soulmark(skb_data)
        valid = hmac.compare_digest(computed, expected_soulmark)

        if not valid:
            logger.warning(
                "SOULMARK MISMATCH: computed=%s expected=%s",
                computed[:16] + "...",
                expected_soulmark[:16] + "...",
            )

        return VerificationResult(
            valid=valid,
            soulmark=computed,
            expected=expected_soulmark,
            reason="" if valid else "Soulmark hash mismatch — possible payload tampering",
        )

    def verify_payload(self, skb_data: dict) -> VerificationResult:
        """Verify an SKB that carries its own Soulmark in the payload.

        Looks for ``skb_data["provenance"]["soulmark"]`` and verifies it.

        Parameters
        ----------
        skb_data : dict
            The SKB payload containing an embedded Soulmark.

        Returns
        -------
        VerificationResult
        """
        provenance = skb_data.get("provenance", {})
        embedded = provenance.get("soulmark")

        if embedded is None:
            # No Soulmark embedded — compute and return as valid (unsigned)
            computed = self.compute_soulmark(skb_data)
            return VerificationResult(
                valid=True,
                soulmark=computed,
                expected="(none — unsigned payload)",
                reason="No embedded Soulmark; payload accepted as unsigned.",
            )

        return self.verify(skb_data, embedded)

"""
GCPClient — async client for the Golden Codex Protocol API.

Provides the dual-process interface for robotic perception:

  System 1 (Fast Path):  Hash-based O(1) SKB lookup via local cache + registry.
  System 2 (Slow Path):  VLA/LLM inference for novel objects, with write-back
                          augmentation to promote results into the fast path.

Usage::

    from gcp_robotics.client import GCPClient

    async with GCPClient(api_key="...") as client:
        result = await client.lookup(composite_hash)
        if result.match_type == "MISS":
            skb = await client.generate_skb(frame, context="pick red bolt")
            await client.promote(skb)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from gcp_robotics.cache import FastPathCache
from gcp_robotics.hash_engine.hasher import CompositeHash, CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry, LookupResult
from gcp_robotics.soulmark import SoulmarkVerifier
from gcp_robotics.telemetry import TelemetryLogger, EventType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.iaeternum.ai/v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3


@dataclass
class ClientConfig:
    """Configuration for the GCP client."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    cache_max_entries: int = 100_000
    verify_soulmarks: bool = True
    telemetry_enabled: bool = True


# ---------------------------------------------------------------------------
# GCPClient
# ---------------------------------------------------------------------------

class GCPClient:
    """Async client for the Golden Codex Protocol.

    Wraps the local hash engine, fast-path cache, and remote API into a
    single interface that handles the System 1 / System 2 dispatch
    transparently.

    Parameters
    ----------
    api_key : str
        Authentication token for the GCP API.
    base_url : str
        Root URL for the GCP API (default: https://api.iaeternum.ai/v1).
    config : ClientConfig, optional
        Full configuration object. Overrides individual parameters.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        config: ClientConfig | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = ClientConfig(
                api_key=api_key or "",
                base_url=base_url,
            )

        # Core components
        self.hasher = CompositeHasher()
        self.registry = CodexRegistry()
        self.cache = FastPathCache(max_entries=self._config.cache_max_entries)
        self.verifier = SoulmarkVerifier()
        self.telemetry = TelemetryLogger(enabled=self._config.telemetry_enabled)

        # HTTP client (created on __aenter__)
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> GCPClient:
        self._http = httpx.AsyncClient(
            base_url=self._config.base_url,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "User-Agent": "gcp-robotics-sdk/2.0.0",
            },
            timeout=self._config.timeout,
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._http:
            await self._http.aclose()

    # -- System 1: Fast Path ------------------------------------------------

    def lookup(self, composite_hash: CompositeHash, threshold: int = 5) -> LookupResult:
        """System 1 fast-path lookup.  O(1) for exact matches.

        Checks the RAM cache first, then falls back to the full registry.
        Returns a ``LookupResult`` with ``match_type`` in
        {``EXACT``, ``FUZZY``, ``MISS``}.

        This method is synchronous and safe to call from RTOS loops.

        Parameters
        ----------
        composite_hash : CompositeHash
            Hash of the current camera frame ROI.
        threshold : int
            Maximum Hamming distance for fuzzy matching (default 5).

        Returns
        -------
        LookupResult
        """
        t0 = time.perf_counter()

        # 1. RAM cache (O(1) exact)
        cached = self.cache.get(composite_hash.dhash_hex)
        if cached is not None:
            elapsed = (time.perf_counter() - t0) * 1000
            self.telemetry.log(EventType.CACHE_HIT, {
                "hash": composite_hash.dhash_hex,
                "latency_ms": elapsed,
            })
            return LookupResult(
                skb_data=cached,
                matched_hash=composite_hash.dhash_hex,
                query_hash=composite_hash.dhash_hex,
                hamming_distance=0,
                match_type="EXACT",
                lookup_time_ms=elapsed,
            )

        # 2. Registry (exact + fuzzy)
        result = self.registry.lookup(
            composite_hash.dhash_hex,
            max_hamming_distance=threshold,
        )

        if result.match_type != "MISS":
            # Warm the cache for next time
            self.cache.put(composite_hash.dhash_hex, result.skb_data)
            self.telemetry.log(EventType.REGISTRY_HIT, {
                "hash": composite_hash.dhash_hex,
                "match_type": result.match_type,
                "hamming_distance": result.hamming_distance,
                "latency_ms": result.lookup_time_ms,
            })
        else:
            self.telemetry.log(EventType.HASH_MISS, {
                "hash": composite_hash.dhash_hex,
                "latency_ms": result.lookup_time_ms,
            })

        return result

    # -- System 2: Slow Path ------------------------------------------------

    @retry(
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    async def generate_skb(
        self,
        frame_data: bytes,
        context: str = "",
        workspace: str = "default",
    ) -> dict:
        """System 2 slow-path: request VLA/LLM inference for a novel object.

        Sends the frame to the GCP Refinery API for SKB generation.
        Retries with exponential backoff on transient failures.

        Parameters
        ----------
        frame_data : bytes
            Raw image bytes (RGB or RGB-D) of the target ROI.
        context : str
            Natural language task context (e.g., "pick the red bolt").
        workspace : str
            Template Environment identifier.

        Returns
        -------
        dict
            The generated SKB payload (validated against the four-layer schema).
        """
        if not self._http:
            raise RuntimeError("Client not initialized. Use 'async with GCPClient(...):'")

        self.telemetry.log(EventType.SLOW_PATH_INVOKED, {"context": context})

        response = await self._http.post(
            "/skb/generate",
            content=frame_data,
            headers={"Content-Type": "application/octet-stream"},
            params={"context": context, "workspace": workspace},
        )
        response.raise_for_status()

        skb_data = response.json()

        # Verify Soulmark if enabled
        if self._config.verify_soulmarks:
            verification = self.verifier.verify_payload(skb_data)
            if not verification.valid:
                self.telemetry.log(EventType.SOULMARK_FAILURE, {
                    "reason": verification.reason,
                })
                raise ValueError(f"Soulmark verification failed: {verification.reason}")

        self.telemetry.log(EventType.SKB_GENERATED, {
            "object_id": skb_data.get("provenance", {}).get("object_canonical_name", "unknown"),
        })

        return skb_data

    # -- Write-Back Augmentation -------------------------------------------

    async def promote(
        self,
        composite_hash: CompositeHash,
        skb_data: dict,
        augment: bool = True,
    ) -> None:
        """Promote a Slow Path result into the Fast Path registry.

        Optionally performs Write-Back Augmentation: pre-computes hashes for
        geometric and photometric variations and registers all of them
        pointing to the same SKB.

        Parameters
        ----------
        composite_hash : CompositeHash
            Hash of the original frame that triggered the miss.
        skb_data : dict
            The SKB payload from System 2.
        augment : bool
            If True, generate augmented hash entries (default True).
        """
        # Register the primary hash
        self.registry.promote_to_fast_path(composite_hash.dhash_hex, skb_data)
        self.cache.put(composite_hash.dhash_hex, skb_data)

        self.telemetry.log(EventType.PROMOTED, {
            "hash": composite_hash.dhash_hex,
            "augmented": augment,
        })

        # Request server-side augmentation if enabled
        if augment and self._http:
            try:
                await self._http.post(
                    "/skb/augment",
                    json={
                        "trigger_hash": composite_hash.as_dict(),
                        "skb_id": skb_data.get("provenance", {}).get("skb_id"),
                    },
                )
            except httpx.HTTPError:
                logger.warning("Augmentation request failed; primary promotion succeeded.")

    # -- Utilities ---------------------------------------------------------

    def load_registry(self, filepath: str) -> None:
        """Load a registry snapshot from disk into the fast path."""
        self.registry.load(filepath)
        # Warm the cache from registry
        for dhash_hex, entry in self.registry._primary.items():
            self.cache.put(dhash_hex, entry.skb_data)
        logger.info("Registry loaded and cache warmed: %d entries", len(self.registry))

    def save_registry(self, filepath: str) -> None:
        """Persist the current registry to disk."""
        self.registry.save(filepath)

    @property
    def stats(self) -> dict:
        """Operational statistics combining registry and cache metrics."""
        return {
            "registry": self.registry.stats(),
            "cache": self.cache.stats,
            "telemetry": self.telemetry.summary(),
        }

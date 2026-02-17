"""
PCOController — Perceptual Compute Offloading pipeline for simulation.

Implements the dual-process architecture:
  - System 1 (Fast Path): hash the camera frame, look up in CodexRegistry.
    Sub-millisecond response when a match is found.
  - System 2 (Slow Path): on a hash miss, generate a mock SKB (simulating
    VLM inference) and promote the result into the registry for future hits.

Also provides a VLA baseline simulation for side-by-side comparison.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image

from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProcessResult:
    """Result of a single frame through the PCO or VLA pipeline."""

    match_type: str          # "EXACT", "FUZZY", "MISS", "VLA_BASELINE"
    skb_data: Optional[dict]
    latency_ms: float
    dhash_hex: str
    pipeline: str            # "FAST_PATH", "SLOW_PATH", "VLA_BASELINE"
    hamming_distance: int = 0


# ---------------------------------------------------------------------------
# PCOController
# ---------------------------------------------------------------------------

class PCOController:
    """Implements the PCO fast-path / slow-path pipeline for simulation.

    On each ``process_frame`` call the controller:

    1. Computes the perceptual dHash of the camera frame.
    2. Looks up the hash in the ``CodexRegistry``.
    3. On a **hit** (exact or fuzzy): returns the cached SKB immediately
       (System 1 fast path).
    4. On a **miss**: invokes the simulated slow path to generate a mock
       SKB, promotes it into the registry, and returns it (System 2).

    Parameters
    ----------
    registry : CodexRegistry, optional
        Pre-populated registry.  A fresh one is created if not provided.
    hasher : CompositeHasher, optional
        Hash computation engine.  A fresh one is created if not provided.
    hamming_threshold : int
        Maximum Hamming distance for a fuzzy match (default 5).
    slow_path_latency_ms : float
        Simulated VLM inference latency for the slow path (default 150 ms).
    """

    def __init__(
        self,
        registry: Optional[CodexRegistry] = None,
        hasher: Optional[CompositeHasher] = None,
        hamming_threshold: int = 5,
        slow_path_latency_ms: float = 150.0,
    ) -> None:
        self._registry = registry or CodexRegistry()
        self._hasher = hasher or CompositeHasher()
        self._hamming_threshold = hamming_threshold
        self._slow_path_latency_ms = slow_path_latency_ms

        # Bookkeeping
        self._total_frames: int = 0
        self._fast_path_hits: int = 0
        self._slow_path_invocations: int = 0
        self._latencies_fast: List[float] = []
        self._latencies_slow: List[float] = []

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: Image.Image,
        context: Optional[dict] = None,
    ) -> ProcessResult:
        """Run the full PCO pipeline on a camera frame.

        Parameters
        ----------
        frame : PIL.Image.Image
            Camera frame rendered from the simulation.
        context : dict, optional
            Additional context (e.g. scene metadata). Not currently used
            but reserved for future VLM prompt injection.

        Returns
        -------
        ProcessResult
            Includes match type, SKB data, latency, and the computed hash.
        """
        self._total_frames += 1
        t0 = time.perf_counter()

        # Step 1: compute dHash
        dhash_hex = self._hasher.compute_dhash(frame)

        # Step 2: registry lookup
        lookup = self._registry.lookup(
            dhash_hex, max_hamming_distance=self._hamming_threshold
        )

        if lookup.match_type in ("EXACT", "FUZZY"):
            # --- System 1: Fast Path ---
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._fast_path_hits += 1
            self._latencies_fast.append(elapsed_ms)

            logger.debug(
                "FAST PATH %s (hamming=%d) in %.3f ms",
                lookup.match_type, lookup.hamming_distance, elapsed_ms,
            )
            return ProcessResult(
                match_type=lookup.match_type,
                skb_data=lookup.skb_data,
                latency_ms=elapsed_ms,
                dhash_hex=dhash_hex,
                pipeline="FAST_PATH",
                hamming_distance=lookup.hamming_distance,
            )

        # --- System 2: Slow Path (hash miss) ---
        skb = self._simulate_slow_path(frame, dhash_hex)

        # Promote to registry for future fast-path hits
        composite = self._hasher.compute_composite(frame)
        self._registry.promote_to_fast_path(dhash_hex, skb)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._slow_path_invocations += 1
        self._latencies_slow.append(elapsed_ms)

        logger.debug("SLOW PATH miss -> promoted in %.3f ms", elapsed_ms)
        return ProcessResult(
            match_type="MISS",
            skb_data=skb,
            latency_ms=elapsed_ms,
            dhash_hex=dhash_hex,
            pipeline="SLOW_PATH",
            hamming_distance=-1,
        )

    # ------------------------------------------------------------------
    # Pre-registration
    # ------------------------------------------------------------------

    def register_object(
        self,
        image: Image.Image,
        skb_data: dict,
        label: Optional[str] = None,
    ) -> str:
        """Register an object image and its SKB in the registry.

        This pre-populates the fast path so the first encounter is already
        a hit.

        Parameters
        ----------
        image : PIL.Image.Image
            Reference image of the object.
        skb_data : dict
            The Spatial Kinematic Blueprint to associate.
        label : str, optional
            Human-readable label (stored in the SKB if not already present).

        Returns
        -------
        str
            The dHash hex string used as the registry key.
        """
        if label and "label" not in skb_data:
            skb_data = {**skb_data, "label": label}

        composite = self._hasher.compute_composite(image)
        self._registry.register(
            dhash_hex=composite.dhash_hex,
            skb_data=skb_data,
            phash_hex=composite.phash_hex,
            ppp_256=composite.ppp_256bit,
        )

        logger.info(
            "Registered object label=%s dhash=%s",
            label, composite.dhash_hex,
        )
        return composite.dhash_hex

    # ------------------------------------------------------------------
    # VLA baseline simulation
    # ------------------------------------------------------------------

    def simulate_vla_inference(
        self,
        frame: Image.Image,
        latency_ms: float = 300.0,
    ) -> ProcessResult:
        """Simulate a VLA baseline inference.

        Sleeps for ``latency_ms`` then returns a mock result.  This is
        the comparison point: what happens when every frame must go
        through a full vision-language-action model with no caching.

        Parameters
        ----------
        frame : PIL.Image.Image
            Camera frame (hashed for record-keeping but not looked up).
        latency_ms : float
            Simulated inference latency in milliseconds.

        Returns
        -------
        ProcessResult
        """
        t0 = time.perf_counter()

        dhash_hex = self._hasher.compute_dhash(frame)
        time.sleep(latency_ms / 1000.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        skb = {
            "action": "pick_and_place",
            "source": "VLA_BASELINE",
            "frame_hash": dhash_hex,
        }

        return ProcessResult(
            match_type="VLA_BASELINE",
            skb_data=skb,
            latency_ms=elapsed_ms,
            dhash_hex=dhash_hex,
            pipeline="VLA_BASELINE",
            hamming_distance=-1,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Return pipeline statistics."""
        total = self._total_frames or 1  # avoid division by zero
        return {
            "total_frames": self._total_frames,
            "fast_path_hits": self._fast_path_hits,
            "slow_path_invocations": self._slow_path_invocations,
            "hit_rate": self._fast_path_hits / total,
            "miss_rate": self._slow_path_invocations / total,
            "mean_fast_latency_ms": (
                sum(self._latencies_fast) / len(self._latencies_fast)
                if self._latencies_fast else 0.0
            ),
            "mean_slow_latency_ms": (
                sum(self._latencies_slow) / len(self._latencies_slow)
                if self._latencies_slow else 0.0
            ),
            "registry_size": len(self._registry),
        }

    @property
    def registry(self) -> CodexRegistry:
        """Direct access to the underlying registry."""
        return self._registry

    @property
    def hasher(self) -> CompositeHasher:
        """Direct access to the underlying hasher."""
        return self._hasher

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_slow_path(
        self,
        frame: Image.Image,
        dhash_hex: str,
    ) -> dict:
        """Simulate VLM-based SKB generation (slow path).

        In production this would call Claude or another VLM.  Here we
        sleep to simulate latency and return a mock SKB.
        """
        time.sleep(self._slow_path_latency_ms / 1000.0)

        return {
            "skb_id": str(uuid.uuid4()),
            "action": "pick_and_place",
            "source": "SLOW_PATH_VLM",
            "frame_hash": dhash_hex,
            "layers": {
                "L1_provenance": {"hash": dhash_hex, "origin": "simulation"},
                "L2_semantic": {"objects_detected": ["unknown"]},
                "L3_affordance": {"graspable": True, "grasp_type": "top_down"},
                "L4_safety": {"collision_free": True},
            },
        }

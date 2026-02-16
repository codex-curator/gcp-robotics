"""
CodexRegistry — in-memory Fast Path lookup table for the Golden Codex Protocol.

Maps composite perceptual hashes to SKB (Spatial Kinematic Blueprint) JSON
payloads.  Provides O(1) exact-match lookup and linear-scan fuzzy lookup
within a configurable Hamming distance threshold.

Lifecycle
---------
1. On startup the registry is loaded from a JSON snapshot on disk.
2. Incoming camera frames are hashed by ``CompositeHasher`` and looked up here.
3. If a **Fast Path hit** is found the SKB is returned immediately.
4. If a **miss** occurs the frame is sent to the Slow Path (VLM pipeline).
5. Once the Slow Path produces an SKB, ``promote_to_fast_path()`` registers it
   so subsequent encounters resolve instantly (loop-closure).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from gcp_robotics.hash_engine.hasher import CompositeHasher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LookupResult:
    """Result returned by every ``lookup`` call."""

    skb_data: Optional[dict]
    matched_hash: Optional[str]
    query_hash: str
    hamming_distance: int
    match_type: Literal["EXACT", "FUZZY", "MISS"]
    lookup_time_ms: float


@dataclass
class _RegistryEntry:
    """Internal storage record for a single registered hash."""

    dhash_hex: str
    skb_data: dict
    phash_hex: Optional[str] = None
    ppp_256: Optional[str] = None
    promoted: bool = False


# ---------------------------------------------------------------------------
# CodexRegistry
# ---------------------------------------------------------------------------

class CodexRegistry:
    """In-memory Fast Path lookup table.

    Primary index is on ``dhash_hex`` (exact O(1) dict lookup).  Secondary
    indices on ``phash_hex`` and ``ppp_256`` are maintained for
    cross-validation and provenance queries.

    Fuzzy matching
    ~~~~~~~~~~~~~~
    When no exact dHash match is found the registry performs a **linear scan**
    over all entries to find the nearest neighbour within
    ``max_hamming_distance``.

    .. note::

        For production scale (>100K entries) the linear scan should be
        replaced with **LSH band indexing** — partition each 64-bit hash into
        16 bands of 4 hex characters and maintain an inverted index per band.
        Two hashes sharing at least one band are candidate neighbours,
        reducing the scan to a small fraction of the full registry.
    """

    def __init__(self) -> None:
        # Primary index: dhash_hex -> _RegistryEntry
        self._primary: Dict[str, _RegistryEntry] = {}

        # Secondary indices (optional, best-effort)
        self._by_phash: Dict[str, str] = {}   # phash_hex -> dhash_hex
        self._by_ppp: Dict[str, str] = {}     # ppp_256 -> dhash_hex

        # Bookkeeping
        self._promotion_count: int = 0
        self._promotion_log: list[dict] = []
        self._total_lookups: int = 0
        self._exact_hits: int = 0
        self._fuzzy_hits: int = 0
        self._misses: int = 0

    # -- Registration ------------------------------------------------------

    def register(
        self,
        dhash_hex: str,
        skb_data: dict,
        phash_hex: Optional[str] = None,
        ppp_256: Optional[str] = None,
    ) -> None:
        """Store an SKB indexed by its dHash.

        Optionally store secondary indices on ``phash_hex`` and ``ppp_256``
        for cross-validation and global provenance resolution.
        """
        entry = _RegistryEntry(
            dhash_hex=dhash_hex,
            skb_data=skb_data,
            phash_hex=phash_hex,
            ppp_256=ppp_256,
        )
        self._primary[dhash_hex] = entry

        if phash_hex is not None:
            self._by_phash[phash_hex] = dhash_hex
        if ppp_256 is not None:
            self._by_ppp[ppp_256] = dhash_hex

        logger.info(
            "Registered dHash=%s (phash=%s, ppp=%s)",
            dhash_hex,
            phash_hex,
            ppp_256,
        )

    # -- Lookup ------------------------------------------------------------

    def lookup(
        self, dhash_hex: str, max_hamming_distance: int = 5
    ) -> LookupResult:
        """Look up an SKB by dHash.

        1. **Exact match** — O(1) dict lookup.
        2. **Fuzzy match** — linear scan for nearest neighbour within
           ``max_hamming_distance``.  Returns the closest match.
        3. **Miss** — no entry within threshold.

        .. note::

            For production scale (>100K entries) replace the linear scan with
            LSH band indexing (16 bands of 4 hex chars) to achieve sub-linear
            fuzzy lookup.
        """
        self._total_lookups += 1
        t0 = time.perf_counter()

        # --- exact match ---
        entry = self._primary.get(dhash_hex)
        if entry is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._exact_hits += 1
            logger.debug(
                "EXACT hit for %s in %.3f ms", dhash_hex, elapsed_ms
            )
            return LookupResult(
                skb_data=entry.skb_data,
                matched_hash=dhash_hex,
                query_hash=dhash_hex,
                hamming_distance=0,
                match_type="EXACT",
                lookup_time_ms=elapsed_ms,
            )

        # --- fuzzy scan ---
        best_distance = max_hamming_distance + 1
        best_entry: Optional[_RegistryEntry] = None
        best_hash: Optional[str] = None

        for stored_hash, stored_entry in self._primary.items():
            try:
                dist = CompositeHasher.hamming_distance(dhash_hex, stored_hash)
            except ValueError:
                # Length mismatch — skip silently (different hash families)
                continue

            if dist < best_distance:
                best_distance = dist
                best_entry = stored_entry
                best_hash = stored_hash

                # Early exit on perfect fuzzy match (distance 1)
                if dist <= 1:
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if best_entry is not None and best_distance <= max_hamming_distance:
            self._fuzzy_hits += 1
            logger.info(
                "FUZZY hit for %s -> %s (distance=%d) in %.3f ms",
                dhash_hex,
                best_hash,
                best_distance,
                elapsed_ms,
            )
            return LookupResult(
                skb_data=best_entry.skb_data,
                matched_hash=best_hash,
                query_hash=dhash_hex,
                hamming_distance=best_distance,
                match_type="FUZZY",
                lookup_time_ms=elapsed_ms,
            )

        # --- miss ---
        self._misses += 1
        logger.info("MISS for %s in %.3f ms", dhash_hex, elapsed_ms)
        return LookupResult(
            skb_data=None,
            matched_hash=None,
            query_hash=dhash_hex,
            hamming_distance=-1,
            match_type="MISS",
            lookup_time_ms=elapsed_ms,
        )

    # -- Convenience: hash-then-lookup -------------------------------------

    def lookup_by_image(
        self,
        image_path_or_pil: Any,
        hasher: CompositeHasher,
        max_hamming_distance: int = 5,
    ) -> LookupResult:
        """Hash the image with *hasher* then look up the dHash."""
        dhash_hex = hasher.compute_dhash(image_path_or_pil)
        return self.lookup(dhash_hex, max_hamming_distance=max_hamming_distance)

    # -- Loop-closure promotion --------------------------------------------

    def promote_to_fast_path(
        self, trigger_hash: str, skb_data: dict
    ) -> None:
        """Register a Slow Path result so future encounters are Fast Path hits.

        This is the loop-closure mechanism: when the Slow Path (VLM pipeline)
        produces an SKB for a previously unseen image, the hash/SKB pair is
        promoted into the registry.
        """
        self.register(trigger_hash, skb_data)

        # Mark the entry as a promotion
        self._primary[trigger_hash].promoted = True
        self._promotion_count += 1

        event = {
            "timestamp": time.time(),
            "trigger_hash": trigger_hash,
            "skb_keys": list(skb_data.keys()),
        }
        self._promotion_log.append(event)

        logger.info(
            "PROMOTED to Fast Path: %s (total promotions: %d)",
            trigger_hash,
            self._promotion_count,
        )

    # -- Persistence -------------------------------------------------------

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize the entire registry to a JSON file on disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "version": 1,
            "entries": {},
            "stats": self.stats(),
            "promotion_log": self._promotion_log,
        }

        for dhash_hex, entry in self._primary.items():
            payload["entries"][dhash_hex] = {
                "skb_data": entry.skb_data,
                "phash_hex": entry.phash_hex,
                "ppp_256": entry.ppp_256,
                "promoted": entry.promoted,
            }

        filepath.write_text(json.dumps(payload, indent=2))
        logger.info("Registry saved to %s (%d entries)", filepath, len(self._primary))

    def load(self, filepath: Union[str, Path]) -> None:
        """Load the registry from a JSON file, replacing current contents."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Registry file not found: {filepath}")

        raw = json.loads(filepath.read_text())

        # Reset state
        self._primary.clear()
        self._by_phash.clear()
        self._by_ppp.clear()

        entries = raw.get("entries", {})
        for dhash_hex, blob in entries.items():
            self.register(
                dhash_hex=dhash_hex,
                skb_data=blob["skb_data"],
                phash_hex=blob.get("phash_hex"),
                ppp_256=blob.get("ppp_256"),
            )
            if blob.get("promoted", False):
                self._primary[dhash_hex].promoted = True

        # Restore promotion log if present
        self._promotion_log = raw.get("promotion_log", [])
        self._promotion_count = sum(
            1 for e in self._primary.values() if e.promoted
        )

        logger.info(
            "Registry loaded from %s (%d entries, %d promotions)",
            filepath,
            len(self._primary),
            self._promotion_count,
        )

    # -- Stats -------------------------------------------------------------

    def stats(self) -> dict:
        """Return operational statistics for monitoring / dashboards."""
        return {
            "total_entries": len(self._primary),
            "total_lookups": self._total_lookups,
            "exact_hits": self._exact_hits,
            "fuzzy_hits": self._fuzzy_hits,
            "misses": self._misses,
            "promotion_count": self._promotion_count,
            "secondary_phash_entries": len(self._by_phash),
            "secondary_ppp_entries": len(self._by_ppp),
        }

    def __len__(self) -> int:
        return len(self._primary)

    def __contains__(self, dhash_hex: str) -> bool:
        return dhash_hex in self._primary

    def __repr__(self) -> str:
        return (
            f"<CodexRegistry entries={len(self._primary)} "
            f"promotions={self._promotion_count}>"
        )

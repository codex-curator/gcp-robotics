#!/usr/bin/env python3
"""
Golden Codex Protocol 2.0 -- Fleet Learning & Registry Sync Demo
=================================================================

Demonstrates fleet-wide knowledge propagation:

    "One robot learns, the whole fleet knows."

Workflow:
    1. Robot A encounters a new object, generates an SKB, signs it with
       the fleet HMAC secret, and registers it in its local CodexRegistry.
    2. Robot A saves its registry to disk (JSON snapshot).
    3. Robot B loads Robot A's registry -- simulating a fleet sync event.
    4. Robot B encounters the SAME object  -> EXACT match (no LLM call).
    5. Robot B encounters a SIMILAR view   -> FUZZY match (still no LLM call).
    6. Fleet stats are printed showing knowledge propagation.

The fleet_secret HMAC ensures only authorized robots can contribute
valid SKBs.  A forged payload without the secret will fail verification.

Run:
    python examples/fleet_sync.py
"""

from __future__ import annotations

import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path setup -- ensure gcp_robotics is importable regardless of CWD
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from gcp_robotics.hash_engine.registry import CodexRegistry
from gcp_robotics.soulmark import SoulmarkVerifier

# ---------------------------------------------------------------------------
# Shared fleet secret -- in production this is distributed via secure
# key exchange to every authorized robot in the fleet.
# ---------------------------------------------------------------------------
FLEET_SECRET = "metavolve-fleet-alpha-2026"

# Mock perceptual hash (16 hex chars = 64-bit dHash)
OBJECT_DHASH = "0xA3B2C1D4E5F60718"

# Mock SKB payload (minimal four-layer structure for demo purposes)
MOCK_SKB = {
    "object_label": "red_coffee_mug",
    "semantic_topology": {
        "classification": "Rigid_Body_Graspable",
        "mass_kg": 0.35,
        "material": "ceramic",
    },
    "affordance_layer": {
        "primitive": "GRASP_PARALLEL_JAW",
        "grip_force_n": 12.0,
        "approach_vector": [0.0, 0.0, -1.0],
    },
    "safety_layer": {
        "hazard_class": "NONE",
        "max_force_n": 25.0,
        "fragile": True,
    },
}


def main() -> None:
    verifier = SoulmarkVerifier(fleet_secret=FLEET_SECRET)

    # ── Step 1: Robot A encounters a new object ──────────────────────────
    print("=" * 65)
    print("  FLEET LEARNING DEMO -- Golden Codex Protocol 2.0")
    print("=" * 65)

    print("\n[Robot A] Encountered new object: red_coffee_mug")
    print("[Robot A] No registry entry -- Slow Path (VLM) generates SKB...")

    # Sign the SKB with the fleet HMAC secret
    signed = verifier.sign(MOCK_SKB)
    print(f"[Robot A] Soulmark (HMAC-SHA256): {signed.soulmark[:32]}...")

    # Register in Robot A's local CodexRegistry
    registry_a = CodexRegistry()
    skb_with_soulmark = {**MOCK_SKB, "provenance": {"soulmark": signed.soulmark}}
    registry_a.promote_to_fast_path(OBJECT_DHASH, skb_with_soulmark)
    print(f"[Robot A] Promoted to Fast Path. Registry: {registry_a}")

    # ── Step 2: Robot A saves registry to disk ───────────────────────────
    sync_file = os.path.join(tempfile.gettempdir(), "fleet_sync_snapshot.json")
    registry_a.save(sync_file)
    print(f"\n[Robot A] Registry saved -> {sync_file}")

    # ── Step 3: Robot B loads the fleet snapshot ─────────────────────────
    print("\n[Robot B] Fleet sync event -- loading Robot A's registry...")
    registry_b = CodexRegistry()
    registry_b.load(sync_file)
    print(f"[Robot B] Registry loaded. {registry_b}")

    # Verify the Soulmark before trusting the SKB
    entry = registry_b.lookup(OBJECT_DHASH)
    vr = verifier.verify(entry.skb_data, entry.skb_data["provenance"]["soulmark"])
    print(f"[Robot B] Soulmark verification: {'PASS' if vr.valid else 'FAIL'}")

    # ── Step 4: Robot B sees the SAME object -> EXACT match ──────────────
    print("\n[Robot B] Camera sees red_coffee_mug (same angle)...")
    result_exact = registry_b.lookup(OBJECT_DHASH)
    print(f"[Robot B] Match type : {result_exact.match_type}")
    print(f"[Robot B] Hamming    : {result_exact.hamming_distance}")
    print(f"[Robot B] Object     : {result_exact.skb_data['object_label']}")
    print(f"[Robot B] Action     : {result_exact.skb_data['affordance_layer']['primitive']}")
    print("  --> No LLM call needed. Fast Path served the SKB.")

    # ── Step 5: Robot B sees a SLIGHTLY different view -> FUZZY match ────
    # Flip two bits in the hash to simulate a slightly different viewpoint
    modified_hash = OBJECT_DHASH[:4] + "B4" + OBJECT_DHASH[6:]
    print(f"\n[Robot B] Camera sees red_coffee_mug (different angle)...")
    print(f"  Original hash : {OBJECT_DHASH}")
    print(f"  New hash      : {modified_hash}")

    result_fuzzy = registry_b.lookup(modified_hash, max_hamming_distance=8)
    print(f"[Robot B] Match type : {result_fuzzy.match_type}")
    print(f"[Robot B] Hamming    : {result_fuzzy.hamming_distance}")
    print(f"[Robot B] Object     : {result_fuzzy.skb_data['object_label']}")
    print("  --> Fuzzy match. Still no LLM call -- fleet knowledge reused.")

    # ── Step 6: Demonstrate HMAC rejection of unauthorized SKBs ──────────
    print("\n[Rogue Bot] Attempting to inject a forged SKB...")
    rogue_verifier = SoulmarkVerifier(fleet_secret="wrong-secret")
    rogue_signed = rogue_verifier.sign(MOCK_SKB)
    rogue_check = verifier.verify(MOCK_SKB, rogue_signed.soulmark)
    print(f"[Fleet]    Soulmark verification: {'PASS' if rogue_check.valid else 'REJECT'}")
    print("  --> Fleet HMAC prevents unauthorized knowledge injection.")

    # ── Step 7: Fleet stats ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FLEET STATS")
    print("=" * 65)
    for label, reg in [("Robot A", registry_a), ("Robot B", registry_b)]:
        s = reg.stats()
        print(f"\n  {label}:")
        print(f"    Entries      : {s['total_entries']}")
        print(f"    Promotions   : {s['promotion_count']}")
        print(f"    Lookups      : {s['total_lookups']}")
        print(f"    Exact hits   : {s['exact_hits']}")
        print(f"    Fuzzy hits   : {s['fuzzy_hits']}")
        print(f"    Misses       : {s['misses']}")

    print("\n  One robot learns, the whole fleet knows.")
    print("=" * 65)

    # Cleanup
    os.remove(sync_file)


if __name__ == "__main__":
    main()

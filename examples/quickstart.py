"""
GCP-Robotics SDK -- Quickstart
==============================

The simplest possible Perceptual Codex Object (PCO) demo.

This script demonstrates the core "System 1" fast path of the Golden Codex
Protocol: hash an image, look it up in the registry, get back a Spatial
Kinematic Blueprint (SKB) in sub-millisecond time.  No hardware, no API keys,
no external files required -- everything runs in memory.

Workflow:
  1. Create a CompositeHasher and an empty CodexRegistry.
  2. Hash a synthetic test image (generated with PIL).
  3. Look it up -- MISS on first encounter (object is unknown).
  4. Register the object with an SKB payload (simulating Slow Path output).
  5. Look it up again -- EXACT match, SKB returned in < 1 ms.
"""

import numpy as np
from PIL import Image

from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry


def main() -> None:
    # --- Initialize the hash engine and an empty registry ---
    hasher = CompositeHasher()
    registry = CodexRegistry()

    # --- Create a synthetic test image (red bolt on black background) ---
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[16:48, 24:40] = [200, 30, 30]  # red rectangle as a stand-in object
    bolt_image = Image.fromarray(arr)

    # --- Compute the composite perceptual hash ---
    composite = hasher.compute_composite(bolt_image)
    print(f"dHash  : {composite.dhash_hex}")
    print(f"pHash  : {composite.phash_hex}")
    print(f"PPP-256: {composite.ppp_256bit[:24]}...")

    # --- First lookup: registry is empty, so this is a MISS ---
    result = registry.lookup(composite.dhash_hex)
    print(f"\nFirst lookup  -> {result.match_type}")
    assert result.match_type == "MISS", "Expected a MISS on empty registry"

    # --- Register the object with an SKB (simulating Slow Path output) ---
    skb_payload = {
        "object_label": "M6 hex bolt (red, anodized)",
        "affordance_layer": {
            "action_primitive": "PICK_AND_PLACE",
            "grip_force_n": 5.0,
            "approach_vector": [0, 0, -1],
        },
        "safety_layer": {
            "max_grip_force_n": 12.0,
            "drop_hazard": False,
        },
    }
    registry.register(
        dhash_hex=composite.dhash_hex,
        skb_data=skb_payload,
        phash_hex=composite.phash_hex,
        ppp_256=composite.ppp_256bit,
    )
    print(f"Registered object. Registry size: {len(registry)}")

    # --- Second lookup: EXACT match, sub-millisecond ---
    result = registry.lookup(composite.dhash_hex)
    print(f"Second lookup -> {result.match_type} in {result.lookup_time_ms:.4f} ms")
    assert result.match_type == "EXACT", "Expected an EXACT match after registration"

    # --- Print the retrieved SKB ---
    skb = result.skb_data
    print(f"\n--- Retrieved SKB ---")
    print(f"  Object : {skb['object_label']}")
    print(f"  Action : {skb['affordance_layer']['action_primitive']}")
    print(f"  Force  : {skb['affordance_layer']['grip_force_n']} N")
    print(f"  Safe?  : grip <= {skb['safety_layer']['max_grip_force_n']} N")

    print(f"\nRegistry stats: {registry.stats()}")


if __name__ == "__main__":
    main()

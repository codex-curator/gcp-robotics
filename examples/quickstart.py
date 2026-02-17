"""
GCP Robotics SDK — Quickstart Example

Self-contained demo: generates a synthetic image in memory, hashes it,
registers it, and performs a sub-millisecond lookup. No external files needed.
"""

import numpy as np
from PIL import Image
from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry

# 1. Initialize the hash engine and registry
hasher = CompositeHasher()
registry = CodexRegistry()

# 2. Generate a synthetic "red bolt" image in memory (no files needed)
arr = np.zeros((64, 64, 3), dtype=np.uint8)
arr[16:48, 24:40] = [200, 30, 30]  # red rectangle
bolt_img = Image.fromarray(arr)

# 3. Register the object
registry.register(
    dhash_hex=hasher.compute_dhash(bolt_img),
    skb_data={
        "action_primitive": "PICK_AND_PLACE",
        "grip_force_n": 5.0,
        "approach_vector": [0, 0, -1],
    },
)

# 4. Fast-path lookup (< 1 ms)
query_hash = hasher.compute_dhash(bolt_img)
result = registry.lookup(query_hash)

if result.match_type != "MISS":
    print(f"SKB retrieved in {result.lookup_time_ms:.3f} ms")
    print(f"Match type: {result.match_type}")
    print(f"Action: {result.skb_data['action_primitive']}")
    print(f"Force: {result.skb_data['grip_force_n']} N")
else:
    print("Unknown object — routing to System 2 (VLA inference)")

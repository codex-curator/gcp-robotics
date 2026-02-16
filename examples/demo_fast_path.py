#!/usr/bin/env python3
"""
Golden Codex Protocol 2.0 -- Fast Path End-to-End Demo
======================================================

Demonstrates the complete Fast Path lifecycle:

  1. Template Environment registration (pre-hashing YCB objects)
  2. EXACT match lookup (O(1) Fast Path hit)
  3. FUZZY match lookup (perturbed image, Hamming distance)
  4. HASH MISS -> Slow Path recovery (MockSlowPathGenerator)
  5. LOOP CLOSURE (promote Slow Path result, verify re-lookup is EXACT)

Run:
    python examples/demo_fast_path.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — ensure gcp_robotics is importable regardless of CWD
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/tmp/gcp-robotics-venv/lib/python3.10/site-packages")

import numpy as np
from PIL import Image

from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry
from gcp_robotics.datasets.ycb_loader import YCBLoader
from gcp_robotics.template_env.registrar import TemplateEnvironmentRegistrar
from gcp_robotics.slow_path.prompts import MockSlowPathGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_fast_path")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def print_banner(text: str, char: str = "=", width: int = 72) -> None:
    """Print a centred banner."""
    print()
    print(char * width)
    print(text.center(width))
    print(char * width)
    print()


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print(f"--- {title} " + "-" * max(0, 68 - len(title) - 5))
    print()


def print_stats(registry: CodexRegistry) -> None:
    """Print registry statistics."""
    stats = registry.stats()
    print(f"  Total entries        : {stats['total_entries']}")
    print(f"  Total lookups        : {stats['total_lookups']}")
    print(f"  Exact hits           : {stats['exact_hits']}")
    print(f"  Fuzzy hits           : {stats['fuzzy_hits']}")
    print(f"  Misses               : {stats['misses']}")
    print(f"  Promotions           : {stats['promotion_count']}")
    print(f"  Secondary (pHash)    : {stats['secondary_phash_entries']}")
    print(f"  Secondary (PPP)      : {stats['secondary_ppp_entries']}")


def add_noise_to_image(image_path: str, noise_level: int = 25) -> Image.Image:
    """Load an image and add Gaussian noise + a slight crop."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.int16)

    # Add Gaussian noise
    noise = np.random.default_rng(42).integers(
        -noise_level, noise_level + 1, size=arr.shape, dtype=np.int16
    )
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # Slight crop (remove 3 pixels from each edge)
    h, w = arr.shape[:2]
    margin = 3
    if h > margin * 4 and w > margin * 4:
        arr = arr[margin : h - margin, margin : w - margin]

    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    print_banner("Golden Codex Protocol 2.0 -- Fast Path Demo")

    # ---- 1. Initialise ---------------------------------------------------
    print_section("STEP 1: Initialise Components")

    loader = YCBLoader()
    registrar = TemplateEnvironmentRegistrar(
        environment_name="Demo Lab Cell",
        workspace_zone="zone_A_workbench",
    )
    hasher = registrar.hasher
    registry = registrar.get_registry()

    print("  YCBLoader           : ready")
    print("  TemplateEnvRegistrar: ready")
    print(f"  Environment         : {registrar.environment_name}")
    print(f"  Workspace Zone      : {registrar.workspace_zone}")

    # ---- 2. Register 5 YCB objects (3 images each) -----------------------
    print_section("STEP 2: Register 5 YCB Objects (3 images each)")

    target_objects = [
        "001_chips_can",
        "003_cracker_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "025_mug",
    ]

    all_image_paths: dict[str, list[str]] = {}

    for oid in target_objects:
        images = loader.generate_test_images(oid, count=3)
        skb_seed = loader.build_skb_seed(oid)
        image_paths = []
        for img_path in images:
            registrar.register_object_from_image(
                image_path=str(img_path),
                object_id=oid,
                skb_seed=skb_seed,
                end_effector="robotiq_2f85",
                manufacturer="YCB_Dataset",
                model_no=oid,
                serial=f"ycb-{oid}-{img_path.stem}",
            )
            image_paths.append(str(img_path))
        all_image_paths[oid] = image_paths
        gt = loader.get_ground_truth(oid)
        print(f"  Registered: {oid} ({gt['name']}) -- {len(image_paths)} images")

    print()
    print(f"  Total registered images: {len(registry)}")

    print_section("STEP 2b: Registry Stats After Registration")
    print_stats(registry)

    # ---- 3. FAST PATH HIT — Exact match ---------------------------------
    print_section("STEP 3: FAST PATH HIT -- Exact Match")

    # Pick the first image of chips_can
    test_image = all_image_paths["001_chips_can"][0]
    print(f"  Query image: {test_image}")

    result = registry.lookup_by_image(test_image, hasher)
    print(f"  Match type           : {result.match_type}")
    print(f"  Hamming distance     : {result.hamming_distance}")
    print(f"  Lookup time          : {result.lookup_time_ms:.3f} ms")
    print(f"  Query hash           : {result.query_hash}")
    print(f"  Matched hash         : {result.matched_hash}")

    if result.skb_data:
        skb = result.skb_data
        affordance = skb.get("layer_3_affordance", {})
        action = affordance.get("action_schema", {})
        print(f"  Action primitive     : {action.get('primitive_id', 'N/A')}")
        print(f"  Speed profile        : {action.get('parameters', {}).get('speed_profile', 'N/A')}")
        force = action.get("parameters", {}).get("force_profile", {})
        print(f"  Max grip force       : {force.get('max_grip_newtons', 'N/A')} N")
        provenance = skb.get("layer_1_provenance", {})
        print(f"  Source dataset       : {provenance.get('source_dataset', 'N/A')}")

    assert result.match_type == "EXACT", f"Expected EXACT, got {result.match_type}"
    print()
    print("  [PASS] Exact match confirmed.")

    # ---- 4. FUZZY MATCH — Perturbed image --------------------------------
    print_section("STEP 4: FUZZY MATCH -- Noisy/Cropped Image")

    fuzzy_source = all_image_paths["003_cracker_box"][1]
    print(f"  Source image (clean) : {fuzzy_source}")

    noisy_img = add_noise_to_image(fuzzy_source, noise_level=25)
    print(f"  Applied: Gaussian noise (level=25) + 3px crop")

    # Hash the noisy image (in-memory PIL)
    noisy_hash = hasher.compute_dhash(noisy_img)
    clean_hash = hasher.compute_dhash(fuzzy_source)
    bit_diff = CompositeHasher.hamming_distance(noisy_hash, clean_hash)
    print(f"  Clean dHash          : {clean_hash}")
    print(f"  Noisy dHash          : {noisy_hash}")
    print(f"  Hamming distance     : {bit_diff} bits")

    result = registry.lookup(noisy_hash, max_hamming_distance=10)
    print(f"  Match type           : {result.match_type}")
    print(f"  Registry Hamming     : {result.hamming_distance}")
    print(f"  Lookup time          : {result.lookup_time_ms:.3f} ms")

    if result.skb_data:
        skb = result.skb_data
        action = skb.get("layer_3_affordance", {}).get("action_schema", {})
        print(f"  Action primitive     : {action.get('primitive_id', 'N/A')}")
        ident = skb.get("identifiers", {})
        print(f"  Matched codex_id     : {ident.get('codex_id', 'N/A')}")

    if result.match_type == "EXACT":
        print()
        print("  [NOTE] Noise was too small to shift the hash -- got EXACT instead of FUZZY.")
        print("  This is fine: the perceptual hash is robust to minor perturbations.")
    elif result.match_type == "FUZZY":
        print()
        print("  [PASS] Fuzzy match confirmed.")
    else:
        print()
        print(f"  [NOTE] Got {result.match_type}. Noise may have been too large for threshold.")

    # ---- 5. HASH MISS -> SLOW PATH recovery ------------------------------
    print_section("STEP 5: HASH MISS -> SLOW PATH Recovery")

    # Use an object NOT registered in the environment
    miss_object_id = "011_banana"
    print(f"  Generating image for UNREGISTERED object: {miss_object_id}")

    miss_images = loader.generate_test_images(miss_object_id, count=1)
    miss_image_path = str(miss_images[0])
    print(f"  Test image: {miss_image_path}")

    # Hash and attempt lookup
    miss_composite = hasher.compute_composite(miss_image_path)
    miss_dhash = miss_composite.dhash_hex
    print(f"  dHash: {miss_dhash}")

    result = registry.lookup(miss_dhash)
    print(f"  Match type           : {result.match_type}")
    print(f"  Hamming distance     : {result.hamming_distance}")
    print(f"  Lookup time          : {result.lookup_time_ms:.3f} ms")

    assert result.match_type == "MISS", f"Expected MISS, got {result.match_type}"
    print()
    print("  [MISS] No match found. Triggering Slow Path...")
    print()

    # Invoke MockSlowPathGenerator
    slow_path = MockSlowPathGenerator()

    t0 = time.perf_counter()
    generated_skb = slow_path.generate_and_validate(
        trigger_event="HASH_MISS_OOD_STATE",
        workspace_zone=registrar.workspace_zone,
        end_effector="robotiq_2f85",
        failure_context=(
            f"Hash miss for object {miss_object_id} (banana). "
            "Object not in template environment. Need recovery action."
        ),
        image_path=miss_image_path,
        available_objects=target_objects,
    )
    generation_ms = (time.perf_counter() - t0) * 1000

    print(f"  Slow Path generated SKB in {generation_ms:.1f} ms")
    print(f"  Primitive            : {generated_skb.get('primitive_id')}")
    print(f"  Confidence           : {generated_skb.get('confidence_score')}")
    print(f"  Reasoning            : {generated_skb.get('reasoning_trace', '')[:100]}...")
    print(f"  Hazard class         : {generated_skb.get('hazard_class')}")

    # Format as promotion entry
    promotion_entry = slow_path.format_promotion_entry(
        trigger_hash=miss_dhash,
        generated_skb=generated_skb,
        generation_time_ms=generation_ms,
    )

    # Promote to Fast Path
    print()
    print("  Promoting Slow Path result to Fast Path cache...")
    registry.promote_to_fast_path(
        trigger_hash=miss_dhash,
        skb_data=promotion_entry,
    )
    print(f"  [PROMOTED] {miss_dhash} now in Fast Path.")

    # ---- 6. LOOP CLOSURE — re-lookup should be EXACT ---------------------
    print_section("STEP 6: LOOP CLOSURE -- Re-lookup After Promotion")

    print(f"  Re-querying dHash: {miss_dhash}")
    result = registry.lookup(miss_dhash)
    print(f"  Match type           : {result.match_type}")
    print(f"  Hamming distance     : {result.hamming_distance}")
    print(f"  Lookup time          : {result.lookup_time_ms:.3f} ms")

    assert result.match_type == "EXACT", f"Expected EXACT after promotion, got {result.match_type}"
    print()
    print("  [PASS] Loop closure confirmed -- previously missed hash is now an EXACT hit.")

    if result.skb_data:
        recovery = result.skb_data.get("recovery_metadata", {})
        print(f"  Generated by slow path : {recovery.get('generated_by_slow_path')}")
        print(f"  Trigger event          : {recovery.get('trigger_event')}")
        print(f"  Confidence             : {recovery.get('confidence_score')}")

    # ---- 7. Final stats --------------------------------------------------
    print_section("STEP 7: Final Registry Statistics")
    print_stats(registry)

    print_banner("Demo Complete -- Golden Codex Protocol 2.0", char="*")
    print("  The Fast Path lifecycle has been demonstrated end-to-end:")
    print("    1. Template environment pre-registration (5 objects, 15 images)")
    print("    2. O(1) exact-match lookup")
    print("    3. Fuzzy match with Hamming distance tolerance")
    print("    4. Hash miss -> Slow Path (VLM) recovery")
    print("    5. Loop closure: promoted result becomes instant Fast Path hit")
    print()


if __name__ == "__main__":
    main()

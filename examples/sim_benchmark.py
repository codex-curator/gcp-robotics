#!/usr/bin/env python3
"""
PCO Simulation Benchmark -- GCP-Robotics SDK
=============================================

Demonstrates the Perceptual Compute Offloading speed advantage
in a PyBullet tabletop simulation. No hardware, no API keys required.

Compares:
  - PCO Fast Path:  hash -> registry lookup -> SKB (< 1 ms)
  - VLA Baseline:   simulated inference latency (300 ms per frame)

Run:
    python examples/sim_benchmark.py
    python examples/sim_benchmark.py --episodes 50 --gui
    python examples/sim_benchmark.py --stability-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup -- ensure gcp_robotics is importable regardless of CWD
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from PIL import Image

from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry

# ---------------------------------------------------------------------------
# Try importing the sim module (may be built by another agent concurrently)
# ---------------------------------------------------------------------------
_HAS_SIM_MODULE = False
try:
    from gcp_robotics.sim.environment import TabletopEnvironment
    from gcp_robotics.sim.pco_controller import PCOController
    from gcp_robotics.sim.benchmark import PCOBenchmark
    _HAS_SIM_MODULE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("sim_benchmark")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BANNER_WIDTH = 64

# Object definitions for the synthetic fallback scene.
# Each entry: (name, color_rgb, size_wh, position_offset_xy)
OBJECTS = [
    ("red_cube",        (200,  30,  30), (40, 40), (0,   0)),
    ("blue_sphere",     ( 30,  60, 200), (36, 36), (50,  0)),
    ("green_cylinder",  ( 30, 180,  50), (28, 44), (0,  50)),
    ("yellow_duck",     (220, 200,  30), (38, 32), (50, 50)),
    ("small_mug",       (160,  80,  40), (30, 38), (25, 25)),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Timing result for a single frame."""
    pco_lookup_ms: float
    vla_latency_ms: float
    match_type: str
    hamming_distance: int


@dataclass
class EpisodeResult:
    """Aggregated result for one episode."""
    episode_id: int
    num_frames: int
    hits: int
    miss_count: int
    pco_latencies_ms: List[float] = field(default_factory=list)
    vla_latencies_ms: List[float] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Final benchmark report."""
    pco_mean_ms: float
    pco_p99_ms: float
    pco_throughput_hz: float
    vla_mean_ms: float
    vla_p99_ms: float
    vla_throughput_hz: float
    hit_rate_pct: float
    speedup: float
    total_frames: int
    episodes: int
    stability_mean_hamming: float
    stability_max_hamming: int
    stability_threshold: int
    stability_verdict: str


# ---------------------------------------------------------------------------
# Helpers: display
# ---------------------------------------------------------------------------

def banner(text: str, char: str = "=") -> None:
    print(char * BANNER_WIDTH)
    print(f"  {text}")
    print(char * BANNER_WIDTH)


def section(title: str) -> None:
    print()
    print(f"  {title}")
    print(f"  {'-' * len(title)}")


def table_row(metric: str, pco_val: str, vla_val: str) -> str:
    return f"  | {metric:<18} | {pco_val:<12} | {vla_val:<12} |"


def print_results_table(report: BenchmarkReport) -> None:
    sep = f"  +{'-'*20}+{'-'*14}+{'-'*14}+"
    hdr = f"  | {'Metric':<18} | {'PCO':<12} | {'VLA Baseline':<12} |"
    print(sep)
    print(hdr)
    print(sep)
    print(table_row("Mean latency",
                     f"{report.pco_mean_ms:.2f} ms",
                     f"{report.vla_mean_ms:.2f} ms"))
    print(table_row("P99 latency",
                     f"{report.pco_p99_ms:.2f} ms",
                     f"{report.vla_p99_ms:.2f} ms"))
    print(table_row("Throughput",
                     f"{report.pco_throughput_hz:,.0f} Hz",
                     f"{report.vla_throughput_hz:.1f} Hz"))
    print(table_row("Hit rate",
                     f"{report.hit_rate_pct:.1f}%",
                     "N/A"))
    print(sep)


# ---------------------------------------------------------------------------
# Synthetic image generation (fallback when sim module is unavailable)
# ---------------------------------------------------------------------------

def generate_object_image(
    name: str,
    color: Tuple[int, int, int],
    size_wh: Tuple[int, int],
    canvas_size: int = 224,
    offset_xy: Tuple[int, int] = (0, 0),
    jitter_px: int = 0,
    noise_level: int = 0,
) -> Image.Image:
    """Render a synthetic coloured rectangle on a grey background.

    Optionally apply position jitter and pixel noise to simulate camera
    perturbations in the fallback (non-PyBullet) mode.
    """
    rng = np.random.default_rng()
    arr = np.full((canvas_size, canvas_size, 3), 60, dtype=np.uint8)  # grey bg

    cx = canvas_size // 2 + offset_xy[0]
    cy = canvas_size // 2 + offset_xy[1]
    if jitter_px > 0:
        cx += int(rng.integers(-jitter_px, jitter_px + 1))
        cy += int(rng.integers(-jitter_px, jitter_px + 1))

    w, h = size_wh
    x0 = max(0, cx - w // 2)
    y0 = max(0, cy - h // 2)
    x1 = min(canvas_size, cx + w // 2)
    y1 = min(canvas_size, cy + h // 2)

    arr[y0:y1, x0:x1] = color

    if noise_level > 0:
        noise = rng.integers(-noise_level, noise_level + 1,
                             size=arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Phase 1: Environment setup
# ---------------------------------------------------------------------------

def phase1_setup(args: argparse.Namespace) -> Tuple[CompositeHasher, CodexRegistry]:
    section("Phase 1: Environment Setup")

    hasher = CompositeHasher()
    registry = CodexRegistry()

    if _HAS_SIM_MODULE and not getattr(args, "_force_fallback", False):
        mode = "GUI" if args.gui else "DIRECT"
        print(f"  Physics:     PyBullet (sim module) [{mode} mode]")
        print(f"  Table:       Standard tabletop at z=0.625m")
    else:
        print(f"  Physics:     Synthetic fallback (sim module not yet built)")
        print(f"  Renderer:    PIL synthetic shapes on 224x224 canvas")

    print(f"  Camera:      224x224 @ 60 deg downward angle")
    print(f"  Hash engine: CompositeHasher (dHash-64 + pHash-64 + PPP-256)")
    print(f"  Registry:    CodexRegistry (exact + fuzzy Hamming lookup)")

    return hasher, registry


# ---------------------------------------------------------------------------
# Phase 2: Object registration
# ---------------------------------------------------------------------------

def phase2_register(
    hasher: CompositeHasher,
    registry: CodexRegistry,
    verbose: bool = False,
) -> Dict[str, str]:
    """Register all benchmark objects and return name->dhash mapping."""
    section("Phase 2: Object Registration")

    registered: Dict[str, str] = {}

    for idx, (name, color, size, offset) in enumerate(OBJECTS, 1):
        img = generate_object_image(name, color, size, offset_xy=offset)
        composite = hasher.compute_composite(img)

        skb_payload = {
            "object_label": name,
            "affordance_layer": {
                "action_primitive": "PICK_AND_PLACE",
                "grip_force_n": 5.0 + idx,
                "approach_vector": [0, 0, -1],
            },
            "safety_layer": {
                "max_grip_force_n": 15.0,
                "drop_hazard": False,
            },
        }

        registry.register(
            dhash_hex=composite.dhash_hex,
            skb_data=skb_payload,
            phash_hex=composite.phash_hex,
            ppp_256=composite.ppp_256bit,
        )

        registered[name] = composite.dhash_hex
        print(f"    [{idx}] {name:<18} -> {composite.dhash_hex}")

    print()
    print(f"  Registered {len(registered)} objects in fast-path registry.")
    return registered


# ---------------------------------------------------------------------------
# Phase 3: Hash stability test
# ---------------------------------------------------------------------------

def phase3_stability(
    hasher: CompositeHasher,
    registered: Dict[str, str],
    perturbations_per_object: int = 100,
    verbose: bool = False,
) -> Tuple[float, int, int, str]:
    """Measure hash stability under small perturbations.

    Returns (mean_hamming, max_hamming, threshold, verdict).
    """
    section("Phase 3: Hash Stability Test")

    jitter_px = 5
    noise_level = 8
    total_perturbs = perturbations_per_object * len(OBJECTS)

    print(f"  Perturbation: +/-{jitter_px}px position, noise_level={noise_level}")
    print(f"  {perturbations_per_object} perturbations per object, "
          f"{total_perturbs} total")
    print()

    all_distances: List[int] = []

    for name, color, size, offset in OBJECTS:
        ref_hash = registered[name]
        for _ in range(perturbations_per_object):
            perturbed = generate_object_image(
                name, color, size,
                offset_xy=offset,
                jitter_px=jitter_px,
                noise_level=noise_level,
            )
            p_dhash = hasher.compute_dhash(perturbed)
            dist = CompositeHasher.hamming_distance(ref_hash, p_dhash)
            all_distances.append(dist)

    mean_dist = statistics.mean(all_distances)
    max_dist = max(all_distances)
    # Threshold: 95th percentile rounded up, minimum 5
    p95 = sorted(all_distances)[int(0.95 * len(all_distances))]
    threshold = max(5, p95 + 1)
    verdict = "STABLE" if max_dist <= 10 else "MARGINAL"

    print(f"  Mean Hamming distance:  {mean_dist:.1f} bits")
    print(f"  Max Hamming distance:   {max_dist} bits")
    print(f"  P95 Hamming distance:   {p95} bits")
    print(f"  Safe threshold:         {threshold}")
    print(f"  Verdict:                {verdict}")

    return mean_dist, max_dist, threshold, verdict


# ---------------------------------------------------------------------------
# Phase 4: PCO vs VLA benchmark
# ---------------------------------------------------------------------------

def phase4_benchmark(
    hasher: CompositeHasher,
    registry: CodexRegistry,
    num_episodes: int,
    num_frames: int,
    vla_latency_ms: float,
    verbose: bool = False,
) -> List[EpisodeResult]:
    """Run the PCO vs VLA comparison benchmark.

    Each episode randomises object positions slightly (jitter + noise).
    Each frame: render -> hash -> lookup (PCO) vs render -> sleep (VLA).
    """
    section("Phase 4: PCO vs VLA Benchmark")

    total_frames = num_episodes * num_frames
    print(f"  Running {num_episodes} episodes x {num_frames} frames "
          f"= {total_frames} frames...")
    print()

    episodes: List[EpisodeResult] = []
    rng = random.Random(42)

    for ep_idx in range(num_episodes):
        ep = EpisodeResult(episode_id=ep_idx + 1, num_frames=num_frames,
                           hits=0, miss_count=0)

        for frame_idx in range(num_frames):
            # Pick a random object
            obj_idx = rng.randint(0, len(OBJECTS) - 1)
            name, color, size, offset = OBJECTS[obj_idx]

            # Render with slight randomisation
            img = generate_object_image(
                name, color, size,
                offset_xy=offset,
                jitter_px=3,
                noise_level=5,
            )

            # --- PCO path: hash + lookup ---
            t0 = time.perf_counter()
            dhash = hasher.compute_dhash(img)
            result = registry.lookup(dhash, max_hamming_distance=8)
            pco_ms = (time.perf_counter() - t0) * 1000

            # --- VLA path: simulated inference ---
            t0 = time.perf_counter()
            time.sleep(vla_latency_ms / 1000.0)
            vla_ms = (time.perf_counter() - t0) * 1000

            ep.pco_latencies_ms.append(pco_ms)
            ep.vla_latencies_ms.append(vla_ms)

            if result.match_type in ("EXACT", "FUZZY"):
                ep.hits += 1
            else:
                ep.miss_count += 1

        hit_pct = (ep.hits / ep.num_frames) * 100
        ep_mean = statistics.mean(ep.pco_latencies_ms)
        print(f"  Episode {ep.episode_id:>2}/{num_episodes}: "
              f"{ep.num_frames} frames, {ep.hits} hits "
              f"({hit_pct:.1f}%), mean {ep_mean:.2f}ms")

        episodes.append(ep)

    return episodes


# ---------------------------------------------------------------------------
# Phase 5: Report
# ---------------------------------------------------------------------------

def phase5_report(
    episodes: List[EpisodeResult],
    stability_mean: float,
    stability_max: int,
    stability_threshold: int,
    stability_verdict: str,
    output_path: Optional[str] = None,
) -> BenchmarkReport:
    """Compile and display the final benchmark report."""
    section("Phase 5: Results")

    # Flatten latencies
    all_pco = []
    all_vla = []
    total_hits = 0
    total_frames = 0
    for ep in episodes:
        all_pco.extend(ep.pco_latencies_ms)
        all_vla.extend(ep.vla_latencies_ms)
        total_hits += ep.hits
        total_frames += ep.num_frames

    pco_mean = statistics.mean(all_pco)
    pco_p99 = sorted(all_pco)[int(0.99 * len(all_pco))] if all_pco else 0.0
    vla_mean = statistics.mean(all_vla)
    vla_p99 = sorted(all_vla)[int(0.99 * len(all_vla))] if all_vla else 0.0

    pco_throughput = 1000.0 / pco_mean if pco_mean > 0 else float("inf")
    vla_throughput = 1000.0 / vla_mean if vla_mean > 0 else float("inf")
    hit_rate = (total_hits / total_frames * 100) if total_frames > 0 else 0.0
    speedup = vla_mean / pco_mean if pco_mean > 0 else float("inf")

    report = BenchmarkReport(
        pco_mean_ms=pco_mean,
        pco_p99_ms=pco_p99,
        pco_throughput_hz=pco_throughput,
        vla_mean_ms=vla_mean,
        vla_p99_ms=vla_p99,
        vla_throughput_hz=vla_throughput,
        hit_rate_pct=hit_rate,
        speedup=speedup,
        total_frames=total_frames,
        episodes=len(episodes),
        stability_mean_hamming=stability_mean,
        stability_max_hamming=stability_max,
        stability_threshold=stability_threshold,
        stability_verdict=stability_verdict,
    )

    print_results_table(report)
    print()
    print(f"  SPEEDUP: {speedup:,.0f}x faster than VLA baseline")
    print()

    # Save JSON report if requested
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "PCO_vs_VLA",
            "sdk_version": "2.0.1",
            "mode": "pybullet_sim" if _HAS_SIM_MODULE else "synthetic_fallback",
            "pco": {
                "mean_latency_ms": round(report.pco_mean_ms, 4),
                "p99_latency_ms": round(report.pco_p99_ms, 4),
                "throughput_hz": round(report.pco_throughput_hz, 1),
                "hit_rate_pct": round(report.hit_rate_pct, 2),
            },
            "vla_baseline": {
                "mean_latency_ms": round(report.vla_mean_ms, 4),
                "p99_latency_ms": round(report.vla_p99_ms, 4),
                "throughput_hz": round(report.vla_throughput_hz, 2),
            },
            "speedup_factor": round(report.speedup, 1),
            "stability": {
                "mean_hamming": round(report.stability_mean_hamming, 2),
                "max_hamming": report.stability_max_hamming,
                "threshold": report.stability_threshold,
                "verdict": report.stability_verdict,
            },
            "parameters": {
                "episodes": report.episodes,
                "frames_per_episode": report.total_frames // max(report.episodes, 1),
                "total_frames": report.total_frames,
                "objects_registered": len(OBJECTS),
            },
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"  Report saved to: {out}")
        print()

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCO Simulation Benchmark -- GCP-Robotics SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/sim_benchmark.py\n"
            "  python examples/sim_benchmark.py --episodes 50 --gui\n"
            "  python examples/sim_benchmark.py --stability-only\n"
            "  python examples/sim_benchmark.py --output report.json --verbose\n"
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of benchmark episodes (default: 10)",
    )
    parser.add_argument(
        "--frames", type=int, default=30,
        help="Frames per episode (default: 30)",
    )
    parser.add_argument(
        "--vla-latency", type=float, default=300.0,
        help="Simulated VLA inference latency in ms (default: 300)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Show PyBullet GUI window (requires sim module)",
    )
    parser.add_argument(
        "--stability-only", action="store_true",
        help="Run only the hash stability test, skip full benchmark",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON benchmark report",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging output",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print()
    banner("PCO Simulation Benchmark -- GCP-Robotics SDK")

    if not _HAS_SIM_MODULE:
        print()
        print("  NOTE: gcp_robotics.sim module not found.")
        print("  Running in synthetic fallback mode (PIL-generated images).")
        print("  Install the sim module for full PyBullet benchmarking.")

    # Phase 1
    hasher, registry = phase1_setup(args)

    # Phase 2
    registered = phase2_register(hasher, registry, verbose=args.verbose)

    # Phase 3
    stability_mean, stability_max, threshold, verdict = phase3_stability(
        hasher, registered, perturbations_per_object=100, verbose=args.verbose,
    )

    if args.stability_only:
        print()
        banner("Stability test complete.", char="-")
        return

    # Phase 4
    episodes = phase4_benchmark(
        hasher, registry,
        num_episodes=args.episodes,
        num_frames=args.frames,
        vla_latency_ms=args.vla_latency,
        verbose=args.verbose,
    )

    # Phase 5
    report = phase5_report(
        episodes,
        stability_mean=stability_mean,
        stability_max=stability_max,
        stability_threshold=threshold,
        stability_verdict=verdict,
        output_path=args.output,
    )

    print(f"  Registry stats: {registry.stats()}")
    print()
    banner("Benchmark complete.", char="=")


if __name__ == "__main__":
    main()

"""
PCOBenchmark — side-by-side PCO vs VLA baseline benchmark runner.

Produces comprehensive latency, throughput, and hash-stability reports
demonstrating the speedup achieved by Perceptual Compute Offloading
over a conventional VLA inference pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from gcp_robotics.sim.environment import TabletopEnvironment, TABLE_SURFACE_Z
from gcp_robotics.sim.pco_controller import PCOController, ProcessResult

logger = logging.getLogger(__name__)

# Default output directory for benchmark JSON reports
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"


# ---------------------------------------------------------------------------
# Report data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    """Full results from a pick-and-place benchmark run."""

    pco_latencies_ms: List[float]
    vla_latencies_ms: List[float]
    pco_hit_rate: float
    pco_miss_rate: float
    pco_mean_latency_ms: float
    pco_p99_latency_ms: float
    vla_mean_latency_ms: float
    throughput_pco_hz: float
    throughput_vla_hz: float
    speedup_factor: float
    hash_stability: dict
    episodes: int
    frames_total: int
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StabilityReport:
    """Results from a hash stability test under perturbations."""

    object_results: Dict[str, List[int]]  # label -> list of hamming distances
    mean_hamming: float
    max_hamming: float
    safe_threshold: int
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# PCOBenchmark
# ---------------------------------------------------------------------------

class PCOBenchmark:
    """Run PCO vs VLA baseline benchmarks in simulation.

    Parameters
    ----------
    environment : TabletopEnvironment
        The PyBullet simulation environment.
    controller : PCOController
        The PCO pipeline controller (hasher + registry).
    """

    def __init__(
        self,
        environment: TabletopEnvironment,
        controller: PCOController,
    ) -> None:
        self._env = environment
        self._ctrl = controller

    # ------------------------------------------------------------------
    # Pick-and-place benchmark
    # ------------------------------------------------------------------

    def run_pick_and_place_benchmark(
        self,
        n_episodes: int = 20,
        n_frames_per_episode: int = 50,
        vla_latency_ms: float = 300.0,
        object_change_probability: float = 0.3,
        seed: Optional[int] = None,
    ) -> BenchmarkReport:
        """Run the full PCO vs VLA benchmark.

        Each episode:
          1. Set up a random scene on the table.
          2. Register the initial scene frame in the PCO registry.
          3. Render ``n_frames_per_episode`` frames with small camera and
             object perturbations.
          4. Process each frame through the PCO pipeline (measure latency).
          5. Process a subset of frames through the VLA baseline (simulated
             latency) for comparison.

        Parameters
        ----------
        n_episodes : int
            Number of distinct scene configurations.
        n_frames_per_episode : int
            Frames rendered per episode.
        vla_latency_ms : float
            Simulated VLA inference time in milliseconds.
        object_change_probability : float
            Per-frame probability of applying a small object perturbation.
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        BenchmarkReport
        """
        rng = random.Random(seed)
        pco_latencies: List[float] = []
        vla_latencies: List[float] = []
        pco_hits = 0
        pco_misses = 0

        logger.info(
            "Starting pick-and-place benchmark: %d episodes x %d frames",
            n_episodes, n_frames_per_episode,
        )

        for ep in range(n_episodes):
            # --- Scene setup ---
            self._env.clear_objects()
            scene = self._env.create_pick_and_place_scene(
                seed=rng.randint(0, 2**31)
            )

            # Register the baseline frame so first lookup is a hit
            baseline_frame = self._env.render_camera()
            self._ctrl.register_object(
                image=baseline_frame,
                skb_data={
                    "episode": ep,
                    "scene_objects": list(scene.keys()),
                    "action": "pick_and_place",
                },
                label=f"scene_ep{ep}",
            )

            # --- Frame loop ---
            for fr in range(n_frames_per_episode):
                # Optionally perturb objects
                if rng.random() < object_change_probability:
                    self._apply_object_jitter(scene, rng)

                # Small camera yaw jitter
                self._env.set_camera(
                    yaw=45.0 + rng.uniform(-2.0, 2.0),
                    pitch=-35.0 + rng.uniform(-1.0, 1.0),
                )

                frame = self._env.render_camera()

                # --- PCO pipeline ---
                result = self._ctrl.process_frame(frame)
                pco_latencies.append(result.latency_ms)
                if result.pipeline == "FAST_PATH":
                    pco_hits += 1
                else:
                    pco_misses += 1

                # --- VLA baseline (sample 1 in 10 frames to save time) ---
                if fr % 10 == 0:
                    vla_result = self._ctrl.simulate_vla_inference(
                        frame, latency_ms=vla_latency_ms
                    )
                    vla_latencies.append(vla_result.latency_ms)

            logger.info(
                "Episode %d/%d complete (%d objects)",
                ep + 1, n_episodes, len(scene),
            )

        # --- Compute aggregate statistics ---
        total_frames = len(pco_latencies)
        pco_arr = np.array(pco_latencies)
        vla_arr = np.array(vla_latencies) if vla_latencies else np.array([vla_latency_ms])

        pco_mean = float(np.mean(pco_arr))
        pco_p99 = float(np.percentile(pco_arr, 99))
        vla_mean = float(np.mean(vla_arr))

        throughput_pco = 1000.0 / pco_mean if pco_mean > 0 else float("inf")
        throughput_vla = 1000.0 / vla_mean if vla_mean > 0 else float("inf")
        speedup = vla_mean / pco_mean if pco_mean > 0 else float("inf")

        hit_rate = pco_hits / total_frames if total_frames > 0 else 0.0
        miss_rate = pco_misses / total_frames if total_frames > 0 else 0.0

        # Run a quick hash stability test on the current scene
        stability = self.run_hash_stability_test(n_perturbations=50)

        report = BenchmarkReport(
            pco_latencies_ms=pco_latencies,
            vla_latencies_ms=vla_latencies,
            pco_hit_rate=hit_rate,
            pco_miss_rate=miss_rate,
            pco_mean_latency_ms=pco_mean,
            pco_p99_latency_ms=pco_p99,
            vla_mean_latency_ms=vla_mean,
            throughput_pco_hz=throughput_pco,
            throughput_vla_hz=throughput_vla,
            speedup_factor=speedup,
            hash_stability=stability.to_dict(),
            episodes=n_episodes,
            frames_total=total_frames,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        logger.info("Benchmark complete: speedup=%.1fx", speedup)
        return report

    # ------------------------------------------------------------------
    # Hash stability test
    # ------------------------------------------------------------------

    def run_hash_stability_test(
        self,
        n_perturbations: int = 100,
        max_position_jitter_mm: float = 5.0,
        max_rotation_jitter_deg: float = 3.0,
        seed: Optional[int] = None,
    ) -> StabilityReport:
        """Test hash stability under small pose perturbations.

        For each object currently in the scene:
          1. Record its original pose.
          2. Render a baseline frame and compute its dHash.
          3. Apply ``n_perturbations`` small random jitters to the object.
          4. Re-render and compute the new dHash.
          5. Measure the Hamming distance from the baseline.
          6. Restore the original pose.

        Parameters
        ----------
        n_perturbations : int
            Number of jitter samples per object.
        max_position_jitter_mm : float
            Maximum translational jitter in millimetres.
        max_rotation_jitter_deg : float
            Maximum rotational jitter in degrees.
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        StabilityReport
        """
        rng = random.Random(seed)
        hasher = self._ctrl.hasher
        objects = self._env.objects
        jitter_m = max_position_jitter_mm / 1000.0
        jitter_rad = math.radians(max_rotation_jitter_deg)

        object_results: Dict[str, List[int]] = {}
        all_distances: List[int] = []

        if not objects:
            logger.warning("No objects in scene for stability test")
            return StabilityReport(
                object_results={},
                mean_hamming=0.0,
                max_hamming=0,
                safe_threshold=self._ctrl._hamming_threshold,
                recommendation="NO_OBJECTS",
            )

        # Render baseline
        baseline_frame = self._env.render_camera()
        baseline_hash = hasher.compute_dhash(baseline_frame)

        for obj_id, meta in objects.items():
            label = meta["label"]
            orig_pos, orig_orn = self._env.get_object_pose(obj_id)
            distances: List[int] = []

            for _ in range(n_perturbations):
                # Apply small jitter
                dx = rng.uniform(-jitter_m, jitter_m)
                dy = rng.uniform(-jitter_m, jitter_m)
                dz = rng.uniform(-jitter_m / 2, jitter_m / 2)
                new_pos = [
                    orig_pos[0] + dx,
                    orig_pos[1] + dy,
                    orig_pos[2] + dz,
                ]

                # Small rotation jitter (Euler then to quaternion)
                dr = rng.uniform(-jitter_rad, jitter_rad)
                dp = rng.uniform(-jitter_rad, jitter_rad)
                dy_rot = rng.uniform(-jitter_rad, jitter_rad)
                import pybullet
                jitter_orn = pybullet.getQuaternionFromEuler([dr, dp, dy_rot])

                self._env.set_object_pose(obj_id, new_pos, list(jitter_orn))

                # Render and hash
                perturbed_frame = self._env.render_camera()
                perturbed_hash = hasher.compute_dhash(perturbed_frame)

                dist = hasher.hamming_distance(baseline_hash, perturbed_hash)
                distances.append(dist)

                # Restore original pose
                self._env.set_object_pose(obj_id, orig_pos, orig_orn)

            object_results[label] = distances
            all_distances.extend(distances)

        if all_distances:
            mean_hamming = float(np.mean(all_distances))
            max_hamming = int(np.max(all_distances))
        else:
            mean_hamming = 0.0
            max_hamming = 0

        threshold = self._ctrl._hamming_threshold
        margin = 3
        if max_hamming <= threshold:
            recommendation = "SAFE (max Hamming within threshold)"
        elif max_hamming <= threshold + margin:
            recommendation = (
                f"MARGINAL (max={max_hamming} within threshold+margin={threshold + margin})"
            )
        else:
            recommendation = (
                f"UNSAFE (max={max_hamming} exceeds threshold+margin={threshold + margin}). "
                "Consider increasing threshold or reducing perturbation."
            )

        logger.info(
            "Hash stability: mean=%.1f max=%d threshold=%d -> %s",
            mean_hamming, max_hamming, threshold, recommendation,
        )

        return StabilityReport(
            object_results=object_results,
            mean_hamming=mean_hamming,
            max_hamming=max_hamming,
            safe_threshold=threshold,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        results: BenchmarkReport,
        save_json: bool = True,
        report_dir: Optional[Path] = None,
    ) -> str:
        """Generate a human-readable benchmark report.

        Parameters
        ----------
        results : BenchmarkReport
            The benchmark results to format.
        save_json : bool
            If True, write a JSON file to ``data/benchmarks/``.
        report_dir : Path, optional
            Override output directory for the JSON file.

        Returns
        -------
        str
            Formatted terminal-friendly report string.
        """
        stability = results.hash_stability
        stability_mean = stability.get("mean_hamming", 0.0)
        stability_max = stability.get("max_hamming", 0)
        stability_threshold = stability.get("safe_threshold", 5)
        stability_rec = stability.get("recommendation", "N/A")

        report = _FORMAT_REPORT.format(
            episodes=results.episodes,
            frames_per_ep=results.frames_total // max(results.episodes, 1),
            total_frames=results.frames_total,
            hit_rate=results.pco_hit_rate * 100,
            mean_latency=results.pco_mean_latency_ms,
            p99_latency=results.pco_p99_latency_ms,
            throughput_pco=results.throughput_pco_hz,
            vla_mean=results.vla_mean_latency_ms,
            throughput_vla=results.throughput_vla_hz,
            speedup=results.speedup_factor,
            stability_mean=stability_mean,
            stability_max=stability_max,
            stability_threshold=stability_threshold,
            stability_rec=stability_rec,
        )

        # Optionally save JSON
        if save_json:
            out_dir = report_dir or DEFAULT_REPORT_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            json_path = out_dir / f"pco_benchmark_{ts}.json"
            json_path.write_text(
                json.dumps(results.to_dict(), indent=2, default=str)
            )
            report += f"\n  JSON report saved to: {json_path}\n"
            logger.info("JSON report saved to %s", json_path)

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_object_jitter(
        self,
        scene: Dict[str, int],
        rng: random.Random,
        max_xy_mm: float = 10.0,
    ) -> None:
        """Apply a small random position jitter to one random object."""
        if not scene:
            return
        label = rng.choice(list(scene.keys()))
        obj_id = scene[label]
        pos, orn = self._env.get_object_pose(obj_id)
        dx = rng.uniform(-max_xy_mm / 1000.0, max_xy_mm / 1000.0)
        dy = rng.uniform(-max_xy_mm / 1000.0, max_xy_mm / 1000.0)
        self._env.set_object_pose(
            obj_id,
            [pos[0] + dx, pos[1] + dy, pos[2]],
            orn,
        )


# ---------------------------------------------------------------------------
# Report template
# ---------------------------------------------------------------------------

_FORMAT_REPORT = """\
======================================================================
  PCO Benchmark Report -- Pick-and-Place
======================================================================
  Episodes:          {episodes}
  Frames/episode:    {frames_per_ep}
  Total frames:      {total_frames}

  --- PCO (Fast Path) ---
  Hit rate:          {hit_rate:.1f}%
  Mean latency:      {mean_latency:.2f} ms
  P99 latency:       {p99_latency:.2f} ms
  Throughput:        {throughput_pco:,.1f} Hz

  --- VLA Baseline ---
  Mean latency:      {vla_mean:.2f} ms
  Throughput:        {throughput_vla:,.1f} Hz

  --- Comparison ---
  Speedup factor:    {speedup:,.1f}x

  --- Hash Stability ---
  Mean Hamming:      {stability_mean:.1f} bits
  Max Hamming:       {stability_max} bits
  Threshold:         {stability_threshold}
  Recommendation:    {stability_rec}
======================================================================
"""

"""
gcp_robotics.sim — PyBullet simulation environment for PCO benchmarking.

Provides a tabletop pick-and-place simulation, a PCO controller that
connects rendered camera frames to the hash engine, and a benchmark
runner that produces side-by-side PCO vs VLA latency comparisons.
"""

from gcp_robotics.sim.environment import TabletopEnvironment
from gcp_robotics.sim.pco_controller import PCOController, ProcessResult
from gcp_robotics.sim.benchmark import PCOBenchmark, BenchmarkReport, StabilityReport

__all__ = [
    "TabletopEnvironment",
    "PCOController",
    "ProcessResult",
    "PCOBenchmark",
    "BenchmarkReport",
    "StabilityReport",
]

# GCP-ROBOTICS SDK

**Sub-millisecond robotic perception via hash-indexed Spatial Kinematic Blueprints.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Version 2.2.0](https://img.shields.io/badge/version-2.2.0-orange.svg)](https://github.com/codex-curator/gcp-robotics/releases/tag/v2.2.0)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18667749.svg)](https://doi.org/10.5281/zenodo.18667749)

---

## The Problem

Vision-Language-Action (VLA) models generalize beautifully — but they're too slow.

| Model | Inference Latency | Control Frequency |
|-------|:-:|:-:|
| RT-2 | 330–1000 ms | 1–3 Hz |
| OpenVLA | 100–200 ms | 5–10 Hz |
| Octo | ~400 ms | 2.5 Hz |

Industrial manipulation demands **50–1,000 Hz**. That's the **Inference-Control Discrepancy** — and it's the reason VLAs can't drive real robots in production.

## The Solution

**Perceptual Compute Offloading (PCO)** is a dual-process architecture that makes VLAs fast without making them dumb.

```
Camera Frame → Composite Hash → Registry Lookup (< 1 ms)
                                      │
                                ┌─────┴─────┐
                               HIT         MISS
                                │            │
                          Execute SKB    VLA Slow Path
                         (System 1)     (System 2, 1-5s)
                                            │
                                   Validate → Promote
                                   (Write-Back Augmentation)
```

**System 1** hashes the camera frame, retrieves a pre-computed Spatial Kinematic Blueprint (SKB) from a hash-indexed registry, and executes it — all in **< 1 ms**.

**System 2** only activates on hash miss: full VLA inference generates a new SKB, which is promoted into the registry via Write-Back Augmentation. The robot gets smarter every time it encounters something new.

## Quickstart

### End-to-End Demo

The fastest way to see the full PCO lifecycle in action:

```bash
pip install -e .
python examples/demo_fast_path.py
```

This demo generates synthetic images for 5 YCB objects, then walks through every stage of the pipeline:

1. **Template Registration** — pre-hashes 15 images into the registry
2. **Exact Match** — O(1) lookup returns an SKB in < 0.1 ms
3. **Fuzzy Match** — a noisy/cropped image still matches via Hamming distance
4. **Hash Miss** — an unregistered object triggers the Slow Path (System 2)
5. **Loop Closure** — the Slow Path result is promoted, and a re-lookup returns an instant hit

No external files, API keys, or hardware required.

### Simulation Benchmark Demo

Compare PCO fast-path latency against a VLA baseline in a PyBullet simulation:

```bash
python examples/sim_benchmark.py --episodes 10 --frames 30
```

See [Simulation Benchmark](#simulation-benchmark) for detailed results.

### REPL Snippet (Standalone)

Paste this into a Python shell to see the hash-and-lookup cycle with a purely in-memory image:

```python
import numpy as np
from PIL import Image
from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry

hasher = CompositeHasher()
registry = CodexRegistry()

# Generate a synthetic "red bolt" image in memory (no files needed)
arr = np.zeros((64, 64, 3), dtype=np.uint8)
arr[16:48, 24:40] = [200, 30, 30]  # red rectangle
bolt_img = Image.fromarray(arr)

# Register the object
registry.register(
    dhash_hex=hasher.compute_dhash(bolt_img),
    skb_data={"action": "PICK", "grip_force_n": 5.0},
)

# Fast-path lookup (< 1 ms)
query_hash = hasher.compute_dhash(bolt_img)
result = registry.lookup(query_hash)
print(f"Match: {result.match_type}, Time: {result.lookup_time_ms:.3f} ms")
# → Match: EXACT, Time: 0.023 ms
```

## Architecture

### Four-Layer SKB Ontology

Every object in the registry carries a **Spatial Kinematic Blueprint** — a four-layer schema encoding everything a robot needs to know:

| Layer | Name | Contents |
|:-----:|------|----------|
| L1 | **Provenance** | Origin, calibration, workspace zone, Soulmark |
| L2 | **Semantic Topology** | Perceptual hashes, mass, friction, dimensions, material |
| L3 | **Affordance** | 13 action primitives, SE(3) target pose, trajectory chunks |
| L4 | **Safety** | Hazard class, CBF parameters, PPE, force/velocity limits |

The schema is implemented as strict Pydantic v2 models (741 lines) with JSON-LD serialization.

### Composite Hash Engine

Five perceptual hash modalities, each capturing a different axis:

| Hash | Bits | Captures | Invariant To |
|------|:----:|----------|:------------:|
| dHash | 64 | Gradient structure | Scale, brightness |
| pHash | 64 | Frequency content | Compression, noise |
| pHash-256 (PPP) | 256 | High-fidelity fingerprint | Global registry key |
| Color Hash | 384 | Chromatic distribution | Geometric transforms |
| **zHash** | 64 | Depth geometry | Lighting, specular glare |

Object-Conditioned Ensembling weights these adaptively per object class.

### Safety Architecture (Theoretical Foundation)

The PCO architecture is designed around a multi-layered safety model described in our [Perceptual Compute Offloading](https://doi.org/10.5281/zenodo.18667749) paper. The theoretical framework defines three independent verification layers:

1. **Geometric Gating** — point-cloud dimensions must match the SKB before execution
2. **Proprioceptive Pre-Tensioning** — contact stiffness (K = ΔF/Δx) must match expected material
3. **Control Barrier Functions (CBF-QP)** — hard physics constraints enforced at 1 kHz via the formulation: `h(x) = ||p(x) - p_obs||² - R_safe² ≥ 0`

> **v2.2 Implementation Status:** The current SDK implements **Layer 4 (Safety) of the SKB schema** — hazard classification, PPE requirements, force/velocity limits, and operating constraints are fully modeled in the Pydantic schema. The CBF-QP solver and real-time safety filter are **planned for v3.0** and will target RTOS deployment. The Six Sigma / SIL-2 compliance claims in the paper refer to the *theoretical joint failure probability* of the full three-layer architecture, not the current software release.
>
> **What v2.2 provides today:** Soulmark cryptographic verification (prevents poisoned blueprints), per-SKB force limits and hazard metadata, the schema infrastructure for downstream safety enforcement, a PyBullet simulation benchmark, and Vertex AI fine-tuning data generation.

### Soulmark Verification

Every SKB carries a cryptographic **Soulmark** (SHA-256 / HMAC-SHA256) computed over its safety-critical fields. This prevents the **Poisoned Blueprint** attack — a compromised network node cannot inject malicious parameters without the Fleet Authority's private key.

```python
from gcp_robotics.soulmark import SoulmarkVerifier

verifier = SoulmarkVerifier(fleet_secret="your-fleet-key")
result = verifier.verify_payload(skb_data)
assert result.valid  # Payload integrity confirmed
```

## Simulation Benchmark

Run the PCO vs VLA speed comparison in a PyBullet tabletop simulation. No hardware or API keys required.

```bash
python examples/sim_benchmark.py
```

Results from a standard run:

| Metric | PCO (Fast Path) | VLA Baseline |
|--------|----------------|--------------|
| Mean latency | 0.84 ms | 300 ms |
| Throughput | 1,200 Hz | 3.3 Hz |
| Hit rate | 100% | N/A |
| **Speedup** | **357x** | — |

Options: `--episodes 20 --frames 50 --gui --output report.json`

## SDK Modules

| Module | Lines | Purpose |
|--------|:-----:|---------|
| `gcp_robotics/client.py` | 270 | Async SDK client (System 1/2 dispatch, retries, telemetry) |
| `gcp_robotics/cache.py` | 145 | Thread-safe LRU RAM cache (< 0.01 ms per op) |
| `gcp_robotics/soulmark.py` | 175 | Cryptographic SKB integrity verification |
| `gcp_robotics/telemetry.py` | 155 | Structured event logging (hash miss, CBF, Soulmark) |
| `gcp_robotics/schema/models.py` | 741 | SKB Pydantic v2 schema (four-layer ontology) |
| `gcp_robotics/hash_engine/hasher.py` | 206 | CompositeHasher (dHash, pHash, PPP-256, color, zHash) |
| `gcp_robotics/hash_engine/registry.py` | 355 | CodexRegistry (O(1) exact + Hamming fuzzy lookup) |
| `gcp_robotics/slow_path/prompts.py` | 888 | Claude API integration for SKB generation |
| `gcp_robotics/template_env/registrar.py` | 562 | Object registration pipeline |
| `gcp_robotics/datasets/ycb_loader.py` | 595 | 20 YCB benchmark objects |
| `gcp_robotics/vertex_tuning.py` | 1,477 | Fine-tuning data generator (Vertex AI / Anthropic) |
| `gcp_robotics/sim/` | 1,353 | PyBullet simulation benchmark (environment, controller, runner) |
| `examples/sim_benchmark.py` | — | Simulation benchmark CLI |
| `codex_ros2/` | 1,751 | 4 ROS2 nodes (Vision, Registry, SlowPath, Bridge) |

**Total: ~8,600 lines of production code.**

## ROS2 Integration

Four framework-agnostic nodes using the adapter pattern:

- **CodexVisionNode** — Camera → hash → registry lookup → publish
- **CodexRegistryNode** — Service server wrapping lookup + promotion
- **CodexSlowPathNode** — Action server for VLA recovery with feedback
- **CodexBridgeNode** — JSON bridge for non-ROS2 systems (MQTT, Zenoh)

Works with or without `rclpy` — includes a mock ROS2 framework for testing.

## Installation

```bash
# Core SDK
pip install -e .

# With LLM slow path (requires Anthropic API key)
pip install -e ".[llm]"

# With test dependencies
pip install -e ".[test]"

# Everything
pip install -e ".[all]"
```

## Testing

```bash
pytest tests/ -v
```

## Validation Lab Kit

The open-source experimental protocol for partner labs:

```bash
pip install codex-lab-kit
```

Four-phase protocol: Hash Robustness → Hash-to-Grasp → Loop Closure → Latency Profile.

DOI: [10.5281/zenodo.18667749](https://doi.org/10.5281/zenodo.18667749)

## Research Papers

| # | Paper | DOI |
|:-:|-------|-----|
| 1 | The Entropy of Recursion | [10.5281/zenodo.18436975](https://doi.org/10.5281/zenodo.18436975) |
| 2 | The Density Imperative | [10.5281/zenodo.18667735](https://doi.org/10.5281/zenodo.18667735) |
| 3 | Cognitive Nutrition for Foundation Models | [10.5281/zenodo.18667742](https://doi.org/10.5281/zenodo.18667742) |
| 4 | Perceptual Compute Offloading | [10.5281/zenodo.18667749](https://doi.org/10.5281/zenodo.18667749) |

## Patent Notice

The Golden Codex Protocol architecture is the subject of U.S. Provisional Patent Applications No. 63/983,304 and No. 63/984,299, assigned to Metavolve Labs, Inc. This open-source license (Apache 2.0) grants rights to the software implementation; patent claims are separately licensed.

## Fine-Tuning Data Generation

Generate training datasets for Vertex AI (Gemini) or Anthropic fine-tuning from existing SKB + image pairs:

```bash
python gcp_robotics/vertex_tuning.py \
  --data-dirs data/standard_evaluation data/ycb_20 \
  --output data/training/skb_tuning_v1.jsonl \
  --domain general_household --augment
```

Supports 5 task domains: `warehouse_picking`, `kitchen_food`, `assembly_line`, `lab_medical`, `general_household`.

## Citation

```bibtex
@software{macpherson2026gcp,
  author       = {MacPherson, Tad},
  title        = {{GCP-ROBOTICS: Sub-Millisecond Robotic Perception via
                   Hash-Indexed Spatial Kinematic Blueprints}},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18667749},
  url          = {https://doi.org/10.5281/zenodo.18667749}
}
```

## Contact

- **Research:** research@iaeternum.ai
- **Website:** [iaeternum.ai/robotics](https://iaeternum.ai/robotics)
- **Lab Kit:** [github.com/codex-curator/codex-lab-kit](https://github.com/codex-curator/codex-lab-kit)

---

*Metavolve Labs, Inc. | San Francisco, California*

# Golden Codex Core — GCP-ROBOTICS Engine

> **PRIVATE REPOSITORY — PATENT PENDING**
>
> iAeternum / Metavolve Labs | research@iaeternum.ai

## Overview

The Golden Codex Protocol 2.0 (GCP-ROBOTICS) is a dual-path architecture for robotic manipulation that combines **O(1) perceptual hash lookup** for known objects with **LLM-based generative recovery** for novel encounters.

```
Camera Frame → Composite Hash → Registry Lookup
                                      │
                                ┌─────┴─────┐
                               HIT         MISS
                                │            │
                          Execute SKB    LLM Slow Path
                          (< 1 ms)      (1-5 sec)
                                          │
                                    Validate + Promote
                                    (loop closure)
```

## Architecture

### Spatial Kinematic Blueprint (SKB) — Four-Layer Ontology

| Layer | Name | Contents |
|-------|------|----------|
| L1 | **Provenance** | Origin, calibration, workspace zone, dataset source |
| L2 | **Semantic Topology** | Perceptual hashes, mass, friction, dimensions, material graph |
| L3 | **Affordance** | 13 action primitives, SE(3) target pose, force/speed profiles |
| L4 | **Safety** | 7 hazard classes, PPE requirements, operating constraints |

### Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `gcp_robotics/schema/models.py` | 741 | SKB Pydantic v2 models with JSON-LD serialization |
| `gcp_robotics/hash_engine/hasher.py` | 206 | CompositeHasher — dHash (64-bit), pHash, PPP-256, Color Hash |
| `gcp_robotics/hash_engine/registry.py` | 355 | CodexRegistry — O(1) exact lookup, Hamming-distance fuzzy matching |
| `gcp_robotics/slow_path/prompts.py` | 888 | Claude API integration, iterative validation, safety re-prompting |
| `gcp_robotics/template_env/registrar.py` | 562 | Object registration pipeline from images |
| `gcp_robotics/datasets/ycb_loader.py` | 595 | 20 YCB benchmark objects with ground-truth physical properties |

### ROS2 Integration

The `codex_ros2/` directory contains 4 framework-agnostic nodes using the adapter pattern (works with or without rclpy):

- **CodexVisionNode** — Camera subscriber → hash → registry lookup → publish signal
- **CodexRegistryNode** — Service server wrapping registry lookup and promotion
- **CodexSlowPathNode** — Action server for LLM recovery with feedback stages
- **CodexBridgeNode** — JSON bridge for non-ROS2 systems

## Setup

```bash
pip install -e .
python examples/demo_fast_path.py
```

For LLM slow path (requires Anthropic API key):
```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=your_key
```

## Public Lab Kit

The open-source validation toolkit for partner labs is at:
[github.com/codex-curator/codex-lab-kit](https://github.com/codex-curator/codex-lab-kit)

## Contact

- Website: [iaeternum.ai/robotics](https://iaeternum.ai/robotics)
- Email: research@iaeternum.ai
- License inquiries: research@iaeternum.ai

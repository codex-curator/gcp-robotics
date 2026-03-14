# CLAUDE.md -- GCP-Robotics SDK

**What this is**: The production GCP-Robotics SDK (v2.2.0). Sub-millisecond robotic perception via hash-indexed Spatial Kinematic Blueprints. Pushed to `codex-curator/gcp-robotics` on GitHub. DOI: 10.5281/zenodo.18668113.

**Local path**: `/mnt/d/NeuralNet/golden-codex-core/`

---

## Workspace Map

| Directory | What It Is | Local Path |
|-----------|-----------|------------|
| **This repo** (SDK) | Core GCP-Robotics SDK | `/mnt/d/NeuralNet/golden-codex-core/` |
| **Codex Lab Kit** | Validation toolkit (separate repo, pip-installable) | `/mnt/d/NeuralNet/codex-lab-kit/` |
| **Robotics Lab** | Research strategy, LAIR, open questions | `/mnt/d/NeuralNet/LABS/robotics-research/` |
| **golden-codex-robotics** | STALE dev copy -- missing client, cache, soulmark, telemetry, api. Do not use. | `/mnt/d/NeuralNet/golden-codex-robotics/` |

---

## SDK Structure

```
gcp_robotics/
  api.py                  FastAPI SKB API server
  client.py               Async SDK client (System 1/2 dispatch)
  cache.py                Thread-safe LRU RAM cache (< 0.01 ms)
  soulmark.py             SHA-256 / HMAC-SHA256 integrity verification
  telemetry.py            Structured event logging
  schema/models.py        Four-layer SKB Pydantic v2 schema (741 lines)
  hash_engine/hasher.py   CompositeHasher (dHash, pHash, PPP-256, color, zHash)
  hash_engine/registry.py CodexRegistry -- O(1) exact + Hamming fuzzy lookup
  slow_path/prompts.py    Claude API integration for SKB generation
  template_env/registrar.py  Object registration pipeline
  datasets/ycb_loader.py  20 YCB benchmark objects
  vertex_tuning.py        Vertex AI fine-tuning data generator (1,477 lines)
  sim/
    __init__.py            Module init
    environment.py         PyBullet tabletop with camera rendering (540 lines)
    pco_controller.py      System 1/2 pipeline for sim (323 lines)
    benchmark.py           Configurable PCO vs VLA benchmark runner (490 lines)

codex_ros2/               ROS2 nodes (4 nodes, mock framework included)
  codex_nodes/            Vision, Registry, SlowPath, Bridge nodes
  launch/                 ROS2 launch file
  examples/               sim_pipeline.py

examples/
  demo_fast_path.py       Full PCO lifecycle demo (no hardware needed)
  fusion_poc.py           YCB -> SKB -> Registry -> Lab Kit
  batch_registry_builder.py  6-stage batch pipeline (organize/hash/enrich/sign/register/validate)
  sim_benchmark.py        Researcher-facing PCO vs VLA sim benchmark CLI (617 lines)
  quickstart.py           Minimal usage example

data/
  standard_evaluation/    5 geometric primitives (cube, sphere, cylinder, pyramid, L-bracket)
  ycb_20/                 20 YCB benchmark objects with generated images
  ycb_assets/             Fusion registry, SKBs, scene manifest, experiment protocol

tests/
  test_models.py          SKB schema tests
  test_cache.py           LRU cache tests
  test_soulmark.py        Cryptographic verification tests
  test_batch_pipeline.py  Batch pipeline integration tests
```

---

## Key Commands

```bash
# Install (editable, all extras)
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Run the end-to-end demo (no hardware, no API keys)
python examples/demo_fast_path.py

# Run the fusion proof-of-concept (YCB objects)
python examples/fusion_poc.py

# Build a full registry from scratch (6-stage batch pipeline)
python examples/batch_registry_builder.py --template standard_evaluation

# Start the FastAPI SKB API
uvicorn gcp_robotics.api:app --reload

# Run the PCO vs VLA simulation benchmark (no hardware needed)
python examples/sim_benchmark.py
python examples/sim_benchmark.py --episodes 20 --frames 50 --gui

# Generate Vertex AI fine-tuning data from SKBs
python gcp_robotics/vertex_tuning.py --data-dirs data/standard_evaluation data/ycb_20 --output data/training/skb_tuning_v1.jsonl --domain general_household --augment

# Run codex-lab-kit tests (separate repo)
cd /mnt/d/NeuralNet/codex-lab-kit && python3 -m pytest tests/ -v
```

---

## Architecture (Quick Reference)

**Dual-Process (System 1 / System 2)**:
- System 1 (Fast Path): Camera -> dHash -> Registry lookup -> Execute SKB. Under 1 ms.
- System 2 (Slow Path): Hash miss -> Claude generates SKB -> validate -> promote to registry.

**Batch Pipeline** (`examples/batch_registry_builder.py`):
1. ORGANIZE -- collect object images
2. HASH -- compute composite hashes
3. ENRICH -- generate SKBs via Claude
4. SIGN -- Soulmark + verify
5. REGISTER -- populate CodexRegistry
6. VALIDATE -- run codex-lab-kit 4-phase protocol

**SKB Four Layers**: Provenance (L1) -> Semantic Topology (L2) -> Affordance (L3) -> Safety (L4)

---

## Related Repos & Workspaces

| Resource | Path | Notes |
|----------|------|-------|
| **Codex Lab Kit** | `/mnt/d/NeuralNet/codex-lab-kit/` | GitHub: `codex-curator/codex-lab-kit`, DOI: 10.5281/zenodo.18668110 |
| **Robotics Research LAIR** | `/mnt/d/NeuralNet/LABS/robotics-research/ROBOT_LAB_START_HERE.md` | Full research strategy + quick start |
| **Artiswa Pipeline (art)** | `/mnt/d/NeuralNet/STUDIOS/Artiswa/pipeline/` | 6-stage batch processor — same architecture adapted for art enrichment |
| **Artiswa LAIR** | `/mnt/d/NeuralNet/STUDIOS/Artiswa/LAIR.md` | Art pipeline context + commands |
| **NeuralNet root** | `/mnt/d/NeuralNet/START_HERE.md` | D: drive session map (AI/ML workspace) |
| **Business root** | `/mnt/c/Users/atmta/source/repos/golden_codex_pipeline/lairs/START_HERE.md` | C: drive business/product lairs |
| **LABS directory** | `/mnt/d/NeuralNet/LABS/` | AI training experiments (CN, SPARK, ai-partner, mamba) |

## Warning: Stale Copy

`/mnt/d/NeuralNet/golden-codex-robotics/` is an older dev snapshot. It is missing `client.py`, `cache.py`, `soulmark.py`, `telemetry.py`, and `api.py`. It has `codex_lab_kit` and `codex_ros2` bundled inline (now split into separate repos). **Do not develop there.** Use this directory (`golden-codex-core`) as the canonical SDK.

---

## Published DOIs & References

### All Published DOIs

| # | Item | DOI |
|:-:|------|-----|
| 1 | Alexandria Aeternum (dataset) | [10.5281/zenodo.18359131](https://doi.org/10.5281/zenodo.18359131) |
| 2 | Entropy of Recursion (paper 1) | [10.5281/zenodo.18436975](https://doi.org/10.5281/zenodo.18436975) |
| 3 | The Density Imperative (paper 2) | [10.5281/zenodo.18667735](https://doi.org/10.5281/zenodo.18667735) |
| 4 | Cognitive Nutrition (paper 3) | [10.5281/zenodo.18667742](https://doi.org/10.5281/zenodo.18667742) |
| 5 | Perceptual Compute Offloading (paper 4) | [10.5281/zenodo.18667749](https://doi.org/10.5281/zenodo.18667749) |
| 6 | codex-lab-kit v1.0.1 (software) | [10.5281/zenodo.18668110](https://doi.org/10.5281/zenodo.18668110) |
| 7 | gcp-robotics v2.0.1 (software) | [10.5281/zenodo.18668113](https://doi.org/10.5281/zenodo.18668113) |
| 8 | gcp-robotics v2.1.0 (software) | [10.5281/zenodo.18669470](https://doi.org/10.5281/zenodo.18669470) |

### Patents

- U.S. Provisional Patent Application No. 63/983,304
- U.S. Provisional Patent Application No. 63/984,299
- U.S. Provisional Patent Application No. 63/985,213 (CIP — Multi-Modal Provenance, Domain-Conditioned Fine-Tuning, Hierarchical Verification, Cross-Modal Reflexive Hashing; filed 02/18/2026)
- Assigned to: Metavolve Labs, Inc.

### GitHub Repositories

- [codex-curator/gcp-robotics](https://github.com/codex-curator/gcp-robotics)
- [codex-curator/codex-lab-kit](https://github.com/codex-curator/codex-lab-kit)

### HuggingFace

- [Metavolve-Labs/alexandria-aeternum-genesis](https://huggingface.co/datasets/Metavolve-Labs/alexandria-aeternum-genesis)

### Websites

- https://iaeternum.ai
- https://iaeternum.ai/robotics

### Paper Source Locations

| Paper | Local Path |
|-------|------------|
| Entropy of Recursion | `/mnt/c/TadFile2025/Metavolve Labs/PAPERS/Entropy of Recursion/Entropy of Recursion.tex` |
| The Density Imperative | `/mnt/c/TadFile2025/Metavolve Labs/PAPERS/The Density Imperative/the_density_imperative.tex` |
| Cognitive Nutrition | `/mnt/c/TadFile2025/Metavolve Labs/PAPERS/Cognitive Nutrition/cognitive_nutrition_architecture.tex` |
| Perceptual Compute Offloading | `/mnt/c/TadFile2025/Metavolve Labs/PAPERS/Perceptual Compute Offloading/perceptual_compute_offloading.tex` |

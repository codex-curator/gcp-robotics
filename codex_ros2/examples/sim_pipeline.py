#!/usr/bin/env python3
"""
Golden Codex Protocol 2.0 -- ROS2 Simulated Pipeline Demo
==========================================================

Replicates the full Fast Path lifecycle from demo_fast_path.py,
but routes everything through the mock ROS2 pub/sub, service,
and action layer.

Pipeline:  camera image → CodexVisionNode → [HIT: GoldenCodexSignal]
                                            [MISS: SlowPathTrigger → CodexSlowPathNode
                                                   → auto-promote → SlowPathResult]

Run:
    python codex_ros2/examples/sim_pipeline.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, List

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, "/tmp/gcp-robotics-venv/lib/python3.10/site-packages")

# ---------------------------------------------------------------------------
# Imports — existing library
# ---------------------------------------------------------------------------
from gcp_robotics.hash_engine.hasher import CompositeHasher
from gcp_robotics.hash_engine.registry import CodexRegistry
from gcp_robotics.datasets.ycb_loader import YCBLoader
from gcp_robotics.template_env.registrar import TemplateEnvironmentRegistrar

# ---------------------------------------------------------------------------
# Imports — mock ROS2 layer + nodes
# ---------------------------------------------------------------------------
from codex_ros2.codex_mock_ros import MockMessageBus, MockServiceRegistry, MockNode
from codex_ros2.codex_nodes.codex_nodes.codex_vision_node import CodexVisionNode
from codex_ros2.codex_nodes.codex_nodes.codex_registry_node import CodexRegistryNode
from codex_ros2.codex_nodes.codex_nodes.codex_slow_path_node import CodexSlowPathNode
from codex_ros2.codex_nodes.codex_nodes.codex_bridge_node import CodexBridgeNode
from codex_ros2.codex_nodes.codex_nodes.qos_profiles import (
    TOPIC_CODEX_SIGNAL,
    TOPIC_SLOW_PATH_RESULT,
    TOPIC_STATS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sim_pipeline")


# ---------------------------------------------------------------------------
# Simulated camera message
# ---------------------------------------------------------------------------
@dataclass
class SimImageMsg:
    """Simulates a sensor_msgs/Image with path to source image."""
    image_path: str
    frame_id: str = "camera_link"
    pil_image: Any = None
    previous_action: str = ""
    available_objects: list = None

    def __post_init__(self):
        if self.available_objects is None:
            self.available_objects = []


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def print_banner(text: str, char: str = "=", width: int = 72) -> None:
    print()
    print(char * width)
    print(text.center(width))
    print(char * width)
    print()


def print_section(title: str) -> None:
    print()
    print(f"--- {title} " + "-" * max(0, 68 - len(title) - 5))
    print()


def print_stats(registry: CodexRegistry) -> None:
    stats = registry.stats()
    print(f"  Total entries        : {stats['total_entries']}")
    print(f"  Total lookups        : {stats['total_lookups']}")
    print(f"  Exact hits           : {stats['exact_hits']}")
    print(f"  Fuzzy hits           : {stats['fuzzy_hits']}")
    print(f"  Misses               : {stats['misses']}")
    print(f"  Promotions           : {stats['promotion_count']}")
    print(f"  Secondary (pHash)    : {stats['secondary_phash_entries']}")
    print(f"  Secondary (PPP)      : {stats['secondary_ppp_entries']}")


# ---------------------------------------------------------------------------
# Collectors — capture messages flowing through the pipeline
# ---------------------------------------------------------------------------
class MessageCollector:
    """Subscribes to topics and collects messages for verification."""

    def __init__(self, bus: MockMessageBus) -> None:
        self.signals: List[Any] = []
        self.slow_path_results: List[Any] = []
        self.stats_msgs: List[Any] = []

        bus.subscribe(TOPIC_CODEX_SIGNAL, self.signals.append)
        bus.subscribe(TOPIC_SLOW_PATH_RESULT, self.slow_path_results.append)
        bus.subscribe(TOPIC_STATS, self.stats_msgs.append)


# ---------------------------------------------------------------------------
# Main pipeline demo
# ---------------------------------------------------------------------------
def main() -> None:
    print_banner("Golden Codex ROS2 Pipeline — Mock Simulation")

    # ---- 1. Initialize shared infrastructure --------------------------------
    print_section("STEP 1: Initialize Mock ROS2 Infrastructure")

    bus = MockMessageBus()
    service_registry = MockServiceRegistry()

    # Create mock nodes (shared bus + service registry)
    vision_mock = MockNode("codex_vision", bus, service_registry)
    registry_mock = MockNode("codex_registry", bus, service_registry)
    slow_path_mock = MockNode("codex_slow_path", bus, service_registry)
    bridge_mock = MockNode("codex_bridge", bus, service_registry)

    print("  MockMessageBus       : ready")
    print("  MockServiceRegistry  : ready")
    print("  Mock nodes created   : 4")

    # ---- 2. Initialize Golden Codex components ------------------------------
    print_section("STEP 2: Initialize Golden Codex Components")

    loader = YCBLoader()
    registrar = TemplateEnvironmentRegistrar(
        environment_name="ROS2 Sim Lab",
        workspace_zone="zone_A_workbench",
    )
    hasher = registrar.hasher
    registry = registrar.get_registry()

    print(f"  Environment          : {registrar.environment_name}")
    print(f"  Workspace Zone       : {registrar.workspace_zone}")

    # ---- 3. Register YCB objects (pre-populate fast path) -------------------
    print_section("STEP 3: Register 5 YCB Objects (3 images each)")

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

    print(f"\n  Total registered images: {len(registry)}")

    # ---- 4. Wire up ROS2 nodes (adapter pattern) ----------------------------
    print_section("STEP 4: Wire Up ROS2 Nodes")

    # Registry node (services + stats publisher) — must come first
    registry_node = CodexRegistryNode(
        node=registry_mock,
        registry=registry,
        stats_interval=0,  # No background timer for sim
    )

    # Slow path node (action server + trigger subscriber)
    slow_path_node = CodexSlowPathNode(
        node=slow_path_mock,
        registry=registry,
        auto_promote=True,
    )

    # Vision node (camera subscriber + signal/trigger publishers)
    vision_node = CodexVisionNode(
        node=vision_mock,
        registry=registry,
        hasher=hasher,
        workspace_zone="zone_A_workbench",
        end_effector="robotiq_2f85",
    )

    # Bridge node (topic → JSON collector)
    bridge_node = CodexBridgeNode(node=bridge_mock)

    # Collector for verification
    collector = MessageCollector(bus)

    print("  CodexRegistryNode    : services registered")
    print("  CodexSlowPathNode    : action server + trigger subscriber ready")
    print("  CodexVisionNode      : camera subscriber + publishers ready")
    print("  CodexBridgeNode      : JSON bridge ready")
    print(f"  Active topics        : {bus.get_topic_names()}")
    print(f"  Active services      : {service_registry.list_services()}")

    # ---- 5. EXACT HIT — publish registered image to camera topic ------------
    print_section("STEP 5: EXACT HIT — Publish Registered Image")

    test_image = all_image_paths["001_chips_can"][0]
    print(f"  Publishing image: {test_image}")

    from codex_ros2.codex_nodes.codex_nodes.qos_profiles import (
        TOPIC_CAMERA_IMAGE,
        SENSOR_IMAGE_QOS,
    )

    bus.publish(
        TOPIC_CAMERA_IMAGE,
        SimImageMsg(image_path=test_image),
        SENSOR_IMAGE_QOS,
    )

    # Check collector
    assert len(collector.signals) >= 1, "Expected at least 1 GoldenCodexSignal"
    signal = collector.signals[-1]
    print(f"  Signal received      : match_type={signal.match_type}")
    print(f"  dHash                : {signal.dhash_hex}")
    print(f"  Hamming distance     : {signal.hamming_distance}")
    print(f"  Lookup time          : {signal.lookup_time_ms:.3f} ms")
    print(f"  Pipeline status      : {signal.pipeline_status}")
    assert signal.match_type == "EXACT", f"Expected EXACT, got {signal.match_type}"
    print("\n  [PASS] Exact hit routed through pub/sub pipeline.")

    # ---- 6. HASH MISS → Slow Path → Auto-Promote ---------------------------
    print_section("STEP 6: HASH MISS → Slow Path Recovery → Loop Closure")

    miss_object_id = "011_banana"
    print(f"  Generating image for UNREGISTERED object: {miss_object_id}")

    miss_images = loader.generate_test_images(miss_object_id, count=1)
    miss_image_path = str(miss_images[0])
    print(f"  Test image: {miss_image_path}")

    # Track counts before
    signals_before = len(collector.signals)
    results_before = len(collector.slow_path_results)

    bus.publish(
        TOPIC_CAMERA_IMAGE,
        SimImageMsg(
            image_path=miss_image_path,
            available_objects=target_objects,
        ),
        SENSOR_IMAGE_QOS,
    )

    # The pipeline should have: vision → trigger → slow_path → result
    # Give a tiny bit of time for action server thread
    time.sleep(0.2)

    # No new signal (it was a miss, not a hit)
    new_signals = collector.signals[signals_before:]
    print(f"  New GoldenCodexSignals : {len(new_signals)} (expected 0 for MISS)")

    # Should have a slow path result
    new_results = collector.slow_path_results[results_before:]
    assert len(new_results) >= 1, "Expected at least 1 SlowPathResult"
    sp_result = new_results[-1]
    print(f"  SlowPathResult received:")
    print(f"    Trigger hash       : {sp_result.trigger_dhash_hex}")
    print(f"    Primitive          : {sp_result.primitive_id}")
    print(f"    Confidence         : {sp_result.confidence_score}")
    print(f"    Generation time    : {sp_result.generation_time_ms:.1f} ms")
    print(f"    Auto-promoted      : {sp_result.promoted}")

    assert sp_result.promoted, "Expected auto-promotion to succeed"
    print("\n  [PASS] Slow path recovery completed and auto-promoted via pub/sub.")

    # ---- 7. LOOP CLOSURE — re-publish same image, should be EXACT -----------
    print_section("STEP 7: LOOP CLOSURE — Re-publish Missed Image")

    signals_before = len(collector.signals)
    bus.publish(
        TOPIC_CAMERA_IMAGE,
        SimImageMsg(image_path=miss_image_path),
        SENSOR_IMAGE_QOS,
    )

    new_signals = collector.signals[signals_before:]
    assert len(new_signals) >= 1, "Expected GoldenCodexSignal after loop closure"
    closure_signal = new_signals[-1]
    print(f"  Signal received      : match_type={closure_signal.match_type}")
    print(f"  Hamming distance     : {closure_signal.hamming_distance}")
    print(f"  Lookup time          : {closure_signal.lookup_time_ms:.3f} ms")
    assert closure_signal.match_type == "EXACT", (
        f"Expected EXACT after promotion, got {closure_signal.match_type}"
    )
    print("\n  [PASS] Loop closure confirmed — MISS → Slow Path → EXACT hit.")

    # ---- 8. Service calls via bridge ----------------------------------------
    print_section("STEP 8: Service Calls via Bridge Node")

    # Use the hasher to get the miss hash for a direct service lookup
    miss_hash = hasher.compute_dhash(miss_image_path)

    lookup_cmd = json.dumps({
        "cmd": "lookup",
        "dhash_hex": miss_hash,
        "max_hamming_distance": 5,
    })
    resp_json = bridge_node.handle_command(lookup_cmd)
    resp = json.loads(resp_json)
    print(f"  Bridge lookup result : {resp.get('match_type', resp)}")
    assert resp.get("match_type") == "EXACT", f"Expected EXACT via bridge, got {resp}"
    print("  [PASS] Service call via bridge node works.")

    # ---- 9. Publish registry stats ------------------------------------------
    print_section("STEP 9: Registry Statistics (published via node)")

    stats_msg = registry_node.publish_stats()
    print(f"  Total entries        : {stats_msg.total_entries}")
    print(f"  Total lookups        : {stats_msg.total_lookups}")
    print(f"  Exact hits           : {stats_msg.exact_hits}")
    print(f"  Fuzzy hits           : {stats_msg.fuzzy_hits}")
    print(f"  Misses               : {stats_msg.misses}")
    print(f"  Promotions           : {stats_msg.promotion_count}")

    # ---- 10. Verify bridge collected everything -----------------------------
    print_section("STEP 10: Verify Bridge Message History")

    bridge_signals = bridge_node.get_signal_history()
    bridge_results = bridge_node.get_result_history()
    print(f"  Bridge signal history  : {len(bridge_signals)} messages")
    print(f"  Bridge result history  : {len(bridge_results)} messages")

    # ---- Final summary ------------------------------------------------------
    print_section("Final Registry Statistics")
    print_stats(registry)

    # Cleanup
    registry_node.destroy()

    print_banner("ROS2 Sim Pipeline Complete", char="*")
    print("  The full Golden Codex ROS2 pipeline has been demonstrated:")
    print("    1. Mock ROS2 infrastructure (bus, services, actions)")
    print("    2. Template environment pre-registration (5 objects, 15 images)")
    print("    3. EXACT hit routed through VisionNode → GoldenCodexSignal")
    print("    4. MISS → SlowPathTrigger → SlowPathNode → auto-promote")
    print("    5. Loop closure: re-published image yields EXACT hit")
    print("    6. Bridge node forwarding + service proxy")
    print("    7. Registry stats telemetry")
    print()


if __name__ == "__main__":
    main()

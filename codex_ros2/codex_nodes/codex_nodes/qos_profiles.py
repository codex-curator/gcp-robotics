"""
QoS profile definitions for the Golden Codex ROS2 pipeline.

These profiles are used by both real rclpy and the mock layer.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# Try to import real rclpy QoS; fall back to mock.
try:
    from rclpy.qos import (
        QoSProfile as _RclQoS,
        ReliabilityPolicy,
        DurabilityPolicy,
        HistoryPolicy,
    )

    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False


if _HAS_RCLPY:
    SENSOR_IMAGE_QOS = _RclQoS(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    SKB_RESULT_QOS = _RclQoS(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )
    SLOW_PATH_TRIGGER_QOS = _RclQoS(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )
    STATS_QOS = _RclQoS(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
else:
    # Import mock QoS types
    from codex_ros2.codex_mock_ros.mock_pubsub import (
        QoSProfile as _MockQoS,
        QoSReliability,
        QoSDurability,
    )

    SENSOR_IMAGE_QOS = _MockQoS(
        reliability=QoSReliability.BEST_EFFORT,
        durability=QoSDurability.VOLATILE,
        depth=1,
    )
    SKB_RESULT_QOS = _MockQoS(
        reliability=QoSReliability.RELIABLE,
        durability=QoSDurability.TRANSIENT_LOCAL,
        depth=10,
    )
    SLOW_PATH_TRIGGER_QOS = _MockQoS(
        reliability=QoSReliability.RELIABLE,
        durability=QoSDurability.VOLATILE,
        depth=5,
    )
    STATS_QOS = _MockQoS(
        reliability=QoSReliability.BEST_EFFORT,
        durability=QoSDurability.VOLATILE,
        depth=1,
    )

# Topic names
TOPIC_CAMERA_IMAGE = "/camera/image_raw"
TOPIC_CODEX_SIGNAL = "/codex/signal"
TOPIC_SLOW_PATH_TRIGGER = "/codex/slow_path_trigger"
TOPIC_SLOW_PATH_RESULT = "/codex/slow_path_result"
TOPIC_STATS = "/codex/stats"

# Service names
SERVICE_LOOKUP = "/codex/lookup"
SERVICE_PROMOTE = "/codex/promote"

# Action names
ACTION_SLOW_PATH_RECOVER = "/codex/slow_path_recover"

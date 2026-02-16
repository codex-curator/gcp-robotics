"""
codex_nodes — ROS2 node implementations for the Golden Codex pipeline.

All nodes use an adapter pattern: core logic is framework-agnostic
(no rclpy imports). Publishers, subscribers, and services are injected,
allowing operation with both real rclpy and the mock layer.
"""

__version__ = "0.1.0"

"""
ROS2 launch file for the full Golden Codex pipeline.

Launches: codex_vision_node, codex_registry_node, codex_slow_path_node,
          codex_bridge_node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="codex_nodes",
            executable="codex_registry_node",
            name="codex_registry",
            output="screen",
        ),
        Node(
            package="codex_nodes",
            executable="codex_slow_path_node",
            name="codex_slow_path",
            output="screen",
        ),
        Node(
            package="codex_nodes",
            executable="codex_vision_node",
            name="codex_vision",
            output="screen",
        ),
        Node(
            package="codex_nodes",
            executable="codex_bridge_node",
            name="codex_bridge",
            output="screen",
        ),
    ])

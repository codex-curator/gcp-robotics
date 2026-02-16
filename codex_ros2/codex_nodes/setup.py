from setuptools import setup, find_packages

package_name = "codex_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Golden Codex Robotics",
    maintainer_email="codex@example.com",
    description="ROS2 nodes for the Golden Codex dual-path robotics pipeline.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "codex_vision_node = codex_nodes.codex_vision_node:main",
            "codex_registry_node = codex_nodes.codex_registry_node:main",
            "codex_slow_path_node = codex_nodes.codex_slow_path_node:main",
            "codex_bridge_node = codex_nodes.codex_bridge_node:main",
        ],
    },
)

"""Launch file for the communication node (standalone, for hardware check)."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="humanoid_arm_communication",
            executable="communication_node",
            name="communication_node",
            output="screen",
            parameters=[],
            emulate_tty=True,
        ),
    ])

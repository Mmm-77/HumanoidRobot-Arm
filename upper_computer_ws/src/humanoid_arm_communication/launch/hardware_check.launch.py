"""Launch the communication node in hardware-check mode (debug logging, quick timeout)."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="humanoid_arm_communication",
            executable="communication_node",
            name="communication_node",
            output="screen",
            parameters=[
                {"reconnect.feedback_timeout_s": 2.0},
                {"reconnect.max_backoff_s": 5.0},
            ],
            arguments=["--ros-args", "--log-level", "debug"],
            emulate_tty=True,
        ),
    ])

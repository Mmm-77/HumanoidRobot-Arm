"""Launch node stack in simulation mode (no hardware).

Launches vision, kinematics, and runtime.  communication_node is also
launched but will remain disconnected until a serial device is attached.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="humanoid_arm_vision",
            executable="vision_node",
            name="vision_node",
            output="screen",
            parameters=[],
            emulate_tty=True,
        ),
        Node(
            package="humanoid_arm_kinematics",
            executable="kinematics_node",
            name="kinematics_node",
            output="screen",
            parameters=[],
            emulate_tty=True,
        ),
        Node(
            package="humanoid_arm_communication",
            executable="communication_node",
            name="communication_node",
            output="screen",
            parameters=[],
            emulate_tty=True,
        ),
        Node(
            package="humanoid_arm_runtime",
            executable="runtime_node",
            name="runtime_node",
            output="screen",
            parameters=[],
            emulate_tty=True,
        ),
    ])

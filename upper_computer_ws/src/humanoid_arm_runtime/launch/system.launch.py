"""Launch all 4 packages together: vision + kinematics + communication + runtime.

Start order:
  1. vision_node       — RealSense camera → AprilTag → camera_pose
  2. kinematics_node   — camera_pose → IK → joint_target
  3. communication_node — joint_target → serial → motors
  4. runtime_node       — orchestrates the above

All nodes share a single terminal; use `launch --screen` for log output.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # --- Vision ---
        Node(
            package="humanoid_arm_vision",
            executable="vision_node",
            name="vision_node",
            output="screen",
            parameters=[],
            emulate_tty=True,
        ),
        # --- Kinematics ---
        Node(
            package="humanoid_arm_kinematics",
            executable="kinematics_node",
            name="kinematics_node",
            output="screen",
            parameters=[],
            emulate_tty=True,
        ),
        # --- Communication ---
        Node(
            package="humanoid_arm_communication",
            executable="communication_node",
            name="communication_node",
            output="screen",
            parameters=[],
            emulate_tty=True,
        ),
        # --- Runtime ---
        Node(
            package="humanoid_arm_runtime",
            executable="runtime_node",
            name="runtime_node",
            output="screen",
            parameters=[],
            emulate_tty=True,
        ),
    ])

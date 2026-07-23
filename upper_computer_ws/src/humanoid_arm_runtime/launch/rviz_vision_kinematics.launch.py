"""RViz-only integration check for vision -> runtime -> kinematics.

This launch intentionally excludes communication and hardware nodes. A fresh
camera pose and simulated FK feedback establish the relative-follow baseline,
then FOLLOW starts automatically.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    description_share = Path(
        get_package_share_directory("humanoid_arm_description")
    )
    kinematics_share = Path(
        get_package_share_directory("humanoid_arm_kinematics")
    )
    runtime_share = Path(get_package_share_directory("humanoid_arm_runtime"))
    vision_share = Path(get_package_share_directory("humanoid_arm_vision"))

    robot_description = (
        description_share / "urdf" / "humanoid_arm.urdf"
    ).read_text(encoding="utf-8")

    vision_config = DeclareLaunchArgument(
        "vision_config",
        default_value=str(vision_share / "config" / "vision.yaml"),
        description="Vision node ROS parameter file.",
    )
    runtime_config = DeclareLaunchArgument(
        "runtime_config",
        default_value=str(runtime_share / "config" / "runtime.yaml"),
        description="Runtime node ROS parameter file.",
    )
    rviz_config = DeclareLaunchArgument(
        "rviz_config",
        default_value=str(
            runtime_share / "config" / "vision_kinematics.rviz"
        ),
        description="RViz configuration for the integration check.",
    )

    return LaunchDescription(
        [
            vision_config,
            runtime_config,
            rviz_config,
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="camera_tracking_frame_publisher",
                output="screen",
                # The integration check maps calibrated camera axes directly
                # onto the robot base axes. This transform is display-only.
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "base_link",
                    "camera_tracking",
                ],
            ),
            Node(
                package="humanoid_arm_kinematics",
                executable="kinematics_node",
                name="kinematics_node",
                output="screen",
                parameters=[
                    {
                        "config_file": str(
                            kinematics_share / "config" / "kinematics.yaml"
                        ),
                        "simulation.publish_joint_states": True,
                    }
                ],
            ),
            Node(
                package="humanoid_arm_runtime",
                executable="runtime_node",
                name="runtime_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("runtime_config"),
                    {
                        "follow.auto_start": True,
                        "topics.joint_state": "joint_states",
                    },
                ],
            ),
            Node(
                package="humanoid_arm_vision",
                executable="vision_node",
                name="vision_node",
                output="screen",
                parameters=[LaunchConfiguration("vision_config")],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="vision_kinematics_rviz",
                output="screen",
                arguments=["-d", LaunchConfiguration("rviz_config")],
            ),
        ]
    )

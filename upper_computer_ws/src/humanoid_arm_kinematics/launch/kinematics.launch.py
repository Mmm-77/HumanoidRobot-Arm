"""Run the URDF-backed position kinematics node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("humanoid_arm_kinematics"))
    config_argument = DeclareLaunchArgument(
        "config",
        default_value=str(package_share / "config" / "kinematics.yaml"),
        description="Path to the kinematics configuration YAML file",
    )
    node = Node(
        package="humanoid_arm_kinematics",
        executable="kinematics_node",
        name="kinematics_node",
        output="screen",
        parameters=[{"config_file": LaunchConfiguration("config")}],
    )
    return LaunchDescription([config_argument, node])

"""Launch file to run the kinematics node independently for offline testing."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("humanoid_arm_kinematics")

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=f"{pkg_share}/config/kinematics.yaml",
        description="Path to the kinematics configuration YAML file.",
    )

    limits_arg = DeclareLaunchArgument(
        "limits_config",
        default_value=f"{pkg_share}/config/joint_limits.yaml",
        description="Path to the joint limits configuration YAML file.",
    )

    kinematics_node = Node(
        package="humanoid_arm_kinematics",
        executable="kinematics_node",
        name="kinematics_node",
        output="screen",
        parameters=[{
            "config_file": LaunchConfiguration("config"),
            "limits_config_file": LaunchConfiguration("limits_config"),
        }],
    )

    return LaunchDescription([config_arg, limits_arg, kinematics_node])

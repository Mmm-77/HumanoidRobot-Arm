from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("humanoid_arm_vision")
    default_config = f"{package_share}/config/vision.yaml"

    config_argument = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Absolute path to the vision node ROS 2 parameter file.",
    )
    vision_node = Node(
        package="humanoid_arm_vision",
        executable="vision_node",
        name="vision_node",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )

    return LaunchDescription([config_argument, vision_node])

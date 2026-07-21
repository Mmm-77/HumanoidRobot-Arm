"""Backward-compatible entry point for the Gazebo camera-follow simulation."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    description_share = get_package_share_directory("humanoid_arm_description")
    launch_file = Path(description_share) / "launch" / "gazebo_camera_follow.launch.py"
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(launch_file)))
    ])

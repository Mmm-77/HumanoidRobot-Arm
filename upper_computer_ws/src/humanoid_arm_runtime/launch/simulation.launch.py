"""Entry point for the RViz camera-follow simulation (no hardware I/O).

Launches vision → kinematics → runtime with simulated joint feedback,
displayed in RViz.  No Gazebo or lower-computer is required.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    runtime_share = get_package_share_directory("humanoid_arm_runtime")
    launch_file = (
        Path(runtime_share) / "launch" / "rviz_vision_kinematics.launch.py"
    )
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_file))
        )
    ])

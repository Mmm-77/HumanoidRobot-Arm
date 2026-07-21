"""Display the 4-DOF arm in RViz with draggable joint sliders.

The launch API used here is available in ROS 2 Foxy.  No xacro processing or
newer launch substitutions are required.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("humanoid_arm_description"))
    robot_description = (package_share / "urdf" / "humanoid_arm.urdf").read_text(
        encoding="utf-8"
    )
    rviz_config = str(package_share / "config" / "humanoid_arm.rviz")
    jsp_config = str(package_share / "config" / "joint_state_publisher.yaml")
    use_gui = LaunchConfiguration("use_gui")

    rsp_parameters = [{"robot_description": robot_description}]
    jsp_parameters = [{"robot_description": robot_description, "rate": 30.0}]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_gui",
                default_value="true",
                description="Start joint_state_publisher_gui instead of the headless publisher",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=rsp_parameters,
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                name="joint_state_publisher_gui",
                output="screen",
                parameters=[rsp_parameters[0], jsp_config],
                condition=IfCondition(use_gui),
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="joint_state_publisher",
                output="screen",
                parameters=jsp_parameters,
                condition=UnlessCondition(use_gui),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            ),
            Node(
                package="humanoid_arm_description",
                executable="tip_position_gui.py",
                name="tip_position_gui",
                output="screen",
            ),
        ]
    )

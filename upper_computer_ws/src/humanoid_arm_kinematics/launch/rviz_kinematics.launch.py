"""Visualize position-IK commands in RViz without hardware or joint limits."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    description_share = Path(
        get_package_share_directory("humanoid_arm_description")
    )
    kinematics_share = Path(
        get_package_share_directory("humanoid_arm_kinematics")
    )
    robot_description = (
        description_share / "urdf" / "humanoid_arm.urdf"
    ).read_text(encoding="utf-8")

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
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
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=[
                    "-d",
                    str(description_share / "config" / "humanoid_arm.rviz"),
                ],
            ),
        ]
    )

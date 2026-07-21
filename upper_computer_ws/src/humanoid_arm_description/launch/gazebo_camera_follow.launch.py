"""Gazebo + real-camera FOLLOW chain; intentionally excludes hardware I/O."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    description_share = get_package_share_directory("humanoid_arm_description")
    gazebo_share = get_package_share_directory("gazebo_ros")
    kinematics_share = get_package_share_directory("humanoid_arm_kinematics")
    runtime_share = get_package_share_directory("humanoid_arm_runtime")
    vision_share = get_package_share_directory("humanoid_arm_vision")

    default_world = str(Path(description_share) / "worlds" / "empty.world")
    robot_description = (Path(description_share) / "urdf" / "humanoid_arm.urdf").read_text(
        encoding="utf-8"
    )

    world_argument = DeclareLaunchArgument(
        "world", default_value=default_world, description="Gazebo world file"
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(gazebo_share) / "launch" / "gazebo.launch.py")
        ),
        launch_arguments={"world": LaunchConfiguration("world")}.items(),
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
        remappings=[("joint_states", "/kinematics/joint_state")],
    )
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_humanoid_arm",
        output="screen",
        arguments=["-entity", "humanoid_arm", "-topic", "robot_description"],
    )

    # Give Gazebo and the model plugin time to advertise their services/topics.
    upper_chain = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="humanoid_arm_kinematics",
                executable="kinematics_node",
                name="kinematics_node",
                output="screen",
                parameters=[{
                    "config_file": str(
                        Path(kinematics_share) / "config" / "kinematics.yaml"
                    ),
                }],
            ),
            Node(
                package="humanoid_arm_runtime",
                executable="runtime_node",
                name="runtime_node",
                output="screen",
                parameters=[str(Path(runtime_share) / "config" / "runtime.yaml")],
            ),
            Node(
                package="humanoid_arm_vision",
                executable="vision_node",
                name="vision_node",
                output="screen",
                parameters=[str(Path(vision_share) / "config" / "vision.yaml")],
            ),
        ],
    )

    return LaunchDescription([
        world_argument,
        gazebo,
        state_publisher,
        spawn_robot,
        upper_chain,
    ])

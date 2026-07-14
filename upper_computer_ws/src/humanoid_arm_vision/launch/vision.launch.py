from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("humanoid_arm_vision")
    default_config = f"{package_share}/config/vision.yaml"

    config_argument = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Absolute path to the vision node ROS 2 parameter file.",
    )
    launch_realsense_argument = DeclareLaunchArgument(
        "launch_realsense",
        default_value="true",
        description="Launch the official realsense2_camera driver for the D435i.",
    )
    serial_number_argument = DeclareLaunchArgument(
        "serial_no",
        default_value="''",
        description="Optional D435i serial number, using realsense2_camera syntax.",
    )
    color_profile_argument = DeclareLaunchArgument(
        "color_profile",
        default_value="640,480,30",
        description="D435i RGB profile as width,height,fps.",
    )
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("launch_realsense")),
        launch_arguments={
            "camera_namespace": "camera",
            "camera_name": "camera",
            "serial_no": LaunchConfiguration("serial_no"),
            "device_type": "d435i",
            "enable_color": "true",
            "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
            "enable_depth": "false",
            "enable_infra": "false",
            "enable_infra1": "false",
            "enable_infra2": "false",
            "enable_gyro": "false",
            "enable_accel": "false",
            "pointcloud.enable": "false",
            "align_depth.enable": "false",
        }.items(),
    )
    vision_node = Node(
        package="humanoid_arm_vision",
        executable="vision_node",
        name="vision_node",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )

    return LaunchDescription(
        [
            config_argument,
            launch_realsense_argument,
            serial_number_argument,
            color_profile_argument,
            realsense,
            vision_node,
        ]
    )

"""Launch all 5 packages together: vision + kinematics + communication + runtime + RViz.

Pipeline:
  vision_node       — RealSense camera → AprilTag → camera_pose
  kinematics_node   — camera_pose delta → IK → joint_command
  communication_node — joint_command → serial → motors → joint_state
  runtime_node       — orchestrates the above with safety and state machine

By default all nodes share a single terminal; use ``launch --screen`` for
combined log output, or omit ``emulate_tty`` to split each node into their
own pty.
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
    communication_share = Path(
        get_package_share_directory("humanoid_arm_communication")
    )

    robot_description = (
        description_share / "urdf" / "humanoid_arm.urdf"
    ).read_text(encoding="utf-8")

    vision_config = DeclareLaunchArgument(
        "vision_config",
        default_value=str(vision_share / "config" / "vision.yaml"),
        description="Vision node ROS parameter file.",
    )
    kinematics_config = DeclareLaunchArgument(
        "kinematics_config",
        default_value=str(
            kinematics_share / "config" / "kinematics.yaml"
        ),
        description="Kinematics node ROS parameter file.",
    )
    communication_config = DeclareLaunchArgument(
        "communication_config",
        default_value=str(
            communication_share / "config" / "communication.yaml"
        ),
        description="Communication node ROS parameter file.",
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
        description="RViz configuration for the full system.",
    )

    return LaunchDescription(
        [
            vision_config,
            kinematics_config,
            communication_config,
            runtime_config,
            rviz_config,

            # --- Robot model ---
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                remappings=[
                    (
                        "joint_states",
                        "kinematics/joint_state",
                    )
                ],
            ),

            # --- Static transform: camera_tracking → base_link ---
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="camera_tracking_frame_publisher",
                output="screen",
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

            # --- Vision ---
            Node(
                package="humanoid_arm_vision",
                executable="vision_node",
                name="vision_node",
                output="screen",
                parameters=[LaunchConfiguration("vision_config")],
                emulate_tty=True,
            ),

            # --- Kinematics ---
            Node(
                package="humanoid_arm_kinematics",
                executable="kinematics_node",
                name="kinematics_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("kinematics_config"),
                    {"config_file": str(
                        kinematics_share / "config" / "kinematics.yaml"
                    )},
                ],
                emulate_tty=True,
            ),

            # --- Communication (hardware I/O) ---
            Node(
                package="humanoid_arm_communication",
                executable="communication_node",
                name="communication_node",
                output="screen",
                parameters=[LaunchConfiguration("communication_config")],
                emulate_tty=True,
            ),

            # --- Runtime orchestrator ---
            Node(
                package="humanoid_arm_runtime",
                executable="runtime_node",
                name="runtime_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("runtime_config"),
                    {
                        # Override: hardware joint feedback comes from
                        # communication_node, not from kinematics sim.
                        "topics.joint_state": "kinematics/joint_state",
                    },
                ],
                emulate_tty=True,
            ),

            # --- RViz ---
            Node(
                package="rviz2",
                executable="rviz2",
                name="system_rviz",
                output="screen",
                arguments=["-d", LaunchConfiguration("rviz_config")],
            ),
        ]
    )

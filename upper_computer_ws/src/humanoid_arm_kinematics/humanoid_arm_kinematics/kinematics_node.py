"""ROS 2 node for URDF-backed position kinematics.

The target interface remains ``PoseStamped`` for compatibility, but only the
position is controlled.  Target orientation is intentionally ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from .forward_solver import ForwardSolver
from .inverse_solver import IKConfig, InverseSolver
from .jacobian import JacobianSolver
from .robot_model import ModelError, RobotModel
from .solution_policy import is_solution_acceptable
from .target_shaper import ShaperConfig, TargetShaper


class KinematicsNode(Node):
    """Solve ``tip_frame`` position targets and publish joint commands/FK."""

    def __init__(self) -> None:
        super().__init__("kinematics_node")
        self._declare_parameters()
        self._load_config_file()
        self._load_model()
        self._build_pipeline()
        self._last_joint_angles: Optional[np.ndarray] = None
        self._latest_target: Optional[PoseStamped] = None
        self._consecutive_ik_failures = 0
        self._max_consecutive_failures = 10
        self._setup_ros()

    def _declare_parameters(self) -> None:
        defaults = {
            "config_file": "",
            "processing_rate_hz": 30.0,
            "model.description_package": "humanoid_arm_description",
            "model.urdf_relative_path": "urdf/humanoid_arm.urdf",
            "model.base_link": "base_link",
            "model.tip_link": "tip_frame",
            "frame_id.base": "base_link",
            "topics.target": "kinematics/target",
            "topics.joint_command": "kinematics/joint_command",
            "topics.joint_state": "kinematics/joint_state",
            "topics.simulated_joint_state": "joint_states",
            "topics.end_effector_pose": "kinematics/end_effector_pose",
            "topics.diagnostics": "kinematics/diagnostics",
            "topics.visualization": "kinematics/visualization",
            "simulation.publish_joint_states": False,
            "ik_solver.max_iterations": 250,
            "ik_solver.position_tolerance_m": 0.001,
            "ik_solver.initial_lambda": 0.03,
            "ik_solver.lambda_increase_factor": 2.0,
            "ik_solver.lambda_decrease_factor": 0.5,
            "ik_solver.lambda_min": 1e-5,
            "ik_solver.lambda_max": 1.0,
            "ik_solver.multi_start_attempts": 8,
            "ik_solver.multi_start_perturbation_rad": 0.5,
            "ik_solver.continuity_gain": 0.02,
            "ik_solver.max_step_rad": 0.25,
            "validation.max_position_error_m": 0.005,
            "shaper.position_dead_zone_m": 0.001,
            "shaper.max_position_step_m": 0.05,
            "shaper.position_alpha": 0.3,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _load_config_file(self) -> None:
        self._config: Dict[str, Any] = {}
        path = str(self.get_parameter("config_file").value)
        if not path:
            return
        with open(path, "r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        self._config = loaded

    def _value(self, name: str) -> Any:
        value: Any = self._config
        for part in name.split("."):
            if not isinstance(value, dict) or part not in value:
                return self.get_parameter(name).value
            value = value[part]
        return value

    def _load_model(self) -> None:
        package = str(self._value("model.description_package"))
        relative_path = str(self._value("model.urdf_relative_path"))
        urdf_path = Path(get_package_share_directory(package)) / relative_path
        self._model = RobotModel.from_urdf_file(
            urdf_path,
            base_link=str(self._value("model.base_link")),
            tip_link=str(self._value("model.tip_link")),
        )
        self.get_logger().info(
            f"Loaded URDF chain {self._model.base_link} -> {self._model.tip_link}: "
            f"{', '.join(self._model.joint_names)}"
        )

    def _build_pipeline(self) -> None:
        self._fk = ForwardSolver(self._model)
        self._jac = JacobianSolver(self._model)
        self._ik = InverseSolver(
            self._fk,
            self._jac,
            IKConfig(
                max_iterations=int(self._value("ik_solver.max_iterations")),
                position_tolerance_m=float(
                    self._value("ik_solver.position_tolerance_m")
                ),
                initial_lambda=float(self._value("ik_solver.initial_lambda")),
                lambda_increase_factor=float(
                    self._value("ik_solver.lambda_increase_factor")
                ),
                lambda_decrease_factor=float(
                    self._value("ik_solver.lambda_decrease_factor")
                ),
                lambda_min=float(self._value("ik_solver.lambda_min")),
                lambda_max=float(self._value("ik_solver.lambda_max")),
                multi_start_attempts=int(
                    self._value("ik_solver.multi_start_attempts")
                ),
                multi_start_perturbation_rad=float(
                    self._value("ik_solver.multi_start_perturbation_rad")
                ),
                continuity_gain=float(self._value("ik_solver.continuity_gain")),
                max_step_rad=float(self._value("ik_solver.max_step_rad")),
            ),
        )
        self._shaper = TargetShaper(
            ShaperConfig(
                position_dead_zone_m=float(
                    self._value("shaper.position_dead_zone_m")
                ),
                max_position_step_m=float(
                    self._value("shaper.max_position_step_m")
                ),
                position_alpha=float(self._value("shaper.position_alpha")),
            )
        )

    def _setup_ros(self) -> None:
        qos = QoSProfile(depth=5)
        self._target_sub = self.create_subscription(
            PoseStamped,
            str(self._value("topics.target")),
            self._target_callback,
            qos,
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            str(self._value("topics.joint_state")),
            self._joint_state_callback,
            qos,
        )
        self._command_pub = self.create_publisher(
            JointTrajectory, str(self._value("topics.joint_command")), qos
        )
        self._end_effector_pub = self.create_publisher(
            PoseStamped, str(self._value("topics.end_effector_pose")), qos
        )
        self._diag_pub = self.create_publisher(
            DiagnosticArray, str(self._value("topics.diagnostics")), qos
        )
        self._visualization_pub = self.create_publisher(
            MarkerArray, str(self._value("topics.visualization")), qos
        )
        self._simulation_pub = None
        if bool(self._value("simulation.publish_joint_states")):
            self._simulation_pub = self.create_publisher(
                JointState, str(self._value("topics.simulated_joint_state")), qos
            )
            self._last_joint_angles = np.zeros(self._model.num_joints)
            self._publish_simulated_joint_state(self._last_joint_angles)
            self._publish_end_effector_pose(self._last_joint_angles)
        rate = float(self._value("processing_rate_hz"))
        self._timer = self.create_timer(1.0 / rate, self._process_target)

    def _target_callback(self, message: PoseStamped) -> None:
        self._latest_target = message

    def _joint_state_callback(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        if message.name:
            if not all(name in by_name for name in self._model.joint_names):
                return
            values = [by_name[name] for name in self._model.joint_names]
        elif len(message.position) >= self._model.num_joints:
            values = message.position[: self._model.num_joints]
        else:
            return
        angles = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(angles)):
            return
        self._last_joint_angles = angles
        self._publish_end_effector_pose(angles, message.header.stamp)

    def _process_target(self) -> None:
        try:
            self._process_target_impl()
        except Exception:
            self.get_logger().exception(
                "Unhandled exception in kinematics tick; skipping this cycle"
            )
            self._publish_diagnostics(
                "internal_error", DiagnosticStatus.ERROR,
                {"detail": "unhandled exception, see node log"},
            )

    def _process_target_impl(self) -> None:
        if self._simulation_pub is not None and self._last_joint_angles is not None:
            self._publish_simulated_joint_state(self._last_joint_angles)
            self._publish_end_effector_pose(self._last_joint_angles)
        if self._latest_target is None:
            return
        raw = np.array(
            [
                self._latest_target.pose.position.x,
                self._latest_target.pose.position.y,
                self._latest_target.pose.position.z,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(raw)):
            self._publish_diagnostics("invalid_target", DiagnosticStatus.WARN, {})
            return
        if self._last_joint_angles is None:
            self._last_joint_angles = np.zeros(self._model.num_joints)
        if not self._shaper.initialized:
            self._shaper.shape(self._fk.solve(self._last_joint_angles).position)
        target = self._shaper.shape(raw)
        result = self._ik.solve(target, self._last_joint_angles)
        validation_limit = float(
            self._value("validation.max_position_error_m")
        )
        if not is_solution_acceptable(
            result.success,
            result.position_error_m,
            validation_limit,
        ):
            self._on_ik_unreachable(target, result)
            return

        self._consecutive_ik_failures = 0
        previous = self._last_joint_angles.copy()
        self._last_joint_angles = result.joint_angles_rad.copy()
        self._publish_joint_command(
            self._last_joint_angles,
            (self._last_joint_angles - previous)
            * float(self._value("processing_rate_hz")),
            self._latest_target.header.stamp,
        )
        if self._simulation_pub is not None:
            self._publish_simulated_joint_state(self._last_joint_angles)
        self._publish_end_effector_pose(self._last_joint_angles)
        actual = self._fk.solve(self._last_joint_angles).position
        self._publish_visualization(target, actual)
        reason = "valid" if result.success else "valid_within_validation"
        level = DiagnosticStatus.OK if result.success else DiagnosticStatus.WARN
        self._publish_diagnostics(
            reason,
            level,
            {
                "position_error_m": f"{result.position_error_m:.6f}",
                "iterations": str(result.iterations),
                "near_position_singularity": str(result.near_singular),
            },
        )

    def _on_ik_unreachable(
        self,
        target: np.ndarray,
        result: object,
    ) -> None:
        """Handle unreachable target by moving to the closest reachable pose.

        The IK solver's Levenberg-Marquardt iteration naturally converges
        toward the workspace boundary when the target is outside it.  We
        accept that partial result as a clamped solution, reset the
        TargetShaper so it doesn't drift, and continue publishing.
        """
        self._consecutive_ik_failures += 1
        if self._consecutive_ik_failures >= self._max_consecutive_failures:
            self._shaper.reset()
            self.get_logger().warn(
                f"TargetShaper reset after {self._consecutive_ik_failures} "
                "consecutive IK failures"
            )
            self._consecutive_ik_failures = 0
            return

        # Use the IK solver's partial result as the closest reachable config.
        if result.forward_result is not None and np.isfinite(result.position_error_m):
            clamped_angles = result.joint_angles_rad
            clamped_pos = result.forward_result.position

            # Reset TargetShaper to the clamped boundary position so it
            # doesn't keep drifting toward the unreachable target.
            self._shaper.reset()
            self._shaper.shape(clamped_pos)

            previous = self._last_joint_angles.copy()
            self._last_joint_angles = clamped_angles.copy()
            self._publish_joint_command(
                clamped_angles,
                (clamped_angles - previous)
                * float(self._value("processing_rate_hz")),
                self._latest_target.header.stamp,
            )
            if self._simulation_pub is not None:
                self._publish_simulated_joint_state(clamped_angles)
            self._publish_end_effector_pose(clamped_angles)
            self._publish_visualization(target, clamped_pos)
            self._publish_diagnostics(
                "ik_clamped",
                DiagnosticStatus.WARN,
                {"position_error_m": f"{result.position_error_m:.6f}"},
            )
            self._consecutive_ik_failures = 0
        else:
            # No usable partial result — truly stuck.
            actual = self._fk.solve(self._last_joint_angles).position
            self._publish_visualization(target, actual)
            self._publish_diagnostics(
                "ik_failed",
                DiagnosticStatus.WARN,
                {"position_error_m": f"{result.position_error_m:.6f}"},
            )

    def _publish_joint_command(
        self, angles: np.ndarray, velocities: np.ndarray, stamp: Any
    ) -> None:
        message = JointTrajectory()
        message.header.stamp = stamp
        message.header.frame_id = str(self._value("frame_id.base"))
        message.joint_names = self._model.joint_names
        point = JointTrajectoryPoint()
        point.positions = angles.tolist()
        point.velocities = velocities.tolist()
        point.time_from_start.nanosec = int(
            1.0e9 / float(self._value("processing_rate_hz"))
        )
        message.points = [point]
        self._command_pub.publish(message)

    def _publish_simulated_joint_state(self, angles: np.ndarray) -> None:
        if self._simulation_pub is None:
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = self._model.joint_names
        message.position = angles.tolist()
        self._simulation_pub.publish(message)

    def _publish_end_effector_pose(self, angles: np.ndarray, stamp: Any = None) -> None:
        result = self._fk.solve(angles)
        message = PoseStamped()
        message.header.stamp = stamp or self.get_clock().now().to_msg()
        message.header.frame_id = str(self._value("frame_id.base"))
        message.pose.position.x = float(result.position[0])
        message.pose.position.y = float(result.position[1])
        message.pose.position.z = float(result.position[2])
        message.pose.orientation.x = float(result.quaternion_xyzw[0])
        message.pose.orientation.y = float(result.quaternion_xyzw[1])
        message.pose.orientation.z = float(result.quaternion_xyzw[2])
        message.pose.orientation.w = float(result.quaternion_xyzw[3])
        self._end_effector_pub.publish(message)

    def _publish_visualization(
        self, target: np.ndarray, actual: np.ndarray
    ) -> None:
        """Show target, achieved tip position, and Cartesian error in RViz."""
        stamp = self.get_clock().now().to_msg()
        frame = str(self._value("frame_id.base"))

        target_marker = Marker()
        target_marker.header.stamp = stamp
        target_marker.header.frame_id = frame
        target_marker.ns = "kinematics_target"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position = Point(
            x=float(target[0]), y=float(target[1]), z=float(target[2])
        )
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.025
        target_marker.scale.y = 0.025
        target_marker.scale.z = 0.025
        target_marker.color.r = 1.0
        target_marker.color.g = 0.15
        target_marker.color.b = 0.05
        target_marker.color.a = 0.95

        actual_marker = Marker()
        actual_marker.header.stamp = stamp
        actual_marker.header.frame_id = frame
        actual_marker.ns = "kinematics_tip"
        actual_marker.id = 1
        actual_marker.type = Marker.SPHERE
        actual_marker.action = Marker.ADD
        actual_marker.pose.position = Point(
            x=float(actual[0]), y=float(actual[1]), z=float(actual[2])
        )
        actual_marker.pose.orientation.w = 1.0
        actual_marker.scale.x = 0.018
        actual_marker.scale.y = 0.018
        actual_marker.scale.z = 0.018
        actual_marker.color.r = 0.05
        actual_marker.color.g = 1.0
        actual_marker.color.b = 0.15
        actual_marker.color.a = 0.95

        error_marker = Marker()
        error_marker.header.stamp = stamp
        error_marker.header.frame_id = frame
        error_marker.ns = "kinematics_error"
        error_marker.id = 2
        error_marker.type = Marker.LINE_LIST
        error_marker.action = Marker.ADD
        error_marker.scale.x = 0.004
        error_marker.color.r = 1.0
        error_marker.color.g = 0.85
        error_marker.color.b = 0.05
        error_marker.color.a = 0.95
        error_marker.points = [target_marker.pose.position, actual_marker.pose.position]

        text_marker = Marker()
        text_marker.header.stamp = stamp
        text_marker.header.frame_id = frame
        text_marker.ns = "kinematics_error_text"
        text_marker.id = 3
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        midpoint = 0.5 * (target + actual)
        text_marker.pose.position = Point(
            x=float(midpoint[0]),
            y=float(midpoint[1]),
            z=float(midpoint[2] + 0.025),
        )
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.018
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 0.95
        error_mm = 1000.0 * float(np.linalg.norm(target - actual))
        text_marker.text = f"tip error: {error_mm:.2f} mm"

        self._visualization_pub.publish(
            MarkerArray(
                markers=[target_marker, actual_marker, error_marker, text_marker]
            )
        )

    def _publish_diagnostics(
        self, reason: str, level: int, metrics: Dict[str, str]
    ) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "humanoid_arm_kinematics/pipeline"
        status.hardware_id = "urdf_position_kinematics"
        status.level = level
        status.message = reason
        values = [("reason", reason)] + list(metrics.items())
        status.values = [KeyValue(key=key, value=value) for key, value in values]
        array.status = [status]
        self._diag_pub.publish(array)

    def reset(self) -> None:
        self._shaper.reset()
        self._latest_target = None
        self._last_joint_angles = None


def main(args: Optional[Iterable[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[KinematicsNode] = None
    try:
        node = KinematicsNode()
        rclpy.spin(node)
    except (ModelError, ValueError) as exc:
        rclpy.logging.get_logger("kinematics_node").fatal(str(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

"""ROS 2 node that owns the complete kinematics pipeline.

Subscribes to task-space targets from the runtime package and publishes
joint-angle / joint-velocity commands for the communication package.
"""

from __future__ import annotations

import os

_ros_lib = "/opt/ros/foxy/lib"
_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if _ros_lib not in _ld_path:
    _extra = (
        "/opt/ros/foxy/opt/yaml_cpp_vendor/lib"
        ":/opt/ros/foxy/opt/rviz_ogre_vendor/lib"
        ":/opt/ros/foxy/lib/x86_64-linux-gnu"
    )
    os.environ["LD_LIBRARY_PATH"] = (
        f"{_ros_lib}:{_extra}:{_ld_path}" if _ld_path else f"{_ros_lib}:{_extra}"
    )

import math
from typing import Any, Iterable, Optional

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import yaml

from .forward_solver import ForwardSolver
from .inverse_solver import IKConfig, InverseSolver
from .jacobian import JacobianSolver
from .robot_model import RobotModel
from .solution_selector import JointLimits, SolutionSelector
from .solution_validator import SolutionValidator
from .target_shaper import ShaperConfig, TargetShaper
from .workspace_guard import GuardReason, WorkspaceGuard


class KinematicsNode(Node):
    """ROS 2 node for the kinematics pipeline.

    Pipeline:
      PoseStamped target → WorkspaceGuard → InverseSolver (multi-start)
        → SolutionSelector → SolutionValidator → TargetShaper
        → JointTrajectory published
    """

    def __init__(self) -> None:
        super().__init__("kinematics_node")
        self._declare_parameters()
        self._load_model()
        self._load_limits()
        self._build_pipeline()
        self._setup_ros()
        self._last_joint_angles: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        defaults: dict[str, object] = {
            "processing_rate_hz": 30.0,
            "topics.target": "kinematics/target",
            "topics.joint_command": "kinematics/joint_command",
            "topics.joint_state": "kinematics/joint_state",
            "topics.diagnostics": "kinematics/diagnostics",
            "frame_id.base": "base",
            "dh_params": [],
            "tool_offset.translation": [0.0, 0.0, 0.0],
            "tool_offset.rotation_deg": [0.0, 0.0, 0.0],
            "controllable_axis": "base_z",
            # IK solver
            "ik_solver.max_iterations": 200,
            "ik_solver.position_tolerance_m": 0.001,
            "ik_solver.orientation_tolerance_rad": 0.01,
            "ik_solver.initial_lambda": 0.1,
            "ik_solver.lambda_increase_factor": 2.0,
            "ik_solver.lambda_decrease_factor": 0.5,
            "ik_solver.lambda_min": 1e-6,
            "ik_solver.lambda_max": 1.0,
            "ik_solver.multi_start_attempts": 5,
            "ik_solver.multi_start_perturbation_rad": 0.3,
            # Singularity
            "singularity.manipulability_threshold": 0.001,
            # Validation
            "validation.max_position_error_m": 0.005,
            "validation.max_orientation_error_rad": 0.02,
            # Shaper
            "shaper.position_dead_zone_m": 0.001,
            "shaper.orientation_dead_zone_rad": 0.005,
            "shaper.max_position_step_m": 0.05,
            "shaper.max_orientation_step_rad": 0.1,
            "shaper.position_alpha": 0.3,
            "shaper.orientation_alpha": 0.3,
            # Joint limits
            "joints.joint_1.angle_min_deg": -85.0,
            "joints.joint_1.angle_max_deg": 175.0,
            "joints.joint_1.max_velocity_deg_per_s": 180.0,
            "joints.joint_2.angle_min_deg": -10.0,
            "joints.joint_2.angle_max_deg": 150.0,
            "joints.joint_2.max_velocity_deg_per_s": 180.0,
            "joints.joint_3.angle_min_deg": -100.0,
            "joints.joint_3.angle_max_deg": 100.0,
            "joints.joint_3.max_velocity_deg_per_s": 180.0,
            "joints.joint_4.angle_min_deg": -40.0,
            "joints.joint_4.angle_max_deg": 100.0,
            "joints.joint_4.max_velocity_deg_per_s": 180.0,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _value(self, name: str) -> Any:
        return self.get_parameter(name).value

    def _load_model(self) -> None:
        dh_params = self._value("dh_params")
        if not dh_params:
            raise ValueError("dh_params must be provided in the config file")

        tool_t = self._value("tool_offset.translation")
        tool_r = self._value("tool_offset.rotation_deg")
        self._model = RobotModel.from_config(dh_params, tool_t, tool_r)
        self.get_logger().info(
            f"Loaded DH model with {self._model.num_joints} joints"
        )

    def _load_limits(self) -> None:
        limits = JointLimits(
            angle_min_rad=np.deg2rad([
                float(self._value("joints.joint_1.angle_min_deg")),
                float(self._value("joints.joint_2.angle_min_deg")),
                float(self._value("joints.joint_3.angle_min_deg")),
                float(self._value("joints.joint_4.angle_min_deg")),
            ]),
            angle_max_rad=np.deg2rad([
                float(self._value("joints.joint_1.angle_max_deg")),
                float(self._value("joints.joint_2.angle_max_deg")),
                float(self._value("joints.joint_3.angle_max_deg")),
                float(self._value("joints.joint_4.angle_max_deg")),
            ]),
            max_velocity_rad_per_s=np.deg2rad([
                float(self._value("joints.joint_1.max_velocity_deg_per_s")),
                float(self._value("joints.joint_2.max_velocity_deg_per_s")),
                float(self._value("joints.joint_3.max_velocity_deg_per_s")),
                float(self._value("joints.joint_4.max_velocity_deg_per_s")),
            ]),
        )
        self._limits = limits

    def _build_pipeline(self) -> None:
        self._fk = ForwardSolver(self._model)
        self._jac = JacobianSolver(self._model)

        ik_cfg = IKConfig(
            max_iterations=int(self._value("ik_solver.max_iterations")),
            position_tolerance_m=float(self._value("ik_solver.position_tolerance_m")),
            orientation_tolerance_rad=float(
                self._value("ik_solver.orientation_tolerance_rad")
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
            multi_start_attempts=int(self._value("ik_solver.multi_start_attempts")),
            multi_start_perturbation_rad=float(
                self._value("ik_solver.multi_start_perturbation_rad")
            ),
        )
        self._ik = InverseSolver(self._fk, self._jac, ik_cfg)
        self._selector = SolutionSelector(
            self._limits,
            manipulability_threshold=float(
                self._value("singularity.manipulability_threshold")
            ),
        )
        self._validator = SolutionValidator(
            self._fk,
            self._limits,
            max_position_error_m=float(self._value("validation.max_position_error_m")),
            max_orientation_error_rad=float(
                self._value("validation.max_orientation_error_rad")
            ),
        )
        self._shaper = TargetShaper(
            ShaperConfig(
                position_dead_zone_m=float(self._value("shaper.position_dead_zone_m")),
                orientation_dead_zone_rad=float(
                    self._value("shaper.orientation_dead_zone_rad")
                ),
                max_position_step_m=float(self._value("shaper.max_position_step_m")),
                max_orientation_step_rad=float(
                    self._value("shaper.max_orientation_step_rad")
                ),
                position_alpha=float(self._value("shaper.position_alpha")),
                orientation_alpha=float(self._value("shaper.orientation_alpha")),
            ),
            dt_s=1.0 / float(self._value("processing_rate_hz")),
        )
        self._guard = WorkspaceGuard(self._limits)

    def _setup_ros(self) -> None:
        state_qos = QoSProfile(depth=5)
        self._target_sub = self.create_subscription(
            PoseStamped,
            str(self._value("topics.target")),
            self._target_callback,
            state_qos,
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            str(self._value("topics.joint_state")),
            self._joint_state_callback,
            state_qos,
        )
        self._command_pub = self.create_publisher(
            JointTrajectory,
            str(self._value("topics.joint_command")),
            state_qos,
        )
        self._diag_pub = self.create_publisher(
            DiagnosticArray,
            str(self._value("topics.diagnostics")),
            state_qos,
        )

        processing_rate_hz = float(self._value("processing_rate_hz"))
        self._timer = self.create_timer(
            1.0 / processing_rate_hz, self._process_pipeline
        )

        # Cached target from subscription
        self._latest_target: Optional[PoseStamped] = None

        self.get_logger().info(
            f"Kinematics node running at {processing_rate_hz} Hz, "
            f"subscribing to {self._value('topics.target')}"
        )

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _target_callback(self, msg: PoseStamped) -> None:
        self._latest_target = msg

    def _joint_state_callback(self, msg: JointState) -> None:
        if len(msg.position) >= self._model.num_joints:
            self._last_joint_angles = np.array(
                msg.position[: self._model.num_joints], dtype=np.float64
            )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _process_pipeline(self) -> None:
        target = self._latest_target
        if target is None:
            self._publish_diagnostics(
                "no_target", DiagnosticStatus.WARN, {"reason": "No target received yet"}
            )
            return

        stamp = target.header.stamp
        target_pos = np.array([
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
        ], dtype=np.float64)
        # Extract yaw from the quaternion
        qx, qy, qz, qw = (
            target.pose.orientation.x,
            target.pose.orientation.y,
            target.pose.orientation.z,
            target.pose.orientation.w,
        )
        target_yaw = float(math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        ))

        # Step 1: Workspace guard (basic validity check)
        guard = self._guard.check_target(target_pos, target_yaw)
        if not guard.allowed:
            self._publish_diagnostics(
                guard.reason.value,
                DiagnosticStatus.WARN,
                {"reason": guard.reason.value},
            )
            return

        # Step 2: If no current joints, use zero / mid-range as initial guess
        if self._last_joint_angles is None:
            self._last_joint_angles = np.zeros(self._model.num_joints, dtype=np.float64)
            self.get_logger().info("No joint state received yet, using zero guess")

        initial_guess = self._last_joint_angles.copy()

        # Step 3: IK solve (multi-start, damped least squares)
        manipulability_threshold = float(
            self._value("singularity.manipulability_threshold")
        )
        ik_result = self._ik.solve(
            target_pos, target_yaw, initial_guess, manipulability_threshold
        )

        if not ik_result.success:
            # Multi-start failed; try once more from zero config
            self.get_logger().debug(
                "Multi-start IK failed, retrying from zero configuration"
            )
            ik_result = self._ik.solve(
                target_pos, target_yaw,
                np.zeros(self._model.num_joints, dtype=np.float64),
                manipulability_threshold,
            )

        if not ik_result.success:
            self._publish_diagnostics(
                "ik_failed",
                DiagnosticStatus.WARN,
                {
                    "reason": "Inverse kinematics did not converge",
                    "iterations": str(self._ik.config.max_iterations),
                },
            )
            return

        # Step 4: Solution selection (single result from single-start per call;
        #         multi-start is handled inside InverseSolver.solve)
        candidates = [ik_result]
        best = self._selector.select(candidates, self._last_joint_angles)
        if best is None:
            self._publish_diagnostics(
                "no_valid_solution",
                DiagnosticStatus.WARN,
                {"reason": "No solution passes joint limits or singularity check"},
            )
            return

        # Step 5: Validation (FK back-substitution)
        validation = self._validator.validate(best, target_pos, target_yaw)
        if not validation.valid:
            self._publish_diagnostics(
                "validation_failed",
                DiagnosticStatus.WARN,
                {
                    "position_error_m": f"{validation.position_error_m:.6f}",
                    "orientation_error_rad": f"{validation.orientation_error_rad:.6f}",
                    "within_limits": str(validation.within_joint_limits),
                },
            )
            return

        # Step 6: Guard check on joint angles
        joint_guard = self._guard.check_joint_angles(best.joint_angles_rad)
        if not joint_guard.allowed:
            self._publish_diagnostics(
                joint_guard.reason.value,
                DiagnosticStatus.WARN,
                {"reason": joint_guard.reason.value},
            )
            return

        # Step 7: Target shaping
        if ik_result.forward_result is not None:
            fk_pos = ik_result.forward_result.position
            fk_yaw = ik_result.forward_result.yaw_rad
        else:
            fk = self._fk.solve(best.joint_angles_rad)
            fk_pos = fk.position
            fk_yaw = fk.yaw_rad

        shaped = self._shaper.shape(best.joint_angles_rad, fk_pos, fk_yaw)

        # Step 8: Guard check on velocities
        vel_guard = self._guard.check_joint_velocities(
            shaped.joint_velocities_rad_per_s
        )
        if not vel_guard.allowed:
            self._publish_diagnostics(
                vel_guard.reason.value,
                DiagnosticStatus.WARN,
                {"reason": vel_guard.reason.value},
            )
            return

        # Step 9: Publish joint command
        self._publish_joint_command(shaped, stamp)

        # Update state
        self._last_joint_angles = shaped.joint_angles_rad.copy()

        # Diagnostics
        self._publish_diagnostics(
            "valid",
            DiagnosticStatus.OK,
            {
                "position_error_m": f"{validation.position_error_m:.6f}",
                "orientation_error_rad": f"{validation.orientation_error_rad:.6f}",
                "joint_distance_rad": f"{best.joint_distance_rad:.6f}",
                "ik_iterations": str(ik_result.iterations),
                "near_singular": str(ik_result.near_singular),
            },
        )

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_joint_command(
        self, shaped: object, stamp
    ) -> None:
        """Publish a JointTrajectory with one point containing 4 joint angles + velocities."""
        from .target_shaper import ShapedTarget
        shaped_target: ShapedTarget = shaped

        msg = JointTrajectory()
        msg.header.stamp = stamp
        msg.header.frame_id = str(self._value("frame_id.base"))
        msg.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4"]

        point = JointTrajectoryPoint()
        point.positions = shaped_target.joint_angles_rad.tolist()
        point.velocities = shaped_target.joint_velocities_rad_per_s.tolist()
        # Convert the duration from processing rate
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(
            (1.0 / float(self._value("processing_rate_hz"))) * 1e9
        )

        msg.points = [point]
        self._command_pub.publish(msg)

    def _publish_diagnostics(
        self,
        reason: str,
        level: int,
        metrics: dict[str, str],
    ) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.name = "humanoid_arm_kinematics/pipeline"
        status.hardware_id = "kinematics"
        status.message = reason
        status.level = level

        values = [("reason", reason)]
        values.extend(metrics.items())
        status.values = [self._kv(k, v) for k, v in values]

        array.status = [status]
        self._diag_pub.publish(array)

    @staticmethod
    def _kv(key: str, value: str) -> KeyValue:
        item = KeyValue()
        item.key = key
        item.value = value
        return item

    def reset(self) -> None:
        """Reset internal state (shaper, last joints)."""
        self._shaper.reset()
        self._last_joint_angles = None
        self.get_logger().info("Kinematics node state reset")


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node: Optional[KinematicsNode] = None
    try:
        node = KinematicsNode()
        rclpy.spin(node)
    except (ModelError, ValueError) as exc:
        rclpy.logging.get_logger("kinematics_node").fatal(
            f"invalid kinematics configuration: {exc}"
        )
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

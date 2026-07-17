"""ROS 2 system orchestration node: connects vision → kinematics → communication.

Publishes 4-DOF task targets based on camera deltas relative to a recorded
baseline.  Monitors safety conditions and controls the state machine.

Subscriptions:
  - ``vision/camera_pose``  (PoseStamped)
  - ``kinematics/joint_state`` / ``communication/joint_state`` (JointState)
  - ``communication/link_state`` (Bool)

Publications:
  - ``kinematics/target`` (PoseStamped)   — 4-DOF task target
  - ``communication/ctrl_mode`` (UInt8)   — control mode override
  - ``runtime/diagnostics`` (DiagnosticArray)
  - ``runtime/state`` (String)
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional, Tuple

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String, UInt8
from std_srvs.srv import Trigger

from .baseline_manager import BaselineManager
from .diagnostics import DiagnosticsAggregator
from .follow_mapper import FollowMapper
from .safety_manager import (
    ControlAction,
    SafetyManager,
    SafetyResult,
)
from .state_machine import (
    SystemState,
    StateMachine,
    Transition,
)
from .system_context import (
    JointSnapshot,
    PoseSnapshot,
    SystemContext,
)
from .task_projector import TaskProjector
from .watchdog import Watchdog, WatchdogConfig


def _quat_from_msg(msg: Quaternion) -> Tuple[float, float, float, float]:
    return (msg.x, msg.y, msg.z, msg.w)


def _quat_to_msg(xyzw: np.ndarray) -> Quaternion:
    q = Quaternion()
    q.x = float(xyzw[0])
    q.y = float(xyzw[1])
    q.z = float(xyzw[2])
    q.w = float(xyzw[3])
    return q


class RuntimeNode(Node):
    """System orchestrator for the 4-DOF humanoid arm."""

    def __init__(self) -> None:
        super().__init__("runtime_node")
        self._declare_params()

        # --- Core components ---
        self._context = SystemContext()
        self._fsm = StateMachine(SystemState.INIT)
        self._watchdog = Watchdog(
            self._context,
            WatchdogConfig(
                vision_fresh_s=float(self._p("watchdog.vision_fresh_s")),
                vision_stale_s=float(self._p("watchdog.vision_stale_s")),
                communication_fresh_s=float(self._p("watchdog.communication_fresh_s")),
                communication_stale_s=float(self._p("watchdog.communication_stale_s")),
                ik_fresh_s=float(self._p("watchdog.ik_fresh_s")),
            ),
            monotonic_clock=time.monotonic,
        )
        self._safety = SafetyManager(
            self._fsm,
            self._watchdog,
            max_lost_frames=int(self._p("safety.max_vision_lost_frames")),
            max_ik_failures=int(self._p("safety.max_ik_failures")),
        )
        self._baseline = BaselineManager(
            self._context,
            max_pose_age_s=float(self._p("baseline.max_pose_age_s")),
            max_joint_age_s=float(self._p("baseline.max_joint_age_s")),
            monotonic_clock=time.monotonic,
        )
        self._mapper = FollowMapper(
            axis_signs=(
                int(self._p("follow.axis_sign_x")),
                int(self._p("follow.axis_sign_y")),
                int(self._p("follow.axis_sign_z")),
            ),
            position_scale=float(self._p("follow.position_scale")),
        )
        self._projector = TaskProjector(
            max_position_step_m=float(self._p("projector.max_position_step_m")),
            max_yaw_step_rad=float(self._p("projector.max_yaw_step_rad")),
        )
        self._diagnostics = DiagnosticsAggregator(self)

        # --- ROS 2 interfaces ---
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        state_qos = QoSProfile(depth=10)

        # Subscriptions
        self._pose_sub = self.create_subscription(
            PoseStamped,
            str(self._p("topics.camera_pose")),
            self._pose_callback,
            sensor_qos,
        )
        self._joint_sub = self.create_subscription(
            JointState,
            str(self._p("topics.joint_state")),
            self._joint_callback,
            sensor_qos,
        )
        self._link_sub = self.create_subscription(
            Bool,
            str(self._p("topics.link_state")),
            self._link_callback,
            sensor_qos,
        )

        # Publications
        self._target_pub = self.create_publisher(
            PoseStamped,
            str(self._p("topics.kinematics_target")),
            state_qos,
        )
        self._ctrl_mode_pub = self.create_publisher(
            UInt8,
            str(self._p("topics.ctrl_mode")),
            state_qos,
        )
        self._state_pub = self.create_publisher(
            String,
            str(self._p("topics.state")),
            state_qos,
        )
        self._diag_pub = self.create_publisher(
            DiagnosticArray,
            str(self._p("topics.diagnostics")),
            state_qos,
        )

        # Services
        self._srv_start = self.create_service(Trigger, "~/start", self._start_cb)
        self._srv_hold = self.create_service(Trigger, "~/hold", self._hold_cb)
        self._srv_unhold = self.create_service(Trigger, "~/unhold", self._unhold_cb)
        self._srv_reset = self.create_service(Trigger, "~/reset", self._reset_cb)

        # Timer
        rate_hz = float(self._p("processing_rate_hz"))
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

        # --- Internal state ---
        self._frame_id = str(self._p("frame_id"))
        self._clock = time.monotonic
        self._last_ik_joints: Optional[np.ndarray] = None

        self.get_logger().info(f"Runtime node started at {rate_hz} Hz")

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _declare_params(self) -> None:
        defaults: dict[str, object] = {
            "processing_rate_hz": 60.0,
            "frame_id": "base",

            # Watchdog
            "watchdog.vision_fresh_s": 0.2,
            "watchdog.vision_stale_s": 0.5,
            "watchdog.communication_fresh_s": 0.2,
            "watchdog.communication_stale_s": 0.5,
            "watchdog.ik_fresh_s": 0.5,

            # Safety
            "safety.max_vision_lost_frames": 10,
            "safety.max_ik_failures": 5,

            # Baseline
            "baseline.max_pose_age_s": 0.2,
            "baseline.max_joint_age_s": 0.2,

            # Follow mapper
            "follow.axis_sign_x": 1,
            "follow.axis_sign_y": 1,
            "follow.axis_sign_z": 1,
            "follow.position_scale": 1.0,

            # Task projector
            "projector.max_position_step_m": 0.05,
            "projector.max_yaw_step_rad": 0.1,

            # Topics
            "topics.camera_pose": "vision/camera_pose",
            "topics.joint_state": "kinematics/joint_state",
            "topics.link_state": "communication/link_state",
            "topics.kinematics_target": "kinematics/target",
            "topics.ctrl_mode": "communication/ctrl_mode",
            "topics.state": "runtime/state",
            "topics.diagnostics": "runtime/diagnostics",
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _p(self, name: str) -> object:
        return self.get_parameter(name).value

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _pose_callback(self, msg: PoseStamped) -> None:
        """Cache the latest camera pose (with quality gate checked upstream)."""
        now = self._clock()
        pose = PoseSnapshot(
            timestamp_s=now,
            position=np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ], dtype=np.float64),
            quaternion_xyzw=np.array(
                _quat_from_msg(msg.pose.orientation), dtype=np.float64
            ),
            valid=True,
        )
        self._context.set_pose(pose)

    def _joint_callback(self, msg: JointState) -> None:
        """Cache the latest joint state from communication."""
        now = self._clock()
        positions = np.array(msg.position[:4], dtype=np.float64)
        velocities = np.array(msg.velocity[:4], dtype=np.float64)

        # Pad if shorter
        if len(positions) < 4:
            positions = np.pad(positions, (0, 4 - len(positions)))
        if len(velocities) < 4:
            velocities = np.pad(velocities, (0, 4 - len(velocities)))

        joints = JointSnapshot(
            timestamp_s=now,
            positions_rad=positions,
            velocities_rad_per_s=velocities,
            any_error=False,
        )
        self._context.set_joints(joints)

    def _link_callback(self, msg: Bool) -> None:
        self._context.link_ok = msg.data

    # ------------------------------------------------------------------
    # Service callbacks
    # ------------------------------------------------------------------

    def _start_cb(self, request, response):
        try:
            change = self._fsm.transition(Transition.START, self._clock())
            self.get_logger().info(f"State: {change.previous.value} → {change.current.value}")

            if self._fsm.state == SystemState.FOLLOW:
                # Try to capture baseline immediately
                captured = self._capture_baseline()
                response.success = captured
                response.message = "FOLLOW started" if captured else "FOLLOW started (baseline pending)"
            else:
                response.success = True
                response.message = f"State: {self._fsm.state.value}"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _hold_cb(self, request, response):
        try:
            change = self._fsm.transition(Transition.HOLD, self._clock())
            response.success = True
            response.message = f"HOLD → {change.current.value}"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _unhold_cb(self, request, response):
        try:
            change = self._fsm.transition(Transition.UNHOLD, self._clock())
            response.success = True
            response.message = f"UNHOLD → {change.current.value}"
            # Re-capture baseline
            if self._fsm.state == SystemState.FOLLOW:
                self._capture_baseline()
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _reset_cb(self, request, response):
        try:
            change = self._fsm.transition(Transition.RESET, self._clock())
            self._context.clear_baseline()
            self._watchdog.reset()
            response.success = True
            response.message = f"RESET → {change.current.value}"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    # ------------------------------------------------------------------
    # Timer tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now = self._clock()

        # If all subsystems are online → READY
        if self._fsm.state == SystemState.INIT:
            if self._context.latest_pose is not None and self._context.latest_joints is not None:
                try:
                    self._fsm.transition(Transition.SYSTEMS_READY, now)
                    self.get_logger().info("All systems ready → READY")
                except Exception:
                    pass

        # Safety evaluation
        result = self._safety.evaluate()

        # Always publish control mode and state
        self._publish_ctrl_mode(result.action)
        self._publish_state()

        # Publish task target if permitted
        if result.action == ControlAction.PERMIT:
            self._publish_follow_target(now)
        else:
            # If HOLD or SAFE, keep publishing last valid target (or zero)
            if result.action == ControlAction.SAFE:
                self._publish_safe_target(now)

        # Diagnostics
        diag = self._diagnostics.build(
            state=self._fsm.state,
            action=result.action,
            vision_status=self._watchdog.vision_status(),
            comm_status=self._watchdog.communication_status(),
            consecutive_vision_lost=self._watchdog.consecutive_vision_lost,
            consecutive_ik_failures=self._watchdog.consecutive_ik_failures,
        )
        self._diag_pub.publish(diag)

    # ------------------------------------------------------------------
    # Target generation
    # ------------------------------------------------------------------

    def _publish_follow_target(self, now: float) -> None:
        """Compute and publish a 4-DOF follow target from camera delta."""
        pose = self._context.get_pose()
        if pose is None or not pose.valid:
            self._watchdog.on_vision_lost()
            return

        self._watchdog.on_vision_ok()

        # Ensure baseline exists
        if not self._context.has_baseline():
            self._capture_baseline()

        # Map camera delta → base-frame 4-DOF target
        baseline_pose = self._context.baseline_pose
        baseline_ee_pos = self._context.baseline_ee_position_m
        baseline_ee_yaw = self._context.baseline_ee_yaw_rad

        if baseline_pose is None or baseline_ee_pos is None or baseline_ee_yaw is None:
            return

        delta_pos_base, delta_yaw_base = self._mapper.map(
            baseline_pose.position,
            baseline_pose.quaternion_xyzw,
            pose.position,
            pose.quaternion_xyzw,
        )

        # Add delta to baseline end-effector position
        target_pos = baseline_ee_pos + delta_pos_base
        target_yaw = baseline_ee_yaw + delta_yaw_base

        # Project (clip)
        clipped_pos, clipped_yaw = self._projector.project(
            delta_pos_base, delta_yaw_base,
        )
        final_pos = baseline_ee_pos + clipped_pos
        final_yaw = baseline_ee_yaw + clipped_yaw

        self._publish_pose_target(now, final_pos, final_yaw)

    def _publish_safe_target(self, now: float) -> None:
        """Publish a zero-velocity hold target (last known)."""
        # Use last known IK result as hold target
        pose = self._context.get_pose()
        if pose is None:
            return

        # Publish current position as target (freeze)
        self._publish_pose_target(now, pose.position, 0.0)

    def _publish_pose_target(
        self,
        now: float,
        pos_m: np.ndarray,
        yaw_rad: float,
    ) -> None:
        """Publish a 4-DOF PoseStamped [x, y, z, yaw about base Z]."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = float(pos_m[0])
        msg.pose.position.y = float(pos_m[1])
        msg.pose.position.z = float(pos_m[2])

        # Represent yaw as quaternion about Z
        cy = math.cos(yaw_rad / 2.0)
        sy = math.sin(yaw_rad / 2.0)
        msg.pose.orientation.z = float(sy)
        msg.pose.orientation.w = float(cy)
        # x=y=0 (pure Z rotation)

        self._target_pub.publish(msg)

    def _capture_baseline(self) -> bool:
        """Attempt to capture the FOLLOW baseline from current FK result."""
        joints = self._context.get_joints()
        if joints is None:
            return False

        # Approximate FK: we need the end-effector position and yaw.
        # This requires calling the forward kinematics. Since we don't
        # have a direct FK API here, we use the last published IK result
        # as the current end-effector pose. In production this should
        # call the FK solver directly.
        if self._last_ik_joints is None:
            # No previous FK result; can't baseline
            return False

        # The end-effector pose from FK is stored as the baseline target.
        # For now we use a placeholder: the FK is computed by the kinematics
        # package and published as joint_command—but we need FK, not IK.
        #
        # We'll capture based on the latest known joint positions and use
        # a simple approximation: the last target position is the FK.
        # TODO: replace with actual FK call when kinematics package exposes
        #       FK as a ROS topic or library call.

        # For now, use the last known IK result as baseline EE position.
        # This assumes the arm is at the last commanded position.
        ee_pos = self._last_ik_joints.copy()  # placeholder (needs FK)
        ee_yaw = 0.0

        pose = self._context.get_pose()
        if pose is None:
            return False

        return self._baseline.capture(ee_pos, ee_yaw)

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_ctrl_mode(self, action: ControlAction) -> None:
        msg = UInt8()
        if action == ControlAction.SAFE:
            msg.data = 0x00  # Weak mode (disable motors)
        else:
            msg.data = 0x02  # MIT mode
        self._ctrl_mode_pub.publish(msg)

    def _publish_state(self) -> None:
        msg = String()
        msg.data = self._fsm.state.value
        self._state_pub.publish(msg)

    def destroy_node(self) -> bool:
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[RuntimeNode] = None
    try:
        node = RuntimeNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

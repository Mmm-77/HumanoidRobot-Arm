"""ROS 2 node: forward joint commands to STM32 and publish feedback.

Pipeline:
  JointTrajectory (kinematics) → command_codec → serial_transport → STM32
  STM32 → serial_transport → frame_parser → feedback_codec → JointState
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, Optional

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, UInt8
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .command_codec import CommandFrame, CodecError, encode_command
from .feedback_codec import FeedbackError, FeedbackFrame, decode_feedback
from .frame_parser import FrameParser
from .protocol import (
    ControlMode,
    DOWNLOAD_FRAME_SIZE,
    MotorErrorBits,
    RIGHT_ARM_MOTORS,
    UPLOAD_FRAME_SIZE,
)
from .reconnect_manager import (
    LinkState,
    ReconnectConfig,
    ReconnectManager,
)
from .serial_transport import SerialConfig, SerialError, SerialTransport


class CommunicationNode(Node):
    """ROS 2 node for USB serial communication with the STM32 lower computer.

    Subscriptions:
      - ``kinematics/joint_command`` (JointTrajectory)
      - ``communication/ctrl_mode`` (UInt8)

    Publications:
      - ``kinematics/joint_state`` (JointState)
      - ``communication/link_state`` (Bool)
      - ``communication/diagnostics`` (DiagnosticArray)
    """

    def __init__(self) -> None:
        super().__init__("communication_node")
        self._declare_parameters()

        # --- Serial transport ---
        self._transport = SerialTransport(
            SerialConfig(
                port=str(self._value("serial.port")),
                baudrate=int(self._value("serial.baudrate")),
                timeout=float(self._value("serial.read_timeout_s")),
                write_timeout=float(self._value("serial.write_timeout_s")),
            )
        )
        self._parser = FrameParser(UPLOAD_FRAME_SIZE)

        # --- Reconnect manager ---
        self._reconnect = ReconnectManager(
            ReconnectConfig(
                feedback_timeout_s=float(self._value("reconnect.feedback_timeout_s")),
                degraded_threshold_s=float(
                    self._value("reconnect.degraded_threshold_s")
                ),
                initial_backoff_s=float(self._value("reconnect.initial_backoff_s")),
                max_backoff_s=float(self._value("reconnect.max_backoff_s")),
                backoff_multiplier=float(
                    self._value("reconnect.backoff_multiplier")
                ),
                max_consecutive_crc_failures=int(
                    self._value("reconnect.max_consecutive_crc_failures")
                ),
            )
        )

        # --- MIT gains ---
        self._kp = np.array([
            float(self._value("mit.kp_1")),
            float(self._value("mit.kp_2")),
            float(self._value("mit.kp_3")),
            float(self._value("mit.kp_4")),
        ], dtype=np.float32)
        self._kd = np.array([
            float(self._value("mit.kd_1")),
            float(self._value("mit.kd_2")),
            float(self._value("mit.kd_3")),
            float(self._value("mit.kd_4")),
        ], dtype=np.float32)
        self._tff = np.array([
            float(self._value("mit.tff_1")),
            float(self._value("mit.tff_2")),
            float(self._value("mit.tff_3")),
            float(self._value("mit.tff_4")),
        ], dtype=np.float32)

        # --- ROS 2 interfaces ---
        state_qos = QoSProfile(depth=5)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self._command_sub = self.create_subscription(
            JointTrajectory,
            str(self._value("topics.joint_command")),
            self._command_callback,
            state_qos,
        )
        self._ctrl_mode_sub = self.create_subscription(
            UInt8,
            str(self._value("topics.ctrl_mode")),
            self._ctrl_mode_callback,
            state_qos,
        )

        self._joint_state_pub = self.create_publisher(
            JointState,
            str(self._value("topics.joint_state")),
            sensor_qos,
        )
        self._link_state_pub = self.create_publisher(
            Bool,
            str(self._value("topics.link_state")),
            state_qos,
        )
        self._diag_pub = self.create_publisher(
            DiagnosticArray,
            str(self._value("topics.diagnostics")),
            state_qos,
        )

        self._frame_id = str(self._value("frame_id"))

        # --- Timer ---
        rate_hz = float(self._value("processing_rate_hz"))
        self._timer = self.create_timer(1.0 / rate_hz, self._timer_callback)

        # --- State ---
        self._latest_command: Optional[CommandFrame] = None
        self._ctrl_mode: ControlMode = ControlMode.MIT
        self._frame_count = 0
        self._crc_fail_count = 0

        # Try to open serial initially
        self._try_open()

        self.get_logger().info(
            f"Communication node running at {rate_hz} Hz, "
            f"serial: {self._value('serial.port')} @ {self._value('serial.baudrate')}"
        )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _declare_parameters(self) -> None:
        defaults: dict[str, object] = {
            "serial.port": "/dev/ttyACM0",
            "serial.baudrate": 115200,
            "serial.read_timeout_s": 0.01,
            "serial.write_timeout_s": 0.05,
            "processing_rate_hz": 200.0,
            "reconnect.feedback_timeout_s": 0.5,
            "reconnect.degraded_threshold_s": 0.2,
            "reconnect.initial_backoff_s": 0.5,
            "reconnect.max_backoff_s": 10.0,
            "reconnect.backoff_multiplier": 2.0,
            "reconnect.max_consecutive_crc_failures": 5,
            "mit.kp_1": 1.0,
            "mit.kp_2": 1.0,
            "mit.kp_3": 1.0,
            "mit.kp_4": 1.0,
            "mit.kd_1": 0.05,
            "mit.kd_2": 0.05,
            "mit.kd_3": 0.05,
            "mit.kd_4": 0.05,
            "mit.tff_1": 0.0,
            "mit.tff_2": 0.0,
            "mit.tff_3": 0.0,
            "mit.tff_4": 0.0,
            "topics.joint_command": "kinematics/joint_command",
            "topics.joint_state": "kinematics/joint_state",
            "topics.ctrl_mode": "communication/ctrl_mode",
            "topics.link_state": "communication/link_state",
            "topics.diagnostics": "communication/diagnostics",
            "frame_id": "base",
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _value(self, name: str) -> Any:
        return self.get_parameter(name).value

    # ------------------------------------------------------------------
    # Transport management
    # ------------------------------------------------------------------

    def _try_open(self) -> bool:
        try:
            self._transport.open()
            self._parser.reset()
            self._reconnect.on_reconnect_succeeded()
            self.get_logger().info(
                f"Opened serial port {self._value('serial.port')}"
            )
            return True
        except SerialError as exc:
            self.get_logger().warn(f"Serial open failed: {exc}")
            self._reconnect.on_reconnect_failed()
            return False

    def _ensure_open(self) -> bool:
        if self._transport.is_open:
            return True
        if self._reconnect.should_reconnect:
            return self._try_open()
        return False

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _command_callback(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return
        point: JointTrajectoryPoint = msg.points[0]

        positions = np.array(point.positions[:4], dtype=np.float64)
        velocities = (
            np.array(point.velocities[:4], dtype=np.float64)
            if len(point.velocities) >= 4
            else np.zeros(4, dtype=np.float64)
        )

        # Pad to 4 elements if shorter
        if len(positions) < 4:
            positions = np.pad(positions, (0, 4 - len(positions)))
        if len(velocities) < 4:
            velocities = np.pad(velocities, (0, 4 - len(velocities)))

        try:
            self._latest_command = encode_command(
                positions,
                velocities,
                ctrl_mode=self._ctrl_mode,
                kp=self._kp,
                kd=self._kd,
                tff=self._tff,
            )
        except CodecError as exc:
            self.get_logger().error(f"Command encoding failed: {exc}")

    def _ctrl_mode_callback(self, msg: UInt8) -> None:
        try:
            self._ctrl_mode = ControlMode(msg.data)
        except ValueError:
            self.get_logger().warn(f"Unknown control mode: {msg.data}")

    # ------------------------------------------------------------------
    # Timer callback (main I/O loop)
    # ------------------------------------------------------------------

    def _timer_callback(self) -> None:
        if not self._ensure_open():
            self._publish_diagnostics("disconnected", DiagnosticStatus.WARN)
            return

        # --- Write (if we have a command) ---
        if self._latest_command is not None:
            try:
                self._transport.write(self._latest_command.data)
            except SerialError as exc:
                self.get_logger().error(f"Serial write error: {exc}")
                self._transport.close()
                self._reconnect.on_reconnect_failed()
                return

        # --- Read (collect incoming bytes) ---
        try:
            raw = self._transport.read_available()
        except SerialError as exc:
            self.get_logger().error(f"Serial read error: {exc}")
            self._transport.close()
            self._reconnect.on_reconnect_failed()
            return

        if not raw:
            # No data yet – check link health
            if self._reconnect.link_state == LinkState.DISCONNECTED:
                self._transport.close()
                self._reconnect.on_reconnect_failed()
                self._publish_diagnostics("feedback_timeout", DiagnosticStatus.WARN)
            else:
                self._publish_diagnostics("no_data", DiagnosticStatus.OK)
            return

        # --- Parse frames ---
        frames = self._parser.feed(raw)
        for frame in frames:
            try:
                feedback = decode_feedback(frame, verify_crc=True)
            except FeedbackError:
                continue

            if feedback is None:
                # CRC failure
                self._crc_fail_count += 1
                self._reconnect.on_crc_failure()
                continue

            self._reconnect.on_feedback_received()
            self._frame_count += 1

            # Publish JointState
            self._publish_joint_state(feedback)

        # --- Link diagnostics ---
        self._publish_link_state()
        self._publish_diagnostics("ok", DiagnosticStatus.OK)

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------

    def _publish_joint_state(self, feedback: FeedbackFrame) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.name = [
            "joint_1", "joint_2", "joint_3", "joint_4",
        ]
        msg.position = feedback.joint_angles_rad.tolist()
        msg.velocity = feedback.joint_velocities_rad_per_s.tolist()
        msg.effort = feedback.joint_torque_nm.tolist()
        self._joint_state_pub.publish(msg)

    def _publish_link_state(self) -> None:
        msg = Bool()
        msg.data = self._reconnect.link_state == LinkState.CONNECTED
        self._link_state_pub.publish(msg)

    def _publish_diagnostics(self, reason: str, level: int) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.name = "humanoid_arm_communication/link"
        status.hardware_id = str(self._value("serial.port"))
        status.message = reason
        status.level = level

        values = [
            ("reason", reason),
            ("link_state", self._reconnect.link_state.value),
            ("frames_received", str(self._frame_count)),
            ("crc_failures", str(self._crc_fail_count)),
        ]
        elapsed = self._reconnect.time_since_last_feedback
        if elapsed is not None:
            values.append(("time_since_last_feedback_s", f"{elapsed:.3f}"))

        status.values = [self._kv(k, v) for k, v in values]
        array.status = [status]
        self._diag_pub.publish(array)

    @staticmethod
    def _kv(key: str, value: str) -> KeyValue:
        item = KeyValue()
        item.key = key
        item.value = value
        return item

    def destroy_node(self) -> bool:
        self._transport.close()
        return super().destroy_node()


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node: Optional[CommunicationNode] = None
    try:
        node = CommunicationNode()
        rclpy.spin(node)
    except ValueError as exc:
        rclpy.logging.get_logger("communication_node").fatal(
            f"invalid communication configuration: {exc}"
        )
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

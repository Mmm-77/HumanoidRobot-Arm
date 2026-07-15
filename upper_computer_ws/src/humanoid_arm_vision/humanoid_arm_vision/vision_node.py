"""ROS 2 node that owns the complete camera-to-valid-pose vision pipeline."""

from __future__ import annotations

import os

# Ensure ROS 2 shared libraries are discoverable even when the process
# environment does not inherit LD_LIBRARY_PATH (e.g. sandboxed shells).
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
import time
from typing import Any, Iterable

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool

from .apriltag_detector import AprilTagConfig, AprilTagDetection, AprilTagDetector
from .camera_calibration import CalibrationError, CameraCalibration
from .camera_driver import CameraConfig, CameraError, CameraFrame, RealSenseCamera
from .pose_filter import PoseFilter, PoseFilterConfig
from .pose_solver import AprilTagPoseSolver, PoseEstimate, PoseSolverError
from .quality_gate import PoseQualityGate, QualityConfig, QualityDecision, QualityReason


class VisionNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_node")
        self._declare_parameters()
        self._tag_frame = str(self._value("frames.tag"))
        self._camera_frame = str(self._value("frames.camera"))
        if not self._tag_frame or not self._camera_frame:
            raise ValueError("tag and camera frame ids must not be empty")

        camera_config = CameraConfig(
            serial_number=str(self._value("camera.serial_number")).strip(),
            width=int(self._value("camera.width")),
            height=int(self._value("camera.height")),
            fps=int(self._value("camera.fps")),
            frame_timeout_ms=int(self._value("camera.frame_timeout_ms")),
            reopen_after_failures=int(self._value("camera.reopen_after_failures")),
            reopen_delay_s=float(self._value("camera.reopen_delay_s")),
        )
        self._camera = RealSenseCamera(camera_config)
        self._detector = AprilTagDetector(
            AprilTagConfig(
                family=str(self._value("tag.family")),
                target_id=int(self._value("tag.id")),
                corner_refinement=bool(self._value("tag.corner_refinement")),
                quad_decimate=float(self._value("tag.quad_decimate")),
            )
        )
        self._solver = AprilTagPoseSolver(float(self._value("tag.size_m")))
        self._quality_gate = PoseQualityGate(
            QualityConfig(
                max_reprojection_error_px=float(
                    self._value("quality.max_reprojection_error_px")
                ),
                min_tag_area_ratio=float(self._value("quality.min_tag_area_ratio")),
                min_decision_margin=float(self._value("quality.min_decision_margin")),
                min_camera_distance_m=float(
                    self._value("quality.min_camera_distance_m")
                ),
                max_camera_distance_m=float(
                    self._value("quality.max_camera_distance_m")
                ),
                max_position_jump_m=float(self._value("quality.max_position_jump_m")),
                max_orientation_jump_rad=float(
                    self._value("quality.max_orientation_jump_rad")
                ),
                max_frame_age_s=float(self._value("quality.max_frame_age_s")),
                continuity_reset_s=float(self._value("quality.continuity_reset_s")),
            )
        )
        self._pose_filter = PoseFilter(
            PoseFilterConfig(
                position_alpha=float(self._value("filter.position_alpha")),
                orientation_alpha=float(self._value("filter.orientation_alpha")),
                reset_gap_s=float(self._value("filter.reset_gap_s")),
            )
        )
        self._bridge = CvBridge()
        self._publish_raw = bool(self._value("publish_raw_image"))
        self._publish_debug = bool(self._value("publish_debug_image"))

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        state_qos = QoSProfile(depth=5)
        self._pose_publisher = self.create_publisher(
            PoseStamped, str(self._value("topics.pose")), state_qos
        )
        self._valid_publisher = self.create_publisher(
            Bool, str(self._value("topics.valid")), state_qos
        )
        self._camera_info_publisher = self.create_publisher(
            CameraInfo, str(self._value("topics.camera_info")), sensor_qos
        )
        self._diagnostics_publisher = self.create_publisher(
            DiagnosticArray, str(self._value("topics.diagnostics")), state_qos
        )
        self._raw_image_publisher = (
            self.create_publisher(
                Image, str(self._value("topics.raw_image")), sensor_qos
            )
            if self._publish_raw
            else None
        )
        self._debug_image_publisher = (
            self.create_publisher(
                Image, str(self._value("topics.debug_image")), sensor_qos
            )
            if self._publish_debug
            else None
        )

        processing_rate_hz = float(self._value("processing_rate_hz"))
        if not math.isfinite(processing_rate_hz) or processing_rate_hz <= 0.0:
            raise ValueError("processing_rate_hz must be a positive finite value")
        self._timer = self.create_timer(
            1.0 / processing_rate_hz, self._process_realsense_frame
        )
        self.get_logger().info(
            "vision pipeline using the RealSense SDK, configured for tag "
            f"{self._detector.config.family}:{self._detector.config.target_id} "
            f"in frame {self._tag_frame!r}"
        )

    def _declare_parameters(self) -> None:
        defaults: dict[str, object] = {
            "camera.serial_number": "",
            "camera.width": 640,
            "camera.height": 480,
            "camera.fps": 30,
            "camera.frame_timeout_ms": 1000,
            "camera.reopen_after_failures": 3,
            "camera.reopen_delay_s": 1.0,
            "processing_rate_hz": 30.0,
            "tag.family": "tag36h11",
            "tag.id": 0,
            "tag.size_m": 0.0,
            "tag.corner_refinement": True,
            "tag.quad_decimate": 1.0,
            "quality.max_reprojection_error_px": 2.5,
            "quality.min_tag_area_ratio": 0.002,
            "quality.min_decision_margin": 0.0,
            "quality.min_camera_distance_m": 0.05,
            "quality.max_camera_distance_m": 3.0,
            "quality.max_position_jump_m": 0.20,
            "quality.max_orientation_jump_rad": 0.70,
            "quality.max_frame_age_s": 0.20,
            "quality.continuity_reset_s": 0.50,
            "filter.position_alpha": 0.35,
            "filter.orientation_alpha": 0.30,
            "filter.reset_gap_s": 0.50,
            "frames.tag": "tag",
            "frames.camera": "camera",
            "topics.pose": "vision/camera_pose",
            "topics.valid": "vision/valid",
            "topics.camera_info": "vision/camera_info",
            "topics.diagnostics": "vision/diagnostics",
            "topics.raw_image": "vision/raw_image",
            "topics.debug_image": "vision/debug_image",
            "publish_raw_image": False,
            "publish_debug_image": False,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)

    def _value(self, name: str) -> Any:
        return self.get_parameter(name).value

    def _process_realsense_frame(self) -> None:
        try:
            frame = self._camera.read()
        except CameraError as exc:
            self._publish_invalid(QualityReason.CAMERA_ERROR, detail=str(exc))
            return
        stamp = self.get_clock().now().to_msg()
        self._process_image(
            frame, self._camera.calibration, stamp=stamp, now_s=time.monotonic()
        )

    def _process_image(
        self,
        frame: CameraFrame,
        calibration: CameraCalibration,
        *,
        stamp,
        now_s: float,
    ) -> None:
        if self._raw_image_publisher is not None:
            self._publish_image(self._raw_image_publisher, frame.image, stamp)
        self._camera_info_publisher.publish(self._camera_info(calibration, stamp))

        detection: AprilTagDetection | None = None
        estimate: PoseEstimate | None = None
        detail = ""
        try:
            detection = self._detector.detect(frame.image)
            if detection is not None:
                estimate = self._solver.solve(detection, calibration)
        except PoseSolverError as exc:
            detail = str(exc)
        except Exception as exc:  # Keep a malformed frame from killing the ROS node.
            decision = PoseQualityGate.invalid(QualityReason.INTERNAL_ERROR)
            self._publish_result(
                decision,
                frame=frame,
                stamp=stamp,
                detection=detection,
                detail=f"{type(exc).__name__}: {exc}",
            )
            return

        height, width = frame.image.shape[:2]
        decision = self._quality_gate.evaluate(
            detection,
            estimate,
            image_width=width,
            image_height=height,
            capture_time_s=frame.capture_time_s,
            now_s=now_s,
        )
        if decision.valid:
            assert estimate is not None
            try:
                filtered = self._pose_filter.update(
                    estimate.position, estimate.orientation_xyzw, frame.capture_time_s
                )
            except ValueError:
                # A rosbag loop or /clock reset can move time backwards. A fresh
                # filter state avoids mixing samples from different time epochs.
                self._pose_filter.reset()
                self._quality_gate.reset()
                filtered = self._pose_filter.update(
                    estimate.position, estimate.orientation_xyzw, frame.capture_time_s
                )
            self._pose_publisher.publish(
                self._pose_message(filtered.position, filtered.orientation_xyzw, stamp)
            )
        self._publish_result(
            decision,
            frame=frame,
            stamp=stamp,
            detection=detection,
            estimate=estimate,
            detail=detail,
        )

    def _publish_invalid(
        self, reason: QualityReason, *, detail: str = "", stamp=None
    ) -> None:
        if stamp is None:
            stamp = self.get_clock().now().to_msg()
        self._publish_result(
            PoseQualityGate.invalid(reason), stamp=stamp, detail=detail
        )

    def _publish_result(
        self,
        decision: QualityDecision,
        *,
        stamp,
        frame: CameraFrame | None = None,
        detection: AprilTagDetection | None = None,
        estimate: PoseEstimate | None = None,
        detail: str = "",
    ) -> None:
        valid_message = Bool()
        valid_message.data = decision.valid
        self._valid_publisher.publish(valid_message)
        self._diagnostics_publisher.publish(
            self._diagnostics(
                decision,
                stamp=stamp,
                frame=frame,
                detection=detection,
                estimate=estimate,
                detail=detail,
            )
        )
        if frame is not None and self._debug_image_publisher is not None:
            debug_image = self._annotate(frame.image, detection, decision.reason.value)
            self._publish_image(self._debug_image_publisher, debug_image, stamp)

    def _pose_message(
        self, position: np.ndarray, orientation: np.ndarray, stamp
    ) -> PoseStamped:
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self._tag_frame
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        message.pose.orientation.x = float(orientation[0])
        message.pose.orientation.y = float(orientation[1])
        message.pose.orientation.z = float(orientation[2])
        message.pose.orientation.w = float(orientation[3])
        return message

    def _camera_info(self, calibration: CameraCalibration, stamp) -> CameraInfo:
        message = CameraInfo()
        message.header.stamp = stamp
        message.header.frame_id = self._camera_frame
        message.width = calibration.image_width
        message.height = calibration.image_height
        message.distortion_model = calibration.distortion_model
        message.d = calibration.distortion_coefficients.tolist()
        message.k = calibration.camera_matrix.reshape(-1).tolist()
        message.r = np.eye(3, dtype=float).reshape(-1).tolist()
        projection = np.zeros((3, 4), dtype=float)
        projection[:3, :3] = calibration.camera_matrix
        message.p = projection.reshape(-1).tolist()
        return message

    def _diagnostics(
        self,
        decision: QualityDecision,
        *,
        stamp,
        frame: CameraFrame | None,
        detection: AprilTagDetection | None,
        estimate: PoseEstimate | None,
        detail: str,
    ) -> DiagnosticArray:
        array = DiagnosticArray()
        array.header.stamp = stamp
        status = DiagnosticStatus()
        status.name = "humanoid_arm_vision/pipeline"
        status.hardware_id = self._camera.hardware_id
        status.message = decision.reason.value
        if decision.valid:
            status.level = DiagnosticStatus.OK
        elif decision.reason in {
            QualityReason.CAMERA_ERROR,
            QualityReason.CALIBRATION_ERROR,
            QualityReason.INTERNAL_ERROR,
        }:
            status.level = DiagnosticStatus.ERROR
        else:
            status.level = DiagnosticStatus.WARN

        values: list[tuple[str, object]] = [
            ("valid", decision.valid),
            ("reason", decision.reason.value),
            ("tag_family", self._detector.config.family),
            ("target_tag_id", self._detector.config.target_id),
        ]
        if frame is not None:
            values.append(("frame_sequence", frame.sequence))
        if detection is not None:
            values.extend(
                [
                    ("detected_tag_id", detection.tag_id),
                    ("tag_pixel_area", detection.pixel_area),
                    ("tag_perimeter_px", detection.perimeter_px),
                ]
            )
        if estimate is not None:
            values.append(("reprojection_error_px", estimate.reprojection_error_px))
            values.append(("camera_distance_m", estimate.camera_distance_m))
        values.extend(decision.metrics.items())
        if detail:
            values.append(("detail", detail))
        status.values = [self._key_value(key, value) for key, value in values]
        array.status = [status]
        return array

    @staticmethod
    def _key_value(key: str, value: object) -> KeyValue:
        item = KeyValue()
        item.key = str(key)
        item.value = str(value)
        return item

    def _publish_image(self, publisher, image: np.ndarray, stamp) -> None:
        if image.ndim == 2:
            encoding = "mono8"
        elif image.shape[2] == 4:
            encoding = "bgra8"
        else:
            encoding = "bgr8"
        # cv_bridge in Foxy is incompatible with OpenCV >= 5; use passthrough
        # and set the encoding manually.
        message = self._bridge.cv2_to_imgmsg(image, encoding="passthrough")
        message.encoding = encoding
        message.header.stamp = stamp
        message.header.frame_id = self._camera_frame
        publisher.publish(message)

    @staticmethod
    def _annotate(
        image: np.ndarray, detection: AprilTagDetection | None, label: str
    ) -> np.ndarray:
        if image.ndim == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            output = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            output = image.copy()
        color = (0, 200, 0) if label == QualityReason.VALID.value else (0, 0, 255)
        if detection is not None:
            corners = np.rint(detection.corners).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [corners], True, color, 2, cv2.LINE_AA)
            center = tuple(np.rint(detection.center).astype(int))
            cv2.circle(output, center, 3, color, -1)
        cv2.putText(output, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return output

    def destroy_node(self) -> bool:
        self._camera.close()
        return super().destroy_node()


def main(args: Iterable[str] | None = None) -> None:
    rclpy.init(args=args)
    node: VisionNode | None = None
    try:
        node = VisionNode()
        rclpy.spin(node)
    except (CalibrationError, CameraError, ValueError) as exc:
        rclpy.logging.get_logger("vision_node").fatal(
            f"invalid vision configuration: {exc}"
        )
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

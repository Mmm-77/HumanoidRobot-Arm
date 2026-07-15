#!/usr/bin/env python3
"""AprilTag 位姿可视化工具 —— 独立验证程序，监听 ROS 话题并通过 OpenCV 窗口显示。

订阅话题:
  - vision/debug_image  (sensor_msgs/Image)     已标注 AprilTag 框的调试图像
  - vision/camera_pose  (geometry_msgs/PoseStamped)  相机在标签坐标系下的 6-DOF 位姿

显示内容:
  - 彩色视频流
  - 右上角叠加 6-DOF 位姿文字 (位置: m, 姿态: deg)
  - 图像中已有 AprilTag 红/绿框 (由 debug_image 提供)
"""

from __future__ import annotations

import math
import signal
import sys
import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def quaternion_to_euler_zyx(
    x: float, y: float, z: float, w: float
) -> tuple[float, float, float]:
    """将四元数 (ROS 顺序: x, y, z, w) 转换为 ZYX 欧拉角 (roll, pitch, yaw)，单位弧度。"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class PoseVisualizer(Node):
    """监听 debug_image 和 camera_pose 话题，在 OpenCV 窗口中叠加位姿信息。"""

    WINDOW_NAME = "AprilTag Pose Visualizer"

    def __init__(self) -> None:
        super().__init__("pose_visualizer")

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        # 最新数据缓存
        self._latest_image: Optional[np.ndarray] = None
        self._latest_position: Optional[tuple[float, float, float]] = None
        self._latest_euler_deg: Optional[tuple[float, float, float]] = None
        self._pose_valid: bool = False
        self._image_received: bool = False

        # 订阅 debug_image (已画好 AprilTag 框) — 与 vision_node 的 sensor_qos 匹配
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Image, "vision/debug_image", self._image_callback, sensor_qos
        )
        # 订阅 camera_pose — 与 vision_node 的 state_qos 匹配
        self.create_subscription(
            PoseStamped, "vision/camera_pose", self._pose_callback, 10
        )

        self._image_count = 0
        self._pose_count = 0

        # 定时诊断日志，帮助排查回调是否被触发
        self._diag_timer = self.create_timer(2.0, self._diag_callback)

        self.get_logger().info("PoseVisualizer 已启动，按 ESC 或关闭窗口退出")
        self.get_logger().info(
            "订阅话题: vision/debug_image, vision/camera_pose"
        )

    def _diag_callback(self) -> None:
        with self._lock:
            img_ok = self._latest_image is not None
            pose_ok = self._latest_position is not None
            img_cnt = self._image_count
            pose_cnt = self._pose_count
        self.get_logger().info(
            f"状态: image={'OK' if img_ok else 'WAIT'}({img_cnt})  "
            f"pose={'OK' if pose_ok else 'WAIT'}({pose_cnt})"
        )

    def _image_callback(self, msg: Image) -> None:
        self._image_count += 1
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"图像转换失败: {exc}")
            return
        with self._lock:
            self._latest_image = image
            self._image_received = True

    def _pose_callback(self, msg: PoseStamped) -> None:
        self._pose_count += 1
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        roll, pitch, yaw = quaternion_to_euler_zyx(qx, qy, qz, qw)
        with self._lock:
            self._latest_position = (px, py, pz)
            self._latest_euler_deg = (
                math.degrees(roll),
                math.degrees(pitch),
                math.degrees(yaw),
            )
            self._pose_valid = True

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """获取最新图像帧并叠加位姿文字，返回带标注的副本。"""
        with self._lock:
            if self._latest_image is None:
                return None
            display = self._latest_image.copy()
            position = self._latest_position
            euler = self._latest_euler_deg

        self._overlay_pose(display, position, euler)
        return display

    @staticmethod
    def _overlay_pose(
        image: np.ndarray,
        position: Optional[tuple[float, float, float]],
        euler_deg: Optional[tuple[float, float, float]],
    ) -> None:
        """在图像右上角叠加位姿文字。"""
        h, w = image.shape[:2]

        # 半透明背景面板
        panel_w = 280
        panel_h = 130
        panel_x = w - panel_w - 10
        panel_y = 10
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (30, 30, 30),
            -1,
        )
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        line_spacing = 22
        text_color = (220, 220, 220)
        label_color = (150, 150, 150)

        x0 = panel_x + 10
        y = panel_y + 20

        # 标题
        cv2.putText(
            image, "Camera Pose (tag frame)", (x0, y),
            font, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )
        y += line_spacing

        if position is None:
            cv2.putText(
                image, "Waiting for pose...", (x0, y),
                font, font_scale, (120, 120, 120), thickness, cv2.LINE_AA,
            )
            return

        # 位置 (m)
        cv2.putText(
            image, "Pos (m):", (x0, y),
            font, font_scale, label_color, thickness, cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"x:{position[0]: 8.4f}  y:{position[1]: 8.4f}  z:{position[2]: 8.4f}",
            (x0 + 80, y),
            font, font_scale, text_color, thickness, cv2.LINE_AA,
        )
        y += line_spacing

        if euler_deg is None:
            return

        # 姿态 (deg)
        cv2.putText(
            image, "Ori (deg):", (x0, y),
            font, font_scale, label_color, thickness, cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"R:{euler_deg[0]: 7.2f}  P:{euler_deg[1]: 7.2f}  Y:{euler_deg[2]: 7.2f}",
            (x0 + 80, y),
            font, font_scale, text_color, thickness, cv2.LINE_AA,
        )


def _make_placeholder(width: int = 640, height: int = 480) -> np.ndarray:
    """生成等待提示画布。"""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    lines = [
        "Waiting for debug_image...",
        "",
        "Enable with:",
        "  ros2 param set /vision_node publish_debug_image true",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (180, 180, 180)
    y0 = height // 2 - 40
    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        y = y0 + i * 30
        cv2.putText(canvas, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return canvas


def main() -> None:
    rclpy.init(args=sys.argv)

    visualizer = PoseVisualizer()

    shutdown_flag = threading.Event()
    signal.signal(signal.SIGINT, lambda sig, frame: shutdown_flag.set())

    cv2.namedWindow(PoseVisualizer.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PoseVisualizer.WINDOW_NAME, 960, 720)

    window_shown = False   # 跟踪是否已调用过 imshow

    try:
        while rclpy.ok() and not shutdown_flag.is_set():
            rclpy.spin_once(visualizer, timeout_sec=0.01)

            frame = visualizer.get_latest_frame()
            if frame is not None:
                cv2.imshow(PoseVisualizer.WINDOW_NAME, frame)
                window_shown = True
            elif not window_shown:
                # 尚未收到图像时显示等待提示
                placeholder = _make_placeholder()
                cv2.imshow(PoseVisualizer.WINDOW_NAME, placeholder)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if window_shown:
                try:
                    if cv2.getWindowProperty(PoseVisualizer.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    finally:
        cv2.destroyAllWindows()
        visualizer.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
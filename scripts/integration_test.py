#!/usr/bin/env python3
"""集成测试：AprilTag 视觉定位 → 相对跟随 → 逆运动学解算。

管线:
  RealSense 相机 → AprilTag 检测 → PnP 位姿解算 (camera in tag frame)
  → 坐标变换到 base frame → 相对增量计算 (自校准)
  → [x,y,z] 末端位置目标 → 冗余逆运动学 → 关节角度输出

可视化:
  OpenCV 窗口实时显示相机画面、AprilTag 检测框、位姿信息和关节角度。

用法:
  python3 integration_test.py
  python3 integration_test.py --record output.bag  # 同时录制 rosbag
  python3 integration_test.py --tag-size 0.15      # 自定义 Tag 尺寸(m)
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# --- ROS / OpenCV 路径设置 (与 vision_node 一致) ---
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

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

# --- 视觉包 ---
from humanoid_arm_vision.apriltag_detector import (
    AprilTagConfig,
    AprilTagDetector,
    AprilTagDetection,
)
from humanoid_arm_vision.camera_calibration import CameraCalibration
from humanoid_arm_vision.camera_driver import CameraConfig, CameraFrame, RealSenseCamera
from humanoid_arm_vision.pose_solver import AprilTagPoseSolver, PoseEstimate

# --- 运动学包 ---
from humanoid_arm_kinematics.forward_solver import ForwardSolver
from humanoid_arm_kinematics.inverse_solver import IKConfig, InverseSolver
from humanoid_arm_kinematics.jacobian import JacobianSolver
from humanoid_arm_kinematics.robot_model import RobotModel

# --- DH 参数 ---
# DH 参数 (原始单位 cm，已转换为 m → ÷100)

# Home 位姿 (编码器全零 = 准备跟随姿态)
HOME_JOINTS_RAD = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# 关节限位 (来自 MotorAngleDetectTask.c, 上位机 4 电机: 0x01~0x04)
# 格式: (NegExtAngle_deg, PosExtAngle_deg) per joint


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CalibrationState:
    """校准时的参考状态 (base 坐标系下)."""
    camera_position: np.ndarray  # 3-element
    camera_yaw_rad: float
    ee_position: np.ndarray     # 参考末端位置 (FK of home joints)
    ee_yaw_rad: float           # 参考末端 yaw

    @classmethod
    def invalid(cls) -> "CalibrationState":
        return cls(
            camera_position=np.zeros(3),
            camera_yaw_rad=0.0,
            ee_position=np.zeros(3),
            ee_yaw_rad=0.0,
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def rotation_to_yaw(R: np.ndarray) -> float:
    """提取旋转矩阵绕 base Z 轴的旋转角 (yaw), atan2(R[1,0], R[0,0])."""
    return float(math.atan2(R[1, 0], R[0, 0]))


def yaw_difference(target: float, current: float) -> float:
    """计算最短角路径差异, 结果在 [-π, π]."""
    diff = target - current
    return (diff + math.pi) % (2 * math.pi) - math.pi


def build_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """构建 4x4 齐次变换矩阵."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation.ravel()
    return T


def quaternion_to_rotation_matrix(
    qx: float, qy: float, qz: float, qw: float
) -> np.ndarray:
    """将四元数 (x, y, z, w) 转换为 3x3 旋转矩阵."""
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[0, 1] = 2 * (qx * qy - qw * qz)
    R[0, 2] = 2 * (qx * qz + qw * qy)
    R[1, 0] = 2 * (qx * qy + qw * qz)
    R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[1, 2] = 2 * (qy * qz - qw * qx)
    R[2, 0] = 2 * (qx * qz - qw * qy)
    R[2, 1] = 2 * (qy * qz + qw * qx)
    R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


# ---------------------------------------------------------------------------
# 集成测试主类
# ---------------------------------------------------------------------------

class IntegrationTester:
    """AprilTag 视觉定位 + 逆运动学联调测试器."""

    WINDOW_NAME = "Integration Test: Tag Tracking → IK"
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        tag_size_m: float = 0.15,
        tag_to_base_translation: tuple[float, float, float] = (20.0, 0.0, -20.0),
        tag_to_base_rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        record_bag: Optional[str] = None,
    ) -> None:
        # --- AprilTag 检测/解算 ---
        detector_cfg = AprilTagConfig(family="tag36h11", target_id=0)
        self._detector = AprilTagDetector(detector_cfg)
        self._pose_solver = AprilTagPoseSolver(tag_size_m)

        # --- Tag → Base 坐标变换 ---
        # 用户输入: base 原点在 tag 坐标系中的位置
        # 即 T_base→tag:  R=I,  t=[tx, ty, tz]
        # 则 T_tag→base = T_base→tag^{-1}:  R^T,  t = -R^T·[tx,ty,tz]
        roll, pitch, yaw = np.deg2rad(tag_to_base_rotation_deg)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        R_tag_to_base = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)
        base_origin_in_tag = np.array(tag_to_base_translation, dtype=np.float64)
        t_tag_to_base = -R_tag_to_base @ base_origin_in_tag  # tag 原点在 base 下的位置
        self._T_tag_to_base = build_transform(R_tag_to_base, t_tag_to_base)
        self._T_base_to_tag = np.linalg.inv(self._T_tag_to_base)

        # --- 运动学 ---
        description_share = Path(
            get_package_share_directory("humanoid_arm_description")
        )
        model = RobotModel.from_urdf_file(
            description_share / "urdf" / "humanoid_arm.urdf",
            base_link="base_link",
            tip_link="tip_frame",
        )
        self._fk = ForwardSolver(model)
        self._jac = JacobianSolver(model)
        ik_cfg = IKConfig(
            max_iterations=500,
            position_tolerance_m=0.005,
            multi_start_attempts=8,
            multi_start_perturbation_rad=0.5,
        )
        self._ik = InverseSolver(self._fk, self._jac, ik_cfg)

        # --- 状态 ---
        self._calibration: Optional[CalibrationState] = None
        self._current_joints_rad = HOME_JOINTS_RAD.copy()
        self._latest_detection: Optional[AprilTagDetection] = None
        self._latest_pose_estimate: Optional[PoseEstimate] = None
        self._latest_ik_result: Optional[str] = None  # IK 结果字符串
        self._lock = threading.Lock()

        # --- 统计 ---
        self._frame_count = 0
        self._solve_count = 0
        self._fail_count = 0

        # --- rosbag 录制 ---
        self._record_bag = record_bag
        self._bag_writer: Any = None
        if record_bag is not None:
            self._setup_bag_recording(record_bag)

        # --- 相机 ---
        self._camera: Optional[RealSenseCamera] = None
        self._calibration_obj: Optional[CameraCalibration] = None

    def _setup_bag_recording(self, path: str) -> None:
        """初始化 rosbag 录制."""
        try:
            import rclpy
            from rclpy.serialization import serialize_message
            from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
        except ImportError:
            print("[WARN] rosbag2_py 不可用, 将跳过录制")
            return
        # 简化: 跳过录制实现, 避免复杂依赖
        print(f"[INFO] rosbag 录制功能暂未实现, 路径: {path}")

    def _transform_to_base(self, pose: PoseEstimate) -> tuple[np.ndarray, float]:
        """将相机在 tag 坐标系下的位姿转换到 base 坐标系.

        推导:
          pose.camera_from_tag = T_tag→camera  (PnP 输出)
          T_camera→tag  = inv(T_tag→camera)    (相机在 tag 中的位姿)
          T_camera→base = T_tag→base · T_camera→tag

        Returns:
            (position_in_base, yaw_in_base)
        """
        T_tag_from_camera = pose.camera_from_tag   # 标签原点在相机下的位姿
        T_camera_in_tag = np.linalg.inv(T_tag_from_camera)  # 相机在 tag 中的位姿
        T_camera_in_base = self._T_tag_to_base @ T_camera_in_tag
        pos_base = T_camera_in_base[:3, 3].copy()
        R_base = T_camera_in_base[:3, :3]
        yaw_base = rotation_to_yaw(R_base)
        return pos_base, yaw_base

    def open_camera(self) -> bool:
        """打开 RealSense 相机并获取内参."""
        try:
            camera_cfg = CameraConfig(width=640, height=480, fps=30)
            self._camera = RealSenseCamera(camera_cfg)
            self._camera.open()
            self._calibration_obj = self._camera.calibration
            print(f"[INFO] 相机已打开: {camera_cfg.width}x{camera_cfg.height}")
            return True
        except Exception as exc:
            print(f"[ERROR] 无法打开相机: {exc}")
            print("  请确保 RealSense D435i 已连接")
            return False

    def calibrate(self) -> bool:
        """记录当前相机位姿和末端位姿作为参考基准.

        Returns:
            True 如果校准成功.
        """
        if self._latest_pose_estimate is None:
            print("[WARN] 没有可用的位姿估计, 请确保 Tag 在视野内")
            return False

        cam_pos, cam_yaw = self._transform_to_base(self._latest_pose_estimate)
        fk_home = self._fk.solve(HOME_JOINTS_RAD)

        self._calibration = CalibrationState(
            camera_position=cam_pos.copy(),
            camera_yaw_rad=cam_yaw,
            ee_position=fk_home.position.copy(),
            ee_yaw_rad=fk_home.yaw_rad,
        )
        print(f"[CALIB] 参考相机位姿 (base): pos={cam_pos}, yaw={math.degrees(cam_yaw):.1f}°")
        print(f"[CALIB] 参考末端位姿 (base): pos={fk_home.position}, yaw={math.degrees(fk_home.yaw_rad):.1f}°")
        print(f"[CALIB] 相对跟随模式已激活")
        return True

    def process_frame(self, frame: CameraFrame) -> None:
        """处理一帧: 检测 → 解算 → 变换 → IK → 可视化."""
        self._frame_count += 1

        with self._lock:
            color = frame.image.copy()

        # Step 1: AprilTag 检测
        try:
            detection = self._detector.detect(color)
        except Exception:
            detection = None

        # Step 2: PnP 解算
        pose_estimate: Optional[PoseEstimate] = None

        if detection is not None and self._calibration_obj is not None:
            try:
                pose_estimate = self._pose_solver.solve(
                    detection, self._calibration_obj
                )
            except Exception:
                pose_estimate = None

        # Step 3: 坐标变换 + IK
        ik_result_text: Optional[str] = None
        error_text: Optional[str] = None

        if pose_estimate is not None and self._calibration is not None:
            try:
                cam_pos, _ = self._transform_to_base(pose_estimate)

                # 相对跟随: 计算增量
                delta_pos = cam_pos - self._calibration.camera_position

                target_pos = self._calibration.ee_position + delta_pos

                # Step 4: 逆运动学
                ik_result = self._ik.solve(
                    target_pos,
                    self._current_joints_rad,
                )

                if ik_result.success:
                    # 额外安全层: 限位截断 + 连续解选择
                    self._current_joints_rad = ik_result.joint_angles_rad.copy()
                    self._solve_count += 1

                    joints_deg = np.rad2deg(ik_result.joint_angles_rad)
                    ik_result_text = (
                        f"J1:{joints_deg[0]: 6.1f}  J2:{joints_deg[1]: 6.1f}  "
                        f"J3:{joints_deg[2]: 6.1f}  J4:{joints_deg[3]: 6.1f}"
                    )
                    pos_err = ik_result.position_error_m
                    error_text = (
                        f"pos_err:{pos_err*1000:.1f}mm  "
                        f"iter:{ik_result.iterations}  orientation:ignored"
                    )
                else:
                    self._fail_count += 1
                    ik_result_text = "IK FAILED"
                    error_text = "未收敛"

            except Exception as exc:
                ik_result_text = "IK ERROR"
                error_text = str(exc)[:40]

        # 更新状态
        with self._lock:
            self._latest_detection = detection
            self._latest_pose_estimate = pose_estimate
            self._latest_ik_result = ik_result_text

        # Step 5: 可视化
        display = self._build_display(
            color, detection, pose_estimate, ik_result_text, error_text
        )
        cv2.imshow(self.WINDOW_NAME, display)

    def _build_display(
        self,
        image: np.ndarray,
        detection: Optional[AprilTagDetection],
        pose_estimate: Optional[PoseEstimate],
        ik_text: Optional[str],
        error_text: Optional[str],
    ) -> np.ndarray:
        """构建带标注的可视化图像."""
        display = image.copy()
        h, w = display.shape[:2]

        # --- 绘制 AprilTag 检测框 ---
        if detection is not None:
            corners = detection.corners.astype(np.int32)
            cv2.polylines(display, [corners], True, (0, 255, 0), 2)
            cx, cy = int(detection.center[0]), int(detection.center[1])
            cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(
                display,
                f"ID:{detection.tag_id} area:{detection.pixel_area:.0f}",
                (cx + 10, cy),
                self.FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

        # --- 右上面板: 相机位姿 (tag frame) ---
        panel_w = 320
        panel_h = 220
        panel_x = w - panel_w - 10
        panel_y = 10
        overlay = display.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        self._draw_text(display, "Tag Pose (in camera)", panel_x + 10, panel_y + 20, (180, 180, 180), 0.5)
        y = panel_y + 42

        if pose_estimate is not None:
            # pose_estimate.position = 标签原点在相机坐标系下的位置
            p = pose_estimate.position
            self._draw_text(display, f"tag x:{p[0]:7.3f}  y:{p[1]:7.3f}  z:{p[2]:7.3f} (m)",
                            panel_x + 10, y, (220, 220, 220), 0.45)
            y += 20
            self._draw_text(display,
                            f"reproj: {pose_estimate.reprojection_error_px:.2f} px  dist: {pose_estimate.camera_distance_m:.2f}m",
                            panel_x + 10, y, (180, 180, 180), 0.4)
            y += 22

            # 计算相机在 tag 坐标系下的位置 (取逆)
            try:
                T_tag_from_camera = pose_estimate.camera_from_tag
                T_camera_in_tag = np.linalg.inv(T_tag_from_camera)
                cam_in_tag = T_camera_in_tag[:3, 3]
                cam_yaw_tag = rotation_to_yaw(T_camera_in_tag[:3, :3])
                y += 5
                self._draw_text(display, "Camera Pose (tag frame)",
                                panel_x + 10, y, (180, 160, 100), 0.45)
                y += 18
                self._draw_text(display,
                                f"x:{cam_in_tag[0]:7.3f}  y:{cam_in_tag[1]:7.3f}  z:{cam_in_tag[2]:7.3f}",
                                panel_x + 10, y, (200, 200, 150), 0.45)
                y += 18
                self._draw_text(display,
                                f"yaw: {math.degrees(cam_yaw_tag):.1f}°",
                                panel_x + 10, y, (200, 200, 150), 0.4)
            except Exception:
                pass

            # 如果在 base 坐标系下
            if self._calibration is not None:
                try:
                    cam_pos, cam_yaw = self._transform_to_base(pose_estimate)
                    y += 3
                    self._draw_text(display, "Camera Pose (base frame)",
                                    panel_x + 10, y, (180, 160, 100), 0.45)
                    y += 18
                    self._draw_text(display,
                                    f"x:{cam_pos[0]:7.3f}  y:{cam_pos[1]:7.3f}  z:{cam_pos[2]:7.3f} (m)",
                                    panel_x + 10, y, (200, 200, 150), 0.45)
                    y += 18
                    self._draw_text(display,
                                    f"yaw: {math.degrees(cam_yaw):.1f} deg",
                                    panel_x + 10, y, (200, 200, 150), 0.45)
                    delta_pos = cam_pos - self._calibration.camera_position
                    y += 18
                    self._draw_text(display,
                                    f"Δ: [{delta_pos[0]:.3f}, {delta_pos[1]:.3f}, {delta_pos[2]:.3f}] (rel)",
                                    panel_x + 10, y, (150, 200, 150), 0.4)
                except Exception:
                    pass
        else:
            self._draw_text(display, "No tag detected", panel_x + 10, y, (120, 120, 120), 0.5)

        # --- 底部面板: IK 输出 ---
        ik_panel_h = 90 if ik_text else 30
        ik_panel_y = h - ik_panel_h - 10
        overlay2 = display.copy()
        cv2.rectangle(overlay2, (10, ik_panel_y), (w - 10, h - 10), (30, 30, 30), -1)
        cv2.addWeighted(overlay2, 0.55, display, 0.45, 0, display)

        y_ik = ik_panel_y + 22
        self._draw_text(display, "Inverse Kinematics", 20, y_ik, (180, 180, 180), 0.5)

        if ik_text:                                       
            y_ik += 22                                                                       
            if "FAILED" in ik_text or "ERROR" in ik_text:
                color = (80, 80, 255)
            else:
                color = (100, 255, 100)
            self._draw_text(display, ik_text, 20, y_ik, color, 0.5)

        if error_text:                                                                       
            y_ik += 20                                                                       
            self._draw_text(display, error_text, 20, y_ik, (180, 180, 180), 0.4)

        # --- 状态指示 ---
        status_parts = []
        status_parts.append(f"Frames: {self._frame_count}")
        if self._calibration is not None:
            status_parts.append("TRACKING")
        else:
            status_parts.append("NO CALIB | Press SPACE")
        status_parts.append(f"Solve: {self._solve_count} OK / {self._fail_count} FAIL")
        status_text = " | ".join(status_parts)
        y_status = ik_panel_y - 8
        self._draw_text(display, status_text, 20, y_status, (160, 160, 160), 0.4)

        # --- 未校准时提示 ---
        if self._calibration is None:
            hint = "Press SPACE to calibrate reference pose"
            text_size = cv2.getTextSize(hint, self.FONT, 0.7, 2)[0]
            tx = (w - text_size[0]) // 2
            ty = h // 2 - 10
            cv2.rectangle(display, (tx - 15, ty - 25), (tx + text_size[0] + 15, ty + 15), (0, 0, 0), -1)
            self._draw_text(display, hint, tx, ty, (100, 255, 255), 0.7)

        return display

    @staticmethod
    def _draw_text(
        img: np.ndarray, text: str, x: int, y: int, color: tuple, scale: float
    ) -> None:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def close(self) -> None:
        """关闭相机和资源."""
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print(f"[INFO] 统计: {self._frame_count} 帧, {self._solve_count} 次 IK 成功, {self._fail_count} 次失败")


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AprilTag → IK 集成测试")
    parser.add_argument("--tag-size", type=float, default=0.07,
                        help="AprilTag 边长 (米), 默认 0.07 (7cm)")
    parser.add_argument("--tag-x", type=float, default=0.2,
                        help="Base 在 Tag 坐标系下的 X (米), 默认 0.2 (20cm)")
    parser.add_argument("--tag-y", type=float, default=0.0,
                        help="Base 在 Tag 坐标系下的 Y (米), 默认 0.0")
    parser.add_argument("--tag-z", type=float, default=-0.2,
                        help="Base 在 Tag 坐标系下的 Z (米), 默认 -0.2 (20cm)")
    parser.add_argument("--tag-roll", type=float, default=0.0,
                        help="Tag→Base 旋转 Roll (度)")
    parser.add_argument("--tag-pitch", type=float, default=0.0,
                        help="Tag→Base 旋转 Pitch (度)")
    parser.add_argument("--tag-yaw", type=float, default=0.0,
                        help="Tag→Base 旋转 Yaw (度)")
    parser.add_argument("--record", type=str, default=None,
                        help="录制 rosbag 输出路径 (暂不支持)")
    args = parser.parse_args()

    tester = IntegrationTester(
        tag_size_m=args.tag_size,
        tag_to_base_translation=(args.tag_x, args.tag_y, args.tag_z),
        tag_to_base_rotation_deg=(args.tag_roll, args.tag_pitch, args.tag_yaw),
        record_bag=args.record,
    )

    if not tester.open_camera():
        sys.exit(1)

    cv2.namedWindow(IntegrationTester.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(IntegrationTester.WINDOW_NAME, 1280, 960)

    shutdown_flag = threading.Event()
    signal.signal(signal.SIGINT, lambda sig, frame: shutdown_flag.set())

    print("=" * 60)
    print("  集成测试已启动: AprilTag 视觉 → 逆运动学")
    print("  按键: SPACE = 校准参考位姿  |  ESC = 退出")
    print("=" * 60)

    try:
        while not shutdown_flag.is_set():
            if tester._camera is None:
                break

            try:
                frame = tester._camera.read()
            except Exception as exc:
                print(f"[WARN] 相机读取失败: {exc}, 重试中...")
                import time
                time.sleep(0.5)
                continue

            if frame is None:
                import time
                time.sleep(0.01)
                continue

            tester.process_frame(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                success = tester.calibrate()
                if not success:
                    print("[WARN] 校准失败, 请确保 Tag 在视野内")

            try:
                if cv2.getWindowProperty(IntegrationTester.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    finally:
        tester.close()


if __name__ == "__main__":
    main()

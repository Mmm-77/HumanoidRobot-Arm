#!/usr/bin/env python3
"""Display real-time end-effector tip coordinates (6-DOF) in a small GUI window."""
import math
import threading
import tkinter as tk

import rclpy
from rclpy.node import Node
import tf2_ros


def quat_to_rpy(w, x, y, z):
    """Convert quaternion to roll-pitch-yaw (intrinsic ZYX, radians)."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - x * z))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class TipPositionNode(Node):
    def __init__(self):
        super().__init__("tip_position_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._pos = (0.0, 0.0, 0.0)
        self._rpy = (0.0, 0.0, 0.0)
        self._root: tk.Tk | None = None
        self._label: tk.Label | None = None

    def spin(self):
        self._timer = self.create_timer(0.05, self._tf_callback)

    def _tf_callback(self):
        try:
            t = self.tf_buffer.lookup_transform(
                "world", "tip_frame", rclpy.time.Time()
            )
            trans = t.transform.translation
            rot = t.transform.rotation
            self._pos = (trans.x, trans.y, trans.z)
            self._rpy = quat_to_rpy(rot.w, rot.x, rot.y, rot.z)
        except Exception:
            pass

    def run_gui(self) -> None:
        self._root = tk.Tk()
        self._root.title("末端杆尖端 6-DOF")
        self._root.geometry("320x220")

        frame = tk.Frame(self._root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(frame, text="末端杆尖端 (世界坐标系)",
                 font=("sans", 10, "bold")).pack(anchor=tk.W)

        self._label = tk.Label(frame, text="等待 TF 数据...",
                               font=("monospace", 13), justify=tk.LEFT)
        self._label.pack(anchor=tk.W, pady=(8, 0))

        self._refresh()
        self._root.mainloop()

    def _refresh(self) -> None:
        if self._root and self._label:
            x, y, z = self._pos
            roll, pitch, yaw = self._rpy
            dist = (x * x + y * y + z * z) ** 0.5
            self._label.config(
                text=f"位置 (m)\n"
                     f"  X: {x:+7.4f}  Y: {y:+7.4f}  Z: {z:+7.4f}\n"
                     f"  |R| = {dist:.4f}\n"
                     f"\n"
                     f"姿态 (°)\n"
                     f"  R: {math.degrees(roll):+7.1f}  P: {math.degrees(pitch):+7.1f}  Y: {math.degrees(yaw):+7.1f}"
            )
            self._root.after(80, self._refresh)


def main():
    rclpy.init()
    node = TipPositionNode()
    node.spin()

    executor_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    executor_thread.start()

    try:
        node.run_gui()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

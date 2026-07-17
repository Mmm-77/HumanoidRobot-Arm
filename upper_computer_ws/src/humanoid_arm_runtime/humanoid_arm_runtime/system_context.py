"""System context: thread-safe cache for latest data from all subsystems.

Aggregates the most recent pose, joint feedback, IK output, and diagnostic
status in one place for the runtime node.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class PoseSnapshot:
    """A captured pose with a timestamp.

    Attributes:
        timestamp_s: Acquisition time (monotonic seconds).
        position: [x, y, z] in meters.
        quaternion_xyzw: Unit quaternion [x, y, z, w].
        valid: True if the quality gate passed.
    """

    timestamp_s: float
    position: NDArray[np.float64]   # shape (3,)
    quaternion_xyzw: NDArray[np.float64]  # shape (4,)
    valid: bool = True


@dataclass
class JointSnapshot:
    """Current joint state from the lower computer.

    Attributes:
        timestamp_s: Time of reception.
        positions_rad: 4-element array of joint angles.
        velocities_rad_per_s: 4-element array of joint velocities.
        any_error: True if any motor has an error flag.
    """

    timestamp_s: float
    positions_rad: NDArray[np.float64]
    velocities_rad_per_s: NDArray[np.float64]
    any_error: bool = False


@dataclass
class TargetSnapshot:
    """A 4-DOF target [x, y, z, yaw] sent to the kinematics solver.

    Attributes:
        timestamp_s: Generation time.
        position_m: [x, y, z] in meters (base frame).
        yaw_rad: Controllable yaw angle about base Z.
    """

    timestamp_s: float
    position_m: NDArray[np.float64]  # shape (3,)
    yaw_rad: float


@dataclass
class SystemContext:
    """Thread-safe shared context for the runtime node."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Latest vision data
    latest_pose: Optional[PoseSnapshot] = None

    # Latest communication feedback
    latest_joints: Optional[JointSnapshot] = None

    # Latest IK output (for FK back-calculation)
    latest_ik_joints_rad: Optional[NDArray[np.float64]] = None

    # Baseline (recorded on entering FOLLOW)
    baseline_pose: Optional[PoseSnapshot] = None
    baseline_ee_position_m: Optional[NDArray[np.float64]] = None   # from FK at baseline time
    baseline_ee_yaw_rad: Optional[float] = None

    # Current target being sent
    current_target: Optional[TargetSnapshot] = None

    # Communication link state
    link_ok: bool = False
    communication_timeout: bool = False

    def set_pose(self, pose: PoseSnapshot) -> None:
        with self._lock:
            self.latest_pose = pose

    def get_pose(self) -> Optional[PoseSnapshot]:
        with self._lock:
            return self.latest_pose

    def set_joints(self, joints: JointSnapshot) -> None:
        with self._lock:
            self.latest_joints = joints

    def get_joints(self) -> Optional[JointSnapshot]:
        with self._lock:
            return self.latest_joints

    def set_baseline(self, pose: PoseSnapshot, ee_pos: NDArray[np.float64], ee_yaw: float) -> None:
        with self._lock:
            self.baseline_pose = pose
            self.baseline_ee_position_m = ee_pos.copy()
            self.baseline_ee_yaw_rad = ee_yaw

    def clear_baseline(self) -> None:
        with self._lock:
            self.baseline_pose = None
            self.baseline_ee_position_m = None
            self.baseline_ee_yaw_rad = None

    def has_baseline(self) -> bool:
        with self._lock:
            return self.baseline_pose is not None

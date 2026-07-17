"""Baseline manager: capture initial pose and FK result on entering FOLLOW.

Records the camera pose (in tag frame) and the current end-effector pose
(computed via forward kinematics from the latest joint angles) at the moment
the system enters the FOLLOW state.  All subsequent camera-delta calculations
are relative to this baseline.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .system_context import (
    JointSnapshot,
    PoseSnapshot,
    SystemContext,
)


class BaselineError(RuntimeError):
    """Raised when a baseline cannot be established."""


class BaselineManager:
    """Capture and validate the FOLLOW baseline."""

    def __init__(
        self,
        context: SystemContext,
        *,
        max_pose_age_s: float = 0.2,
        max_joint_age_s: float = 0.2,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._context = context
        self._max_pose_age_s = max_pose_age_s
        self._max_joint_age_s = max_joint_age_s
        self._clock = monotonic_clock

    def capture(self, ee_pos_m: np.ndarray, ee_yaw_rad: float) -> bool:
        """Attempt to capture a valid baseline.

        Args:
            ee_pos_m: Current end-effector position in base frame (from FK).
            ee_yaw_rad: Current end-effector yaw about base Z (from FK).

        Returns:
            True if a fresh, valid baseline was captured.
        """
        now = self._clock()
        pose = self._context.get_pose()
        joints = self._context.get_joints()

        if pose is None or not pose.valid:
            return False
        if (now - pose.timestamp_s) > self._max_pose_age_s:
            return False
        if joints is not None and (now - joints.timestamp_s) > self._max_joint_age_s:
            return False

        self._context.set_baseline(pose, ee_pos_m, ee_yaw_rad)
        return True

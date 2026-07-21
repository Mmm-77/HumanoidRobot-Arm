"""Follow mapper: convert camera delta (in tag frame) to arm target (in base frame).

Given a baseline camera pose T_tag→cam0 and the current camera pose T_tag→cam,
the camera displacement Δ_cam = cam⁻¹ · cam0 is the motion of the camera from
baseline to current.

This displacement is transformed from camera frame to base frame using the
(fixed, calibrated) transform T_cam→base and then scaled/axis-mapped to
produce a 4-DOF command [x, y, z, yaw] in the base frame.

Configuration parameters:
  - axis_signs:  ±1 per axis to invert mapping direction
  - scale:       gain factor for position (default 1.0)
  - tag_to_base: fixed 4×4 homogeneous transform from tag frame to base frame
                 (to be calibrated once the tag is physically mounted)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class FollowMapper:
    """Maps a camera-delta into a base-frame 4-DOF target."""

    def __init__(
        self,
        *,
        axis_signs: Tuple[int, int, int] = (1, 1, 1),
        position_scale: float = 1.0,
        tag_to_base: NDArray[np.float64] | None = None,
        camera_pose_convention: str = "camera_in_tag",
        yaw_sign: int = 1,
        yaw_scale: float = 1.0,
    ) -> None:
        self.axis_signs = np.array(axis_signs, dtype=np.float64)
        self.position_scale = position_scale
        if camera_pose_convention not in ("camera_in_tag", "tag_in_camera"):
            raise ValueError(
                "camera_pose_convention must be 'camera_in_tag' or 'tag_in_camera'"
            )
        self.camera_pose_convention = camera_pose_convention
        self.yaw_sign = float(yaw_sign)
        self.yaw_scale = float(yaw_scale)
        # T_tag→base: the fixed transform from tag CS to robot base CS.
        # If None, identity is assumed (tag is aligned with base).
        self._tag_to_base = (
            np.asarray(tag_to_base, dtype=np.float64)
            if tag_to_base is not None
            else np.eye(4, dtype=np.float64)
        )

    def map(
        self,
        baseline_position_cam: NDArray[np.float64],
        baseline_quat_xyzw: NDArray[np.float64],
        current_position_cam: NDArray[np.float64],
        current_quat_xyzw: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        """Compute the arm target from two camera poses.

        Args:
            baseline_position_cam: [x, y, z] of camera in tag frame at baseline.
            baseline_quat_xyzw: [x, y, z, w] quat of camera at baseline.
            current_position_cam: [x, y, z] of camera in tag frame now.
            current_quat_xyzw: [x, y, z, w] quat of camera now.

        Returns:
            target_pos_m: [x, y, z] in base frame.
            target_yaw_rad: Yaw about base Z in radians.
        """
        # Build transforms
        T_tag_cam0 = self._build_transform(baseline_position_cam, baseline_quat_xyzw)
        T_tag_cam = self._build_transform(current_position_cam, current_quat_xyzw)
        if self.camera_pose_convention == "tag_in_camera":
            T_tag_cam0 = np.linalg.inv(T_tag_cam0)
            T_tag_cam = np.linalg.inv(T_tag_cam)

        # Translation is differenced in the common tag frame, then expressed in
        # base coordinates by the calibrated tag-to-base rotation.
        rotation_tag_to_base = self._tag_to_base[:3, :3]
        delta_pos_tag = T_tag_cam[:3, 3] - T_tag_cam0[:3, 3]
        delta_pos_base = rotation_tag_to_base @ delta_pos_tag

        # Apply scaling and axis signs
        delta_pos_base = delta_pos_base * self.axis_signs * self.position_scale

        # Yaw extraction: decompose delta rotation about tag Z → base Z
        delta_rotation_tag = T_tag_cam[:3, :3] @ T_tag_cam0[:3, :3].T
        delta_rotation_base = (
            rotation_tag_to_base
            @ delta_rotation_tag
            @ rotation_tag_to_base.T
        )
        delta_yaw_base = (
            self._extract_yaw_about_z(delta_rotation_base)
            * self.yaw_sign
            * self.yaw_scale
        )

        return delta_pos_base, delta_yaw_base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_transform(
        position: NDArray[np.float64],
        quat_xyzw: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Build a 4×4 homogeneous transform from position + quaternion [x,y,z,w]."""
        x, y, z, w = quat_xyzw
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y],
        ], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
        return T

    @staticmethod
    def _extract_yaw_about_z(rotation: NDArray[np.float64]) -> float:
        """Extract rotation about the Z axis from a 3×3 rotation matrix."""
        # R_z(θ) = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        # yaw = atan2(R[1,0], R[0,0])
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

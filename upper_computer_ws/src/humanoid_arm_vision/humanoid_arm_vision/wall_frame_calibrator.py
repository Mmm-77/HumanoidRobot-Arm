"""Convert the AprilTag frame into the wall-mounted tracking convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .transform_utils import (
    normalize_quaternion,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


# Raw AprilTag axes are [right, up, out of wall]. The calibrated wall frame is
# [out of wall, right, up], matching the requested [X, Y, Z] convention.
WALL_FROM_TAG_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class WallFrameCalibration:
    """Calibrated camera pose in the wall tracking frame."""

    position: NDArray[np.float64]
    orientation_xyzw: NDArray[np.float64]


class WallFrameCalibrator:
    """Apply the fixed wall-axis permutation and configurable X origin."""

    def __init__(self, x_origin_m: float) -> None:
        if not np.isfinite(x_origin_m) or x_origin_m <= 0.0:
            raise ValueError("wall-frame x_origin_m must be positive and finite")
        self.x_origin_m = float(x_origin_m)

    def calibrate(
        self,
        position_tag: NDArray[np.floating],
        orientation_tag_xyzw: NDArray[np.floating],
    ) -> WallFrameCalibration:
        """Transform a camera-in-tag pose into the calibrated wall frame."""
        position = np.asarray(position_tag, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(position)):
            raise ValueError("camera position contains a non-finite value")
        orientation = normalize_quaternion(orientation_tag_xyzw)

        calibrated_position = WALL_FROM_TAG_ROTATION @ position
        calibrated_position[0] -= self.x_origin_m
        calibrated_rotation = (
            WALL_FROM_TAG_ROTATION
            @ quaternion_to_rotation_matrix(orientation)
        )
        calibrated_orientation = rotation_matrix_to_quaternion(
            calibrated_rotation
        )
        return WallFrameCalibration(
            position=calibrated_position,
            orientation_xyzw=calibrated_orientation,
        )

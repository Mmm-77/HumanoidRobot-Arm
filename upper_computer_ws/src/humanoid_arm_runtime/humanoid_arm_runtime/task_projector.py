"""Task projector: reduce a 6-DOF pose delta to a 4-DOF task target.

The arm has 4 controllable DOFs: [x, y, z, theta] where theta is the yaw
angle about the base Z axis.  The task projector takes the 6-DOF camera
position+orientation delta and projects it onto this 4-DOF space.

For position (x,y,z): direct pass-through with optional scaling.
For orientation: the rotation matrix is decomposed and only the Z-axis
component is retained as theta (yaw).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class TaskProjector:
    """Project a 6-DOF delta → 4-DOF task target [x, y, z, yaw]."""

    def __init__(
        self,
        max_position_step_m: float = 0.05,
        max_yaw_step_rad: float = 0.1,
    ) -> None:
        self._max_position_step_m = max_position_step_m
        self._max_yaw_step_rad = max_yaw_step_rad

    def project(
        self,
        delta_pos_m: NDArray[np.float64],
        delta_yaw_rad: float,
    ) -> Tuple[NDArray[np.float64], float]:
        """Clip and project the delta into safe bounds.

        Returns:
            clipped_pos_m: [x, y, z] in base frame, per-axis clipped.
            clipped_yaw_rad: yaw about base Z, absolute-clipped.
        """
        pos = np.asarray(delta_pos_m, dtype=np.float64).reshape(3)
        yaw = float(delta_yaw_rad)

        # Clamp position
        pos_norm = float(np.linalg.norm(pos))
        if pos_norm > self._max_position_step_m and pos_norm > 0:
            pos = pos / pos_norm * self._max_position_step_m

        # Clamp yaw
        yaw = max(-self._max_yaw_step_rad, min(self._max_yaw_step_rad, yaw))

        return pos, yaw

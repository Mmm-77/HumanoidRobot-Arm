"""Rate-limit camera deltas before they become arm targets."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class TaskProjector:
    """Rate-limit a camera delta without changing its eventual offset."""

    def __init__(
        self,
        max_position_step_m: float = 0.05,
        max_yaw_step_rad: float = 0.1,
    ) -> None:
        if max_position_step_m <= 0.0:
            raise ValueError("max_position_step_m must be positive")
        if max_yaw_step_rad <= 0.0:
            raise ValueError("max_yaw_step_rad must be positive")
        self._max_position_step_m = max_position_step_m
        self._max_yaw_step_rad = max_yaw_step_rad
        self.reset()

    def project(
        self,
        delta_pos_m: NDArray[np.float64],
        delta_yaw_rad: float,
    ) -> Tuple[NDArray[np.float64], float]:
        """Move the projected output one bounded step toward the input delta."""
        desired_pos = np.asarray(delta_pos_m, dtype=np.float64).reshape(3)
        desired_yaw = float(delta_yaw_rad)
        if not np.all(np.isfinite(desired_pos)) or not math.isfinite(desired_yaw):
            raise ValueError("task delta must contain only finite values")

        position_step = desired_pos - self._position
        step_norm = float(np.linalg.norm(position_step))
        if step_norm > self._max_position_step_m:
            position_step *= self._max_position_step_m / step_norm
        self._position = self._position + position_step

        yaw_step = desired_yaw - self._yaw
        yaw_step = max(
            -self._max_yaw_step_rad,
            min(self._max_yaw_step_rad, yaw_step),
        )
        self._yaw += yaw_step

        return self._position.copy(), self._yaw

    def reset(self) -> None:
        """Reset the projected offset when a new follow baseline is captured."""
        self._position = np.zeros(3, dtype=np.float64)
        self._yaw = 0.0

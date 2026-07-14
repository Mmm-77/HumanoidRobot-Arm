"""Stateful exponential filtering for camera position and orientation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .transform_utils import normalize_quaternion, slerp


@dataclass(frozen=True)
class PoseFilterConfig:
    position_alpha: float = 0.35
    orientation_alpha: float = 0.30
    reset_gap_s: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 < self.position_alpha <= 1.0:
            raise ValueError("position_alpha must be in (0, 1]")
        if not 0.0 < self.orientation_alpha <= 1.0:
            raise ValueError("orientation_alpha must be in (0, 1]")
        if self.reset_gap_s <= 0.0:
            raise ValueError("reset_gap_s must be positive")


@dataclass(frozen=True)
class FilteredPose:
    position: NDArray[np.float64]
    orientation_xyzw: NDArray[np.float64]
    timestamp_s: float
    reset: bool


class PoseFilter:
    def __init__(self, config: PoseFilterConfig) -> None:
        self.config = config
        self._position: NDArray[np.float64] | None = None
        self._orientation: NDArray[np.float64] | None = None
        self._timestamp_s: float | None = None

    @property
    def initialized(self) -> bool:
        return self._timestamp_s is not None

    def reset(self) -> None:
        self._position = None
        self._orientation = None
        self._timestamp_s = None

    def update(
        self,
        position: NDArray[np.floating],
        orientation_xyzw: NDArray[np.floating],
        timestamp_s: float,
    ) -> FilteredPose:
        position_array = np.asarray(position, dtype=np.float64).reshape(3)
        orientation_array = normalize_quaternion(orientation_xyzw)
        if not np.all(np.isfinite(position_array)) or not np.isfinite(timestamp_s):
            raise ValueError("pose filter input must be finite")
        if self._timestamp_s is not None and timestamp_s < self._timestamp_s:
            raise ValueError("pose timestamps must be monotonic")

        must_reset = (
            self._timestamp_s is None
            or timestamp_s - self._timestamp_s > self.config.reset_gap_s
        )
        if must_reset:
            self._position = position_array.copy()
            self._orientation = orientation_array.copy()
        else:
            assert self._position is not None and self._orientation is not None
            alpha = self.config.position_alpha
            self._position = (1.0 - alpha) * self._position + alpha * position_array
            self._orientation = slerp(
                self._orientation, orientation_array, self.config.orientation_alpha
            )
        self._timestamp_s = float(timestamp_s)
        return FilteredPose(
            position=self._position.copy(),
            orientation_xyzw=self._orientation.copy(),
            timestamp_s=self._timestamp_s,
            reset=must_reset,
        )

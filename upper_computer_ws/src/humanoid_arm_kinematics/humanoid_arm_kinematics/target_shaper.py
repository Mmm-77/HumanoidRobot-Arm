"""Stateful shaping of Cartesian position targets before IK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ShaperConfig:
    position_dead_zone_m: float = 0.001
    max_position_step_m: float = 0.05
    position_alpha: float = 0.3

    def __post_init__(self) -> None:
        if self.position_dead_zone_m < 0.0:
            raise ValueError("position_dead_zone_m must be >= 0")
        if self.max_position_step_m <= 0.0:
            raise ValueError("max_position_step_m must be > 0")
        if not 0.0 < self.position_alpha <= 1.0:
            raise ValueError("position_alpha must be in (0, 1]")


class TargetShaper:
    """Apply dead-zone, Cartesian step limiting, and EMA before solving IK."""

    def __init__(self, config: ShaperConfig) -> None:
        self._config = config
        self._position: Optional[np.ndarray] = None

    @property
    def initialized(self) -> bool:
        return self._position is not None

    def reset(self) -> None:
        self._position = None

    def shape(self, position: Sequence[float]) -> np.ndarray:
        target = np.asarray(position, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("Position target must be a finite 3-vector")
        if self._position is None:
            self._position = target.copy()
            return target.copy()

        difference = target - self._position
        distance = float(np.linalg.norm(difference))
        if distance <= self._config.position_dead_zone_m:
            return self._position.copy()
        if distance > self._config.max_position_step_m:
            target = (
                self._position
                + difference / distance * self._config.max_position_step_m
            )
        alpha = self._config.position_alpha
        self._position = alpha * target + (1.0 - alpha) * self._position
        return self._position.copy()

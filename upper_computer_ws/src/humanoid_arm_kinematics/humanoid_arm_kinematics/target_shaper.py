"""Target shaper: dead-zone, smoothing, and velocity limiting for joint targets.

Applies a chain of transformations to raw IK output to produce safe,
smooth joint commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ShaperConfig:
    """Configuration for the target shaper.

    Attributes:
        position_dead_zone_m: Ignore position changes smaller than this.
        orientation_dead_zone_rad: Ignore orientation changes smaller than this.
        max_position_step_m: Maximum position change per step.
        max_orientation_step_rad: Maximum orientation change per step.
        position_alpha: EMA smoothing factor for position (0=no smoothing).
        orientation_alpha: EMA smoothing factor for orientation.
    """

    position_dead_zone_m: float = 0.001
    orientation_dead_zone_rad: float = 0.005
    max_position_step_m: float = 0.05
    max_orientation_step_rad: float = 0.1
    position_alpha: float = 0.3
    orientation_alpha: float = 0.3

    def __post_init__(self) -> None:
        if self.position_dead_zone_m < 0:
            raise ValueError("position_dead_zone_m must be >= 0")
        if self.orientation_dead_zone_rad < 0:
            raise ValueError("orientation_dead_zone_rad must be >= 0")
        if self.max_position_step_m <= 0:
            raise ValueError("max_position_step_m must be > 0")
        if self.max_orientation_step_rad <= 0:
            raise ValueError("max_orientation_step_rad must be > 0")
        if not (0 < self.position_alpha <= 1):
            raise ValueError("position_alpha must be in (0, 1]")
        if not (0 < self.orientation_alpha <= 1):
            raise ValueError("orientation_alpha must be in (0, 1]")


@dataclass(frozen=True)
class ShapedTarget:
    """A shaped joint target ready for publication.

    Attributes:
        joint_angles_rad: 4 joint angles.
        joint_velocities_rad_per_s: 4 joint velocities.
        position_smoothed: 3-element smoothed position.
        yaw_smoothed_rad: Smoothed yaw.
    """

    joint_angles_rad: np.ndarray
    joint_velocities_rad_per_s: np.ndarray
    position_smoothed: np.ndarray
    yaw_smoothed_rad: float


class TargetShaper:
    """Shapes raw IK output through dead-zone, clipping, and EMA smoothing.

    The shaper maintains a small internal state for the smoothed target,
    so a single instance should be used for a single continuous control session.
    """

    def __init__(self, config: ShaperConfig, dt_s: float = 1.0 / 30.0) -> None:
        self._config = config
        self._dt = dt_s
        self._position_smoothed: np.ndarray | None = None
        self._yaw_smoothed: float | None = None
        self._initialized: bool = False

    @property
    def config(self) -> ShaperConfig:
        return self._config

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        """Clear all internal smoothing state."""
        self._position_smoothed = None
        self._yaw_smoothed = None
        self._initialized = False

    def shape(
        self,
        joint_angles_rad: np.ndarray,
        position: np.ndarray,
        yaw_rad: float,
    ) -> ShapedTarget:
        """Apply dead-zone, step clipping, and EMA smoothing.

        Args:
            joint_angles_rad: Raw 4 joint angles from IK.
            position: Raw 3-element position from the task target.
            yaw_rad: Raw yaw from the task target.

        Returns:
            ShapedTarget with smoothed joint angles and estimated velocities.
        """
        pos = np.asarray(position, dtype=np.float64)

        if not self._initialized:
            self._position_smoothed = pos.copy()
            self._yaw_smoothed = yaw_rad
            self._initialized = True
            return ShapedTarget(
                joint_angles_rad=joint_angles_rad.copy(),
                joint_velocities_rad_per_s=np.zeros_like(joint_angles_rad),
                position_smoothed=pos.copy(),
                yaw_smoothed_rad=yaw_rad,
            )

        # Dead zone on position
        pos_diff = pos - self._position_smoothed  # type: ignore[operator]
        pos_dist = float(np.linalg.norm(pos_diff))
        if pos_dist < self._config.position_dead_zone_m:
            pos = self._position_smoothed  # type: ignore[assignment]

        # Step clipping on position
        if pos_dist > self._config.max_position_step_m:
            pos = self._position_smoothed + (pos_diff / pos_dist) * self._config.max_position_step_m  # type: ignore[operator]

        # Dead zone on orientation
        yaw_diff = yaw_rad - self._yaw_smoothed  # type: ignore[operator]
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        if abs(yaw_diff) < self._config.orientation_dead_zone_rad:
            shaped_yaw = self._yaw_smoothed
        else:
            # Step clipping
            clipped_diff = np.clip(
                yaw_diff,
                -self._config.max_orientation_step_rad,
                self._config.max_orientation_step_rad,
            )
            shaped_yaw = self._yaw_smoothed + clipped_diff  # type: ignore[operator]
            shaped_yaw = (shaped_yaw + np.pi) % (2 * np.pi) - np.pi

        # EMA smoothing
        alpha_p = self._config.position_alpha
        alpha_o = self._config.orientation_alpha
        self._position_smoothed = alpha_p * pos + (1 - alpha_p) * self._position_smoothed
        self._yaw_smoothed = alpha_o * shaped_yaw + (1 - alpha_o) * self._yaw_smoothed

        # Estimate joint velocities from position change
        prev_angles = (
            self._prev_joint_angles
            if hasattr(self, "_prev_joint_angles")
            else joint_angles_rad
        )
        velocities = (joint_angles_rad - prev_angles) / self._dt
        self._prev_joint_angles = joint_angles_rad.copy()  # type: ignore[attr-defined]

        return ShapedTarget(
            joint_angles_rad=joint_angles_rad.copy(),
            joint_velocities_rad_per_s=velocities,
            position_smoothed=self._position_smoothed.copy(),
            yaw_smoothed_rad=self._yaw_smoothed,
        )

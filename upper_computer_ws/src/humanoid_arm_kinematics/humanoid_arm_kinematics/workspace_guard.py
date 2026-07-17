"""Workspace guard: validates that task targets are within safe operating bounds.

Checks joint limits, velocity limits, and general feasibility before targets
are sent through the IK pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from .solution_selector import JointLimits


class GuardReason(str, Enum):
    """Reasons why a target was rejected by the workspace guard."""

    VALID = "valid"
    JOINT_ANGLE_LIMIT = "joint_angle_limit"
    JOINT_VELOCITY_LIMIT = "joint_velocity_limit"
    POSITION_NONFINITE = "position_nonfinite"
    ORIENTATION_NONFINITE = "orientation_nonfinite"


@dataclass(frozen=True)
class GuardDecision:
    """Result of workspace guard evaluation.

    Attributes:
        allowed: Whether the target is safe to process.
        reason: Reason if rejected.
        metrics: Additional diagnostic data.
    """

    allowed: bool
    reason: GuardReason
    metrics: Mapping[str, float] = field(default_factory=dict)


class WorkspaceGuard:
    """Checks that joint angles and velocities are within allowed limits."""

    def __init__(self, limits: JointLimits) -> None:
        self._limits = limits

    def check_target(
        self,
        target_position: np.ndarray,
        target_yaw_rad: float,
    ) -> GuardDecision:
        """Check whether a task-space target is physically plausible.

        Args:
            target_position: 3-element [x, y, z].
            target_yaw_rad: Yaw angle.

        Returns:
            GuardDecision.
        """
        if not np.all(np.isfinite(target_position)):
            return GuardDecision(
                allowed=False,
                reason=GuardReason.POSITION_NONFINITE,
            )
        if not np.isfinite(target_yaw_rad):
            return GuardDecision(
                allowed=False,
                reason=GuardReason.ORIENTATION_NONFINITE,
            )
        return GuardDecision(allowed=True, reason=GuardReason.VALID)

    def check_joint_angles(self, joint_angles_rad: np.ndarray) -> GuardDecision:
        """Check whether joint angles are within limits.

        Args:
            joint_angles_rad: 4 joint angles in radians.

        Returns:
            GuardDecision.
        """
        q = np.asarray(joint_angles_rad, dtype=np.float64)

        if not self._limits.is_within_limits(q):
            violations = []
            for i in range(len(q)):
                if q[i] < self._limits.angle_min_rad[i]:
                    violations.append(f"joint_{i+1}={q[i]:.3f} < min={self._limits.angle_min_rad[i]:.3f}")
                elif q[i] > self._limits.angle_max_rad[i]:
                    violations.append(f"joint_{i+1}={q[i]:.3f} > max={self._limits.angle_max_rad[i]:.3f}")
            return GuardDecision(
                allowed=False,
                reason=GuardReason.JOINT_ANGLE_LIMIT,
                metrics={"num_violations": float(len(violations))},
            )
        return GuardDecision(allowed=True, reason=GuardReason.VALID)

    def check_joint_velocities(
        self, joint_velocities_rad_per_s: np.ndarray
    ) -> GuardDecision:
        """Check whether joint velocities are within limits.

        Args:
            joint_velocities_rad_per_s: 4 joint velocities.

        Returns:
            GuardDecision.
        """
        v = np.asarray(joint_velocities_rad_per_s, dtype=np.float64)
        max_vel = self._limits.max_velocity_rad_per_s

        if np.any(np.abs(v) > max_vel):
            max_violation = float(np.max(np.abs(v) - max_vel))
            return GuardDecision(
                allowed=False,
                reason=GuardReason.JOINT_VELOCITY_LIMIT,
                metrics={"max_violation_rad_per_s": max_violation},
            )
        return GuardDecision(allowed=True, reason=GuardReason.VALID)

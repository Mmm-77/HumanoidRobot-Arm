"""Solution selector for inverse kinematics results.

Filters candidate IK solutions by joint limits and singularity distance,
selecting the most continuous solution closest to the current joint positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .inverse_solver import IKResult


class SelectorError(RuntimeError):
    """Raised when no valid solution can be selected."""


@dataclass(frozen=True)
class JointLimits:
    """Per-joint angle and velocity limits.

    Attributes:
        angle_min_rad: Minimum allowed joint angle per joint.
        angle_max_rad: Maximum allowed joint angle per joint.
        max_velocity_rad_per_s: Maximum joint velocity per joint.
    """

    angle_min_rad: np.ndarray
    angle_max_rad: np.ndarray
    max_velocity_rad_per_s: np.ndarray

    def __post_init__(self) -> None:
        for name, arr in [
            ("angle_min_rad", self.angle_min_rad),
            ("angle_max_rad", self.angle_max_rad),
            ("max_velocity_rad_per_s", self.max_velocity_rad_per_s),
        ]:
            if arr.ndim != 1:
                raise ValueError(f"{name} must be a 1-D array")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite")
        if np.any(self.angle_min_rad >= self.angle_max_rad):
            raise ValueError("angle_min must be < angle_max for all joints")
        if np.any(self.max_velocity_rad_per_s <= 0):
            raise ValueError("max_velocity must be > 0 for all joints")

    @property
    def num_joints(self) -> int:
        return len(self.angle_min_rad)

    def is_within_limits(self, angles_rad: np.ndarray) -> bool:
        """Check whether all joint angles are within limits."""
        return bool(np.all(angles_rad >= self.angle_min_rad) and
                    np.all(angles_rad <= self.angle_max_rad))

    def clamp(self, angles_rad: np.ndarray) -> np.ndarray:
        """Clamp joint angles to the allowed range."""
        return np.clip(angles_rad, self.angle_min_rad, self.angle_max_rad)


@dataclass(frozen=True)
class SelectedSolution:
    """A validated and selected IK solution.

    Attributes:
        joint_angles_rad: The selected 4 joint angles.
        ik_result: The original IK result.
        joint_distance_rad: L2 distance from the current joint positions.
        within_limits: Whether the solution respects joint limits.
    """

    joint_angles_rad: np.ndarray
    ik_result: IKResult
    joint_distance_rad: float
    within_limits: bool


class SolutionSelector:
    """Selects the best IK solution from candidates.

    Filtering order:
    1. Discard failed solutions
    2. Discard near-singular solutions
    3. Clamp joints to limits, reject if clamping is too extreme
    4. Select the solution closest (L2 distance) to the current joint positions
    """

    def __init__(
        self,
        limits: JointLimits,
        manipulability_threshold: float = 0.001,
    ) -> None:
        self._limits = limits
        self._manipulability_threshold = manipulability_threshold

    def select(
        self,
        candidates: List[IKResult],
        current_joints_rad: np.ndarray,
    ) -> Optional[SelectedSolution]:
        """Select the best solution from a list of IK candidates.

        Args:
            candidates: IK results from multi-start solve (may include failed ones).
            current_joints_rad: Current joint angles for continuity preference.

        Returns:
            SelectedSolution or None if no valid candidate exists.
        """
        current = np.asarray(current_joints_rad, dtype=np.float64)
        valid: List[tuple[IKResult, float, np.ndarray]] = []

        for result in candidates:
            if not result.success:
                continue

            # Reject near-singular solutions
            if result.near_singular:
                continue

            q = result.joint_angles_rad

            # Check limits (before clamping)
            if not self._limits.is_within_limits(q):
                # Reject if the violation is too large (> 0.1 rad outside)
                violations = np.maximum(
                    self._limits.angle_min_rad - q,
                    q - self._limits.angle_max_rad,
                )
                if np.any(violations > 0.1):
                    continue

            # Clamp to limits
            q_clamped = self._limits.clamp(q)
            distance = float(np.linalg.norm(q_clamped - current))

            valid.append((result, distance, q_clamped))

        if not valid:
            return None

        # Select the one closest to current joint positions (most continuous)
        valid.sort(key=lambda x: x[1])

        best_result, best_distance, best_angles = valid[0]

        return SelectedSolution(
            joint_angles_rad=best_angles,
            ik_result=best_result,
            joint_distance_rad=best_distance,
            within_limits=self._limits.is_within_limits(best_angles),
        )

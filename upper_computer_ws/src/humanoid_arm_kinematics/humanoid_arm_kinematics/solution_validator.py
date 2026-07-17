"""Solution validator: forward-kinematics verification of IK results.

Re-computes FK on each IK candidate and checks position/orientation error
against configurable thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forward_solver import ForwardSolver
from .solution_selector import JointLimits, SelectedSolution


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating an IK solution with forward kinematics.

    Attributes:
        valid: Whether the solution passes all checks.
        position_error_m: Euclidean position error in metres.
        orientation_error_rad: Absolute yaw error in radians.
        within_joint_limits: Whether joints are within the allowed range.
    """

    valid: bool
    position_error_m: float
    orientation_error_rad: float
    within_joint_limits: bool


class SolutionValidator:
    """Validates IK solutions by re-running forward kinematics.

    Checks:
    1. Joint angle limits
    2. Position error (FK vs target)
    3. Orientation/yaw error (FK vs target)
    """

    def __init__(
        self,
        forward_solver: ForwardSolver,
        limits: JointLimits,
        max_position_error_m: float = 0.005,
        max_orientation_error_rad: float = 0.02,
    ) -> None:
        self._fk = forward_solver
        self._limits = limits
        self._max_pos_err = max_position_error_m
        self._max_ori_err = max_orientation_error_rad

    def validate(
        self,
        solution: SelectedSolution,
        target_position: np.ndarray,
        target_yaw_rad: float,
    ) -> ValidationResult:
        """Validate an IK solution against the target.

        Args:
            solution: The selected IK solution.
            target_position: 3-element target [x, y, z].
            target_yaw_rad: Target yaw angle in radians.

        Returns:
            ValidationResult indicating pass/fail.
        """
        q = solution.joint_angles_rad

        # Forward kinematics
        fk = self._fk.solve(q)

        # Position error
        pos_error = float(np.linalg.norm(target_position - fk.position))

        # Yaw error (shortest angular distance)
        yaw_error = target_yaw_rad - fk.yaw_rad
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        yaw_error = float(abs(yaw_error))

        # Joint limits
        within_limits = self._limits.is_within_limits(q)

        valid = (
            pos_error <= self._max_pos_err
            and yaw_error <= self._max_ori_err
            and within_limits
        )

        return ValidationResult(
            valid=valid,
            position_error_m=pos_error,
            orientation_error_rad=yaw_error,
            within_joint_limits=within_limits,
        )

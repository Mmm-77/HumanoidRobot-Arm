"""Numerical inverse kinematics solver using damped least squares.

Solves for joint angles given a 4-DOF task target [x, y, z, θ] where θ is
the controllable yaw angle. Uses Levenberg-Marquardt damping with dynamic
lambda adjustment and multi-start initial guesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .forward_solver import ForwardResult, ForwardSolver
from .jacobian import JacobianResult, JacobianSolver


class InverseSolverError(RuntimeError):
    """Raised when inverse kinematics cannot find a valid solution."""


@dataclass(frozen=True)
class IKConfig:
    """Configuration for the inverse kinematics solver.

    Attributes:
        max_iterations: Maximum iterations per solve attempt.
        position_tolerance_m: Convergence threshold for position error (m).
        orientation_tolerance_rad: Convergence threshold for orientation error (rad).
        initial_lambda: Starting damping factor.
        lambda_increase_factor: Multiplier when error increases.
        lambda_decrease_factor: Multiplier when error decreases.
        lambda_min: Minimum damping value.
        lambda_max: Maximum damping value.
        multi_start_attempts: Number of perturbed initial guesses.
        multi_start_perturbation_rad: Max perturbation for multi-start.
        joint_angle_min_rad: Per-joint minimum allowed angle (rad). None = unbounded.
        joint_angle_max_rad: Per-joint maximum allowed angle (rad). None = unbounded.
    """

    max_iterations: int = 200
    position_tolerance_m: float = 0.001
    orientation_tolerance_rad: float = 0.01
    initial_lambda: float = 0.1
    lambda_increase_factor: float = 2.0
    lambda_decrease_factor: float = 0.5
    lambda_min: float = 1e-6
    lambda_max: float = 1.0
    multi_start_attempts: int = 5
    multi_start_perturbation_rad: float = 0.3
    joint_angle_min_rad: Optional[np.ndarray] = None
    joint_angle_max_rad: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.position_tolerance_m <= 0:
            raise ValueError("position_tolerance_m must be > 0")
        if self.orientation_tolerance_rad <= 0:
            raise ValueError("orientation_tolerance_rad must be > 0")
        if not (0 < self.initial_lambda <= self.lambda_max):
            raise ValueError("initial_lambda must be in (0, lambda_max]")
        if self.lambda_increase_factor <= 1:
            raise ValueError("lambda_increase_factor must be > 1")
        if not (0 < self.lambda_decrease_factor < 1):
            raise ValueError("lambda_decrease_factor must be in (0, 1)")
        if not (0 < self.lambda_min < self.lambda_max):
            raise ValueError("lambda_min must be in (0, lambda_max)")
        if self.multi_start_attempts < 1:
            raise ValueError("multi_start_attempts must be >= 1")


@dataclass(frozen=True)
class IKResult:
    """Result of an inverse kinematics computation.

    Attributes:
        success: Whether a valid solution was found.
        joint_angles_rad: The 4 joint angles in radians (if successful).
        forward_result: FK result for the solution (if successful).
        iterations: Number of iterations used.
        final_error: Final task-space error norm.
        error_norm: Norm of the final error.
        position_error_m: Final position error magnitude.
        orientation_error_rad: Final orientation error magnitude.
        near_singular: Whether the solution is near a singularity.
    """

    success: bool
    joint_angles_rad: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    forward_result: Optional[ForwardResult] = None
    iterations: int = 0
    final_error: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )
    error_norm: float = 0.0
    position_error_m: float = 0.0
    orientation_error_rad: float = 0.0
    near_singular: bool = False

    @classmethod
    def failed(cls, iterations: int = 0) -> "IKResult":
        """Convenience factory for a failed result."""
        return cls(success=False, iterations=iterations)


class InverseSolver:
    """Numerical inverse kinematics solver using damped least squares.

    Usage:
        ik = InverseSolver(forward_solver, jacobian_solver, config)
        result = ik.solve(target_pos, target_yaw, initial_guess)
    """

    def __init__(
        self,
        forward_solver: ForwardSolver,
        jacobian_solver: JacobianSolver,
        config: IKConfig,
    ) -> None:
        self._fk = forward_solver
        self._jac = jacobian_solver
        self._config = config

    @property
    def config(self) -> IKConfig:
        return self._config

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to configured limits (if any)."""
        lo = self._config.joint_angle_min_rad
        hi = self._config.joint_angle_max_rad
        if lo is not None and hi is not None:
            return np.clip(q, lo, hi)
        return q

    def solve(
        self,
        target_position: np.ndarray,
        target_yaw_rad: float,
        initial_guess_rad: np.ndarray,
        manipulability_threshold: float = 0.001,
    ) -> IKResult:
        """Solve inverse kinematics using damped least squares with multi-start.

        Args:
            target_position: 3-element [x, y, z] target position in metres.
            target_yaw_rad: Target yaw angle (rotation about base Z) in radians.
            initial_guess_rad: 4-element starting joint angles in radians.
            manipulability_threshold: Minimum manipulability for non-singular.

        Returns:
            IKResult with the best solution found.
        """
        # Wrap yaw to [-π, π] range for the target
        target_yaw = (target_yaw_rad + np.pi) % (2 * np.pi) - np.pi

        # Generate multi-start guesses
        guesses = self._generate_guesses(initial_guess_rad)

        best_result: Optional[IKResult] = None

        for guess in guesses:
            result = self._solve_single(
                target_position, target_yaw, guess, manipulability_threshold
            )
            if result.success:
                if best_result is None or result.error_norm < best_result.error_norm:
                    best_result = result

        if best_result is not None:
            return best_result

        return IKResult.failed()

    def _generate_guesses(self, base_guess: np.ndarray) -> List[np.ndarray]:
        """Generate perturbed initial guesses for multi-start.

        The first guess is the base guess unchanged. Subsequent guesses add
        random perturbations.
        """
        guesses: List[np.ndarray] = [base_guess.copy()]

        rng = np.random.RandomState(42)
        for _ in range(1, self._config.multi_start_attempts):
            perturbation = rng.uniform(
                -self._config.multi_start_perturbation_rad,
                self._config.multi_start_perturbation_rad,
                size=base_guess.shape,
            )
            guesses.append(base_guess + perturbation)

        return guesses

    def _solve_single(
        self,
        target_position: np.ndarray,
        target_yaw: float,
        initial_guess: np.ndarray,
        manipulability_threshold: float,
    ) -> IKResult:
        """Single-start damped least squares IK iteration."""
        q = np.asarray(initial_guess, dtype=np.float64).copy()
        lam = self._config.initial_lambda
        prev_error = np.inf

        for iteration in range(self._config.max_iterations):
            # Forward kinematics
            fk = self._fk.solve(q)
            jac = self._jac.compute(q)

            # Compute task error
            pos_error = target_position - fk.position

            # Yaw error: shortest angular distance
            yaw_error = target_yaw - fk.yaw_rad
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

            error = np.array([
                pos_error[0], pos_error[1], pos_error[2], yaw_error
            ], dtype=np.float64)

            pos_norm = float(np.linalg.norm(pos_error))
            yaw_abs = float(abs(yaw_error))

            # Check convergence
            if pos_norm < self._config.position_tolerance_m and \
               yaw_abs < self._config.orientation_tolerance_rad:
                return IKResult(
                    success=True,
                    joint_angles_rad=q,
                    forward_result=fk,
                    iterations=iteration + 1,
                    final_error=error,
                    error_norm=float(np.linalg.norm(error)),
                    position_error_m=pos_norm,
                    orientation_error_rad=yaw_abs,
                    near_singular=jac.manipulability < manipulability_threshold,
                )

            # Damped least squares step
            J = jac.jacobian_task
            JTJ = J.T @ J
            damped = JTJ + (lam ** 2) * np.eye(self._fk.model.num_joints)
            try:
                delta_q = np.linalg.solve(damped, J.T @ error)
            except np.linalg.LinAlgError:
                # Matrix inversion failure — increase damping and retry
                lam = min(lam * self._config.lambda_increase_factor,
                          self._config.lambda_max)
                continue

            # Update
            q_new = q + delta_q
            q_new = self._clamp(q_new)  # enforce joint limits

            # Evaluate new error
            fk_new = self._fk.solve(q_new)
            pos_error_new = target_position - fk_new.position
            yaw_error_new = target_yaw - fk_new.yaw_rad
            yaw_error_new = (yaw_error_new + np.pi) % (2 * np.pi) - np.pi
            error_new = np.array([
                pos_error_new[0], pos_error_new[1], pos_error_new[2], yaw_error_new
            ])
            error_norm_new = float(np.linalg.norm(error_new))

            # Adaptive damping
            if error_norm_new < prev_error:
                q = q_new
                prev_error = error_norm_new
                lam = max(lam * self._config.lambda_decrease_factor,
                          self._config.lambda_min)
            else:
                lam = min(lam * self._config.lambda_increase_factor,
                          self._config.lambda_max)

        return IKResult.failed(iterations=self._config.max_iterations)

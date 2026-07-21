"""Position-only inverse kinematics for the 4-DOF URDF chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .forward_solver import ForwardResult, ForwardSolver
from .jacobian import JacobianSolver


class InverseSolverError(RuntimeError):
    """Raised when inverse-kinematics inputs are malformed."""


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 250
    position_tolerance_m: float = 0.001
    initial_lambda: float = 0.03
    lambda_increase_factor: float = 2.0
    lambda_decrease_factor: float = 0.5
    lambda_min: float = 1e-5
    lambda_max: float = 1.0
    multi_start_attempts: int = 8
    multi_start_perturbation_rad: float = 0.5
    continuity_gain: float = 0.02
    max_step_rad: float = 0.25

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be > 0")
        if not 0.0 < self.lambda_min <= self.initial_lambda <= self.lambda_max:
            raise ValueError("Require lambda_min <= initial_lambda <= lambda_max")
        if self.lambda_increase_factor <= 1.0:
            raise ValueError("lambda_increase_factor must be > 1")
        if not 0.0 < self.lambda_decrease_factor < 1.0:
            raise ValueError("lambda_decrease_factor must be in (0, 1)")
        if self.multi_start_attempts < 1:
            raise ValueError("multi_start_attempts must be >= 1")
        if self.multi_start_perturbation_rad < 0.0:
            raise ValueError("multi_start_perturbation_rad must be >= 0")
        if self.continuity_gain < 0.0:
            raise ValueError("continuity_gain must be >= 0")
        if self.max_step_rad <= 0.0:
            raise ValueError("max_step_rad must be > 0")


@dataclass(frozen=True)
class IKResult:
    success: bool
    joint_angles_rad: np.ndarray = field(default_factory=lambda: np.zeros(4))
    forward_result: Optional[ForwardResult] = None
    iterations: int = 0
    final_error: np.ndarray = field(default_factory=lambda: np.zeros(3))
    error_norm: float = float("inf")
    position_error_m: float = float("inf")
    near_singular: bool = False

    @classmethod
    def failed(
        cls, iterations: int = 0, error_norm: float = float("inf")
    ) -> "IKResult":
        return cls(success=False, iterations=iterations, error_norm=error_norm)


def _nearest_equivalent(angles: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Choose the 2*pi-equivalent representation nearest to ``reference``."""
    return reference + (angles - reference + np.pi) % (2.0 * np.pi) - np.pi


class InverseSolver:
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

    def solve(
        self,
        target_position: Sequence[float],
        initial_guess_rad: Sequence[float],
    ) -> IKResult:
        target = np.asarray(target_position, dtype=np.float64)
        reference = np.asarray(initial_guess_rad, dtype=np.float64)
        joint_count = self._fk.model.num_joints
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise InverseSolverError("Target position must be a finite 3-vector")
        if reference.shape != (joint_count,) or not np.all(np.isfinite(reference)):
            raise InverseSolverError(
                f"Initial guess must contain {joint_count} finite angles"
            )

        rng = np.random.RandomState(42)
        guesses = [reference.copy()]
        for _ in range(1, self._config.multi_start_attempts):
            guesses.append(
                reference
                + rng.uniform(
                    -self._config.multi_start_perturbation_rad,
                    self._config.multi_start_perturbation_rad,
                    joint_count,
                )
            )

        successful: List[IKResult] = []
        best_failure = IKResult.failed()
        for guess in guesses:
            result = self._solve_single(target, reference, guess)
            if result.success:
                successful.append(result)
            elif result.error_norm < best_failure.error_norm:
                best_failure = result
        if not successful:
            return best_failure

        def score(result: IKResult) -> float:
            distance = np.linalg.norm(result.joint_angles_rad - reference)
            return result.position_error_m + self._config.continuity_gain * distance

        return min(successful, key=score)

    def _solve_single(
        self, target: np.ndarray, reference: np.ndarray, initial: np.ndarray
    ) -> IKResult:
        q = _nearest_equivalent(initial.copy(), reference)
        damping = self._config.initial_lambda
        best_error = float("inf")

        for iteration in range(1, self._config.max_iterations + 1):
            forward = self._fk.solve(q)
            error = target - forward.position
            error_norm = float(np.linalg.norm(error))
            jacobian_result = self._jac.compute(q)
            if error_norm <= self._config.position_tolerance_m:
                return IKResult(
                    success=True,
                    joint_angles_rad=_nearest_equivalent(q, reference),
                    forward_result=forward,
                    iterations=iteration,
                    final_error=error,
                    error_norm=error_norm,
                    position_error_m=error_norm,
                    near_singular=jacobian_result.near_singular,
                )

            jacobian = jacobian_result.jacobian_task
            regularized = jacobian @ jacobian.T + damping**2 * np.eye(3)
            primary = jacobian.T @ np.linalg.solve(regularized, error)

            # Use the null space to remain close to the current measured pose.
            pseudo_inverse = jacobian.T @ np.linalg.solve(
                regularized, np.eye(3)
            )
            null_space = np.eye(self._fk.model.num_joints) - pseudo_inverse @ jacobian
            continuity = self._config.continuity_gain * (reference - q)
            delta = primary + null_space @ continuity
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > self._config.max_step_rad:
                delta *= self._config.max_step_rad / delta_norm

            candidate = _nearest_equivalent(q + delta, reference)
            candidate_error = float(
                np.linalg.norm(target - self._fk.solve(candidate).position)
            )
            if candidate_error < error_norm:
                q = candidate
                best_error = min(best_error, candidate_error)
                damping = max(
                    damping * self._config.lambda_decrease_factor,
                    self._config.lambda_min,
                )
            else:
                damping = min(
                    damping * self._config.lambda_increase_factor,
                    self._config.lambda_max,
                )

        forward = self._fk.solve(q)
        final_error = target - forward.position
        final_norm = float(np.linalg.norm(final_error))
        return IKResult(
            success=False,
            joint_angles_rad=q,
            forward_result=forward,
            iterations=self._config.max_iterations,
            final_error=final_error,
            error_norm=final_norm,
            position_error_m=final_norm,
            near_singular=self._jac.compute(q).near_singular,
        )

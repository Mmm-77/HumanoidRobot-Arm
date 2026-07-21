"""Geometric and position Jacobians for the URDF-defined chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .robot_model import RobotModel


class JacobianError(RuntimeError):
    """Raised when a Jacobian cannot be evaluated."""


@dataclass(frozen=True)
class JacobianResult:
    joint_angles_rad: np.ndarray
    jacobian_geom: np.ndarray
    jacobian_task: np.ndarray
    manipulability: float
    near_singular: bool


class JacobianSolver:
    """Compute a 6xN geometric Jacobian and 3xN position Jacobian."""

    def __init__(self, model: RobotModel) -> None:
        self._model = model

    def compute(
        self, joint_angles_rad: Sequence[float], singular_value_threshold: float = 1e-5
    ) -> JacobianResult:
        angles = np.asarray(joint_angles_rad, dtype=np.float64)
        if angles.shape != (self._model.num_joints,) or not np.all(
            np.isfinite(angles)
        ):
            raise JacobianError(
                f"Expected {self._model.num_joints} finite joint angles"
            )
        state = self._model.evaluate(angles)
        tip_position = state.tip_transform[:3, 3]
        geometric = np.zeros((6, self._model.num_joints), dtype=np.float64)
        for index, (position, axis) in enumerate(
            zip(state.joint_positions, state.joint_axes)
        ):
            geometric[:3, index] = np.cross(axis, tip_position - position)
            geometric[3:, index] = axis
        task = geometric[:3].copy()
        singular_values = np.linalg.svd(task, compute_uv=False)
        manipulability = float(np.prod(singular_values))
        return JacobianResult(
            joint_angles_rad=angles.copy(),
            jacobian_geom=geometric,
            jacobian_task=task,
            manipulability=manipulability,
            near_singular=bool(singular_values[-1] < singular_value_threshold),
        )

    def is_near_singular(
        self, joint_angles_rad: Sequence[float], threshold: float = 1e-5
    ) -> bool:
        return self.compute(joint_angles_rad, threshold).near_singular

    def compute_joint_velocities(
        self,
        joint_angles_rad: Sequence[float],
        task_velocity: Sequence[float],
        damping: float = 0.01,
    ) -> np.ndarray:
        jacobian = self.compute(joint_angles_rad).jacobian_task
        velocity = np.asarray(task_velocity, dtype=np.float64)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise JacobianError("Task velocity must be a finite 3-vector")
        regularized = jacobian @ jacobian.T + damping**2 * np.eye(3)
        return jacobian.T @ np.linalg.solve(regularized, velocity)

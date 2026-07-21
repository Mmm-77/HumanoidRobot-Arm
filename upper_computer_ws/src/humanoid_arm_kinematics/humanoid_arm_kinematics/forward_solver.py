"""Forward kinematics for the URDF-defined base-to-tip chain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .robot_model import ModelError, RobotModel


class ForwardSolverError(RuntimeError):
    """Raised when forward kinematics cannot be evaluated."""


@dataclass(frozen=True)
class ForwardResult:
    joint_angles_rad: np.ndarray
    transform: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    quaternion_xyzw: np.ndarray
    yaw_rad: float


class ForwardSolver:
    def __init__(self, model: RobotModel) -> None:
        self._model = model

    @property
    def model(self) -> RobotModel:
        return self._model

    def solve(self, joint_angles_rad: Sequence[float]) -> ForwardResult:
        angles = np.asarray(joint_angles_rad, dtype=np.float64)
        if angles.shape != (self._model.num_joints,):
            raise ForwardSolverError(
                f"Expected {self._model.num_joints} joint angles, "
                f"got shape {angles.shape}"
            )
        if not np.all(np.isfinite(angles)):
            raise ForwardSolverError("Joint angles must be finite")
        try:
            transform = self._model.forward_kinematics(angles)
        except ModelError as exc:
            raise ForwardSolverError(str(exc)) from exc
        rotation = self._model.extract_rotation(transform)
        return ForwardResult(
            joint_angles_rad=angles.copy(),
            transform=transform,
            position=self._model.extract_position(transform),
            rotation=rotation,
            quaternion_xyzw=self._model.rotation_matrix_to_quaternion(rotation),
            yaw_rad=self._model.extract_yaw(rotation),
        )

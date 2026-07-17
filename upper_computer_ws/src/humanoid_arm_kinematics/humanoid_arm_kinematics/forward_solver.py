"""Forward kinematics solver for the 4-DOF humanoid arm.

Computes the end-effector pose (position + orientation) given four joint angles
using the Modified DH parameter model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot_model import RobotModel


class ForwardSolverError(RuntimeError):
    """Raised when forward kinematics cannot be computed."""


@dataclass(frozen=True)
class ForwardResult:
    """Result of a forward kinematics computation.

    Attributes:
        joint_angles_rad: The four joint angles [θ₁, θ₂, θ₃, θ₄] in radians.
        transform: 4x4 homogeneous transform of the end-effector in the base frame.
        position: 3-element position [x, y, z] in metres.
        rotation: 3x3 rotation matrix of the end-effector.
        quaternion_xyzw: Orientation as quaternion [x, y, z, w].
        yaw_rad: End-effector yaw angle (rotation about base Z) in radians.
    """

    joint_angles_rad: np.ndarray
    transform: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    quaternion_xyzw: np.ndarray
    yaw_rad: float


class ForwardSolver:
    """Computes the end-effector pose from joint angles.

    Usage:
        model = RobotModel.from_config(dh_params)
        solver = ForwardSolver(model)
        result = solver.solve([0.0, 0.5, -1.0, 0.3])
    """

    def __init__(self, model: RobotModel) -> None:
        self._model = model

    @property
    def model(self) -> RobotModel:
        return self._model

    def solve(self, joint_angles_rad: np.ndarray) -> ForwardResult:
        """Compute the end-effector pose for the given joint angles.

        Args:
            joint_angles_rad: Array of 4 joint angles in radians.

        Returns:
            ForwardResult with position, rotation, quaternion, and yaw.

        Raises:
            ForwardSolverError: If the joint angles are invalid.
        """
        if len(joint_angles_rad) != self._model.num_joints:
            raise ForwardSolverError(
                f"Expected {self._model.num_joints} joint angles, "
                f"got {len(joint_angles_rad)}"
            )
        if not np.all(np.isfinite(joint_angles_rad)):
            raise ForwardSolverError("Joint angles must be finite")

        angles = np.asarray(joint_angles_rad, dtype=np.float64)
        model = self._model.with_joint_angles(angles)

        T = model.forward_kinematics()
        position = model.extract_position(T)
        rotation = model.extract_rotation(T)
        quaternion = model.rotation_matrix_to_quaternion(rotation)
        yaw = model.extract_yaw(rotation)

        return ForwardResult(
            joint_angles_rad=angles.copy(),
            transform=T,
            position=position,
            rotation=rotation,
            quaternion_xyzw=quaternion,
            yaw_rad=yaw,
        )

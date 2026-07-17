"""Jacobian computation and singularity detection for the 4-DOF arm.

Computes the geometric Jacobian and the reduced task Jacobian (4x4) that maps
joint velocities to [ẋ, ẏ, ż, θ̇] where θ is the controllable yaw angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot_model import RobotModel


class JacobianError(RuntimeError):
    """Raised when the Jacobian cannot be computed."""


@dataclass(frozen=True)
class JacobianResult:
    """Result of Jacobian computation.

    Attributes:
        joint_angles_rad: Joint angles used.
        jacobian_geom: Full 6x4 geometric Jacobian.
        jacobian_task: Reduced 4x4 task Jacobian.
        manipulability: Scalar manipulability measure μ = sqrt(det(J*Jᵀ)).
        near_singular: Whether the configuration is near-singular.
    """

    joint_angles_rad: np.ndarray
    jacobian_geom: np.ndarray
    jacobian_task: np.ndarray
    manipulability: float
    near_singular: bool


class JacobianSolver:
    """Computes the geometric and task Jacobians for the robot arm.

    The task Jacobian is a 4x4 matrix where:
      - Rows 0-2: linear velocity of the end-effector (positional Jacobian)
      - Row 3:    angular velocity about the base Z axis (yaw rate)
    """

    def __init__(self, model: RobotModel) -> None:
        self._model = model

    def compute(self, joint_angles_rad: np.ndarray) -> JacobianResult:
        """Compute the Jacobians for the given joint configuration.

        Args:
            joint_angles_rad: 4 joint angles in radians.

        Returns:
            JacobianResult with geometric and task Jacobians.
        """
        angles = np.asarray(joint_angles_rad, dtype=np.float64)
        model = self._model.with_joint_angles(angles)

        frames = model.get_frame_transforms()
        # frames[0] = base (identity), frames[1..4] = link frames, frames[5] = tool
        # In Modified DH, the rotation axis for joint i (θ_i) is the Z axis of
        # frame i (frames[i+1]), because RotZ(θ_i) is applied AFTER RotX and
        # TransX but BEFORE TransZ(d_i).  TransZ(d_i) does not change the Z
        # direction, so frames[i+1][:3,2] is the correct screw-axis direction.

        end_pos = frames[-1][:3, 3]  # tool position in base frame

        J = np.zeros((6, self._model.num_joints), dtype=np.float64)

        for i in range(self._model.num_joints):
            # Rotation axis for joint i is the Z axis of frame i+1 (link frame i)
            z_i = frames[i + 1][:3, 2]
            # A point on the axis: origin of frame i+1 (also on the axis)
            p_i = frames[i + 1][:3, 3]

            # For revolute joints:
            # J_p_i = z_i × (p_n - p_i)
            # J_ω_i = z_i
            J[:3, i] = np.cross(z_i, end_pos - p_i)
            J[3:, i] = z_i

        # Task Jacobian: 4x4
        # Position rows (0,1,2) + yaw-rate row (row 5 of geometric J = Z angular velocity)
        J_task = np.zeros((4, self._model.num_joints), dtype=np.float64)
        J_task[:3, :] = J[:3, :]  # linear velocity
        J_task[3, :] = J[5, :]    # angular velocity about Z

        # Manipulability: μ = sqrt(det(J_task * J_taskᵀ))
        # For a square 4x4, this simplifies to |det(J_task)|
        manipulability = float(np.abs(np.linalg.det(J_task)))

        return JacobianResult(
            joint_angles_rad=angles.copy(),
            jacobian_geom=J,
            jacobian_task=J_task,
            manipulability=manipulability,
            near_singular=False,  # caller sets this based on threshold
        )

    def is_near_singular(
        self, joint_angles_rad: np.ndarray, threshold: float
    ) -> bool:
        """Check whether the configuration is near a singularity.

        Args:
            joint_angles_rad: Joint angles.
            threshold: Minimum manipulability value considered non-singular.

        Returns:
            True if the configuration is near-singular.
        """
        result = self.compute(joint_angles_rad)
        return result.manipulability < threshold

    def compute_joint_velocities(
        self,
        joint_angles_rad: np.ndarray,
        task_velocity: np.ndarray,
        damping: float = 0.1,
    ) -> np.ndarray:
        """Map task-space velocity to joint velocities using damped least squares.

        Δq = (JᵀJ + λ²I)⁻¹ Jᵀ Δx

        Args:
            joint_angles_rad: Current joint angles.
            task_velocity: 4-element task velocity [vx, vy, vz, ωθ].
            damping: Damping factor λ for singularity robustness.

        Returns:
            4-element joint velocity vector.
        """
        result = self.compute(joint_angles_rad)
        J = result.jacobian_task
        v = np.asarray(task_velocity, dtype=np.float64)

        # Damped least squares: (JᵀJ + λ²I)⁻¹ Jᵀ v
        JTJ = J.T @ J
        damping_matrix = (damping ** 2) * np.eye(self._model.num_joints)
        J_pinv = np.linalg.solve(JTJ + damping_matrix, J.T)

        return J_pinv @ v

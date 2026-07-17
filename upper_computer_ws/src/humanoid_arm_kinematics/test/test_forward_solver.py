"""Tests for forward kinematics solver."""

import numpy as np
import pytest

from humanoid_arm_kinematics.forward_solver import ForwardResult, ForwardSolver, ForwardSolverError
from humanoid_arm_kinematics.robot_model import RobotModel


# Modified DH parameters from dh_parameters.md (modified table, cm→m ÷100)
DH_PARAMS = [
    {"alpha_prev_deg": -90.0, "a_prev_m": 0.002, "d_m": 0.003},
    {"alpha_prev_deg": 90.0, "a_prev_m": 0.0289, "d_m": 0.05},
    {"alpha_prev_deg": -90.0, "a_prev_m": 0.0, "d_m": 0.07},
    {"alpha_prev_deg": 90.0, "a_prev_m": 0.0, "d_m": 0.035},
]


@pytest.fixture
def model() -> RobotModel:
    return RobotModel.from_config(DH_PARAMS)


@pytest.fixture
def solver(model: RobotModel) -> ForwardSolver:
    return ForwardSolver(model)


class TestForwardSolver:
    def test_zero_angles_produces_finite_pose(self, solver: ForwardSolver) -> None:
        """Zero joint angles should produce a finite, valid FK result."""
        result = solver.solve(np.zeros(4))

        assert isinstance(result, ForwardResult)
        assert np.all(np.isfinite(result.position))
        assert np.all(np.isfinite(result.rotation))
        assert np.all(np.isfinite(result.quaternion_xyzw))
        assert np.isfinite(result.yaw_rad)

    def test_quaternion_is_normalized(self, solver: ForwardSolver) -> None:
        """FK quaternion output must be unit length."""
        result = solver.solve(np.array([0.1, 0.2, 0.3, 0.4]))
        norm = float(np.linalg.norm(result.quaternion_xyzw))
        assert abs(norm - 1.0) < 1e-12

    def test_transform_is_valid_homogeneous(self, solver: ForwardSolver) -> None:
        """The FK transform must be a valid 4x4 homogeneous matrix."""
        result = solver.solve(np.array([0.5, -0.3, 1.0, -0.7]))
        T = result.transform
        assert T.shape == (4, 4)
        # Bottom row must be [0, 0, 0, 1]
        assert np.allclose(T[3, :], [0, 0, 0, 1])
        # Rotation submatrix must be orthogonal
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_fk_consistency_with_same_angles(self, solver: ForwardSolver) -> None:
        """Repeated FK with the same angles should give identical results."""
        angles = np.array([0.2, 0.5, -0.8, 1.2])
        r1 = solver.solve(angles)
        r2 = solver.solve(angles)
        assert np.allclose(r1.position, r2.position)
        assert np.allclose(r1.quaternion_xyzw, r2.quaternion_xyzw)

    def test_rejects_wrong_number_of_angles(self, solver: ForwardSolver) -> None:
        """Passing the wrong number of joint angles raises an error."""
        with pytest.raises(ForwardSolverError):
            solver.solve(np.array([0.0, 0.0, 0.0]))  # 3 instead of 4

    def test_rejects_nonfinite_angles(self, solver: ForwardSolver) -> None:
        """Non-finite joint angles raise an error."""
        with pytest.raises(ForwardSolverError):
            solver.solve(np.array([0.0, np.nan, 0.0, 0.0]))
        with pytest.raises(ForwardSolverError):
            solver.solve(np.array([0.0, np.inf, 0.0, 0.0]))

    def test_yaw_matches_transform_rotation(self, solver: ForwardSolver) -> None:
        """The extracted yaw should match atan2(R[1,0], R[0,0])."""
        angles = np.array([0.3, 1.0, -0.5, 0.8])
        result = solver.solve(angles)
        expected_yaw = float(np.arctan2(result.rotation[1, 0], result.rotation[0, 0]))
        assert abs(result.yaw_rad - expected_yaw) < 1e-12

    def test_position_changes_with_joint_angles(self, solver: ForwardSolver) -> None:
        """Different joint angles should produce different positions."""
        r1 = solver.solve(np.array([0.0, 0.0, 0.0, 0.0]))
        r2 = solver.solve(np.array([0.5, 0.5, 0.5, 0.5]))
        assert not np.allclose(r1.position, r2.position)

"""Tests for inverse kinematics solver."""

import numpy as np
import pytest

from humanoid_arm_kinematics.forward_solver import ForwardSolver
from humanoid_arm_kinematics.inverse_solver import IKConfig, IKResult, InverseSolver
from humanoid_arm_kinematics.jacobian import JacobianSolver
from humanoid_arm_kinematics.robot_model import RobotModel


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
def fk(model: RobotModel) -> ForwardSolver:
    return ForwardSolver(model)


@pytest.fixture
def jac(model: RobotModel) -> JacobianSolver:
    return JacobianSolver(model)


@pytest.fixture
def ik_config() -> IKConfig:
    return IKConfig(
        max_iterations=300,
        position_tolerance_m=0.001,
        orientation_tolerance_rad=0.01,
        initial_lambda=0.1,
        lambda_increase_factor=2.0,
        lambda_decrease_factor=0.5,
        lambda_min=1e-6,
        lambda_max=1.0,
        multi_start_attempts=8,
        multi_start_perturbation_rad=0.5,
    )


@pytest.fixture
def ik(fk: ForwardSolver, jac: JacobianSolver, ik_config: IKConfig) -> InverseSolver:
    return InverseSolver(fk, jac, ik_config)


class TestInverseSolver:
    def test_recovers_known_pose_from_fk(self, fk: ForwardSolver, ik: InverseSolver) -> None:
        """Given an FK result, IK should recover similar joint angles."""
        original_angles = np.array([0.3, 0.8, -0.5, 0.6])
        fk_result = fk.solve(original_angles)

        ik_result = ik.solve(
            fk_result.position,
            fk_result.yaw_rad,
            initial_guess_rad=original_angles + 0.1,  # perturb slightly
        )

        assert ik_result.success, f"IK should converge; got error_norm={ik_result.error_norm}"
        assert ik_result.position_error_m < 0.01
        assert ik_result.orientation_error_rad < 0.05

        # Verify FK of the solution matches the target
        fk_check = fk.solve(ik_result.joint_angles_rad)
        pos_err = float(np.linalg.norm(fk_result.position - fk_check.position))
        yaw_err = abs(fk_result.yaw_rad - fk_check.yaw_rad)
        yaw_err = min(yaw_err, 2 * np.pi - yaw_err)
        assert pos_err < 0.01
        assert yaw_err < 0.1

    def test_recovers_zero_configuration(self, fk: ForwardSolver, ik: InverseSolver) -> None:
        """IK from a distant initial guess should converge to a zero-config target."""
        zero_angles = np.zeros(4)
        fk_result = fk.solve(zero_angles)

        ik_result = ik.solve(
            fk_result.position,
            fk_result.yaw_rad,
            initial_guess_rad=np.array([0.5, 0.5, -0.5, 0.5]),  # distant guess
        )

        assert ik_result.success
        fk_check = fk.solve(ik_result.joint_angles_rad)
        pos_err = float(np.linalg.norm(fk_result.position - fk_check.position))
        assert pos_err < 0.01

    def test_result_has_finite_joint_angles(self, fk: ForwardSolver, ik: InverseSolver) -> None:
        """A successful IK result must have finite joint angles."""
        angles = np.array([0.2, 0.4, -0.3, 0.1])
        fk_result = fk.solve(angles)

        ik_result = ik.solve(fk_result.position, fk_result.yaw_rad, angles)

        if ik_result.success:
            assert np.all(np.isfinite(ik_result.joint_angles_rad))
            fk_check = fk.solve(ik_result.joint_angles_rad)
            assert np.all(np.isfinite(fk_check.position))

    def test_ik_config_validation(self) -> None:
        """IKConfig should reject invalid values."""
        with pytest.raises(ValueError):
            IKConfig(max_iterations=0)
        with pytest.raises(ValueError):
            IKConfig(position_tolerance_m=0.0)
        with pytest.raises(ValueError):
            IKConfig(multi_start_attempts=0)
        with pytest.raises(ValueError):
            IKConfig(lambda_increase_factor=0.5)  # must be > 1

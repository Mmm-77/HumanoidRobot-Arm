"""Tests for solution validator and solution selector."""

import numpy as np
import pytest

from humanoid_arm_kinematics.forward_solver import ForwardSolver
from humanoid_arm_kinematics.inverse_solver import IKConfig, IKResult, InverseSolver
from humanoid_arm_kinematics.jacobian import JacobianSolver
from humanoid_arm_kinematics.robot_model import RobotModel
from humanoid_arm_kinematics.solution_selector import JointLimits, SolutionSelector
from humanoid_arm_kinematics.solution_validator import SolutionValidator


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
def limits() -> JointLimits:
    return JointLimits(
        angle_min_rad=np.deg2rad([-85, -10, -100, -40]),
        angle_max_rad=np.deg2rad([175, 150, 100, 100]),
        max_velocity_rad_per_s=np.deg2rad([180, 180, 180, 180]),
    )


@pytest.fixture
def validator(fk: ForwardSolver, limits: JointLimits) -> SolutionValidator:
    return SolutionValidator(fk, limits)


@pytest.fixture
def selector(limits: JointLimits) -> SolutionSelector:
    return SolutionSelector(limits, manipulability_threshold=1e-8)


class TestJointLimits:
    def test_valid_angles_pass(self, limits: JointLimits) -> None:
        assert limits.is_within_limits(np.deg2rad([0, 30, -50, 20]))

    def test_out_of_range_fails(self, limits: JointLimits) -> None:
        # Below min
        assert not limits.is_within_limits(np.deg2rad([-90, 30, -50, 20]))
        # Above max
        assert not limits.is_within_limits(np.deg2rad([0, 30, 120, 20]))

    def test_clamp_brings_into_range(self, limits: JointLimits) -> None:
        angles = np.deg2rad([-200, 200, -200, 200])
        clamped = limits.clamp(angles)
        assert np.all(clamped >= limits.angle_min_rad)
        assert np.all(clamped <= limits.angle_max_rad)

    def test_rejects_inverted_limits(self) -> None:
        with pytest.raises(ValueError):
            JointLimits(
                angle_min_rad=np.array([1.0, 0, 0, 0]),
                angle_max_rad=np.array([0.0, 1, 1, 1]),
                max_velocity_rad_per_s=np.ones(4),
            )


class TestSolutionValidator:
    def test_validates_good_solution(
        self, fk: ForwardSolver, validator: SolutionValidator, limits: JointLimits
    ) -> None:
        """A solution obtained by IK on its own FK should validate."""
        from humanoid_arm_kinematics.solution_selector import SelectedSolution

        ik_cfg = IKConfig(max_iterations=300)
        jac = JacobianSolver(fk.model)
        ik = InverseSolver(fk, jac, ik_cfg)

        original = np.array([0.2, 0.5, -0.3, 0.4])
        fk_result = fk.solve(original)

        ik_result = ik.solve(fk_result.position, fk_result.yaw_rad, original)
        assert ik_result.success

        fake_result = IKResult(
            success=True,
            joint_angles_rad=ik_result.joint_angles_rad,
            forward_result=ik_result.forward_result,
            iterations=ik_result.iterations,
        )
        solution = SelectedSolution(
            joint_angles_rad=ik_result.joint_angles_rad,
            ik_result=fake_result,
            joint_distance_rad=0.0,
            within_limits=True,
        )

        val = validator.validate(solution, fk_result.position, fk_result.yaw_rad)
        assert val.valid
        assert val.position_error_m < 0.01
        # Orientation error may wrap, so allow larger tolerance
        assert val.orientation_error_rad < 0.1 or abs(
            val.orientation_error_rad - 2 * np.pi
        ) < 0.1

    def test_rejects_joint_limit_violation(
        self, fk: ForwardSolver, validator: SolutionValidator, limits: JointLimits
    ) -> None:
        """A solution with joints outside limits should fail validation."""
        from humanoid_arm_kinematics.solution_selector import SelectedSolution

        bad_angles = np.deg2rad([-200, 200, -200, 200])
        solution = SelectedSolution(
            joint_angles_rad=bad_angles,
            ik_result=IKResult(success=True),
            joint_distance_rad=0.0,
            within_limits=False,
        )

        fk_result = fk.solve(np.zeros(4))
        val = validator.validate(solution, fk_result.position, fk_result.yaw_rad)
        assert not val.valid
        assert not val.within_joint_limits


class TestSolutionSelector:
    def test_selects_closest_solution(
        self, fk: ForwardSolver, selector: SolutionSelector
    ) -> None:
        """The selector should pick the candidate closest to current joints."""
        current = np.array([0.3, 0.5, -0.3, 0.4])

        # Two fake results: one close, one far
        close_angles = np.array([0.31, 0.51, -0.31, 0.41])
        far_angles = np.array([1.0, 1.0, -1.0, 1.0])

        candidates = [
            IKResult(success=True, joint_angles_rad=far_angles),
            IKResult(success=True, joint_angles_rad=close_angles),
        ]

        selected = selector.select(candidates, current)
        assert selected is not None
        assert np.allclose(selected.joint_angles_rad, close_angles, atol=0.05)

    def test_returns_none_when_all_fail(self, selector: SolutionSelector) -> None:
        """If all candidates fail, selector returns None."""
        candidates = [
            IKResult(success=False),
            IKResult(success=False),
        ]
        assert selector.select(candidates, np.zeros(4)) is None

    def test_filters_near_singular_solutions(
        self, fk: ForwardSolver, limits: JointLimits
    ) -> None:
        """Solutions flagged as near-singular should be rejected."""
        current = np.zeros(4)
        candidates = [
            IKResult(
                success=True,
                joint_angles_rad=np.array([0.1, 0.1, 0.1, 0.1]),
                near_singular=True,
            ),
        ]
        selected = SolutionSelector(limits).select(candidates, current)
        assert selected is None

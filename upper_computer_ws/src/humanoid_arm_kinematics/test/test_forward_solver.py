"""Regression tests for URDF-backed FK and the position Jacobian."""

from pathlib import Path
import unittest

import numpy as np

from humanoid_arm_kinematics.forward_solver import ForwardSolver, ForwardSolverError
from humanoid_arm_kinematics.jacobian import JacobianSolver
from humanoid_arm_kinematics.robot_model import RobotModel


URDF_PATH = (
    Path(__file__).parents[2]
    / "humanoid_arm_description"
    / "urdf"
    / "humanoid_arm.urdf"
)


class TestForwardSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = RobotModel.from_urdf_file(URDF_PATH)
        cls.solver = ForwardSolver(cls.model)
        cls.jacobian = JacobianSolver(cls.model)

    def test_chain_is_selected_from_urdf(self) -> None:
        self.assertEqual(
            self.model.joint_names,
            ["joint_1", "joint_2", "joint_3", "joint_4"],
        )
        self.assertEqual(self.model.base_link, "base_link")
        self.assertEqual(self.model.tip_link, "tip_frame")

    def test_zero_pose_matches_urdf_geometry(self) -> None:
        result = self.solver.solve(np.zeros(4))
        np.testing.assert_allclose(result.position, [0.0, -0.05, -0.375], atol=1e-12)
        np.testing.assert_allclose(result.rotation, np.eye(3), atol=1e-12)

    def test_joint_axes_at_zero_match_urdf(self) -> None:
        state = self.model.evaluate(np.zeros(4))
        expected_positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.05, 0.0],
                [0.0, -0.05, -0.07],
                [0.0, -0.05, -0.105],
            ]
        )
        expected_axes = np.array(
            [
                [0.0, -0.8660254, 0.5],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        expected_axes /= np.linalg.norm(expected_axes, axis=1)[:, None]
        np.testing.assert_allclose(
            state.joint_positions, expected_positions, atol=1e-12
        )
        np.testing.assert_allclose(state.joint_axes, expected_axes, atol=1e-8)

    def test_position_jacobian_matches_finite_difference(self) -> None:
        angles = np.array([0.31, 0.67, -0.42, 0.58])
        analytic = self.jacobian.compute(angles).jacobian_task
        numeric = np.zeros_like(analytic)
        epsilon = 1e-7
        for index in range(4):
            plus = angles.copy()
            minus = angles.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric[:, index] = (
                self.solver.solve(plus).position - self.solver.solve(minus).position
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numeric, atol=1e-7)

    def test_quaternion_is_normalized(self) -> None:
        quaternion = self.solver.solve([0.2, 0.5, -0.3, 0.4]).quaternion_xyzw
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=12)

    def test_rejects_invalid_angles(self) -> None:
        with self.assertRaises(ForwardSolverError):
            self.solver.solve([0.0, 0.0, 0.0])
        with self.assertRaises(ForwardSolverError):
            self.solver.solve([0.0, np.nan, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

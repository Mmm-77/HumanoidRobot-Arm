"""Tests for position-only redundant inverse kinematics."""

from pathlib import Path
import unittest

import numpy as np

from humanoid_arm_kinematics.forward_solver import ForwardSolver
from humanoid_arm_kinematics.inverse_solver import (
    IKConfig,
    InverseSolver,
    InverseSolverError,
)
from humanoid_arm_kinematics.jacobian import JacobianSolver
from humanoid_arm_kinematics.robot_model import RobotModel


URDF_PATH = (
    Path(__file__).parents[2]
    / "humanoid_arm_description"
    / "urdf"
    / "humanoid_arm.urdf"
)


class TestInverseSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model = RobotModel.from_urdf_file(URDF_PATH)
        cls.forward = ForwardSolver(model)
        cls.inverse = InverseSolver(
            cls.forward,
            JacobianSolver(model),
            IKConfig(
                max_iterations=350,
                multi_start_attempts=12,
                multi_start_perturbation_rad=0.8,
            ),
        )

    def test_exact_seed_succeeds(self) -> None:
        angles = np.array([0.3, 0.8, -0.5, 0.6])
        target = self.forward.solve(angles).position
        result = self.inverse.solve(target, angles)
        self.assertTrue(result.success)
        self.assertLess(result.position_error_m, 0.001)

    def test_reaches_known_fk_position_from_perturbed_seed(self) -> None:
        original = np.array([0.35, 0.75, -0.45, 0.55])
        target = self.forward.solve(original).position
        initial = original + np.array([0.25, -0.2, 0.3, -0.2])
        result = self.inverse.solve(target, initial)
        self.assertTrue(result.success, msg=f"error={result.position_error_m}")
        actual = self.forward.solve(result.joint_angles_rad).position
        np.testing.assert_allclose(actual, target, atol=0.001)
        self.assertTrue(np.all(np.abs(result.joint_angles_rad - initial) <= np.pi))

    def test_orientation_is_not_part_of_solver_interface(self) -> None:
        target = self.forward.solve([0.2, 0.6, -0.4, 0.5]).position
        result = self.inverse.solve(target, np.zeros(4))
        self.assertTrue(result.success)
        self.assertEqual(result.final_error.shape, (3,))

    def test_rejects_malformed_input(self) -> None:
        with self.assertRaises(InverseSolverError):
            self.inverse.solve([0.0, 0.0], np.zeros(4))
        with self.assertRaises(InverseSolverError):
            self.inverse.solve(np.zeros(3), [0.0, 0.0, np.nan, 0.0])


if __name__ == "__main__":
    unittest.main()

import unittest

from humanoid_arm_kinematics.solution_policy import is_solution_acceptable


class TestSolutionPolicy(unittest.TestCase):
    def test_accepts_strict_solution(self) -> None:
        self.assertTrue(is_solution_acceptable(True, 0.0005, 0.005))

    def test_accepts_best_solution_inside_validation_limit(self) -> None:
        self.assertTrue(is_solution_acceptable(False, 0.003, 0.005))

    def test_rejects_solution_outside_validation_limit(self) -> None:
        self.assertFalse(is_solution_acceptable(False, 0.006, 0.005))

    def test_rejects_non_finite_error(self) -> None:
        self.assertFalse(is_solution_acceptable(False, float("nan"), 0.005))

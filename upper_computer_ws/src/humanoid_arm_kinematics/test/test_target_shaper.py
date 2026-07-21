"""Tests for pre-IK Cartesian target shaping."""

import unittest

import numpy as np

from humanoid_arm_kinematics.target_shaper import ShaperConfig, TargetShaper


class TestTargetShaper(unittest.TestCase):
    def test_first_target_passes_through(self) -> None:
        shaper = TargetShaper(ShaperConfig())
        target = np.array([0.1, -0.2, -0.3])
        np.testing.assert_allclose(shaper.shape(target), target)

    def test_dead_zone_retains_previous_position(self) -> None:
        shaper = TargetShaper(
            ShaperConfig(position_dead_zone_m=0.01, position_alpha=1.0)
        )
        previous = shaper.shape([0.0, 0.0, 0.0])
        shaped = shaper.shape([0.001, 0.0, 0.0])
        np.testing.assert_allclose(shaped, previous)

    def test_step_limit_and_ema_change_actual_ik_target(self) -> None:
        shaper = TargetShaper(
            ShaperConfig(
                position_dead_zone_m=0.0,
                max_position_step_m=0.1,
                position_alpha=0.5,
            )
        )
        shaper.shape([0.0, 0.0, 0.0])
        shaped = shaper.shape([1.0, 0.0, 0.0])
        np.testing.assert_allclose(shaped, [0.05, 0.0, 0.0], atol=1e-12)

    def test_reset_discards_history(self) -> None:
        shaper = TargetShaper(ShaperConfig())
        shaper.shape([0.0, 0.0, 0.0])
        shaper.reset()
        np.testing.assert_allclose(shaper.shape([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0])

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            ShaperConfig(position_alpha=0.0)
        with self.assertRaises(ValueError):
            ShaperConfig(max_position_step_m=0.0)


if __name__ == "__main__":
    unittest.main()

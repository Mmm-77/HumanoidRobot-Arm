"""Tests for target shaper."""

import numpy as np
import pytest

from humanoid_arm_kinematics.target_shaper import ShaperConfig, TargetShaper


class TestShaperConfig:
    def test_default_config_is_valid(self) -> None:
        cfg = ShaperConfig()
        assert cfg.position_dead_zone_m == 0.001
        assert cfg.position_alpha == 0.3

    def test_rejects_invalid_alpha(self) -> None:
        with pytest.raises(ValueError):
            ShaperConfig(position_alpha=0.0)
        with pytest.raises(ValueError):
            ShaperConfig(position_alpha=1.5)
        with pytest.raises(ValueError):
            ShaperConfig(orientation_alpha=-0.1)

    def test_rejects_negative_dead_zone(self) -> None:
        with pytest.raises(ValueError):
            ShaperConfig(position_dead_zone_m=-0.1)
        with pytest.raises(ValueError):
            ShaperConfig(orientation_dead_zone_rad=-0.01)

    def test_rejects_zero_step_limit(self) -> None:
        with pytest.raises(ValueError):
            ShaperConfig(max_position_step_m=0.0)


class TestTargetShaper:
    @pytest.fixture
    def shaper(self) -> TargetShaper:
        cfg = ShaperConfig(
            position_dead_zone_m=0.01,
            orientation_dead_zone_rad=0.01,
            max_position_step_m=0.1,
            max_orientation_step_rad=0.2,
            position_alpha=1.0,
            orientation_alpha=1.0,
        )
        return TargetShaper(cfg, dt_s=0.033)  # ~30 Hz

    def test_first_update_initializes_state(self, shaper: TargetShaper) -> None:
        """First shape() call should return the input unchanged."""
        joints = np.array([0.1, 0.2, 0.3, 0.4])
        pos = np.array([1.0, 2.0, 3.0])
        yaw = 0.5

        shaped = shaper.shape(joints, pos, yaw)
        assert shaper.initialized
        assert np.allclose(shaped.position_smoothed, pos)
        assert abs(shaped.yaw_smoothed_rad - yaw) < 1e-9
        # Velocities should be zero on first update
        assert np.allclose(shaped.joint_velocities_rad_per_s, 0.0)

    def test_dead_zone_ignores_tiny_moves(self, shaper: TargetShaper) -> None:
        """A position change smaller than dead zone should be ignored."""
        joints = np.array([0.1, 0.2, 0.3, 0.4])
        pos1 = np.array([1.0, 2.0, 3.0])
        yaw1 = 0.5

        shaper.shape(joints, pos1, yaw1)

        # Tiny move
        pos2 = pos1 + np.array([0.001, 0.0, 0.0])  # 1 mm < 10 mm dead zone
        yaw2 = yaw1 + 0.001  # < 0.01 rad dead zone

        shaped2 = shaper.shape(joints, pos2, yaw2)
        # Position and yaw should NOT change
        assert np.allclose(shaped2.position_smoothed, pos1)
        assert abs(shaped2.yaw_smoothed_rad - yaw1) < 1e-9

    def test_large_move_passes_through(self, shaper: TargetShaper) -> None:
        """A position change larger than dead zone should be tracked."""
        joints = np.array([0.1, 0.2, 0.3, 0.4])
        pos1 = np.array([1.0, 2.0, 3.0])
        yaw1 = 0.5

        shaper.shape(joints, pos1, yaw1)

        # Large move
        pos2 = np.array([1.5, 2.5, 3.5])  # 0.5 m > 0.01 m dead zone
        yaw2 = 1.0  # 0.5 rad > 0.01 rad dead zone

        shaped2 = shaper.shape(joints, pos2, yaw2)
        # With alpha=1.0, should jump directly to target (clipped by step limit)
        diff = shaped2.position_smoothed - pos1
        assert float(np.linalg.norm(diff)) > 0

    def test_step_clipping_limits_max_change(self) -> None:
        """With a small step limit, large moves should be clipped."""
        cfg = ShaperConfig(
            position_dead_zone_m=0.0,
            max_position_step_m=0.01,  # very small
            max_orientation_step_rad=np.pi,
            position_alpha=1.0,
            orientation_alpha=1.0,
        )
        shaper = TargetShaper(cfg, dt_s=0.033)

        joints = np.array([0.1, 0.2, 0.3, 0.4])
        pos1 = np.array([0.0, 0.0, 0.0])
        shaper.shape(joints, pos1, 0.0)

        # Target 1 meter away, but max step is 0.01 m
        pos2 = np.array([1.0, 0.0, 0.0])
        shaped = shaper.shape(joints, pos2, 0.0)
        step_dist = float(np.linalg.norm(shaped.position_smoothed - pos1))
        assert step_dist <= 0.011  # allow small epsilon

    def test_reset_clears_state(self, shaper: TargetShaper) -> None:
        """After reset, the shaper should behave as if freshly initialized."""
        joints = np.array([0.1, 0.2, 0.3, 0.4])
        shaper.shape(joints, np.array([1.0, 2.0, 3.0]), 0.5)
        assert shaper.initialized

        shaper.reset()
        assert not shaper.initialized

        # Next update should re-initialize
        shaped = shaper.shape(joints, np.array([5.0, 6.0, 7.0]), 1.0)
        assert shaper.initialized
        assert np.allclose(shaped.position_smoothed, [5.0, 6.0, 7.0])

    def test_ema_smoothing_blends(self) -> None:
        """With alpha < 1, the output should be a blend of old and new."""
        cfg = ShaperConfig(
            position_dead_zone_m=0.0,
            orientation_dead_zone_rad=0.0,
            max_position_step_m=10.0,
            max_orientation_step_rad=10.0,
            position_alpha=0.5,
            orientation_alpha=0.5,
        )
        shaper = TargetShaper(cfg, dt_s=0.033)

        joints = np.array([0.0, 0.0, 0.0, 0.0])
        shaper.shape(joints, np.array([0.0, 0.0, 0.0]), 0.0)

        shaped = shaper.shape(joints, np.array([2.0, 0.0, 0.0]), 0.0)
        # With alpha=0.5: smoothed = 0.5*2.0 + 0.5*0.0 = 1.0
        assert abs(shaped.position_smoothed[0] - 1.0) < 1e-6

    def test_yaw_wraps_correctly(self, shaper: TargetShaper) -> None:
        """Yaw wrapping across ±π should be handled correctly."""
        joints = np.array([0.1, 0.2, 0.3, 0.4])
        shaper.shape(joints, np.array([0.0, 0.0, 0.0]), 3.0)  # ~π

        # Move to -3.0 (also ~π), short path should be taken
        shaped = shaper.shape(joints, np.array([0.0, 0.0, 0.0]), -3.0)
        # The smoothed yaw should be near ±π (short path: ~0.28 rad)
        assert abs(shaped.yaw_smoothed_rad) > 2.5  # near ±π

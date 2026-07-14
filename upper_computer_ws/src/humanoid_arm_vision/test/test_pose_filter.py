import math

import numpy as np
import pytest

from humanoid_arm_vision.pose_filter import PoseFilter, PoseFilterConfig


def test_filters_position_and_normalizes_quaternion() -> None:
    pose_filter = PoseFilter(
        PoseFilterConfig(position_alpha=0.5, orientation_alpha=0.5, reset_gap_s=1.0)
    )
    first = pose_filter.update(np.zeros(3), np.array([0.0, 0.0, 0.0, 2.0]), 0.0)
    second = pose_filter.update(
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        0.1,
    )
    assert first.reset
    assert second.position == pytest.approx([1.0, 0.0, 0.0])
    assert np.linalg.norm(second.orientation_xyzw) == pytest.approx(1.0)
    # Halfway from identity to 180 degrees around z is 90 degrees around z.
    assert abs(second.orientation_xyzw[2]) == pytest.approx(math.sqrt(0.5))
    assert abs(second.orientation_xyzw[3]) == pytest.approx(math.sqrt(0.5))


def test_antipodal_quaternion_does_not_flip() -> None:
    pose_filter = PoseFilter(PoseFilterConfig())
    first = pose_filter.update(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 0.0)
    second = pose_filter.update(np.zeros(3), np.array([0.0, 0.0, 0.0, -1.0]), 0.1)
    assert second.orientation_xyzw == pytest.approx(first.orientation_xyzw)


def test_long_gap_resets_filter() -> None:
    pose_filter = PoseFilter(PoseFilterConfig(reset_gap_s=0.5))
    pose_filter.update(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 0.0)
    result = pose_filter.update(
        np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]), 1.0
    )
    assert result.reset
    assert result.position == pytest.approx([1.0, 2.0, 3.0])

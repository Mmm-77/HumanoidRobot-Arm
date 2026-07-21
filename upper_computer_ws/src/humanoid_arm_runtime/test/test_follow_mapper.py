"""Tests for FollowMapper: camera delta → base-frame 4-DOF target."""

import numpy as np

from humanoid_arm_runtime.follow_mapper import FollowMapper


def test_no_movement():
    """If camera hasn't moved, target should be zero."""
    mapper = FollowMapper()
    pos = np.array([0.5, 0.0, 1.0], dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # identity

    delta_pos, delta_yaw = mapper.map(pos, quat, pos, quat)
    assert np.allclose(delta_pos, [0, 0, 0])
    assert abs(delta_yaw) < 1e-6


def test_pure_translation():
    """A 0.1 m forward translation in camera frame → base frame."""
    mapper = FollowMapper()
    pos0 = np.array([0.5, 0.0, 1.0], dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    pos1 = np.array([0.6, 0.0, 1.0], dtype=np.float64)

    delta_pos, delta_yaw = mapper.map(pos0, quat, pos1, quat)
    # Forward in camera = +X camera → rotated by identity tag→base
    assert abs(delta_pos[0] - 0.1) < 0.01
    assert abs(delta_yaw) < 1e-6


def test_pure_rotation():
    """A pure 90-degree yaw rotation detected."""
    mapper = FollowMapper()
    pos = np.array([0.5, 0.0, 1.0], dtype=np.float64)
    q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)            # 0°
    q90 = np.array([0.0, 0.0, 0.7071, 0.7071], dtype=np.float64)    # +90° ~ Z

    delta_pos, delta_yaw = mapper.map(pos, q0, pos, q90)
    assert np.allclose(delta_pos, [0, 0, 0], atol=0.01)
    # Yaw should be roughly pi/2
    assert abs(delta_yaw - np.pi / 2) < 0.1


def test_axis_sign_inversion():
    """Axis sign = -1 should invert the mapping."""
    mapper = FollowMapper(axis_signs=(-1, -1, -1))
    pos0 = np.array([0.5, 0.0, 1.0], dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    pos1 = np.array([0.6, 0.2, 0.8], dtype=np.float64)

    delta, _ = mapper.map(pos0, quat, pos1, quat)
    # Without inversion: [0.1, 0.2, -0.2]; with all -1: [-0.1, -0.2, +0.2]
    assert delta[0] < 0
    assert delta[1] < 0
    assert delta[2] > 0


def test_position_scale():
    """Scale factor should multiply the position delta."""
    mapper = FollowMapper(position_scale=2.0)
    pos0 = np.array([0.5, 0.0, 1.0], dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    pos1 = np.array([0.6, 0.0, 1.0], dtype=np.float64)

    delta, _ = mapper.map(pos0, quat, pos1, quat)
    assert abs(delta[0] - 0.2) < 0.01


def test_tag_to_base_rotation_maps_translation_and_yaw():
    """A calibrated 90-degree frame rotation is applied to the full delta."""
    rotation = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    mapper = FollowMapper(tag_to_base=rotation)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    q90 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])

    delta, yaw = mapper.map(np.zeros(3), identity, np.array([1.0, 0.0, 0.0]), q90)

    assert np.allclose(delta, [0.0, 1.0, 0.0])
    assert abs(yaw - np.pi / 2) < 1e-6


def test_tag_in_camera_convention_is_inverted_before_differencing():
    mapper = FollowMapper(camera_pose_convention="tag_in_camera")
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    delta, _ = mapper.map(
        np.zeros(3), identity, np.array([0.1, 0.0, 0.0]), identity
    )

    assert np.allclose(delta, [-0.1, 0.0, 0.0])

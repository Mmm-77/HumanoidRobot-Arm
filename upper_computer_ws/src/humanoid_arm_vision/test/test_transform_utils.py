import numpy as np
import pytest

from humanoid_arm_vision.transform_utils import (
    invert_transform,
    make_transform,
    quaternion_angular_distance,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def test_rotation_quaternion_round_trip() -> None:
    quaternion = np.array([0.2, -0.3, 0.1, 0.9])
    quaternion /= np.linalg.norm(quaternion)
    recovered = rotation_matrix_to_quaternion(quaternion_to_rotation_matrix(quaternion))
    assert quaternion_angular_distance(quaternion, recovered) == pytest.approx(0.0)


def test_transform_inverse() -> None:
    rotation = quaternion_to_rotation_matrix(np.array([0.0, 0.0, 0.5, 0.8660254]))
    transform = make_transform(rotation, np.array([1.0, 2.0, 3.0]))
    assert transform @ invert_transform(transform) == pytest.approx(
        np.eye(4), abs=1e-12
    )

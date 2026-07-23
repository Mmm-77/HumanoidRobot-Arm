import numpy as np

from humanoid_arm_vision.transform_utils import quaternion_to_rotation_matrix
from humanoid_arm_vision.wall_frame_calibrator import (
    WALL_FROM_TAG_ROTATION,
    WallFrameCalibrator,
)


def test_requested_wall_axes_and_origin() -> None:
    calibrator = WallFrameCalibrator(x_origin_m=0.5)
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    aligned = calibrator.calibrate(np.array([0.0, 0.0, 0.5]), identity)
    away = calibrator.calibrate(np.array([0.0, 0.0, 0.6]), identity)
    right = calibrator.calibrate(np.array([0.1, 0.0, 0.5]), identity)
    up = calibrator.calibrate(np.array([0.0, 0.1, 0.5]), identity)

    np.testing.assert_allclose(aligned.position, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(away.position, [0.1, 0.0, 0.0])
    np.testing.assert_allclose(right.position, [0.0, 0.1, 0.0])
    np.testing.assert_allclose(up.position, [0.0, 0.0, 0.1])


def test_orientation_is_expressed_in_wall_frame() -> None:
    result = WallFrameCalibrator(0.5).calibrate(
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(
        quaternion_to_rotation_matrix(result.orientation_xyzw),
        WALL_FROM_TAG_ROTATION,
        atol=1e-12,
    )


def test_rejects_invalid_x_origin() -> None:
    for value in (0.0, -0.1, float("nan"), float("inf")):
        try:
            WallFrameCalibrator(value)
        except ValueError:
            continue
        raise AssertionError(f"invalid x origin accepted: {value}")

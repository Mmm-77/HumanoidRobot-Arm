from pathlib import Path

import numpy as np
import pytest

from humanoid_arm_vision.camera_calibration import CalibrationError, CameraCalibration


def valid_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array(
            [[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.zeros(5),
        image_width=640,
        image_height=480,
    )


def test_loads_ros_camera_calibration_yaml(tmp_path: Path) -> None:
    calibration_file = tmp_path / "camera.yaml"
    calibration_file.write_text(
        """
image_width: 640
image_height: 480
camera_name: d435i_color
camera_matrix:
  rows: 3
  cols: 3
  data: [600.0, 0.0, 320.0, 0.0, 610.0, 240.0, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.1, -0.2, 0.0, 0.0, 0.0]
""",
        encoding="utf-8",
    )
    calibration = CameraCalibration.from_file(calibration_file)
    assert calibration.image_width == 640
    assert calibration.camera_matrix[1, 1] == pytest.approx(610.0)
    assert calibration.distortion_coefficients.shape == (5,)


def test_rejects_zero_focal_length() -> None:
    with pytest.raises(CalibrationError, match="focal lengths"):
        matrix = np.eye(3)
        matrix[0, 0] = 0.0
        CameraCalibration(
            camera_matrix=matrix,
            distortion_coefficients=np.zeros(5),
            image_width=640,
            image_height=480,
        )


def test_resolution_mismatch_requires_explicit_scaling() -> None:
    calibration = valid_calibration()
    with pytest.raises(CalibrationError, match="does not match"):
        calibration.for_resolution(1280, 960)
    scaled = calibration.for_resolution(1280, 960, allow_scaling=True)
    assert scaled.camera_matrix[0, 0] == pytest.approx(1200.0)
    assert scaled.camera_matrix[0, 2] == pytest.approx(640.0)


def test_resolution_scaling_rejects_aspect_ratio_change() -> None:
    with pytest.raises(CalibrationError, match="aspect ratios"):
        valid_calibration().for_resolution(1280, 720, allow_scaling=True)

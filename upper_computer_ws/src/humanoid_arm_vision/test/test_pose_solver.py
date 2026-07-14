import cv2
import numpy as np
import pytest

from humanoid_arm_vision.apriltag_detector import AprilTagDetection
from humanoid_arm_vision.camera_calibration import CameraCalibration
from humanoid_arm_vision.pose_solver import AprilTagPoseSolver


def test_recovers_camera_pose_in_tag_frame() -> None:
    calibration = CameraCalibration(
        camera_matrix=np.array(
            [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.zeros(5),
        image_width=640,
        image_height=480,
    )
    solver = AprilTagPoseSolver(0.10)
    # A front-facing tag: tag x aligns with optical x, while tag y/z are the
    # inverse of optical y/z. The camera is one meter along tag +z.
    camera_from_tag_rotation = np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(camera_from_tag_rotation)
    tvec = np.array([0.0, 0.0, 1.0])
    corners, _ = cv2.projectPoints(
        solver.object_points,
        rvec,
        tvec,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
    )
    corners = corners.reshape(4, 2)
    detection = AprilTagDetection(
        tag_id=0,
        corners=corners,
        center=np.mean(corners, axis=0),
        pixel_area=6400.0,
        perimeter_px=320.0,
    )
    estimate = solver.solve(detection, calibration)
    assert estimate.position == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert estimate.reprojection_error_px < 1e-5
    assert np.linalg.norm(estimate.orientation_xyzw) == pytest.approx(1.0)
    assert estimate.camera_from_tag @ estimate.tag_from_camera == pytest.approx(
        np.eye(4)
    )


def test_tag_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        AprilTagPoseSolver(0.0)

import numpy as np

from humanoid_arm_vision.apriltag_detector import AprilTagDetection
from humanoid_arm_vision.pose_solver import PoseEstimate
from humanoid_arm_vision.quality_gate import (
    PoseQualityGate,
    QualityConfig,
    QualityReason,
)


def detection(area: float = 1000.0) -> AprilTagDetection:
    corners = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
    return AprilTagDetection(
        tag_id=0,
        corners=corners,
        center=np.array([25.0, 25.0]),
        pixel_area=area,
        perimeter_px=120.0,
    )


def estimate(position=(0.0, 0.0, 1.0), error: float = 0.2) -> PoseEstimate:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(position)
    return PoseEstimate(
        position=np.asarray(position),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        reprojection_error_px=error,
        tag_from_camera=transform,
        camera_from_tag=np.linalg.inv(transform),
        rvec_tag_to_camera=np.zeros(3),
        tvec_tag_to_camera=np.array([0.0, 0.0, 1.0]),
    )


def make_gate() -> PoseQualityGate:
    return PoseQualityGate(
        QualityConfig(
            min_tag_area_ratio=0.01,
            max_position_jump_m=0.2,
            max_orientation_jump_rad=0.5,
            max_frame_age_s=0.2,
            continuity_reset_s=0.5,
        )
    )


def evaluate(gate, tag_detection, pose_estimate, capture=1.0, now=1.01):
    return gate.evaluate(
        tag_detection,
        pose_estimate,
        image_width=100,
        image_height=100,
        capture_time_s=capture,
        now_s=now,
    )


def test_accepts_good_pose() -> None:
    decision = evaluate(make_gate(), detection(), estimate())
    assert decision.valid
    assert decision.reason is QualityReason.VALID


def test_rejects_missing_and_stale_data() -> None:
    gate = make_gate()
    assert evaluate(gate, None, None).reason is QualityReason.NO_DETECTION
    assert (
        evaluate(gate, detection(), estimate(), capture=1.0, now=1.3).reason
        is QualityReason.STALE_FRAME
    )


def test_rejects_low_area_and_reprojection_error() -> None:
    gate = make_gate()
    assert (
        evaluate(gate, detection(area=10.0), estimate()).reason
        is QualityReason.TAG_AREA_TOO_SMALL
    )
    assert (
        evaluate(gate, detection(), estimate(error=10.0)).reason
        is QualityReason.REPROJECTION_ERROR
    )


def test_rejects_pose_jump_but_reacquires_after_gap() -> None:
    gate = make_gate()
    assert evaluate(gate, detection(), estimate(), capture=1.0).valid
    jump = evaluate(
        gate, detection(), estimate(position=(0.5, 0.0, 1.0)), capture=1.1, now=1.11
    )
    assert jump.reason is QualityReason.POSITION_JUMP
    reacquired = evaluate(
        gate,
        detection(),
        estimate(position=(0.5, 0.0, 1.0)),
        capture=1.6,
        now=1.61,
    )
    assert reacquired.valid

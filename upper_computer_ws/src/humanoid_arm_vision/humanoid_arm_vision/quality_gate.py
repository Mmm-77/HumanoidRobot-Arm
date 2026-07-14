"""Quality and continuity checks applied before a pose can leave the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .apriltag_detector import AprilTagDetection
from .pose_solver import PoseEstimate
from .transform_utils import quaternion_angular_distance


class QualityReason(str, Enum):
    VALID = "valid"
    NO_DETECTION = "no_detection"
    STALE_FRAME = "stale_frame"
    FUTURE_TIMESTAMP = "future_timestamp"
    TAG_AREA_TOO_SMALL = "tag_area_too_small"
    DECISION_MARGIN_UNAVAILABLE = "decision_margin_unavailable"
    DECISION_MARGIN_TOO_LOW = "decision_margin_too_low"
    NONFINITE_POSE = "nonfinite_pose"
    REPROJECTION_ERROR = "reprojection_error"
    DISTANCE_OUT_OF_RANGE = "distance_out_of_range"
    POSITION_JUMP = "position_jump"
    ORIENTATION_JUMP = "orientation_jump"
    CAMERA_ERROR = "camera_error"
    CAMERA_INFO_MISSING = "camera_info_missing"
    CAMERA_INFO_STALE = "camera_info_stale"
    POSE_SOLVER_ERROR = "pose_solver_error"
    CALIBRATION_ERROR = "calibration_error"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class QualityConfig:
    max_reprojection_error_px: float = 2.5
    min_tag_area_ratio: float = 0.002
    min_decision_margin: float = 0.0
    min_camera_distance_m: float = 0.05
    max_camera_distance_m: float = 3.0
    max_position_jump_m: float = 0.20
    max_orientation_jump_rad: float = 0.70
    max_frame_age_s: float = 0.20
    continuity_reset_s: float = 0.50
    future_timestamp_tolerance_s: float = 0.02

    def __post_init__(self) -> None:
        positive = {
            "max_reprojection_error_px": self.max_reprojection_error_px,
            "max_camera_distance_m": self.max_camera_distance_m,
            "max_position_jump_m": self.max_position_jump_m,
            "max_orientation_jump_rad": self.max_orientation_jump_rad,
            "max_frame_age_s": self.max_frame_age_s,
            "continuity_reset_s": self.continuity_reset_s,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.min_tag_area_ratio < 1.0:
            raise ValueError("min_tag_area_ratio must be in [0, 1)")
        if self.min_decision_margin < 0.0:
            raise ValueError("min_decision_margin must be non-negative")
        if self.min_camera_distance_m < 0.0:
            raise ValueError("min_camera_distance_m must be non-negative")
        if self.min_camera_distance_m >= self.max_camera_distance_m:
            raise ValueError("minimum camera distance must be below maximum distance")
        if self.future_timestamp_tolerance_s < 0.0:
            raise ValueError("future timestamp tolerance must be non-negative")


@dataclass(frozen=True)
class QualityDecision:
    valid: bool
    reason: QualityReason
    metrics: Mapping[str, float] = field(default_factory=dict)


class PoseQualityGate:
    def __init__(self, config: QualityConfig) -> None:
        self.config = config
        self._last_position: NDArray[np.float64] | None = None
        self._last_orientation: NDArray[np.float64] | None = None
        self._last_timestamp_s: float | None = None

    def reset(self) -> None:
        self._last_position = None
        self._last_orientation = None
        self._last_timestamp_s = None

    @staticmethod
    def invalid(
        reason: QualityReason, metrics: Mapping[str, float] | None = None
    ) -> QualityDecision:
        return QualityDecision(False, reason, metrics or {})

    def evaluate(
        self,
        detection: AprilTagDetection | None,
        estimate: PoseEstimate | None,
        *,
        image_width: int,
        image_height: int,
        capture_time_s: float,
        now_s: float,
    ) -> QualityDecision:
        age = now_s - capture_time_s
        metrics: dict[str, float] = {"frame_age_s": float(age)}
        if age < -self.config.future_timestamp_tolerance_s:
            return self.invalid(QualityReason.FUTURE_TIMESTAMP, metrics)
        if age > self.config.max_frame_age_s:
            return self.invalid(QualityReason.STALE_FRAME, metrics)
        if detection is None:
            return self.invalid(QualityReason.NO_DETECTION, metrics)
        if image_width <= 0 or image_height <= 0:
            return self.invalid(QualityReason.INTERNAL_ERROR, metrics)

        area_ratio = detection.pixel_area / float(image_width * image_height)
        metrics["tag_area_ratio"] = area_ratio
        if area_ratio < self.config.min_tag_area_ratio:
            return self.invalid(QualityReason.TAG_AREA_TOO_SMALL, metrics)
        if self.config.min_decision_margin > 0.0:
            if detection.decision_margin is None:
                return self.invalid(QualityReason.DECISION_MARGIN_UNAVAILABLE, metrics)
            metrics["decision_margin"] = float(detection.decision_margin)
            if detection.decision_margin < self.config.min_decision_margin:
                return self.invalid(QualityReason.DECISION_MARGIN_TOO_LOW, metrics)
        if estimate is None:
            return self.invalid(QualityReason.POSE_SOLVER_ERROR, metrics)

        finite = (
            np.all(np.isfinite(estimate.position))
            and np.all(np.isfinite(estimate.orientation_xyzw))
            and np.isfinite(estimate.reprojection_error_px)
        )
        if not finite:
            return self.invalid(QualityReason.NONFINITE_POSE, metrics)
        metrics["reprojection_error_px"] = float(estimate.reprojection_error_px)
        if estimate.reprojection_error_px > self.config.max_reprojection_error_px:
            return self.invalid(QualityReason.REPROJECTION_ERROR, metrics)

        distance = estimate.camera_distance_m
        metrics["camera_distance_m"] = distance
        if (
            not self.config.min_camera_distance_m
            <= distance
            <= self.config.max_camera_distance_m
        ):
            return self.invalid(QualityReason.DISTANCE_OUT_OF_RANGE, metrics)

        if (
            self._last_timestamp_s is not None
            and capture_time_s >= self._last_timestamp_s
            and capture_time_s - self._last_timestamp_s
            <= self.config.continuity_reset_s
        ):
            assert (
                self._last_position is not None and self._last_orientation is not None
            )
            position_jump = float(
                np.linalg.norm(estimate.position - self._last_position)
            )
            orientation_jump = quaternion_angular_distance(
                estimate.orientation_xyzw, self._last_orientation
            )
            metrics["position_jump_m"] = position_jump
            metrics["orientation_jump_rad"] = orientation_jump
            if position_jump > self.config.max_position_jump_m:
                return self.invalid(QualityReason.POSITION_JUMP, metrics)
            if orientation_jump > self.config.max_orientation_jump_rad:
                return self.invalid(QualityReason.ORIENTATION_JUMP, metrics)

        self._last_position = estimate.position.copy()
        self._last_orientation = estimate.orientation_xyzw.copy()
        self._last_timestamp_s = float(capture_time_s)
        return QualityDecision(True, QualityReason.VALID, metrics)

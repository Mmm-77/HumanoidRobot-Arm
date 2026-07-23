"""Public, ROS-independent interfaces for the humanoid arm vision package."""

from .apriltag_detector import (
    AprilTagConfig,
    AprilTagDetection,
    AprilTagDetector,
    DetectorError,
)
from .camera_calibration import CalibrationError, CameraCalibration
from .camera_driver import (
    CameraConfig,
    CameraError,
    CameraFrame,
    CameraOpenError,
    CameraReadError,
    RealSenseCamera,
)
from .pose_filter import FilteredPose, PoseFilter, PoseFilterConfig
from .pose_solver import AprilTagPoseSolver, PoseEstimate, PoseSolverError
from .quality_gate import PoseQualityGate, QualityConfig, QualityDecision, QualityReason
from .wall_frame_calibrator import (
    WALL_FROM_TAG_ROTATION,
    WallFrameCalibration,
    WallFrameCalibrator,
)

__all__ = [
    "AprilTagConfig",
    "AprilTagDetection",
    "AprilTagDetector",
    "AprilTagPoseSolver",
    "CalibrationError",
    "CameraCalibration",
    "CameraConfig",
    "CameraError",
    "CameraFrame",
    "CameraOpenError",
    "CameraReadError",
    "DetectorError",
    "FilteredPose",
    "RealSenseCamera",
    "PoseEstimate",
    "PoseFilter",
    "PoseFilterConfig",
    "PoseQualityGate",
    "PoseSolverError",
    "QualityConfig",
    "QualityDecision",
    "QualityReason",
    "WALL_FROM_TAG_ROTATION",
    "WallFrameCalibration",
    "WallFrameCalibrator",
]

"""Planar AprilTag pose estimation and camera-in-tag transform conversion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .apriltag_detector import AprilTagDetection
from .camera_calibration import CameraCalibration
from .transform_utils import (
    invert_transform,
    make_transform,
    rotation_matrix_to_quaternion,
)


class PoseSolverError(RuntimeError):
    """Raised when no physically meaningful PnP solution can be produced."""


@dataclass(frozen=True)
class PoseEstimate:
    """Camera pose expressed in the fixed tag coordinate frame."""

    position: NDArray[np.float64]
    orientation_xyzw: NDArray[np.float64]
    reprojection_error_px: float
    tag_from_camera: NDArray[np.float64]
    camera_from_tag: NDArray[np.float64]
    rvec_tag_to_camera: NDArray[np.float64]
    tvec_tag_to_camera: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=np.float64).reshape(3)
        )
        object.__setattr__(
            self,
            "orientation_xyzw",
            np.asarray(self.orientation_xyzw, dtype=np.float64).reshape(4),
        )

    @property
    def camera_distance_m(self) -> float:
        return float(np.linalg.norm(self.position))


class AprilTagPoseSolver:
    def __init__(self, tag_size_m: float) -> None:
        if not np.isfinite(tag_size_m) or tag_size_m <= 0.0:
            raise ValueError("AprilTag size must be a positive finite value in meters")
        half = float(tag_size_m) / 2.0
        # Required order for SOLVEPNP_IPPE_SQUARE. The tag frame is x-right,
        # y-up, z-out of the printed face.
        self.object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

    def solve(
        self, detection: AprilTagDetection, calibration: CameraCalibration
    ) -> PoseEstimate:
        image_points = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
        distortion = calibration.distortion_coefficients
        distortion_input = distortion.reshape(-1, 1) if distortion.size else None
        try:
            result = cv2.solvePnPGeneric(
                self.object_points,
                image_points,
                calibration.camera_matrix,
                distortion_input,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error as exc:
            raise PoseSolverError(f"OpenCV PnP failed: {exc}") from exc
        success, rvecs, tvecs = result[:3]
        if not success or not rvecs:
            raise PoseSolverError("PnP returned no pose solutions")

        # IPPE is the preferred planar-square solver, but an exactly
        # fronto-parallel square is a degenerate case in some OpenCV builds.
        # Add the iterative result as a candidate and let reprojection error plus
        # the printed-front-face constraint select the physical solution.
        all_rvecs = list(rvecs)
        all_tvecs = list(tvecs)
        iterative_ok, iterative_rvec, iterative_tvec = cv2.solvePnP(
            self.object_points,
            image_points,
            calibration.camera_matrix,
            distortion_input,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if iterative_ok:
            all_rvecs.append(iterative_rvec)
            all_tvecs.append(iterative_tvec)

        candidates: list[tuple[float, NDArray[np.float64], NDArray[np.float64]]] = []
        for rvec, tvec in zip(all_rvecs, all_tvecs, strict=True):
            rotation_vector = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            translation_vector = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            if not np.all(np.isfinite(rotation_vector)) or not np.all(
                np.isfinite(translation_vector)
            ):
                continue
            # A visible tag must be in front of the camera optical plane.
            if translation_vector[2, 0] <= 0.0:
                continue
            rotation_camera_from_tag, _ = cv2.Rodrigues(rotation_vector)
            # Tag +z points out of the printed face. When that face is visible,
            # its normal points back toward the camera, opposite optical +z.
            if rotation_camera_from_tag[2, 2] >= 0.0:
                continue
            projected, _ = cv2.projectPoints(
                self.object_points,
                rotation_vector,
                translation_vector,
                calibration.camera_matrix,
                distortion_input,
            )
            residual = projected.reshape(4, 2) - image_points
            rms_error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            candidates.append((rms_error, rotation_vector, translation_vector))
        if not candidates:
            raise PoseSolverError("PnP returned no finite, positive-depth solution")

        reprojection_error, rvec, tvec = min(candidates, key=lambda item: item[0])
        rotation_camera_from_tag, _ = cv2.Rodrigues(rvec)
        camera_from_tag = make_transform(rotation_camera_from_tag, tvec)
        tag_from_camera = invert_transform(camera_from_tag)
        orientation = rotation_matrix_to_quaternion(tag_from_camera[:3, :3])
        return PoseEstimate(
            position=tag_from_camera[:3, 3].copy(),
            orientation_xyzw=orientation,
            reprojection_error_px=reprojection_error,
            tag_from_camera=tag_from_camera,
            camera_from_tag=camera_from_tag,
            rvec_tag_to_camera=rvec.reshape(3),
            tvec_tag_to_camera=tvec.reshape(3),
        )

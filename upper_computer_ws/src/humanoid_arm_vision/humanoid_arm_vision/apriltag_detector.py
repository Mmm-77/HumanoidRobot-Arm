"""AprilTag detection using OpenCV's ArUco/AprilTag dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import cv2
import numpy as np
from numpy.typing import NDArray


class DetectorError(RuntimeError):
    """Raised when the detector backend or configuration is unusable."""


_FAMILY_ATTRIBUTE: Final[dict[str, str]] = {
    "tag16h5": "DICT_APRILTAG_16h5",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag36h11": "DICT_APRILTAG_36h11",
}


@dataclass(frozen=True)
class AprilTagConfig:
    family: str = "tag36h11"
    target_id: int = 0
    corner_refinement: bool = True
    quad_decimate: float = 1.0

    def __post_init__(self) -> None:
        family = self.family.strip().lower()
        object.__setattr__(self, "family", family)
        if family not in _FAMILY_ATTRIBUTE:
            supported = ", ".join(sorted(_FAMILY_ATTRIBUTE))
            raise ValueError(
                f"unsupported AprilTag family {family!r}; supported: {supported}"
            )
        if self.target_id < 0:
            raise ValueError("target AprilTag id must be non-negative")
        if self.quad_decimate <= 0.0:
            raise ValueError("quad_decimate must be positive")


@dataclass(frozen=True)
class AprilTagDetection:
    tag_id: int
    corners: NDArray[np.float64]
    center: NDArray[np.float64]
    pixel_area: float
    perimeter_px: float
    decision_margin: float | None = None
    hamming: int | None = None

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners, dtype=np.float64).reshape(4, 2)
        center = np.asarray(self.center, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(corners)) or not np.all(np.isfinite(center)):
            raise ValueError("detection coordinates must be finite")
        object.__setattr__(self, "corners", corners.copy())
        object.__setattr__(self, "center", center.copy())


class AprilTagDetector:
    def __init__(self, config: AprilTagConfig) -> None:
        if not hasattr(cv2, "aruco"):
            raise DetectorError("OpenCV was built without the aruco module")
        self.config = config
        attribute = _FAMILY_ATTRIBUTE[config.family]
        dictionary_id = getattr(cv2.aruco, attribute, None)
        if dictionary_id is None:
            # Some OpenCV builds expose the same constants with an uppercase H.
            dictionary_id = getattr(cv2.aruco, attribute.upper(), None)
        if dictionary_id is None:
            raise DetectorError(
                f"this OpenCV build does not provide AprilTag family {config.family}"
            )
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = self._make_parameters(config)
        self._detector: Any | None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self._legacy_dictionary = None
            self._legacy_parameters = None
        else:  # Compatibility with OpenCV versions used by older ROS 2 releases.
            self._detector = None
            self._legacy_dictionary = dictionary
            self._legacy_parameters = parameters

    @staticmethod
    def _make_parameters(config: AprilTagConfig):
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = getattr(cv2.aruco, "DetectorParameters_create")()
        if hasattr(parameters, "cornerRefinementMethod"):
            parameters.cornerRefinementMethod = (
                cv2.aruco.CORNER_REFINE_SUBPIX
                if config.corner_refinement
                else cv2.aruco.CORNER_REFINE_NONE
            )
        if hasattr(parameters, "aprilTagQuadDecimate"):
            parameters.aprilTagQuadDecimate = float(config.quad_decimate)
        return parameters

    def detect(self, image: NDArray[np.uint8]) -> AprilTagDetection | None:
        array = np.asarray(image)
        if array.ndim == 3:
            if array.shape[2] == 3:
                grayscale = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            elif array.shape[2] == 4:
                grayscale = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
            else:
                raise DetectorError(
                    f"unsupported image channel count: {array.shape[2]}"
                )
        elif array.ndim == 2:
            grayscale = array
        else:
            raise DetectorError(f"unsupported image shape: {array.shape}")
        if grayscale.dtype != np.uint8:
            raise DetectorError("detector expects an 8-bit image")

        if self._detector is not None:
            corners_list, ids, _ = self._detector.detectMarkers(grayscale)
        else:
            legacy_detect_markers = getattr(cv2.aruco, "detectMarkers")
            corners_list, ids, _ = legacy_detect_markers(
                grayscale,
                self._legacy_dictionary,
                parameters=self._legacy_parameters,
            )
        if ids is None:
            return None

        matches: list[AprilTagDetection] = []
        for marker_corners, marker_id in zip(
            corners_list, ids.reshape(-1)
        ):
            tag_id = int(marker_id)
            if tag_id != self.config.target_id:
                continue
            corners = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            area = abs(float(cv2.contourArea(corners.astype(np.float32))))
            perimeter = float(cv2.arcLength(corners.astype(np.float32), True))
            matches.append(
                AprilTagDetection(
                    tag_id=tag_id,
                    corners=corners,
                    center=np.mean(corners, axis=0),
                    pixel_area=area,
                    perimeter_px=perimeter,
                )
            )
        if not matches:
            return None
        return max(matches, key=lambda detection: detection.pixel_area)

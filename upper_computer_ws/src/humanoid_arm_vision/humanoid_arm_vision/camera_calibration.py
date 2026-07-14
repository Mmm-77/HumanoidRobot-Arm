"""Loading and validation for ROS/OpenCV camera calibration data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray


class CalibrationError(ValueError):
    """Raised when calibration data cannot be used safely."""


def _matrix_data(value: Any, field_name: str) -> Sequence[float]:
    if isinstance(value, Mapping):
        value = value.get("data")
    if not isinstance(value, (list, tuple)):
        raise CalibrationError(f"{field_name} must be a numeric sequence")
    return value


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: NDArray[np.float64]
    distortion_coefficients: NDArray[np.float64]
    image_width: int
    image_height: int
    distortion_model: str = "plumb_bob"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(
            -1
        )
        object.__setattr__(self, "camera_matrix", matrix.copy())
        object.__setattr__(self, "distortion_coefficients", distortion.copy())
        self.validate()

    def validate(self) -> None:
        matrix = self.camera_matrix
        distortion = self.distortion_coefficients
        if matrix.shape != (3, 3):
            raise CalibrationError(
                f"camera matrix must have shape (3, 3), got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise CalibrationError("camera matrix contains a non-finite value")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise CalibrationError("camera focal lengths fx and fy must be positive")
        if abs(float(matrix[2, 2]) - 1.0) > 1e-9:
            raise CalibrationError("camera matrix K[2,2] must equal 1")
        if not np.all(np.isfinite(distortion)):
            raise CalibrationError("distortion coefficients contain a non-finite value")
        if distortion.size not in (0, 4, 5, 8, 12, 14):
            raise CalibrationError(
                "distortion coefficient count must be one of 0, 4, 5, 8, 12, or 14"
            )
        if self.image_width <= 0 or self.image_height <= 0:
            raise CalibrationError(
                "calibration image width and height must be positive"
            )
        if not self.distortion_model:
            raise CalibrationError("distortion model must not be empty")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CameraCalibration":
        try:
            width = int(data.get("image_width", data.get("width")))
            height = int(data.get("image_height", data.get("height")))
            matrix_values = _matrix_data(data.get("camera_matrix"), "camera_matrix")
            distortion_values = _matrix_data(
                data.get("distortion_coefficients", data.get("distortion")),
                "distortion_coefficients",
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid calibration scalar: {exc}") from exc
        return cls(
            camera_matrix=np.asarray(matrix_values, dtype=np.float64).reshape(3, 3),
            distortion_coefficients=np.asarray(distortion_values, dtype=np.float64),
            image_width=width,
            image_height=height,
            distortion_model=str(data.get("distortion_model", "plumb_bob")),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "CameraCalibration":
        calibration_path = Path(path).expanduser()
        if not calibration_path.is_file():
            raise CalibrationError(
                f"calibration file does not exist: {calibration_path}"
            )
        try:
            with calibration_path.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise CalibrationError(f"failed to read calibration file: {exc}") from exc
        if not isinstance(data, Mapping):
            raise CalibrationError("calibration file root must be a mapping")
        return cls.from_mapping(data)

    def for_resolution(
        self, width: int, height: int, *, allow_scaling: bool = False
    ) -> "CameraCalibration":
        if width == self.image_width and height == self.image_height:
            return self
        if not allow_scaling:
            raise CalibrationError(
                f"capture resolution {width}x{height} does not match calibration "
                f"{self.image_width}x{self.image_height}"
            )
        if width <= 0 or height <= 0:
            raise CalibrationError("capture resolution must be positive")
        source_aspect = self.image_width / self.image_height
        target_aspect = width / height
        if not np.isclose(source_aspect, target_aspect, rtol=1e-6, atol=1e-9):
            raise CalibrationError(
                "calibration cannot be scaled across different aspect ratios"
            )
        scale_x = width / self.image_width
        scale_y = height / self.image_height
        matrix = self.camera_matrix.copy()
        matrix[0, :] *= scale_x
        matrix[1, :] *= scale_y
        matrix[2, :] = self.camera_matrix[2, :]
        return CameraCalibration(
            camera_matrix=matrix,
            distortion_coefficients=self.distortion_coefficients,
            image_width=width,
            image_height=height,
            distortion_model=self.distortion_model,
        )

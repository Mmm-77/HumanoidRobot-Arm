"""Intel RealSense SDK camera ownership, calibration, and reconnect handling."""

from __future__ import annotations

import concurrent.futures
import importlib
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .camera_calibration import CameraCalibration


class CameraError(RuntimeError):
    """Base class for RealSense camera failures."""


class CameraOpenError(CameraError):
    """Raised when the configured RealSense camera cannot be opened."""


class CameraReadError(CameraError):
    """Raised when an opened RealSense camera does not return a color frame."""


@dataclass(frozen=True)
class CameraConfig:
    serial_number: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30
    frame_timeout_ms: int = 1000
    reopen_after_failures: int = 3
    reopen_delay_s: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.fps <= 0:
            raise ValueError("camera fps must be positive")
        if self.frame_timeout_ms <= 0:
            raise ValueError("camera frame timeout must be positive")
        if self.reopen_after_failures <= 0:
            raise ValueError("reopen_after_failures must be positive")
        if self.reopen_delay_s < 0.0:
            raise ValueError("reopen_delay_s must be non-negative")


@dataclass(frozen=True)
class CameraFrame:
    image: NDArray[np.uint8]
    capture_time_s: float
    sequence: int


class RealSenseCamera:
    """Capture BGR frames and factory intrinsics through ``pyrealsense2``."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        rs_module: Any | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        try:
            self._rs = (
                rs_module
                if rs_module is not None
                else importlib.import_module("pyrealsense2")
            )
        except ImportError as exc:
            raise CameraOpenError(
                "pyrealsense2 is not installed for this Python interpreter"
            ) from exc
        self._clock = monotonic_clock
        self._pipeline: Any | None = None
        self._calibration: CameraCalibration | None = None
        self._hardware_id = "Intel RealSense"
        self._sequence = 0
        self._consecutive_failures = 0
        self._next_open_time_s = 0.0

    @property
    def is_open(self) -> bool:
        return self._pipeline is not None

    @property
    def calibration(self) -> CameraCalibration:
        if self._calibration is None:
            raise CameraOpenError("RealSense stream has not been started")
        return self._calibration

    @property
    def hardware_id(self) -> str:
        return self._hardware_id

    def open(self) -> None:
        now = self._clock()
        if self.is_open:
            return
        if now < self._next_open_time_s:
            wait = self._next_open_time_s - now
            raise CameraOpenError(f"camera reconnect backoff active for {wait:.3f}s")

        pipeline = self._rs.pipeline()
        sdk_config = self._rs.config()
        if self.config.serial_number:
            sdk_config.enable_device(self.config.serial_number)
        sdk_config.enable_stream(
            self._rs.stream.color,
            self.config.width,
            self.config.height,
            self._rs.format.bgr8,
            self.config.fps,
        )
        try:
            profile = pipeline.start(sdk_config)
            video_profile = profile.get_stream(
                self._rs.stream.color
            ).as_video_stream_profile()
            intrinsics = video_profile.get_intrinsics()
            calibration = self._calibration_from_intrinsics(intrinsics)
            self._hardware_id = self._device_hardware_id(profile)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            self._schedule_reopen(now)
            raise CameraOpenError(f"failed to start RealSense color stream: {exc}") from exc

        self._pipeline = pipeline
        self._calibration = calibration
        self._consecutive_failures = 0

    def _calibration_from_intrinsics(self, intrinsics: Any) -> CameraCalibration:
        distortion_none = getattr(self._rs.distortion, "none")
        distortion_brown = getattr(self._rs.distortion, "brown_conrady", None)
        distortion_inverse = getattr(self._rs.distortion, "inverse_brown_conrady", None)
        if intrinsics.model == distortion_none:
            coefficients: list[float] = []
        elif distortion_brown is not None and intrinsics.model == distortion_brown:
            coefficients = [float(value) for value in intrinsics.coeffs]
        elif (
            distortion_inverse is not None
            and intrinsics.model == distortion_inverse
        ):
            coefficients = [float(value) for value in intrinsics.coeffs]
        else:
            raise CameraOpenError(
                "unsupported RealSense color distortion model for OpenCV pose "
                f"solving: {intrinsics.model}"
            )
        matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return CameraCalibration(
            camera_matrix=matrix,
            distortion_coefficients=np.asarray(coefficients, dtype=np.float64),
            image_width=int(intrinsics.width),
            image_height=int(intrinsics.height),
            distortion_model="plumb_bob",
        )

    def _device_hardware_id(self, profile: Any) -> str:
        device = profile.get_device()
        parts: list[str] = []
        for label, info in (
            ("name", self._rs.camera_info.name),
            ("serial", self._rs.camera_info.serial_number),
        ):
            try:
                if device.supports(info):
                    parts.append(f"{label}={device.get_info(info)}")
            except Exception:
                continue
        return ", ".join(parts) or "Intel RealSense"

    def _schedule_reopen(self, now: float | None = None) -> None:
        current = self._clock() if now is None else now
        self._next_open_time_s = current + self.config.reopen_delay_s

    def read(self) -> CameraFrame:
        if not self.is_open:
            self.open()
        assert self._pipeline is not None
        try:
            # wrap wait_for_frames in a thread so a physically disconnected
            # camera (where the SDK may hang indefinitely) cannot block the
            # ROS timer callback forever.
            deadline_s = self.config.frame_timeout_ms / 1000.0 + 0.5
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(
                    self._pipeline.wait_for_frames, self.config.frame_timeout_ms
                )
                frames = future.result(timeout=deadline_s)
            finally:
                pool.shutdown(wait=False)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("frameset does not contain a color frame")
            image = np.asanyarray(color_frame.get_data())
            capture_time = self._clock()
            expected_shape = (self.config.height, self.config.width, 3)
            if image.dtype != np.uint8 or image.shape != expected_shape:
                raise RuntimeError(
                    f"unexpected BGR frame shape or dtype: {image.shape}, {image.dtype}"
                )
        except Exception as exc:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
            if failures >= self.config.reopen_after_failures:
                self.close()
                self._schedule_reopen()
            raise CameraReadError(
                f"RealSense frame read failed ({failures}/"
                f"{self.config.reopen_after_failures}): {exc}"
            ) from exc

        self._consecutive_failures = 0
        self._sequence += 1
        return CameraFrame(
            image=image, capture_time_s=capture_time, sequence=self._sequence
        )

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._calibration = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def __enter__(self) -> "RealSenseCamera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

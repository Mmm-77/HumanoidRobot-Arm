"""OpenCV camera ownership, configuration, timestamps, and reconnect handling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


class CameraError(RuntimeError):
    """Base class for camera failures."""


class CameraOpenError(CameraError):
    """Raised when the configured camera cannot be opened."""


class CameraReadError(CameraError):
    """Raised when an opened camera does not return a valid frame."""


class _Capture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class CameraConfig:
    device: int | str = 0
    backend: int = -1
    width: int = 640
    height: int = 480
    fps: float = 30.0
    buffer_size: int = 1
    reopen_after_failures: int = 3
    reopen_delay_s: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.fps <= 0.0:
            raise ValueError("camera fps must be positive")
        if self.buffer_size <= 0:
            raise ValueError("camera buffer size must be positive")
        if self.reopen_after_failures <= 0:
            raise ValueError("reopen_after_failures must be positive")
        if self.reopen_delay_s < 0.0:
            raise ValueError("reopen_delay_s must be non-negative")


@dataclass(frozen=True)
class CameraFrame:
    image: NDArray[np.uint8]
    capture_time_s: float
    sequence: int


CaptureFactory = Callable[..., _Capture]


class OpenCVCamera:
    def __init__(
        self,
        config: CameraConfig,
        *,
        capture_factory: CaptureFactory = cv2.VideoCapture,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._capture_factory = capture_factory
        self._clock = monotonic_clock
        self._capture: _Capture | None = None
        self._sequence = 0
        self._consecutive_failures = 0
        self._next_open_time_s = 0.0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    def open(self) -> None:
        now = self._clock()
        if self.is_open:
            return
        if now < self._next_open_time_s:
            wait = self._next_open_time_s - now
            raise CameraOpenError(f"camera reconnect backoff active for {wait:.3f}s")

        try:
            if self.config.backend >= 0:
                capture = self._capture_factory(self.config.device, self.config.backend)
            else:
                capture = self._capture_factory(self.config.device)
        except Exception as exc:  # OpenCV backends can raise backend-specific errors.
            self._schedule_reopen(now)
            raise CameraOpenError(f"failed to create camera capture: {exc}") from exc
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            self._schedule_reopen(now)
            raise CameraOpenError(
                f"unable to open camera device {self.config.device!r}"
            )

        self._capture = capture
        self._set_if_supported(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width))
        self._set_if_supported(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height))
        self._set_if_supported(cv2.CAP_PROP_FPS, float(self.config.fps))
        self._set_if_supported(cv2.CAP_PROP_BUFFERSIZE, float(self.config.buffer_size))
        self._consecutive_failures = 0

    def _set_if_supported(self, prop_id: int, value: float) -> None:
        if self._capture is not None:
            # Some OpenCV backends return False for unsupported properties. The
            # actual frame size is validated against calibration after capture.
            self._capture.set(prop_id, value)

    def _schedule_reopen(self, now: float | None = None) -> None:
        current = self._clock() if now is None else now
        self._next_open_time_s = current + self.config.reopen_delay_s

    def read(self) -> CameraFrame:
        if not self.is_open:
            self.open()
        assert self._capture is not None
        ok, image = self._capture.read()
        capture_time = self._clock()
        if not ok or image is None or image.size == 0:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.reopen_after_failures:
                self.close()
                self._schedule_reopen(capture_time)
            raise CameraReadError(
                f"camera read failed ({self._consecutive_failures}/"
                f"{self.config.reopen_after_failures})"
            )
        if image.ndim not in (2, 3):
            raise CameraReadError(
                f"camera returned an unsupported image shape: {image.shape}"
            )
        self._consecutive_failures = 0
        self._sequence += 1
        return CameraFrame(
            image=image, capture_time_s=capture_time, sequence=self._sequence
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "OpenCVCamera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

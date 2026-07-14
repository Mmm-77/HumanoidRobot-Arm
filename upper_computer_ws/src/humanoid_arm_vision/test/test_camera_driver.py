import numpy as np
import pytest

from humanoid_arm_vision.camera_driver import (
    CameraConfig,
    CameraReadError,
    OpenCVCamera,
)


class FakeCapture:
    def __init__(self, reads):
        self.opened = True
        self.reads = iter(reads)
        self.properties = []
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return next(self.reads)

    def set(self, prop_id, value):
        self.properties.append((prop_id, value))
        return True

    def release(self):
        self.released = True
        self.opened = False


def test_opens_configures_and_timestamps_frames() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FakeCapture([(True, image)])
    camera = OpenCVCamera(
        CameraConfig(width=640, height=480, fps=30.0),
        capture_factory=lambda *_: capture,
        monotonic_clock=iter([1.0, 1.1]).__next__,
    )
    frame = camera.read()
    assert frame.sequence == 1
    assert frame.capture_time_s == pytest.approx(1.1)
    assert frame.image is image
    assert len(capture.properties) == 4
    camera.close()
    assert capture.released


def test_closes_camera_after_configured_read_failures() -> None:
    capture = FakeCapture([(False, None), (False, None)])
    times = iter([1.0, 1.1, 1.2, 1.3])
    camera = OpenCVCamera(
        CameraConfig(reopen_after_failures=2, reopen_delay_s=0.5),
        capture_factory=lambda *_: capture,
        monotonic_clock=times.__next__,
    )
    with pytest.raises(CameraReadError):
        camera.read()
    with pytest.raises(CameraReadError):
        camera.read()
    assert capture.released
    assert not camera.is_open

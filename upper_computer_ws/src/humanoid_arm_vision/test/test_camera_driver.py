from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_arm_vision.camera_driver import (
    CameraConfig,
    CameraOpenError,
    CameraReadError,
    RealSenseCamera,
)


class FakePipeline:
    def __init__(self, module):
        self.module = module
        self.stopped = False

    def start(self, config):
        self.module.started_config = config
        return self.module.profile

    def wait_for_frames(self, timeout_ms):
        self.module.timeout_ms = timeout_ms
        result = self.module.frames.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def stop(self):
        self.stopped = True


class FakeConfig:
    def __init__(self):
        self.serial = None
        self.stream_args = None

    def enable_device(self, serial):
        self.serial = serial

    def enable_stream(self, *args):
        self.stream_args = args


class FakeDevice:
    def supports(self, _):
        return True

    def get_info(self, info):
        return {"name": "Intel RealSense D435I", "serial": "1234"}[info]


class FakeVideoProfile:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics

    def as_video_stream_profile(self):
        return self

    def get_intrinsics(self):
        return self.intrinsics


class FakeProfile:
    def __init__(self, intrinsics):
        self.video_profile = FakeVideoProfile(intrinsics)

    def get_stream(self, _):
        return self.video_profile

    def get_device(self):
        return FakeDevice()


class FakeColorFrame:
    def __init__(self, image):
        self.image = image

    def get_data(self):
        return self.image


class FakeFrames:
    def __init__(self, image):
        self.color_frame = FakeColorFrame(image)

    def get_color_frame(self):
        return self.color_frame


class FakeRs:
    def __init__(self, frames, *, distortion_model="brown"):
        self.stream = SimpleNamespace(color="color")
        self.format = SimpleNamespace(bgr8="bgr8")
        self.distortion = SimpleNamespace(none="none", brown_conrady="brown")
        self.camera_info = SimpleNamespace(name="name", serial_number="serial")
        intrinsics = SimpleNamespace(
            width=640,
            height=480,
            fx=610.0,
            fy=611.0,
            ppx=319.5,
            ppy=239.5,
            coeffs=[0.1, -0.2, 0.01, 0.02, 0.03],
            model=distortion_model,
        )
        self.profile = FakeProfile(intrinsics)
        self.frames = list(frames)
        self.pipelines = []
        self.configs = []

    def pipeline(self):
        pipeline = FakePipeline(self)
        self.pipelines.append(pipeline)
        return pipeline

    def config(self):
        config = FakeConfig()
        self.configs.append(config)
        return config


def test_opens_sdk_stream_and_uses_factory_intrinsics() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    rs = FakeRs([FakeFrames(image)])
    camera = RealSenseCamera(
        CameraConfig(serial_number="1234", frame_timeout_ms=250),
        rs_module=rs,
        monotonic_clock=iter([1.0, 1.1]).__next__,
    )

    frame = camera.read()

    assert frame.sequence == 1
    assert frame.capture_time_s == pytest.approx(1.1)
    assert frame.image is image
    assert rs.configs[0].serial == "1234"
    assert rs.configs[0].stream_args == ("color", 640, 480, "bgr8", 30)
    assert rs.timeout_ms == 250
    assert camera.calibration.camera_matrix[0, 0] == pytest.approx(610.0)
    assert camera.calibration.distortion_coefficients.tolist() == pytest.approx(
        [0.1, -0.2, 0.01, 0.02, 0.03]
    )
    assert "D435I" in camera.hardware_id
    camera.close()
    assert rs.pipelines[0].stopped


def test_rejects_distortion_model_not_supported_by_opencv_solver() -> None:
    rs = FakeRs([], distortion_model="inverse_brown")
    camera = RealSenseCamera(CameraConfig(), rs_module=rs)

    with pytest.raises(CameraOpenError, match="distortion model"):
        camera.open()

    assert rs.pipelines[0].stopped


def test_closes_pipeline_after_configured_read_failures() -> None:
    rs = FakeRs([RuntimeError("timeout"), RuntimeError("disconnected")])
    times = iter([1.0, 1.1, 1.2, 1.3])
    camera = RealSenseCamera(
        CameraConfig(reopen_after_failures=2, reopen_delay_s=0.5),
        rs_module=rs,
        monotonic_clock=times.__next__,
    )

    with pytest.raises(CameraReadError):
        camera.read()
    with pytest.raises(CameraReadError):
        camera.read()

    assert rs.pipelines[0].stopped
    assert not camera.is_open

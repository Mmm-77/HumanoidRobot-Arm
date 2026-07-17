"""Tests for feedback_codec: UploadData_TypeDef unpacking."""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np

from humanoid_arm_communication.crc import crc16_calculate
from humanoid_arm_communication.feedback_codec import decode_feedback
from humanoid_arm_communication.protocol import (
    HEADER_UPLOAD_UPPER,
    UPLOAD_FRAME_SIZE,
    UPLOAD_PAYLOAD_SIZE,
)


def _build_feedback_raw(
    head: int = HEADER_UPLOAD_UPPER,
    imu_quat: Optional[np.ndarray] = None,
    imu_gyro: Optional[np.ndarray] = None,
    imu_state: int = 0,
    motor_angles_deg: Optional[np.ndarray] = None,
    motor_speeds_dps: Optional[np.ndarray] = None,
    motor_torques: Optional[np.ndarray] = None,
    motor_errors: Optional[np.ndarray] = None,
) -> bytes:
    """Build a raw upload frame with valid CRC."""
    if imu_quat is None:
        imu_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if imu_gyro is None:
        imu_gyro = np.zeros(3, dtype=np.float32)
    if motor_angles_deg is None:
        motor_angles_deg = np.zeros(10, dtype=np.float32)
    if motor_speeds_dps is None:
        motor_speeds_dps = np.zeros(10, dtype=np.float32)
    if motor_torques is None:
        motor_torques = np.zeros(10, dtype=np.float32)
    if motor_errors is None:
        motor_errors = np.zeros(10, dtype=np.uint8)

    payload = struct.pack(
        "<B3x4f3fB3x10f10f10f10B",
        head,
        *imu_quat,
        *imu_gyro,
        imu_state,
        *motor_angles_deg,
        *motor_speeds_dps,
        *motor_torques,
        *motor_errors,
    )
    assert len(payload) == UPLOAD_PAYLOAD_SIZE
    crc = struct.pack("<H", crc16_calculate(payload))
    return payload + crc


def test_decode_basic():
    """Decode a valid frame with known motor angles."""
    angles_deg = np.zeros(10, dtype=np.float32)
    angles_deg[1] = 45.0
    angles_deg[2] = 30.0

    raw = _build_feedback_raw(motor_angles_deg=angles_deg)
    fb = decode_feedback(raw)

    assert fb is not None
    assert fb.is_upper_body
    assert abs(fb.joint_angles_rad[0] - np.deg2rad(45.0)) < 0.01
    assert abs(fb.joint_angles_rad[1] - np.deg2rad(30.0)) < 0.01


def test_decode_crc_fail():
    """CRC failure should return None."""
    angles_deg = np.zeros(10, dtype=np.float32)
    raw = _build_feedback_raw(motor_angles_deg=angles_deg)
    # Corrupt a byte in the payload
    corrupted = bytearray(raw)
    corrupted[10] ^= 0xFF
    fb = decode_feedback(bytes(corrupted), verify_crc=True)
    assert fb is None


def test_decode_motor_errors():
    """Motor error flags should be correctly decoded."""
    errors = np.zeros(10, dtype=np.uint8)
    errors[1] = 0x01  # Under-voltage on motor 1

    raw = _build_feedback_raw(motor_errors=errors)
    fb = decode_feedback(raw)

    assert fb is not None
    assert fb.any_motor_error
    assert 1 in fb.error_motors
    assert fb.joint_errors[0] == 0x01


def test_decode_no_crc_verify():
    """With verify_crc=False, corrupted frames should still decode."""
    raw = _build_feedback_raw()
    corrupted = bytearray(raw)
    corrupted[-1] ^= 0xFF  # corrupt CRC
    fb = decode_feedback(bytes(corrupted), verify_crc=False)
    assert fb is not None


def test_decode_returns_none_width():
    """decode_feedback should return None, not FeedbackFrame, on CRC fail."""
    raw = _build_feedback_raw()
    corrupted = bytearray(raw)
    corrupted[0] = 0xFF
    fb = decode_feedback(bytes(corrupted))
    assert fb is None

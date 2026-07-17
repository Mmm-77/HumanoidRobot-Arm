"""Decode the UploadData_TypeDef binary frame from the lower computer.

The C struct layout (ARM Cortex-M, default alignment):

    offset  size  field
    ------  ----  -----
    0       1     uint8_t  Head              (0x31 upper / 0x41 lower)
    [1]     3     (padding)
    4       16    float     IMU_Quat[4]      quaternion [x, y, z, w]
    20      12    float     IMU_Gyro[3]      angular velocity (rad/s?)
    32      1     uint8_t   IMU_State
    [33]    3     (padding)
    36      40    float     Motor_Angle[10]  current angles, deg, idx 1..9
    76      40    float     Motor_Speed[10]  current speeds, deg/s
    116     40    float     Motor_Torque[10] current torque, N·m
    156     10    uint8_t   Motor_Error[10]   error flag bitmask per motor
    166     2     uint16_t  CRC_16
    ----   ---
    168  total
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntFlag

import numpy as np

from .crc import crc16_verify
from .protocol import (
    HEADER_UPLOAD_LOWER,
    HEADER_UPLOAD_UPPER,
    RIGHT_ARM_MOTORS,
    UPLOAD_PAYLOAD_SIZE,
)
from .unit_converter import deg_to_rad


class MotorErrorFlags(IntFlag):
    """Per-motor error bitmask from the lower computer."""

    NONE = 0
    UNDER_VOLTAGE = 1 << 0
    OVER_VOLTAGE = 1 << 1
    OVER_TEMP = 1 << 2
    SHORT_CIRCUIT = 1 << 3
    STALLED = 1 << 4
    OVER_POSITION = 1 << 5
    LOST = 1 << 6


# struct format for the frame (excluding CRC, which is verified separately).
# Layout: B 3x 4f 3f B 3x 10f 10f 10f 10B
_UPLOAD_STRUCT = struct.Struct("<B3x4f3fB3x10f10f10f10B")


@dataclass(frozen=True)
class FeedbackFrame:
    """Decoded feedback from the lower computer.

    All angles are in radians, velocities in rad/s, torques in N·m.
    Only motors 1..4 (right arm) are populated; others are available
    in the raw arrays if needed.
    """

    # --- Header ---
    is_upper_body: bool  # True = head 0x31; False = head 0x41

    # --- IMU ---
    imu_quaternion_xyzw: np.ndarray  # 4-element
    imu_gyro_rad_per_s: np.ndarray   # 3-element
    imu_state: int

    # --- Motors (right arm, 0-indexed [0..3] = motor 1..4) ---
    joint_angles_rad: np.ndarray    # 4-element
    joint_velocities_rad_per_s: np.ndarray  # 4-element
    joint_torque_nm: np.ndarray     # 4-element
    joint_errors: np.ndarray        # 4-element, MotorErrorFlags bitmask

    # --- Raw full arrays (indices 0..9, motor 1..9 at index 1..9) ---
    all_angles_deg: np.ndarray
    all_speeds_deg_per_s: np.ndarray
    all_torques_nm: np.ndarray
    all_errors: np.ndarray         # uint8

    # --- Diagnostics ---
    any_motor_error: bool = False
    error_motors: list[int] = field(default_factory=list)


class FeedbackError(RuntimeError):
    """Raised when feedback decoding fails."""


def decode_feedback(raw: bytes, verify_crc: bool = True) -> FeedbackFrame | None:
    """Decode a raw UploadData_TypeDef frame from the lower computer.

    Args:
        raw: Exactly UPLOAD_FRAME_SIZE (168) bytes.
        verify_crc: If True, verify the CRC-16 before parsing.

    Returns:
        FeedbackFrame, or None if CRC verification fails.
    """
    from .protocol import UPLOAD_FRAME_SIZE

    if len(raw) != UPLOAD_FRAME_SIZE:
        raise FeedbackError(
            f"Expected {UPLOAD_FRAME_SIZE} bytes, got {len(raw)}"
        )

    # Split payload and CRC
    payload = raw[:UPLOAD_PAYLOAD_SIZE]
    crc_bytes = raw[UPLOAD_PAYLOAD_SIZE:]
    expected_crc = struct.unpack("<H", crc_bytes)[0]

    if verify_crc and not crc16_verify(payload, expected_crc):
        return None  # CRC mismatch

    # Unpack – avoid multiple * in assignment (Python 3.8 compat)
    raw_values = _UPLOAD_STRUCT.unpack(payload)
    head = raw_values[0]
    quat_x, quat_y, quat_z, quat_w = raw_values[1:5]
    gyro_x, gyro_y, gyro_z = raw_values[5:8]
    imu_state = raw_values[8]
    all_angles = raw_values[9:19]
    all_speeds = raw_values[19:29]
    all_torques = raw_values[29:39]
    all_errors = raw_values[39:49]

    # Determine body side
    if head == HEADER_UPLOAD_UPPER:
        is_upper = True
    elif head == HEADER_UPLOAD_LOWER:
        is_upper = False
    else:
        raise FeedbackError(f"Unknown upload header: 0x{head:02X}")

    # Convert to numpy
    angles_deg = np.array(all_angles, dtype=np.float32)    # 10 elements
    speeds_dps = np.array(all_speeds, dtype=np.float32)    # 10 elements
    torques_nm = np.array(all_torques, dtype=np.float32)   # 10 elements
    errors = np.array(all_errors, dtype=np.uint8)           # 10 elements

    # Extract right-arm motors (1..4)
    arm_angles_deg = np.array([angles_deg[i] for i in RIGHT_ARM_MOTORS])
    arm_speeds_dps = np.array([speeds_dps[i] for i in RIGHT_ARM_MOTORS])
    arm_torques = np.array([torques_nm[i] for i in RIGHT_ARM_MOTORS])
    arm_errors = np.array([errors[i] for i in RIGHT_ARM_MOTORS], dtype=np.uint8)

    # Check for motor errors
    error_motors = [
        int(mid) for mid in RIGHT_ARM_MOTORS
        if errors[mid] != 0
    ]

    return FeedbackFrame(
        is_upper_body=is_upper,
        imu_quaternion_xyzw=np.array([quat_x, quat_y, quat_z, quat_w], dtype=np.float32),
        imu_gyro_rad_per_s=deg_to_rad(np.array([gyro_x, gyro_y, gyro_z], dtype=np.float32)),
        imu_state=int(imu_state),
        joint_angles_rad=deg_to_rad(arm_angles_deg).astype(np.float64),
        joint_velocities_rad_per_s=deg_to_rad(arm_speeds_dps).astype(np.float64),
        joint_torque_nm=arm_torques.astype(np.float64),
        joint_errors=arm_errors,
        all_angles_deg=angles_deg,
        all_speeds_deg_per_s=speeds_dps,
        all_torques_nm=torques_nm,
        all_errors=errors,
        any_motor_error=len(error_motors) > 0,
        error_motors=error_motors,
    )

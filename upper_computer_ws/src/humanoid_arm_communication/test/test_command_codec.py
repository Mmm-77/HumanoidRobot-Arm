"""Tests for command_codec: DownloadData_TypeDef packing."""

import struct

import numpy as np

from humanoid_arm_communication.command_codec import CommandFrame, encode_command
from humanoid_arm_communication.protocol import (
    ControlMode,
    DOWNLOAD_FRAME_SIZE,
    DOWNLOAD_PAYLOAD_SIZE,
    HEADER_DOWNLOAD,
    RIGHT_ARM_MOTORS,
    UPPER_MOTOR_TOTAL_NUM,
)


def test_encode_basic():
    """Verify that encode_command produces the correct frame size and header."""
    angles = np.array([0.0, 0.5, -0.3, 0.3])       # rad
    velocities = np.array([0.0, -0.1, 0.2, 0.0])     # rad/s

    frame = encode_command(angles, velocities, ctrl_mode=ControlMode.MIT)

    assert isinstance(frame, CommandFrame)
    assert len(frame.data) == DOWNLOAD_FRAME_SIZE

    # Verify header byte
    assert frame.data[0] == HEADER_DOWNLOAD
    # Verify control mode
    assert frame.data[1] == ControlMode.MIT

    # Verify payload size
    payload = frame.data[:DOWNLOAD_PAYLOAD_SIZE]
    assert len(payload) == DOWNLOAD_PAYLOAD_SIZE


def test_encode_motor_positions():
    """Verify that motor 1-4 angle values end up in the correct struct slots."""
    angles = np.array([0.1, 0.2, 0.3, 0.4])
    velocities = np.zeros(4)

    frame = encode_command(angles, velocities)

    # Unpack the pos array from the payload at the known offset
    # Offset of Motor_MIT_Pos[0]: 4 bytes into the struct
    unpacked = struct.unpack_from("<10f", frame.data, offset=4)

    # Index 0 should be 0 (unused), indices 1-4 should be angles in degrees
    assert unpacked[0] == 0.0
    for i, motor_id in enumerate(RIGHT_ARM_MOTORS):
        expected_deg = np.rad2deg(angles[i])
        assert abs(unpacked[motor_id] - expected_deg) < 0.01


def test_encode_with_custom_gains():
    """Custom Kp/Kd should be packed correctly."""
    angles = np.radians([10, 20, 30, 40])
    velocities = np.zeros(4)
    kp = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    kd = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    frame = encode_command(angles, velocities, kp=kp, kd=kd)

    # Kp starts at offset 84
    kp_unpacked = struct.unpack_from("<10f", frame.data, offset=84)
    # Kd starts at offset 124
    kd_unpacked = struct.unpack_from("<10f", frame.data, offset=124)

    for i, motor_id in enumerate(RIGHT_ARM_MOTORS):
        assert abs(kp_unpacked[motor_id] - kp[i]) < 0.001
        assert abs(kd_unpacked[motor_id] - kd[i]) < 0.001


def test_frame_has_valid_crc():
    """CRC-16 appended to frame should match its payload."""
    from humanoid_arm_communication.crc import crc16_calculate

    angles = np.radians([10, 20, 30, 40])
    velocities = np.zeros(4)
    frame = encode_command(angles, velocities)

    payload = frame.data[:DOWNLOAD_PAYLOAD_SIZE]
    crc_bytes = frame.data[DOWNLOAD_PAYLOAD_SIZE:]
    expected_crc = crc16_calculate(payload)
    actual_crc = struct.unpack("<H", crc_bytes)[0]

    assert actual_crc == expected_crc

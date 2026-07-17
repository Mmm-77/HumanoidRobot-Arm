"""Encode joint commands into the DownloadData_TypeDef binary frame.

The C struct layout (ARM Cortex-M, default alignment):

    offset  size  field
    ------  ----  -----
    0       1     uint8_t  Head         (0x38)
    1       1     uint8_t  CtrlMode     (0=Weak, 1=Pos, 2=MIT)
    [2]     2     (padding)
    4       40    float     Motor_MIT_Pos[10]   degrees, index 0 unused
    44      40    float     Motor_MIT_Spd[10]   deg/s
    84      40    float     Motor_MIT_Kp[10]
    124     40    float     Motor_MIT_Kd[10]
    164     40    float     Motor_MIT_Tff[10]   N·m
    204     2     uint16_t  CRC_16
    ----   ---
    206  total

CRC is computed over bytes [0..203] (everything before the CRC field).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .crc import crc16_calculate
from .protocol import (
    ControlMode,
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_TFF,
    DOWNLOAD_PAYLOAD_SIZE,
    HEADER_DOWNLOAD,
    RIGHT_ARM_MOTORS,
    UPPER_MOTOR_TOTAL_NUM,
)
from .unit_converter import rad_to_deg

# struct format: B B 2x 10f 10f 10f 10f 10f  (little-endian)
_DOWNLOAD_STRUCT = struct.Struct("<BB2x10f10f10f10f10f")


class CodecError(RuntimeError):
    """Raised when a command cannot be encoded."""


@dataclass(frozen=True)
class CommandFrame:
    """An encoded binary frame ready for serial transmission."""

    data: bytes
    joint_angles_deg: np.ndarray = field(default_factory=lambda: np.zeros(4))


def encode_command(
    joint_angles_rad: np.ndarray,
    joint_velocities_rad_per_s: np.ndarray,
    *,
    ctrl_mode: ControlMode = ControlMode.MIT,
    kp: Optional[np.ndarray] = None,
    kd: Optional[np.ndarray] = None,
    tff: Optional[np.ndarray] = None,
) -> CommandFrame:
    """Encode a 4-joint command into a DownloadData_TypeDef frame."""
    angles = np.asarray(joint_angles_rad, dtype=np.float64)
    velocities = np.asarray(joint_velocities_rad_per_s, dtype=np.float64)

    if angles.shape != (4,) or velocities.shape != (4,):
        raise CodecError(
            f"Expected 4 joints, got angles={angles.shape}, velocities={velocities.shape}"
        )
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(velocities)):
        raise CodecError("Joint angles and velocities must be finite")

    angles_deg = np.asarray(rad_to_deg(angles), dtype=np.float32)
    velocities_deg_per_s = np.asarray(rad_to_deg(velocities), dtype=np.float32)

    def _fill(motor_values: np.ndarray) -> np.ndarray:
        arr = np.zeros(UPPER_MOTOR_TOTAL_NUM + 1, dtype=np.float32)
        for i, motor_id in enumerate(RIGHT_ARM_MOTORS):
            arr[motor_id] = motor_values[i]
        return arr

    pos_array = _fill(angles_deg)
    spd_array = _fill(velocities_deg_per_s)

    if kp is None:
        kp = np.full(4, DEFAULT_KP, dtype=np.float32)
    if kd is None:
        kd = np.full(4, DEFAULT_KD, dtype=np.float32)
    if tff is None:
        tff = np.full(4, DEFAULT_TFF, dtype=np.float32)

    kp_array = _fill(np.asarray(kp, dtype=np.float32).flatten())
    kd_array = _fill(np.asarray(kd, dtype=np.float32).flatten())
    tff_array = _fill(np.asarray(tff, dtype=np.float32).flatten())

    # Flatten numpy arrays for struct.pack (expects 52 scalar args)
    payload = _DOWNLOAD_STRUCT.pack(
        HEADER_DOWNLOAD,
        int(ctrl_mode),
        *pos_array.tolist(),
        *spd_array.tolist(),
        *kp_array.tolist(),
        *kd_array.tolist(),
        *tff_array.tolist(),
    )

    if len(payload) != DOWNLOAD_PAYLOAD_SIZE:
        raise CodecError(
            f"Unexpected payload size: {len(payload)} != {DOWNLOAD_PAYLOAD_SIZE}"
        )

    crc = crc16_calculate(payload)
    frame = payload + struct.pack("<H", crc)

    return CommandFrame(data=frame, joint_angles_deg=angles_deg)

"""Protocol constants matching STM32 lower computer definitions.

Based on ComputerCtrlTask.h and ComputerCommTask.h.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class ControlMode(IntEnum):
    """Control modes sent to the lower computer (enum Ctrl_Mode)."""

    WEAK = 0x00  # 卸载模式，电机0力矩
    POS = 0x01   # 位置模式，PID 角度闭环
    MIT = 0x02   # MIT 模式，阻抗控制 τ=Kp·Δθ+Kd·Δω+Tff


class MotorErrorBits(IntEnum):
    """Motor error flags from the lower computer (enum Sensor_State)."""

    UNDER_VOLTAGE = 1 << 0  # 欠压
    OVER_VOLTAGE = 1 << 1   # 过压
    OVER_TEMP = 1 << 2      # 过温
    SHORT_CIRCUIT = 1 << 3  # 短路
    STALLED = 1 << 4        # 堵转
    OVER_POS = 1 << 5       # 超限角
    LOST = 1 << 6           # 通信丢失


# --- Frame headers -----------------------------------------------------------

HEADER_DOWNLOAD: Final[int] = 0x38  # 上位机→下位机：上肢信息
HEADER_UPLOAD_UPPER: Final[int] = 0x31   # 下位机→上位机：上肢
HEADER_UPLOAD_LOWER: Final[int] = 0x41   # 下位机→上位机：下肢

# --- Motor count layout from UpperMotorCtrlTask.h ----------------------------

# Total motors in the upper-body struct (indices 0..9, motors 1..9 used)
UPPER_MOTOR_TOTAL_NUM: Final[int] = 9
# We only control the right arm: motors 1..4
RIGHT_ARM_MOTORS: Final[range] = range(1, 5)
# Right arm motor IDs: 1, 2, 3, 4
RIGHT_ARM_START: Final[int] = 1
RIGHT_ARM_END: Final[int] = 4

# --- Struct sizes (C compiler aligned, ARM Cortex-M, no packing) -------------

# sizeof(DownloadData_TypeDef) for upper body (9 motors)
# Layout: uint8 Head(1) + uint8 CtrlMode(1) + 2B pad +
#         float[10]*5 (200B) + uint16 CRC(2B) = 206
DOWNLOAD_FRAME_SIZE: Final[int] = 206
DOWNLOAD_PAYLOAD_SIZE: Final[int] = DOWNLOAD_FRAME_SIZE - 2  # minus CRC

# sizeof(UploadData_TypeDef) for upper body (9 motors)
# Layout: uint8 Head(1) + 3B pad + float[4](16) + float[3](12) +
#         uint8 State(1) + 3B pad + float[10]*3(120) +
#         uint8[10](10) + uint16 CRC(2) = 168
UPLOAD_FRAME_SIZE: Final[int] = 168
UPLOAD_PAYLOAD_SIZE: Final[int] = UPLOAD_FRAME_SIZE - 2  # minus CRC

# --- Default MIT control gains (tunable) -------------------------------------

# These are passed as Kp/Kd/Kff to the lower computer's MIT controller.
# τ = Kp · (θ_target − θ_current) + Kd · (ω_target − ω_current) + Tff
DEFAULT_KP: Final[float] = 1.0    # proportional gain for angle error
DEFAULT_KD: Final[float] = 0.05   # derivative gain for velocity error
DEFAULT_TFF: Final[float] = 0.0   # feed-forward torque (N·m)

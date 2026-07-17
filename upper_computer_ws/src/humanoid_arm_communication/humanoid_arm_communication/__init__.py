"""USB serial communication with STM32 lower computer.

Protocol: fixed-size binary structs over DMA UART, CRC-16 verified.
"""

from .protocol import ControlMode, MotorErrorBits

__all__ = [
    "ControlMode",
    "MotorErrorBits",
]

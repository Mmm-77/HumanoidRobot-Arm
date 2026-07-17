"""Serial transport layer over pyserial for the STM32 USART7 interface.

The lower computer uses DMA reception (``HAL_UART_Receive_DMA``) which reads
exactly ``sizeof(DownloadData_TypeDef)`` bytes per frame.  On the Python side
we must send the exact struct and read incoming bytes into a frame parser.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

import serial


class SerialError(RuntimeError):
    """Raised when the serial port cannot be opened or read."""


class SerialConfig:
    """Serial port configuration."""

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 115200,
        timeout: float = 0.01,
        write_timeout: float = 0.05,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout


class SerialTransport:
    """Non-blocking serial I/O wrapping ``pyserial``.

    Usage::

        transport = SerialTransport(SerialConfig("/dev/ttyACM0", 115200))
        transport.open()
        transport.write(frame_bytes)
        frames = transport.read_available()
        transport.close()
    """

    def __init__(
        self,
        config: SerialConfig,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = monotonic_clock
        self._port: Optional[serial.Serial] = None

    @property
    def is_open(self) -> bool:
        return self._port is not None and self._port.is_open

    def open(self) -> None:
        """Open the serial port."""
        if self.is_open:
            return
        try:
            self._port = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
                write_timeout=self.config.write_timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
        except serial.SerialException as exc:
            raise SerialError(
                f"Cannot open serial port {self.config.port}: {exc}"
            ) from exc

    def close(self) -> None:
        """Close the serial port."""
        port = self._port
        self._port = None
        if port is not None and port.is_open:
            try:
                port.close()
            except Exception:
                pass

    def write(self, data: bytes) -> None:
        """Write raw bytes to the serial port (blocking until buffered)."""
        if not self.is_open:
            raise SerialError("Serial port is not open")
        assert self._port is not None
        try:
            self._port.write(data)
            self._port.flush()
        except serial.SerialException as exc:
            raise SerialError(f"Serial write failed: {exc}") from exc

    def read_available(self) -> bytes:
        """Read all currently available bytes without blocking.

        Returns:
            All bytes in the receive buffer (may be empty).
        """
        if not self.is_open:
            raise SerialError("Serial port is not open")
        assert self._port is not None
        try:
            waiting = self._port.in_waiting
            if waiting > 0:
                return self._port.read(waiting)
            return b""
        except serial.SerialException as exc:
            raise SerialError(f"Serial read failed: {exc}") from exc

    def discard_input(self) -> None:
        """Discard any buffered input (use after reconnect)."""
        if not self.is_open:
            return
        assert self._port is not None
        try:
            self._port.reset_input_buffer()
        except Exception:
            pass

    def __enter__(self) -> "SerialTransport":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

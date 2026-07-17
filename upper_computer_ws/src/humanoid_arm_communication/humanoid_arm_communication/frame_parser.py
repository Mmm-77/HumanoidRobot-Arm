"""Frame parser: extract fixed-size frames from a byte stream.

The lower computer uses DMA with exact-length reception, so each frame is
exactly DOWNLOAD_FRAME_SIZE (206) or UPLOAD_FRAME_SIZE (168) bytes.
There are no frame delimiters – we rely on the fixed size to split frames.

For the upload direction (lower→upper), frames arrive continuously at
the control rate. The parser maintains a ring buffer and emits complete
frames as they become available.
"""

from __future__ import annotations

from typing import List

from .protocol import UPLOAD_FRAME_SIZE


class ParserError(RuntimeError):
    """Raised when the byte stream cannot be parsed."""


class FrameParser:
    """Split a byte stream into fixed-size upload frames."""

    def __init__(self, frame_size: int = UPLOAD_FRAME_SIZE) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self._frame_size = frame_size
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        """Clear the internal buffer (e.g. after a reconnect)."""
        self._buffer.clear()

    def feed(self, data: bytes) -> List[bytes]:
        """Feed raw bytes and return a list of complete frames.

        Args:
            data: Raw bytes received from the serial port.

        Returns:
            List of complete frames (each UPLOAD_FRAME_SIZE bytes).
        """
        self._buffer.extend(data)
        frames: List[bytes] = []

        while len(self._buffer) >= self._frame_size:
            frame = bytes(self._buffer[:self._frame_size])
            self._buffer = self._buffer[self._frame_size:]
            frames.append(frame)

        return frames

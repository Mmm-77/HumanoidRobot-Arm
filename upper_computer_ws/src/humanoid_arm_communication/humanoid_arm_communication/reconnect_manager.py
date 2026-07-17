"""Reconnect manager: exponential backoff, timeout tracking, and state reporting.

Since the lower computer uses DMA for UART reception, a missing or corrupted
frame does not produce an application-level error until the communication
watchdog on the STM32 side triggers.  The reconnect manager tracks how long
it has been since the last valid feedback frame and decides when to declare
a communication loss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class LinkState(str, Enum):
    """Communication link state."""

    CONNECTED = "connected"
    DEGRADED = "degraded"     # occasional frame drops
    DISCONNECTED = "disconnected"


@dataclass
class ReconnectConfig:
    """Configuration for the reconnect manager.

    Attributes:
        feedback_timeout_s: How long without valid feedback before declaring
            the link disconnected.
        degraded_threshold_s: Shorter threshold for warning-level degradation.
        initial_backoff_s: Seconds to wait before the first reconnect attempt.
        max_backoff_s: Maximum backoff interval (capped).
        backoff_multiplier: Multiplier for each successive retry.
        max_consecutive_crc_failures: If this many CRC failures occur
            consecutively, force a reconnect.
    """

    feedback_timeout_s: float = 0.5
    degraded_threshold_s: float = 0.2
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 10.0
    backoff_multiplier: float = 2.0
    max_consecutive_crc_failures: int = 5


class ReconnectManager:
    """Tracks link health and backoff timing."""

    def __init__(
        self,
        config: ReconnectConfig,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = monotonic_clock
        self._last_feedback_time: Optional[float] = None
        self._consecutive_failures = 0
        self._backoff = config.initial_backoff_s
        self._next_reconnect_time = 0.0

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def time_since_last_feedback(self) -> Optional[float]:
        if self._last_feedback_time is None:
            return None
        return self._clock() - self._last_feedback_time

    @property
    def link_state(self) -> LinkState:
        elapsed = self.time_since_last_feedback
        if elapsed is None:
            return LinkState.DISCONNECTED
        if elapsed > self.config.feedback_timeout_s:
            return LinkState.DISCONNECTED
        if elapsed > self.config.degraded_threshold_s:
            return LinkState.DEGRADED
        return LinkState.CONNECTED

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def should_reconnect(self) -> bool:
        """True if the link should be re-established."""
        if self.link_state != LinkState.DISCONNECTED:
            return False
        return self._clock() >= self._next_reconnect_time

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_feedback_received(self) -> None:
        """Call when a valid (CRC-passed) feedback frame is received."""
        self._last_feedback_time = self._clock()
        self._consecutive_failures = 0
        self._backoff = self.config.initial_backoff_s

    def on_crc_failure(self) -> None:
        """Call when a CRC check fails on an upload frame."""
        self._consecutive_failures += 1

    def on_reconnect_failed(self) -> None:
        """Call when a reconnect attempt fails; schedule next retry."""
        self._next_reconnect_time = self._clock() + self._backoff
        self._backoff = min(
            self._backoff * self.config.backoff_multiplier,
            self.config.max_backoff_s,
        )

    def on_reconnect_succeeded(self) -> None:
        """Call after a successful reconnect."""
        self._backoff = self.config.initial_backoff_s
        self._next_reconnect_time = 0.0

    def reset(self) -> None:
        """Reset all state (e.g. after a fresh open)."""
        self._last_feedback_time = None
        self._consecutive_failures = 0
        self._backoff = self.config.initial_backoff_s
        self._next_reconnect_time = 0.0

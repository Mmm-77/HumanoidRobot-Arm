"""Watchdog: monitor data freshness across vision, communication, and IK.

Detects stale data and counts consecutive failures so the safety_manager
can escalate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .system_context import SystemContext


class WatchdogStatus(str, Enum):
    """Status for a single monitored data source."""

    OK = "ok"
    STALE = "stale"
    LOST = "lost"


@dataclass
class WatchdogConfig:
    """Timeouts (seconds) for each monitored data source."""

    vision_fresh_s: float = 0.2
    vision_stale_s: float = 0.5
    communication_fresh_s: float = 0.2
    communication_stale_s: float = 0.5
    ik_fresh_s: float = 0.5


class Watchdog:
    """Monitor data freshness and failure counts."""

    def __init__(
        self,
        context: SystemContext,
        config: WatchdogConfig,
        *,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ctx = context
        self._config = config
        self._clock = monotonic_clock

        self._consecutive_vision_lost = 0
        self._consecutive_ik_failures = 0

    # ------------------------------------------------------------------
    # Status checks
    # ------------------------------------------------------------------

    def vision_status(self) -> WatchdogStatus:
        pose = self._ctx.get_pose()
        if pose is None:
            return WatchdogStatus.LOST
        age = self._clock() - pose.timestamp_s
        if age > self._config.vision_stale_s:
            return WatchdogStatus.STALE
        if age > self._config.vision_fresh_s:
            return WatchdogStatus.STALE  # marginal
        return WatchdogStatus.OK

    def communication_status(self) -> WatchdogStatus:
        joints = self._ctx.get_joints()
        if joints is None:
            return WatchdogStatus.LOST
        age = self._clock() - joints.timestamp_s
        if age > self._config.communication_stale_s:
            return WatchdogStatus.STALE
        if age > self._config.communication_fresh_s:
            return WatchdogStatus.STALE
        return WatchdogStatus.OK

    @property
    def consecutive_vision_lost(self) -> int:
        return self._consecutive_vision_lost

    @property
    def consecutive_ik_failures(self) -> int:
        return self._consecutive_ik_failures

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def on_vision_ok(self) -> None:
        self._consecutive_vision_lost = 0

    def on_vision_lost(self) -> None:
        self._consecutive_vision_lost += 1

    def on_ik_ok(self) -> None:
        self._consecutive_ik_failures = 0

    def on_ik_failure(self) -> None:
        self._consecutive_ik_failures += 1

    def reset(self) -> None:
        self._consecutive_vision_lost = 0
        self._consecutive_ik_failures = 0

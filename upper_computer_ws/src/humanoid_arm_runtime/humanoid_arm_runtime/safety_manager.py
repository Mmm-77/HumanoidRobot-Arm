"""Safety manager: decide whether to permit arm movement or enter hold/safe.

Consults the watchdog for data freshness and state machine for current mode,
then emits one of three control actions:
  - PERMIT: allow normal operation (FOLLOW sends target)
  - HOLD: freeze at the last valid target
  - SAFE: send immediate stop (CtrlMode=WEAK) to disable motors
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .state_machine import SystemState, StateMachine, Transition
from .watchdog import Watchdog, WatchdogStatus


class ControlAction(str, Enum):
    """Allowed control actions from the safety manager."""

    PERMIT = "permit"
    HOLD = "hold"
    SAFE = "safe"


@dataclass
class SafetyResult:
    """Safety evaluation for this tick."""

    action: ControlAction
    state_transition: Transition | None


class SafetyManager:
    """Gateway that decides what the arm is allowed to do."""

    def __init__(
        self,
        state_machine: StateMachine,
        watchdog: Watchdog,
        *,
        max_lost_frames: int = 10,
        max_ik_failures: int = 5,
    ) -> None:
        self._fsm = state_machine
        self._watchdog = watchdog
        self._max_lost_frames = max_lost_frames
        self._max_ik_failures = max_ik_failures

    def evaluate(self) -> SafetyResult:
        """Evaluate the current system state and return a control action."""
        state = self._fsm.state

        # --- FAULT: always safe ---
        if state == SystemState.FAULT:
            return SafetyResult(ControlAction.SAFE, None)

        # --- INIT: arm should be disabled ---
        if state == SystemState.INIT:
            return SafetyResult(ControlAction.SAFE, None)

        # --- Vision monitoring ---
        vision = self._watchdog.vision_status()
        if vision == WatchdogStatus.OK:
            self._watchdog.on_vision_ok()
        else:
            self._watchdog.on_vision_lost()

        # --- IK monitoring (check via watchdog) ---
        if self._watchdog.consecutive_ik_failures > self._max_ik_failures:
            return SafetyResult(ControlAction.SAFE, None)

        # --- State-based escalation ---
        if state == SystemState.FOLLOW:
            # Vision lost → escalate to SAFE
            if (
                self._watchdog.consecutive_vision_lost > self._max_lost_frames
                or vision == WatchdogStatus.LOST
            ):
                if self._watchdog.consecutive_vision_lost > self._max_lost_frames:
                    self._fsm.force(SystemState.SAFE, "vision lost too long")
                    return SafetyResult(ControlAction.SAFE, Transition.VISION_LOST)
            if vision != WatchdogStatus.OK:
                return SafetyResult(ControlAction.HOLD, None)
            return SafetyResult(ControlAction.PERMIT, None)

        if state == SystemState.HOLD:
            if vision == WatchdogStatus.LOST:
                if self._watchdog.consecutive_vision_lost > self._max_lost_frames:
                    self._fsm.force(SystemState.SAFE, "vision lost during hold")
                    return SafetyResult(ControlAction.SAFE, Transition.VISION_LOST)
            return SafetyResult(ControlAction.HOLD, None)

        if state == SystemState.SAFE:
            return SafetyResult(ControlAction.SAFE, None)

        # READY: permit system to idle, not moving
        if state == SystemState.READY:
            return SafetyResult(ControlAction.HOLD, None)

        return SafetyResult(ControlAction.SAFE, None)

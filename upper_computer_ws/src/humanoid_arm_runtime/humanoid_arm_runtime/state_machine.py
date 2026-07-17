"""System state machine: 6-state FSM for the humanoid arm controller.

States and transitions (from 上位机项目规划书):

    INIT ──(all systems ready)──> READY
    READY ──(user START)──> FOLLOW
    READY ──(HOLD)──> HOLD
    FOLLOW ──(inactivity > timeout)──> HOLD
    FOLLOW ──(vision lost)──> SAFE
    HOLD ──(UNHOLD + valid baseline)──> FOLLOW
    SAFE ──(vision recovered + user ACK)──> READY
    ANY ──(emergency)──> FAULT
    FAULT ──(user reset)──> INIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, FrozenSet, Optional, Tuple


class SystemState(str, Enum):
    """Named control states."""

    INIT = "INIT"               # initialising, waiting for all packages
    READY = "READY"             # all packages online, arm at zero / held
    FOLLOW = "FOLLOW"           # actively tracking the AprilTag
    HOLD = "HOLD"               # commands frozen at last valid target
    SAFE = "SAFE"               # vision lost, arm held in place
    FAULT = "FAULT"             # unrecoverable error, requires reset


class Transition(str, Enum):
    """Events that trigger state transitions."""

    SYSTEMS_READY = auto()     # all ROS nodes confirmed online
    START = auto()             # user or external trigger to begin following
    STOP_BY_TIMEOUT = auto()   # inactivity timer expired
    VISION_LOST = auto()       # quality gate / vision node reports invalid
    VISION_RECOVERED = auto()  # valid pose received after loss
    ACK_SAFE = auto()          # user acknowledges safe state
    HOLD = auto()              # explicit hold command
    UNHOLD = auto()            # explicit resume command
    EMERGENCY = auto()         # emergency stop (immediate)
    RESET = auto()             # user-initiated reset from fault


# Allowed transitions
_TRANSITIONS: Dict[SystemState, Dict[Transition, SystemState]] = {
    SystemState.INIT: {
        Transition.SYSTEMS_READY: SystemState.READY,
        Transition.EMERGENCY: SystemState.FAULT,
    },
    SystemState.READY: {
        Transition.START: SystemState.FOLLOW,
        Transition.HOLD: SystemState.HOLD,
        Transition.EMERGENCY: SystemState.FAULT,
    },
    SystemState.FOLLOW: {
        Transition.HOLD: SystemState.HOLD,
        Transition.STOP_BY_TIMEOUT: SystemState.HOLD,
        Transition.VISION_LOST: SystemState.SAFE,
        Transition.EMERGENCY: SystemState.FAULT,
    },
    SystemState.HOLD: {
        Transition.UNHOLD: SystemState.FOLLOW,
        Transition.VISION_LOST: SystemState.SAFE,
        Transition.EMERGENCY: SystemState.FAULT,
    },
    SystemState.SAFE: {
        Transition.VISION_RECOVERED: SystemState.READY,
        Transition.ACK_SAFE: SystemState.READY,
        Transition.EMERGENCY: SystemState.FAULT,
    },
    SystemState.FAULT: {
        Transition.RESET: SystemState.INIT,
    },
}


class StateMachineError(RuntimeError):
    """Raised on invalid state transitions."""


@dataclass
class StateChange:
    """Record of a state transition."""

    previous: SystemState
    current: SystemState
    trigger: Transition
    timestamp_s: float


class StateMachine:
    """Manage control states and validate transitions."""

    def __init__(
        self,
        initial: SystemState = SystemState.INIT,
    ) -> None:
        self._state: SystemState = initial
        self._history: list[StateChange] = []

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def history(self) -> Tuple[StateChange, ...]:
        return tuple(self._history)

    @property
    def is_follow_active(self) -> bool:
        """True when the arm can accept follow commands."""
        return self._state == SystemState.FOLLOW

    def transition(
        self,
        event: Transition,
        timestamp_s: float = 0.0,
    ) -> StateChange:
        """Request a state transition.

        Args:
            event: The transition to trigger.
            timestamp_s: Seconds (monotonic) for log.

        Returns:
            StateChange describing the transition.

        Raises:
            StateMachineError: if the transition is not allowed.
        """
        allowed = _TRANSITIONS.get(self._state, {})
        if event not in allowed:
            raise StateMachineError(
                f"Transition {event.value} not allowed from {self._state.value}"
            )

        previous = self._state
        self._state = allowed[event]
        change = StateChange(
            previous=previous,
            current=self._state,
            trigger=event,
            timestamp_s=timestamp_s,
        )
        self._history.append(change)
        return change

    def force(
        self,
        target: SystemState,
        reason: str = "",
        timestamp_s: float = 0.0,
    ) -> StateChange:
        """Forcibly set a state (use only for safety-critical overrides)."""
        previous = self._state
        self._state = target
        change = StateChange(
            previous=previous,
            current=target,
            trigger=Transition.EMERGENCY,
            timestamp_s=timestamp_s,
        )
        self._history.append(change)
        return change

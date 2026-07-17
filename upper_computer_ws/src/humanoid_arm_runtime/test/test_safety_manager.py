"""Tests for SafetyManager: control action and escalation logic."""

import time

from humanoid_arm_runtime.safety_manager import (
    ControlAction,
    SafetyManager,
)
from humanoid_arm_runtime.state_machine import (
    SystemState,
    StateMachine,
    Transition,
)
from humanoid_arm_runtime.system_context import SystemContext
from humanoid_arm_runtime.watchdog import Watchdog, WatchdogConfig


def test_init_returns_safe():
    fsm = StateMachine(SystemState.INIT)
    wd = Watchdog(SystemContext(), WatchdogConfig(), monotonic_clock=time.monotonic)
    sm = SafetyManager(fsm, wd)
    result = sm.evaluate()
    assert result.action == ControlAction.SAFE


def test_ready_returns_hold():
    fsm = StateMachine(SystemState.INIT)
    fsm.transition(Transition.SYSTEMS_READY, 0.0)
    wd = Watchdog(SystemContext(), WatchdogConfig(), monotonic_clock=time.monotonic)
    sm = SafetyManager(fsm, wd)
    result = sm.evaluate()
    assert result.action == ControlAction.HOLD


def test_fault_returns_safe():
    fsm = StateMachine(SystemState.FAULT)
    wd = Watchdog(SystemContext(), WatchdogConfig(), monotonic_clock=time.monotonic)
    sm = SafetyManager(fsm, wd)
    result = sm.evaluate()
    assert result.action == ControlAction.SAFE

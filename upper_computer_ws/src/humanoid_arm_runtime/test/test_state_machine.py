"""Tests for the SystemState machine."""

from humanoid_arm_runtime.state_machine import (
    StateMachineError,
    SystemState,
    StateMachine,
    Transition,
)


def test_initial_state():
    fsm = StateMachine()
    assert fsm.state == SystemState.INIT


def test_init_to_ready():
    fsm = StateMachine()
    change = fsm.transition(Transition.SYSTEMS_READY, 1.0)
    assert fsm.state == SystemState.READY
    assert change.previous == SystemState.INIT
    assert change.current == SystemState.READY


def test_ready_to_follow():
    fsm = StateMachine()
    fsm.transition(Transition.SYSTEMS_READY, 0.0)
    change = fsm.transition(Transition.START, 1.0)
    assert fsm.state == SystemState.FOLLOW


def test_follow_to_hold():
    fsm = StateMachine()
    fsm.transition(Transition.SYSTEMS_READY, 0.0)
    fsm.transition(Transition.START, 1.0)
    change = fsm.transition(Transition.HOLD, 2.0)
    assert fsm.state == SystemState.HOLD


def test_follow_vision_lost():
    fsm = StateMachine()
    fsm.transition(Transition.SYSTEMS_READY, 0.0)
    fsm.transition(Transition.START, 1.0)
    fsm.transition(Transition.VISION_LOST, 2.0)
    assert fsm.state == SystemState.SAFE


def test_safe_recovered():
    fsm = StateMachine()
    fsm.transition(Transition.SYSTEMS_READY, 0.0)
    fsm.transition(Transition.START, 1.0)
    fsm.transition(Transition.VISION_LOST, 2.0)
    fsm.transition(Transition.VISION_RECOVERED, 3.0)
    assert fsm.state == SystemState.READY


def test_emergency_from_any():
    fsm = StateMachine()
    fsm.transition(Transition.SYSTEMS_READY, 0.0)
    fsm.transition(Transition.EMERGENCY, 1.0)
    assert fsm.state == SystemState.FAULT


def test_reset_from_fault():
    fsm = StateMachine()
    fsm.transition(Transition.EMERGENCY, 0.0)
    assert fsm.state == SystemState.FAULT
    fsm.transition(Transition.RESET, 1.0)
    assert fsm.state == SystemState.INIT


def test_invalid_transition():
    fsm = StateMachine()
    try:
        fsm.transition(Transition.START, 0.0)
    except StateMachineError:
        return
    assert False, "Should have raised StateMachineError"


def test_history():
    fsm = StateMachine()
    fsm.transition(Transition.SYSTEMS_READY, 0.0)
    fsm.transition(Transition.START, 1.0)
    assert len(fsm.history) == 2
    assert fsm.history[0].trigger == Transition.SYSTEMS_READY
    assert fsm.history[1].trigger == Transition.START


def test_force():
    fsm = StateMachine()
    fsm.transition(Transition.SYSTEMS_READY, 0.0)
    fsm.force(SystemState.FAULT, "test", 1.0)
    assert fsm.state == SystemState.FAULT

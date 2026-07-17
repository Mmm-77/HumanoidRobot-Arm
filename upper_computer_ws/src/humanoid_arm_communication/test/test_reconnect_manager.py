"""Tests for reconnect_manager: backoff and link state."""

import time

from humanoid_arm_communication.reconnect_manager import (
    LinkState,
    ReconnectConfig,
    ReconnectManager,
)


def _fixed_clock(start: float = 1000.0):
    """Return a callable that returns a controllable fake time."""
    state = {"t": start}

    def tick(delta: float = 0.0) -> float:
        state["t"] += delta
        return state["t"]

    def clock() -> float:
        return state["t"]

    return clock, tick


def test_initial_state():
    clock, _ = _fixed_clock()
    mgr = ReconnectManager(ReconnectConfig(), monotonic_clock=clock)
    assert mgr.link_state == LinkState.DISCONNECTED
    assert mgr.time_since_last_feedback is None


def test_feedback_marks_connected():
    clock, tick = _fixed_clock()
    mgr = ReconnectManager(ReconnectConfig(feedback_timeout_s=0.5), monotonic_clock=clock)

    mgr.on_feedback_received()
    assert mgr.link_state == LinkState.CONNECTED

    tick(0.1)
    assert mgr.link_state == LinkState.CONNECTED

    tick(0.3)  # total 0.4 – still within degraded
    assert mgr.link_state == LinkState.DEGRADED


def test_timeout_disconnects():
    clock, tick = _fixed_clock()
    mgr = ReconnectManager(ReconnectConfig(feedback_timeout_s=0.5), monotonic_clock=clock)
    mgr.on_feedback_received()

    tick(0.6)
    assert mgr.link_state == LinkState.DISCONNECTED


def test_reconnect_backoff():
    clock, tick = _fixed_clock()
    mgr = ReconnectManager(
        ReconnectConfig(initial_backoff_s=1.0, backoff_multiplier=2.0, max_backoff_s=5.0),
        monotonic_clock=clock,
    )
    # Mark as disconnected
    mgr.on_feedback_received()
    tick(10.0)  # well past timeout

    assert mgr.should_reconnect  # initially, next_reconnect_time is 0

    mgr.on_reconnect_failed()
    tick(0.5)
    assert not mgr.should_reconnect  # 0.5 < 1.0 backoff
    tick(0.6)  # total 1.1
    assert mgr.should_reconnect


def test_on_feedback_resets_failures():
    mgr = ReconnectManager(ReconnectConfig(), monotonic_clock=lambda: 0.0)
    for _ in range(3):
        mgr.on_crc_failure()
    assert mgr.consecutive_failures == 3
    mgr.on_feedback_received()
    assert mgr.consecutive_failures == 0

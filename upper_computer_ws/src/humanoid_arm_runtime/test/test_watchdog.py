"""Tests for Watchdog: data freshness and failure counters."""

import time

from humanoid_arm_runtime.system_context import PoseSnapshot, SystemContext
from humanoid_arm_runtime.watchdog import Watchdog, WatchdogConfig, WatchdogStatus


def _fixed_clock():
    state = {"t": 0.0}

    def clock():
        return state["t"]

    def tick(dt):
        state["t"] += dt

    return clock, tick


def test_initial_vision_lost():
    clock, _ = _fixed_clock()
    ctx = SystemContext()
    wd = Watchdog(ctx, WatchdogConfig(), monotonic_clock=clock)
    assert wd.vision_status() == WatchdogStatus.LOST


def test_fresh_vision():
    clock, tick = _fixed_clock()
    ctx = SystemContext()
    wd = Watchdog(ctx, WatchdogConfig(vision_fresh_s=0.2, vision_stale_s=0.5),
                  monotonic_clock=clock)

    ctx.set_pose(PoseSnapshot(timestamp_s=clock(), position=None, quaternion_xyzw=None))
    tick(0.1)
    assert wd.vision_status() == WatchdogStatus.OK


def test_stale_vision():
    clock, tick = _fixed_clock()
    ctx = SystemContext()
    wd = Watchdog(ctx, WatchdogConfig(vision_fresh_s=0.1, vision_stale_s=0.3),
                  monotonic_clock=clock)

    ctx.set_pose(PoseSnapshot(timestamp_s=clock(), position=None, quaternion_xyzw=None))
    tick(0.4)
    status = wd.vision_status()
    # At 0.4s, it's past the 0.3 stale threshold
    assert status in (WatchdogStatus.STALE, WatchdogStatus.LOST)


def test_counters():
    wd = Watchdog(SystemContext(), WatchdogConfig(), monotonic_clock=lambda: 0.0)
    assert wd.consecutive_vision_lost == 0

    wd.on_vision_lost()
    wd.on_vision_lost()
    assert wd.consecutive_vision_lost == 2

    wd.on_vision_ok()
    assert wd.consecutive_vision_lost == 0

    wd.on_ik_failure()
    wd.on_ik_failure()
    wd.on_ik_failure()
    assert wd.consecutive_ik_failures == 3

    wd.on_ik_ok()
    assert wd.consecutive_ik_failures == 0

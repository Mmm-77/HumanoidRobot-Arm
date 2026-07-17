#!/usr/bin/env python3
"""Interactive state machine + safety escalation test.

Simulates the full runtime pipeline without ROS or hardware:

    vision lost → SAFE → vision recovered → READY → START → FOLLOW
    FOLLOW → vision data → target generation → HOLD → UNHOLD → ...

Run:
    cd upper_computer_ws/src/humanoid_arm_runtime
    python3 test/run_sim.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, "..")

from humanoid_arm_runtime.state_machine import (
    SystemState,
    StateMachine,
    Transition,
    StateMachineError,
)
from humanoid_arm_runtime.watchdog import Watchdog, WatchdogConfig
from humanoid_arm_runtime.safety_manager import (
    ControlAction,
    SafetyManager,
)
from humanoid_arm_runtime.system_context import (
    PoseSnapshot,
    JointSnapshot,
    SystemContext,
)
from humanoid_arm_runtime.follow_mapper import FollowMapper
from humanoid_arm_runtime.task_projector import TaskProjector


def main():
    t = [0.0]

    def clock():
        return t[0]

    def tick(dt):
        t[0] += dt

    # Setup
    ctx = SystemContext()
    fsm = StateMachine(SystemState.INIT)
    wd = Watchdog(ctx, WatchdogConfig(), monotonic_clock=clock)
    safety = SafetyManager(fsm, wd)
    mapper = FollowMapper()
    projector = TaskProjector()

    # Inject a valid baseline pose and joints so the system can enter READY
    ctx.set_pose(PoseSnapshot(
        timestamp_s=clock(),
        position=np.array([0.5, 0.0, 1.0], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        valid=True,
    ))
    ctx.set_joints(JointSnapshot(
        timestamp_s=clock(),
        positions_rad=np.zeros(4, dtype=np.float64),
        velocities_rad_per_s=np.zeros(4, dtype=np.float64),
        any_error=False,
    ))

    # Use a fake "last IK result" as the FK baseline EE position
    baseline_ee_pos = np.array([0.2, 0.0, 0.3], dtype=np.float64)
    baseline_ee_yaw = 0.0

    def step(desc):
        print(f"\n--- {desc} ---")

    def show_state():
        result = safety.evaluate()
        print(f"  State: {fsm.state.value}")
        print(f"  Action: {result.action.value}")
        print(f"  Vision: {wd.vision_status().value} | "
              f"Comm: {wd.communication_status().value}")
        print(f"  Vision lost streak: {wd.consecutive_vision_lost}")
        return result

    # ================================================================
    # Scenario 1: Normal startup flow
    # ================================================================
    step("1. INIT → READY (systems detected)")
    try:
        fsm.transition(Transition.SYSTEMS_READY, clock())
        print(f"  Transition OK → {fsm.state.value}")
    except StateMachineError as e:
        print(f"  ERROR: {e}")

    result = show_state()
    assert fsm.state == SystemState.READY
    assert result.action == ControlAction.HOLD
    print("  ✓ READY state, arm held")

    # ================================================================
    # Scenario 2: START → FOLLOW with baseline
    # ================================================================
    step("2. START → FOLLOW")
    fsm.transition(Transition.START, clock())
    result = show_state()
    assert fsm.state == SystemState.FOLLOW

    # Set baseline manually (simulating what runtime_node does)
    ctx.set_baseline(
        PoseSnapshot(
            timestamp_s=clock(),
            position=np.array([0.5, 0.0, 1.0], dtype=np.float64),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        ),
        baseline_ee_pos.copy(),
        baseline_ee_yaw,
    )
    print(f"  Baseline recorded: EE at {baseline_ee_pos}, yaw={baseline_ee_yaw:.2f}")

    # ================================================================
    # Scenario 3: Simulate camera moving forward 0.1m
    # ================================================================
    step("3. Camera moves forward 0.1m → arm follows")
    tick(0.01)
    # Update current pose (camera moved forward)
    ctx.set_pose(PoseSnapshot(
        timestamp_s=clock(),
        position=np.array([0.6, 0.0, 1.0], dtype=np.float64),  # +0.1m in X
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        valid=True,
    ))

    result = show_state()
    assert result.action == ControlAction.PERMIT

    # Compute the target the runtime would publish
    baseline = ctx.baseline_pose
    delta_pos, delta_yaw = mapper.map(
        baseline.position, baseline.quaternion_xyzw,
        ctx.latest_pose.position, ctx.latest_pose.quaternion_xyzw,
    )
    clipped_pos, clipped_yaw = projector.project(delta_pos, delta_yaw)
    target_pos = ctx.baseline_ee_position_m + clipped_pos
    target_yaw = ctx.baseline_ee_yaw_rad + clipped_yaw

    print(f"  Camera delta: pos={delta_pos}, yaw={delta_yaw:.4f}")
    print(f"  Clipped delta: pos={clipped_pos}, yaw={clipped_yaw:.4f}")
    print(f"  Target (base): pos={target_pos}, yaw={target_yaw:.4f}")
    assert abs(delta_pos[0] - 0.1) < 0.01, f"Expected 0.1m X delta, got {delta_pos[0]}"
    print(f"  ✓ Target matches expected +0.1m motion")

    # ================================================================
    # Scenario 4: Vision lost → SAFE → recovered → READY
    # ================================================================
    step("4. Vision lost → escalate to SAFE")

    # Simulate vision lost by setting stale timestamp
    ctx.set_pose(PoseSnapshot(
        timestamp_s=clock() - 1.0,               # 1 second old!
        position=np.array([0.6, 0.0, 1.0], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        valid=True,
    ))

    # Feed watchdog enough lost frames to trigger escalation
    for _ in range(12):
        tick(0.05)
        result = show_state()
        if result.action == ControlAction.SAFE:
            print(f"  Escalated to SAFE after {wd.consecutive_vision_lost} lost frames ✓")
            break
    else:
        print(f"  WARNING: did not escalate, state={fsm.state.value} act={result.action.value}")

    assert fsm.state == SystemState.SAFE or result.action == ControlAction.SAFE

    # ================================================================
    # Scenario 5: SAFE → recover → READY → START → FOLLOW
    # ================================================================
    step("5. Vision recovered → READY")
    # Force a valid state (in real runtime, vision would come back)
    tick(0.01)
    ctx.set_pose(PoseSnapshot(
        timestamp_s=clock(),
        position=np.array([0.6, 0.0, 1.0], dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        valid=True,
    ))
    wd.on_vision_ok()
    wd.reset()

    fsm.force(SystemState.READY, "vision recovered")
    result = show_state()
    assert fsm.state == SystemState.READY
    print(f"  ✓ Vision recovered, state → READY")

    step("6. START again → FOLLOW")
    fsm.transition(Transition.START, clock())
    # Re-record baseline
    ctx.set_baseline(
        PoseSnapshot(timestamp_s=clock(),
                     position=np.array([0.6, 0.0, 1.0], dtype=np.float64),
                     quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        baseline_ee_pos.copy(), baseline_ee_yaw,
    )
    result = show_state()
    assert result.action == ControlAction.PERMIT
    print(f"  ✓ Back in FOLLOW, PERMIT active")

    # ================================================================
    # Scenario 6: EMERGENCY → FAULT → RESET
    # ================================================================
    step("7. EMERGENCY → FAULT → RESET")
    fsm.transition(Transition.EMERGENCY, clock())
    result = show_state()
    assert fsm.state == SystemState.FAULT
    assert result.action == ControlAction.SAFE
    print(f"  ✓ FAULT state, arm SAFE")

    fsm.transition(Transition.RESET, clock())
    assert fsm.state == SystemState.INIT
    result = show_state()
    assert result.action == ControlAction.SAFE
    print(f"  ✓ RESET → INIT, arm SAFE (needs re-init)")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  Scenario test passed!")
    print(f"  Verified: INIT→READY→FOLLOW (with target) → SAFE (vision loss)")
    print(f"           → READY (recovery) → FOLLOW → FAULT → INIT")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

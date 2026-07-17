"""Diagnostics: aggregate status from all subsystems into a DiagnosticArray.

Combines vision quality, communication link, kinematics validity, and safety
state into a single diagnostic publication.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node

from .safety_manager import ControlAction
from .state_machine import SystemState
from .watchdog import Watchdog, WatchdogStatus


class DiagnosticsAggregator:
    """Build one DiagnosticArray from all subsystem statuses."""

    def __init__(
        self,
        node: Node,
        *,
        hardware_id: str = "humanoid_arm",
    ) -> None:
        self._node = node
        self._hw_id = hardware_id

    def build(
        self,
        state: SystemState,
        action: ControlAction,
        vision_status: WatchdogStatus,
        comm_status: WatchdogStatus,
        consecutive_vision_lost: int,
        consecutive_ik_failures: int,
    ) -> DiagnosticArray:
        array = DiagnosticArray()
        array.header.stamp = self._node.get_clock().now().to_msg()

        statuses: List[DiagnosticStatus] = []

        # Top-level summary
        summary = DiagnosticStatus()
        summary.name = "humanoid_arm_runtime/summary"
        summary.hardware_id = self._hw_id
        summary.message = f"state={state.value} action={action.value}"

        # Overall level
        if state == SystemState.FAULT:
            summary.level = DiagnosticStatus.ERROR
        elif state == SystemState.SAFE or state == SystemState.INIT:
            summary.level = DiagnosticStatus.WARN
        elif action == ControlAction.PERMIT:
            summary.level = DiagnosticStatus.OK
        else:
            summary.level = DiagnosticStatus.WARN

        summary.values = [
            self._kv("state", state.value),
            self._kv("control_action", action.value),
        ]
        statuses.append(summary)

        # Vision
        vs = DiagnosticStatus()
        vs.name = "humanoid_arm_runtime/vision"
        vs.hardware_id = self._hw_id
        vs.message = vision_status.value
        vs.level = self._level(vision_status)
        vs.values = [
            self._kv("status", vision_status.value),
            self._kv("consecutive_lost", str(consecutive_vision_lost)),
        ]
        statuses.append(vs)

        # Communication
        cs = DiagnosticStatus()
        cs.name = "humanoid_arm_runtime/communication"
        cs.hardware_id = self._hw_id
        cs.message = comm_status.value
        cs.level = self._level(comm_status)
        cs.values = [
            self._kv("status", comm_status.value),
        ]
        statuses.append(cs)

        # Kinematics (IK)
        ks = DiagnosticStatus()
        ks.name = "humanoid_arm_runtime/kinematics"
        ks.hardware_id = self._hw_id
        ks.level = (
            DiagnosticStatus.WARN
            if consecutive_ik_failures > 0
            else DiagnosticStatus.OK
        )
        ks.message = (
            "ik_ok" if consecutive_ik_failures == 0
            else f"ik_failures={consecutive_ik_failures}"
        )
        ks.values = [
            self._kv("consecutive_failures", str(consecutive_ik_failures)),
        ]
        statuses.append(ks)

        array.status = statuses
        return array

    @staticmethod
    def _level(status: WatchdogStatus) -> int:
        if status == WatchdogStatus.OK:
            return int(DiagnosticStatus.OK)
        if status == WatchdogStatus.STALE:
            return int(DiagnosticStatus.WARN)
        return int(DiagnosticStatus.ERROR)

    @staticmethod
    def _kv(key: str, value: str) -> KeyValue:
        item = KeyValue()
        item.key = key
        item.value = value
        return item

"""System orchestration: relative following, safety, and diagnostics for the 4-DOF arm.

Connects vision, kinematics, and communication packages through ROS 2 interfaces.
"""

from .state_machine import SystemState
from .system_context import SystemContext

__all__ = [
    "SystemState",
    "SystemContext",
]

"""Public API for the humanoid_arm_kinematics package.

Re-exports all ROS-independent types so callers can import from a single namespace.
"""

# Robot model
from .robot_model import DHLink, ModelError, RobotModel

# Forward kinematics
from .forward_solver import ForwardResult, ForwardSolver, ForwardSolverError

# Jacobian
from .jacobian import JacobianError, JacobianResult, JacobianSolver

# Inverse kinematics
from .inverse_solver import IKConfig, IKResult, InverseSolver, InverseSolverError

# Solution selection
from .solution_selector import JointLimits, SelectedSolution, SelectorError, SolutionSelector

# Solution validation
from .solution_validator import SolutionValidator, ValidationResult

# Target shaping
from .target_shaper import ShapedTarget, ShaperConfig, TargetShaper

# Workspace guard
from .workspace_guard import GuardDecision, GuardReason, WorkspaceGuard

__all__ = [
    # robot_model
    "DHLink",
    "ModelError",
    "RobotModel",
    # forward_solver
    "ForwardResult",
    "ForwardSolver",
    "ForwardSolverError",
    # jacobian
    "JacobianError",
    "JacobianResult",
    "JacobianSolver",
    # inverse_solver
    "IKConfig",
    "IKResult",
    "InverseSolver",
    "InverseSolverError",
    # solution_selector
    "JointLimits",
    "SelectedSolution",
    "SelectorError",
    "SolutionSelector",
    # solution_validator
    "SolutionValidator",
    "ValidationResult",
    # target_shaper
    "ShapedTarget",
    "ShaperConfig",
    "TargetShaper",
    # workspace_guard
    "GuardDecision",
    "GuardReason",
    "WorkspaceGuard",
]

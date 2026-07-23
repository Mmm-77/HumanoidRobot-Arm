"""Acceptance policy between strict IK convergence and output validation."""

from __future__ import annotations

import math


def is_solution_acceptable(
    solver_success: bool,
    position_error_m: float,
    max_validation_error_m: float,
) -> bool:
    """Accept strict convergence or a finite result inside validation bounds."""
    if not math.isfinite(max_validation_error_m) or max_validation_error_m <= 0.0:
        raise ValueError("max_validation_error_m must be positive and finite")
    return bool(
        math.isfinite(position_error_m)
        and (solver_success or position_error_m <= max_validation_error_m)
    )

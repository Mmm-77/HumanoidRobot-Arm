"""Convert between internal SI units (rad, rad/s) and protocol units (deg, deg/s).

The lower computer uses degrees for all angle/speed fields.
"""

from __future__ import annotations

import math

import numpy as np


def rad_to_deg(radians: np.ndarray | float) -> np.ndarray | float:
    """Convert radians to degrees."""
    return np.rad2deg(radians)


def deg_to_rad(degrees: np.ndarray | float) -> np.ndarray | float:
    """Convert degrees to radians."""
    return np.deg2rad(degrees)


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle in degrees to [-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

"""Small, dependency-light helpers for rigid transforms and quaternions.

Quaternions use ROS ordering ``[x, y, z, w]`` throughout this package.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def normalize_quaternion(quaternion: NDArray[np.floating]) -> FloatArray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(q)):
        raise ValueError("quaternion contains a non-finite value")
    norm = float(np.linalg.norm(q))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion norm must be non-zero")
    return q / norm


def quaternion_to_rotation_matrix(quaternion: NDArray[np.floating]) -> FloatArray:
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation: NDArray[np.floating]) -> FloatArray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix contains a non-finite value")

    # Project tiny numerical errors back onto SO(3), while still rejecting a
    # matrix that is clearly not a rotation.
    orthogonality_error = float(np.linalg.norm(matrix.T @ matrix - np.eye(3)))
    determinant = float(np.linalg.det(matrix))
    if orthogonality_error > 1e-4 or abs(determinant - 1.0) > 1e-4:
        raise ValueError("matrix is not a proper rotation")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif diagonal_index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return normalize_quaternion(quaternion)


def make_transform(
    rotation: NDArray[np.floating], translation: NDArray[np.floating]
) -> FloatArray:
    rotation_matrix = np.asarray(rotation, dtype=np.float64)
    translation_vector = np.asarray(translation, dtype=np.float64).reshape(3)
    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if not np.all(np.isfinite(rotation_matrix)) or not np.all(
        np.isfinite(translation_vector)
    ):
        raise ValueError("transform contains a non-finite value")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_vector
    return transform


def invert_transform(transform: NDArray[np.floating]) -> FloatArray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def slerp(
    start: NDArray[np.floating], end: NDArray[np.floating], fraction: float
) -> FloatArray:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    q0 = normalize_quaternion(start)
    q1 = normalize_quaternion(end)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion(q0 + fraction * (q1 - q0))
    angle = math.acos(dot)
    sine = math.sin(angle)
    return normalize_quaternion(
        math.sin((1.0 - fraction) * angle) / sine * q0
        + math.sin(fraction * angle) / sine * q1
    )


def quaternion_angular_distance(
    first: NDArray[np.floating], second: NDArray[np.floating]
) -> float:
    q0 = normalize_quaternion(first)
    q1 = normalize_quaternion(second)
    dot = abs(float(np.dot(q0, q1)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))

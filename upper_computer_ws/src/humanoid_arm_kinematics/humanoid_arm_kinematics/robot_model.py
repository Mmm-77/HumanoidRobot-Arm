"""Modified DH parameter model for the 4-DOF humanoid robot arm.

The model stores DH parameters, computes per-link homogeneous transforms,
and defines the controllable orientation angle theta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np


class ModelError(ValueError):
    """Raised when the robot model parameters are inconsistent."""


@dataclass(frozen=True)
class DHLink:
    """One link in the Modified DH parameterisation.

    Attributes:
        alpha_prev_rad: Link twist angle α_{i-1} in radians.
        a_prev_m: Link length a_{i-1} in metres.
        d_m: Link offset d_i in metres.
        theta_rad: Joint angle θ_i in radians (the joint variable).
    """

    alpha_prev_rad: float
    a_prev_m: float
    d_m: float
    theta_rad: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.a_prev_m) or self.a_prev_m < 0.0:
            raise ModelError(f"a_prev_m must be finite and >= 0, got {self.a_prev_m}")
        if not np.isfinite(self.d_m):
            raise ModelError(f"d_m must be finite, got {self.d_m}")
        if not np.isfinite(self.alpha_prev_rad):
            raise ModelError(f"alpha_prev_rad must be finite, got {self.alpha_prev_rad}")
        if not np.isfinite(self.theta_rad):
            raise ModelError(f"theta_rad must be finite, got {self.theta_rad}")

    @classmethod
    def from_degrees(
        cls, alpha_prev_deg: float, a_prev_m: float, d_m: float, theta_deg: float
    ) -> "DHLink":
        """Create a DHLink with angles specified in degrees."""
        return cls(
            alpha_prev_rad=np.deg2rad(alpha_prev_deg),
            a_prev_m=a_prev_m,
            d_m=d_m,
            theta_rad=np.deg2rad(theta_deg),
        )

    def with_theta(self, theta_rad: float) -> "DHLink":
        """Return a new DHLink with updated joint angle."""
        return DHLink(
            alpha_prev_rad=self.alpha_prev_rad,
            a_prev_m=self.a_prev_m,
            d_m=self.d_m,
            theta_rad=theta_rad,
        )

    def transform_matrix(self) -> np.ndarray:
        """Compute the 4x4 homogeneous transform T_i^{i-1} for this link.

        Modified DH convention:
            T = Rot_X(α_{i-1}) · Trans_X(a_{i-1}) · Rot_Z(θ_i) · Trans_Z(d_i)
        """
        alpha = self.alpha_prev_rad
        theta = self.theta_rad
        a = self.a_prev_m
        d = self.d_m

        ca = np.cos(alpha)
        sa = np.sin(alpha)
        ct = np.cos(theta)
        st = np.sin(theta)

        return np.array(
            [
                [ct, -st, 0, a],
                [st * ca, ct * ca, -sa, -d * sa],
                [st * sa, ct * sa, ca, d * ca],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RobotModel:
    """Kinematic model of the 4-DOF humanoid arm.

    Attributes:
        links: Four DHLink instances defining the arm geometry.
        tool_offset: 4x4 homogeneous transform from the last link frame to the
                     tool (end-effector) frame.
    """

    links: List[DHLink] = field(default_factory=list)
    tool_offset: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    def __post_init__(self) -> None:
        if len(self.links) == 0:
            raise ModelError("RobotModel must have at least one link")
        if self.tool_offset.shape != (4, 4):
            raise ModelError(
                f"tool_offset must be a 4x4 matrix, got {self.tool_offset.shape}"
            )

    @property
    def num_joints(self) -> int:
        """Number of joints (links)."""
        return len(self.links)

    @classmethod
    def from_config(
        cls,
        dh_params: Sequence[dict],
        tool_translation: Sequence[float] = (0.0, 0.0, 0.0),
        tool_rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "RobotModel":
        """Build a RobotModel from configuration dictionaries.

        Args:
            dh_params: List of dicts with keys 'alpha_prev_deg', 'a_prev_m', 'd_m'.
                       Joint angles are initialised to zero.
            tool_translation: [x, y, z] tool offset in metres.
            tool_rotation_deg: [roll, pitch, yaw] tool rotation in degrees.
        """
        links: List[DHLink] = []
        for entry in dh_params:
            links.append(
                DHLink.from_degrees(
                    alpha_prev_deg=float(entry["alpha_prev_deg"]),
                    a_prev_m=float(entry["a_prev_m"]),
                    d_m=float(entry["d_m"]),
                    theta_deg=0.0,
                )
            )

        # Build tool offset as a homogeneous transform
        roll, pitch, yaw = np.deg2rad([float(tool_rotation_deg[0]),
                                        float(tool_rotation_deg[1]),
                                        float(tool_rotation_deg[2])])
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        rot = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float64)

        tool = np.eye(4, dtype=np.float64)
        tool[:3, :3] = rot
        tool[:3, 3] = [float(tool_translation[0]),
                        float(tool_translation[1]),
                        float(tool_translation[2])]

        return cls(links=links, tool_offset=tool)

    def with_joint_angles(self, angles_rad: Sequence[float]) -> "RobotModel":
        """Return a copy of the model with updated joint angles."""
        if len(angles_rad) != self.num_joints:
            raise ModelError(
                f"Expected {self.num_joints} joint angles, got {len(angles_rad)}"
            )
        new_links = [
            link.with_theta(float(theta))
            for link, theta in zip(self.links, angles_rad)
        ]
        return RobotModel(links=new_links, tool_offset=self.tool_offset.copy())

    def get_link_transform(self, index: int) -> np.ndarray:
        """Get the DH transform for a single link (T_i^{i-1})."""
        return self.links[index].transform_matrix()

    def forward_kinematics(self) -> np.ndarray:
        """Compute the end-effector pose in the base frame.

        Returns a 4x4 homogeneous transform T_tool^base.
        """
        T = np.eye(4, dtype=np.float64)
        for link in self.links:
            T = T @ link.transform_matrix()
        return T @ self.tool_offset

    def get_frame_transforms(self) -> List[np.ndarray]:
        """Compute cumulative transforms from base to each link frame.

        Returns a list of 4x4 matrices T_i^base for i = 0..num_joints,
        where index 0 is identity (base frame) and index num_joints is the
        tool frame.
        """
        frames: List[np.ndarray] = [np.eye(4, dtype=np.float64)]
        T = np.eye(4, dtype=np.float64)
        for link in self.links:
            T = T @ link.transform_matrix()
            frames.append(T.copy())
        # Append tool frame
        frames.append(T @ self.tool_offset)
        return frames

    def extract_position(self, transform: np.ndarray) -> np.ndarray:
        """Extract the 3-element position vector from a 4x4 transform."""
        return transform[:3, 3].copy()

    def extract_rotation(self, transform: np.ndarray) -> np.ndarray:
        """Extract the 3x3 rotation matrix from a 4x4 transform."""
        return transform[:3, :3].copy()

    @staticmethod
    def extract_yaw(rotation: np.ndarray) -> float:
        """Extract Z-axis rotation angle (yaw) from a 3x3 rotation matrix.

        Computed as atan2(R[1,0], R[0,0]).
        Returns angle in radians in [-π, π].
        """
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    @staticmethod
    def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to a quaternion [x, y, z, w]."""
        R = rotation
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        q = np.array([x, y, z, w], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / norm

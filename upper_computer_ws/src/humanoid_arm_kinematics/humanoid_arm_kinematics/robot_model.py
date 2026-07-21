"""URDF-backed serial-chain model for the humanoid arm.

The URDF is the single source of truth.  Joint origins, axes, ordering, and the
fixed transform to ``tip_frame`` are parsed directly from the selected chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Union
import xml.etree.ElementTree as ET

import numpy as np


class ModelError(ValueError):
    """Raised when the requested URDF chain is invalid or unsupported."""


def _parse_vector(text: str | None, default: Sequence[float]) -> np.ndarray:
    values = np.asarray(
        default if text is None else [float(item) for item in text.split()],
        dtype=np.float64,
    )
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ModelError(f"Expected a finite 3-vector, got {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_matrix(origin: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    transform[:3, :3] = _rpy_matrix(
        _parse_vector(origin.get("rpy"), (0.0, 0.0, 0.0))
    )
    transform[:3, 3] = _parse_vector(
        origin.get("xyz"), (0.0, 0.0, 0.0)
    )
    return transform


def _axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a homogeneous Rodrigues rotation around a local-frame axis."""
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    rotation = (
        np.eye(3, dtype=np.float64)
        + np.sin(angle_rad) * skew
        + (1.0 - np.cos(angle_rad)) * (skew @ skew)
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


@dataclass(frozen=True)
class URDFJoint:
    """One fixed or revolute joint in the base-to-tip serial chain."""

    name: str
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray

    @property
    def movable(self) -> bool:
        return self.joint_type in ("revolute", "continuous")


@dataclass(frozen=True)
class ChainState:
    """Transforms and screw axes evaluated at one joint configuration."""

    tip_transform: np.ndarray
    joint_positions: List[np.ndarray]
    joint_axes: List[np.ndarray]


@dataclass(frozen=True)
class RobotModel:
    """A base-to-tip chain parsed directly from a URDF document."""

    chain: List[URDFJoint]
    base_link: str
    tip_link: str

    def __post_init__(self) -> None:
        if not self.chain:
            raise ModelError("URDF chain must contain at least one joint")
        if self.num_joints == 0:
            raise ModelError("URDF chain must contain at least one movable joint")

    @property
    def joint_names(self) -> List[str]:
        return [joint.name for joint in self.chain if joint.movable]

    @property
    def num_joints(self) -> int:
        return sum(joint.movable for joint in self.chain)

    @classmethod
    def from_urdf_file(
        cls,
        urdf_path: Union[str, Path],
        base_link: str = "base_link",
        tip_link: str = "tip_frame",
    ) -> "RobotModel":
        path = Path(urdf_path)
        if not path.is_file():
            raise ModelError(f"URDF file does not exist: {path}")
        return cls.from_urdf_string(
            path.read_text(encoding="utf-8"), base_link, tip_link
        )

    @classmethod
    def from_urdf_string(
        cls,
        urdf_xml: str,
        base_link: str = "base_link",
        tip_link: str = "tip_frame",
    ) -> "RobotModel":
        try:
            root = ET.fromstring(urdf_xml)
        except ET.ParseError as exc:
            raise ModelError(f"Invalid URDF XML: {exc}") from exc

        links = {element.get("name") for element in root.findall("link")}
        if base_link not in links:
            raise ModelError(f"Base link {base_link!r} is not present in the URDF")
        if tip_link not in links:
            raise ModelError(f"Tip link {tip_link!r} is not present in the URDF")

        by_child: Dict[str, ET.Element] = {}
        for element in root.findall("joint"):
            child_element = element.find("child")
            if child_element is None or child_element.get("link") is None:
                raise ModelError("Every URDF joint must declare a child link")
            child = str(child_element.get("link"))
            if child in by_child:
                raise ModelError(f"Link {child!r} has more than one parent joint")
            by_child[child] = element

        elements: List[ET.Element] = []
        current = tip_link
        visited: Set[str] = set()
        while current != base_link:
            if current in visited:
                raise ModelError("Cycle detected while tracing the URDF chain")
            visited.add(current)
            element = by_child.get(current)
            if element is None:
                raise ModelError(
                    f"No chain connects {base_link!r} to {tip_link!r}"
                )
            elements.append(element)
            parent_element = element.find("parent")
            if parent_element is None or parent_element.get("link") is None:
                raise ModelError("Every URDF joint must declare a parent link")
            current = str(parent_element.get("link"))

        chain: List[URDFJoint] = []
        for element in reversed(elements):
            joint_type = str(element.get("type"))
            if joint_type not in ("fixed", "revolute", "continuous"):
                raise ModelError(
                    f"Joint {element.get('name')!r} has unsupported type "
                    f"{joint_type!r}"
                )
            parent = str(element.find("parent").get("link"))  # type: ignore[union-attr]
            child = str(element.find("child").get("link"))  # type: ignore[union-attr]
            axis = _parse_vector(
                element.find("axis").get("xyz")
                if element.find("axis") is not None
                else None,
                (1.0, 0.0, 0.0),
            )
            if joint_type in ("revolute", "continuous"):
                norm = float(np.linalg.norm(axis))
                if norm < 1.0e-12:
                    raise ModelError(f"Joint {element.get('name')!r} has a zero axis")
                axis = axis / norm
            chain.append(
                URDFJoint(
                    name=str(element.get("name")),
                    parent=parent,
                    child=child,
                    joint_type=joint_type,
                    origin=_origin_matrix(element.find("origin")),
                    axis=axis,
                )
            )
        return cls(chain=chain, base_link=base_link, tip_link=tip_link)

    def evaluate(self, angles_rad: Sequence[float]) -> ChainState:
        angles = np.asarray(angles_rad, dtype=np.float64)
        if angles.shape != (self.num_joints,) or not np.all(np.isfinite(angles)):
            raise ModelError(
                f"Expected {self.num_joints} finite joint angles, got {angles_rad}"
            )

        transform = np.eye(4, dtype=np.float64)
        positions: List[np.ndarray] = []
        axes: List[np.ndarray] = []
        angle_index = 0
        for joint in self.chain:
            transform = transform @ joint.origin
            if joint.movable:
                positions.append(transform[:3, 3].copy())
                axes.append(transform[:3, :3] @ joint.axis)
                transform = transform @ _axis_rotation(
                    joint.axis, float(angles[angle_index])
                )
                angle_index += 1
        return ChainState(transform, positions, axes)

    def forward_kinematics(self, angles_rad: Sequence[float]) -> np.ndarray:
        return self.evaluate(angles_rad).tip_transform

    @staticmethod
    def extract_position(transform: np.ndarray) -> np.ndarray:
        return transform[:3, 3].copy()

    @staticmethod
    def extract_rotation(transform: np.ndarray) -> np.ndarray:
        return transform[:3, :3].copy()

    @staticmethod
    def extract_yaw(rotation: np.ndarray) -> float:
        """Provide base-frame yaw for telemetry; IK does not control it."""
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    @staticmethod
    def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix to a normalized quaternion [x, y, z, w]."""
        matrix = np.asarray(rotation, dtype=np.float64)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    0.25 * scale,
                ]
            )
        else:
            index = int(np.argmax(np.diag(matrix)))
            first, second = (index + 1) % 3, (index + 2) % 3
            scale = 2.0 * np.sqrt(
                1.0 + matrix[index, index]
                - matrix[first, first] - matrix[second, second]
            )
            quaternion = np.zeros(4, dtype=np.float64)
            quaternion[index] = 0.25 * scale
            quaternion[3] = (matrix[second, first] - matrix[first, second]) / scale
            quaternion[first] = (matrix[first, index] + matrix[index, first]) / scale
            quaternion[second] = (matrix[second, index] + matrix[index, second]) / scale
        return quaternion / np.linalg.norm(quaternion)

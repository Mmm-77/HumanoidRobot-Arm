"""Regression test for the arm's absolute joint geometry at zero position."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET


URDF_PATH = Path(__file__).parents[1] / "urdf" / "humanoid_arm.urdf"
EXPECTED = {
    "joint_1": ((0.0, 0.0, 0.0), (0.0, -0.8660254, 0.5)),
    "joint_2": ((0.0, -0.05, 0.0), (1.0, 0.0, 0.0)),
    "joint_3": ((0.0, -0.05, -0.07), (0.0, 0.0, 1.0)),
    "joint_4": ((0.0, -0.05, -0.105), (0.0, 1.0, 0.0)),
}

J1_AXIS = EXPECTED["joint_1"][1]


def _matrix_multiply(left, right):
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def _rotate(rotation, vector):
    return tuple(sum(rotation[row][k] * vector[k] for k in range(3)) for row in range(3))


def _rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _normalise(vector):
    length = math.sqrt(sum(value * value for value in vector))
    return tuple(value / length for value in vector)


def test_absolute_joint_geometry_at_zero():
    root = ET.parse(URDF_PATH).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    parent_position = (0.0, 0.0, 0.0)
    parent_rotation = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    for name, (expected_point, expected_axis) in EXPECTED.items():
        joint = joints[name]
        assert joint.attrib["type"] == "revolute"

        origin = joint.find("origin")
        relative_position = tuple(float(value) for value in origin.attrib["xyz"].split())
        rpy = tuple(float(value) for value in origin.attrib["rpy"].split())
        local_axis = tuple(float(value) for value in joint.find("axis").attrib["xyz"].split())

        offset_in_base = _rotate(parent_rotation, relative_position)
        joint_position = tuple(parent_position[i] + offset_in_base[i] for i in range(3))
        joint_rotation = _matrix_multiply(parent_rotation, _rpy_matrix(*rpy))
        joint_axis = _normalise(_rotate(joint_rotation, local_axis))

        assert all(abs(joint_position[i] - expected_point[i]) < 1.0e-12 for i in range(3))
        expected_axis = _normalise(expected_axis)
        assert all(abs(joint_axis[i] - expected_axis[i]) < 1.0e-9 for i in range(3))

        parent_position = joint_position
        parent_rotation = joint_rotation


def test_base_top_face_touches_j1_and_is_coaxial():
    root = ET.parse(URDF_PATH).getroot()
    base = next(link for link in root.findall("link") if link.attrib["name"] == "base_link")
    visual = base.find("visual")
    origin = visual.find("origin")
    cylinder = visual.find("geometry/cylinder")

    centre = tuple(float(value) for value in origin.attrib["xyz"].split())
    rotation = _rpy_matrix(*tuple(float(value) for value in origin.attrib["rpy"].split()))
    cylinder_axis = _normalise(_rotate(rotation, (0.0, 0.0, 1.0)))
    half_length = float(cylinder.attrib["length"]) / 2.0
    top_centre = tuple(centre[i] + half_length * cylinder_axis[i] for i in range(3))

    expected_axis = _normalise(J1_AXIS)
    assert all(abs(cylinder_axis[i] - expected_axis[i]) < 1.0e-8 for i in range(3))
    assert all(abs(value) < 1.0e-8 for value in top_centre)

    link_1 = next(link for link in root.findall("link") if link.attrib["name"] == "link_1")
    joint_visual = link_1.findall("visual")[1]
    joint_origin = joint_visual.find("origin")
    joint_centre = tuple(float(value) for value in joint_origin.attrib["xyz"].split())
    joint_length = float(joint_visual.find("geometry/cylinder").attrib["length"])
    joint_bottom = tuple(
        joint_centre[i] - (joint_length / 2.0) * expected_axis[i] for i in range(3)
    )
    assert all(abs(value) < 1.0e-8 for value in joint_bottom)


if __name__ == "__main__":
    test_absolute_joint_geometry_at_zero()
    test_base_top_face_touches_j1_and_is_coaxial()
    print("zero-pose absolute joint geometry: OK")

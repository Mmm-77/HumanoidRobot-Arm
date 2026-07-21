from glob import glob

from setuptools import find_packages, setup

PACKAGE_NAME = "humanoid_arm_kinematics"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="HumanoidRobot Team",
    maintainer_email="dev@humanoid-arm.local",
    description="4-DOF humanoid arm kinematics and joint target generation.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "kinematics_node = humanoid_arm_kinematics.kinematics_node:main",
        ],
    },
)

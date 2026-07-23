from glob import glob

from setuptools import find_packages, setup

PACKAGE_NAME = "humanoid_arm_vision"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (
            f"share/{PACKAGE_NAME}",
            ["package.xml", "README.md", "TESTING.md", "CALIBRATION.md"],
        ),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    # ROS 2 Foxy uses Python 3.8; newer pyrealsense2 wheels no longer support it.
    install_requires=["setuptools", "pyrealsense2<2.56"],
    zip_safe=True,
    maintainer="HumanoidRobot Team",
    maintainer_email="dev@humanoid-arm.local",
    description="Camera capture and AprilTag-based camera pose estimation.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vision_node = humanoid_arm_vision.vision_node:main",
        ],
    },
)

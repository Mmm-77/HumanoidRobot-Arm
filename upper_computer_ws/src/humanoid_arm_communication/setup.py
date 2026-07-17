from setuptools import find_packages, setup

package_name = "humanoid_arm_communication"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch",
         ["launch/communication.launch.py",
          "launch/hardware_check.launch.py"]),
        ("share/" + package_name + "/config",
         ["config/communication.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="USB serial communication with STM32 lower computer.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "communication_node = humanoid_arm_communication.communication_node:main",
        ],
    },
)

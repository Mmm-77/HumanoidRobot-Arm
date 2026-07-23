from glob import glob

from setuptools import find_packages, setup

package_name = "humanoid_arm_runtime"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="System orchestration: relative following, safety, diagnostics.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "runtime_node = humanoid_arm_runtime.runtime_node:main",
        ],
    },
)

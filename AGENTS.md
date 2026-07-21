# Repository Guidelines

## Project Structure & Module Organization

This repository contains two coordinated systems:

- `HumanoidRobot_V2_1 20251015/` contains STM32H723 firmware: application code under `HuBot-*`, board support under `HuBot-Bsp/`, generated peripherals under `Core/` and `Drivers/`, and the Keil project under `MDK-ARM/`.
- `upper_computer_ws/src/` is a ROS 2 Foxy workspace. Packages cover communication, kinematics, runtime orchestration, vision, and robot description/Gazebo assets.
- ROS packages keep sources in their package directory, launch files in `launch/`, parameters in `config/`, and tests in `test/`.
- `scripts/` contains standalone integration and visualization utilities. Root-level Markdown files document protocols, geometry, and simulation decisions.

## Build, Test, and Development Commands

The current Windows workspace is for code development only; it does not provide a ROS, Gazebo, RealSense, or robot runtime. Use the following commands only in the target Ubuntu 20.04/ROS 2 Foxy environment:

```bash
cd upper_computer_ws
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
colcon test
colcon test-result --verbose
```

Use `colcon build --packages-select humanoid_arm_vision` for focused iteration. On the target, launch simulation with `ros2 launch humanoid_arm_runtime simulation.launch.py`. Windows can inspect `MDK-ARM/HumanoidRobot_V2_1.uvprojx`; building or flashing requires Keil and target hardware. Do not edit generated STM32Cube files outside marked user sections.

## Coding Style & Naming Conventions

Python uses 4-space indentation, `snake_case` functions/modules, `PascalCase` classes, and type hints for public interfaces. Keep lines at 88 characters; Flake8 ignores `E203` and `W503`. Follow ROS naming: `humanoid_arm_<capability>` packages, `*_node.py` executables, `*.launch.py` launch files, and lowercase YAML keys. In C/C++, preserve local style and keep hardware logic inside BSP/driver layers.

## Testing Guidelines

Tests use `pytest` through `ament`/`colcon` and are named `test_<behavior>.py`. Add unit tests beside the affected package and cover nominal, limit, and failure paths, especially for framing, kinematics, safety, and reconnection. Hardware-dependent RealSense checks are documented in `humanoid_arm_vision/TESTING.md`. Do not claim runtime validation from this Windows environment; record tests as “not run” and identify the required target environment.

## Commit & Pull Request Guidelines

History mixes short English and Chinese summaries. Prefer a clear imperative subject, optionally Conventional Commit style, for example `fix: reject stale joint feedback`. Keep commits scoped to one subsystem. Pull requests should explain behavior changes, list validation commands and hardware used, link relevant issues, and include screenshots or ROS topic/log excerpts for RViz, Gazebo, or perception changes. Never commit generated `build/`, `install/`, or `log/` workspace output.

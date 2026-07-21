# Humanoid arm simulation

## RViz 4-DOF joint control (ROS 2 Foxy)

The URDF is defined directly from the joint-axis geometry at the zero
configuration. Coordinates are in metres and directions are expressed in the
base frame at zero position:

| joint | point (m) | axis direction |
|---|---|---|
| joint_1 | `(0, 0, 0)` | `(0, -0.8660254, 0.5)` |
| joint_2 | `(0, -0.05, 0)` | `(1, 0, 0)` |
| joint_3 | `(0, -0.05, -0.07)` | `(0, 0, 1)` |
| joint_4 | `(0, -0.05, -0.105)` | `(0, 1, 0)` |

All link frames are parallel to `base_link` at zero position. Joint origins are
relative to their parent links: `(0,0,0)`, `(0,-0.05,0)`, `(0,0,-0.07)`, and
`(0,0,-0.035)`. This makes the zero-position axes reproduce the absolute data
above without inferring directions from link lengths.

Use Ubuntu 20.04 with ROS 2 Foxy. Install the declared dependencies and build
the workspace:

```bash
cd ~/upper_computer_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select humanoid_arm_description
source install/setup.bash
```

Launch RViz and the slider GUI:

```bash
ros2 launch humanoid_arm_description rviz_joint_control.launch.py
```

Drag the four sliders in the **Joint State Publisher** window. The allowed
ranges come directly from each URDF joint limit, and RViz updates through
`/joint_states` -> `robot_state_publisher` -> `/tf`.

For a headless joint publisher, use:

```bash
ros2 launch humanoid_arm_description rviz_joint_control.launch.py use_gui:=false
```

## Gazebo camera-follow simulation

This package is a display-only Gazebo Classic 11 simulation. It deliberately
does not model the STM32, serial link, motor PID, torque, or communication node.

After building the ROS 2 workspace, start the complete chain with:

```bash
ros2 launch humanoid_arm_description gazebo_camera_follow.launch.py
```

Wait for `/vision/camera_pose`, `/kinematics/joint_state`, and
`/kinematics/end_effector_pose`, then enter FOLLOW with:

```bash
ros2 service call /runtime_node/start std_srvs/srv/Trigger '{}'
```

`runtime.yaml` defines the pose convention and the row-major rotation from tag
coordinates to robot-base coordinates. The default assumes
`/vision/camera_pose` is the camera pose in the tag frame and both frames are
axis-aligned. Calibrate `follow.tag_to_base_rotation`, axis signs, and yaw sign
before using a different physical mounting.

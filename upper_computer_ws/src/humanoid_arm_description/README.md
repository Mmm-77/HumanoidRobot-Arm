# Humanoid arm simulation

## RViz 4-DOF joint control (ROS 2 Foxy)

The URDF implements the four rows in `dh_parameters.md` using the Modified-DH
convention. Lengths are converted from centimetres to metres:

| joint | alpha (deg) | a (m) | d (m) |
|---|---:|---:|---:|
| joint_1 | -90 | 0.0020 | 0.003 |
| joint_2 | 90 | 0.0289 | 0.050 |
| joint_3 | -90 | 0 | 0.070 |
| joint_4 | 90 | 0 | 0.035 |

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

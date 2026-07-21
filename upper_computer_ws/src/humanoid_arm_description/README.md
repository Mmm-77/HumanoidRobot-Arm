# Humanoid arm Gazebo simulation

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

# Vision and kinematics RViz integration check

Use the dedicated launch on Ubuntu 20.04 with ROS 2 Foxy, a configured
RealSense camera, and the fixed AprilTag visible:

```bash
cd upper_computer_ws
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --packages-select \
  humanoid_arm_description humanoid_arm_vision \
  humanoid_arm_kinematics humanoid_arm_runtime
source install/setup.bash
ros2 launch humanoid_arm_runtime rviz_vision_kinematics.launch.py
```

The launch does not start the communication package, serial port, motors, or
Gazebo. It starts FOLLOW automatically once fresh vision, simulated joint
state, and FK feedback have all arrived. The first valid camera pose is the
motion baseline, so move the camera only after the robot model and all RViz
displays appear.

RViz shows:

- blue axes: `/vision/camera_pose`, the wall-calibrated camera pose in
  `camera_tracking`;
- red arrow: `/kinematics/target`, the mapped target in `base_link`;
- green arrow: `/kinematics/end_effector_pose`, the FK result;
- red/green spheres and the error line: `/kinematics/visualization`;
- the annotated RealSense image: `/vision/debug_image`.

The calibrated camera axes map directly onto `base_link`: away from the wall
is +X, right while facing the tag is +Y, and up is +Z. Set the neutral
camera-to-tag distance with `calibration.wall_x_origin_m` in `vision.yaml`.

Expected behavior:

1. With the camera still, target and FK end-effector remain coincident.
2. A small camera translation moves the red target in the configured base-axis
   direction and the robot model follows.
3. The green FK result converges on the target; the displayed tip error remains
   below the configured 5 mm validation threshold.
4. If the AprilTag is lost, new follow targets stop until valid vision resumes.

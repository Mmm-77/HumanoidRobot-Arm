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

- blue axes: `/vision/camera_pose`, the camera pose in the `tag` frame;
- red arrow: `/kinematics/target`, the mapped target in `base_link`;
- green arrow: `/kinematics/end_effector_pose`, the FK result;
- red/green spheres and the error line: `/kinematics/visualization`;
- the annotated RealSense image: `/vision/debug_image`.

The default assumes the `tag` and `base_link` axes are aligned. Before judging
motion direction, calibrate `follow.tag_to_base_rotation` and the axis signs in
`runtime.yaml`; update the static `base_link -> tag` transform in the launch
file to match the same physical installation.

Expected behavior:

1. With the camera still, target and FK end-effector remain coincident.
2. A small camera translation moves the red target in the configured base-axis
   direction and the robot model follows.
3. The green FK result converges on the target; the displayed tip error remains
   below the configured 5 mm validation threshold.
4. If the AprilTag is lost, new follow targets stop until valid vision resumes.

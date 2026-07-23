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
- purple axes: `/runtime/baseline_end_effector`, the fixed zero-position tip
  pose used as the relative-motion origin;
- red arrow: `/kinematics/target`, the mapped target in `base_link`;
- green arrow: `/kinematics/end_effector_pose`, the FK result;
- red/green spheres and the error line: `/kinematics/visualization`;
- the annotated RealSense image: `/vision/debug_image`.

The calibrated camera axes map directly onto `base_link`: away from the wall
is +X, right while facing the tag is +Y, and up is +Z. Set the neutral
camera-to-tag distance with `calibration.wall_x_origin_m` in `vision.yaml`.

Expected behavior:

1. With the camera still, target and FK end-effector remain coincident.
2. The red target equals the purple zero-position tip pose plus the camera
   displacement; the camera and tip do not need to coincide in world space.
3. A camera translation moves the red target by the same displacement in the
   configured base-axis direction and the robot model follows.
4. The green FK result converges on the target; the displayed tip error remains
   below the configured 5 mm validation threshold.
5. If the AprilTag is lost, new follow targets stop until valid vision resumes.

For a stationary model, check the chain in order:

```bash
ros2 topic echo /runtime/state
ros2 topic echo /runtime/baseline_end_effector
ros2 topic echo /kinematics/target
ros2 topic echo /kinematics/diagnostics
ros2 topic hz /joint_states
```

The normal state is `FOLLOW`. A strict IK miss that is still inside the 5 mm
output validation limit is reported as `valid_within_validation` and is
executed; larger errors remain rejected.

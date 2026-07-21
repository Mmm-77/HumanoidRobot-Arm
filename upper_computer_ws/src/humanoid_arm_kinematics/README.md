# Humanoid arm kinematics

This package treats `humanoid_arm_description/urdf/humanoid_arm.urdf` as the
only source of kinematic geometry. It parses the chain from `base_link` to
`tip_frame`; there is no separate DH table.

Inverse kinematics controls only the three-dimensional `tip_frame` position.
The orientation in an incoming `geometry_msgs/PoseStamped` target is ignored.
The fourth joint is used as a redundant degree of freedom, with a null-space
preference for the solution nearest to the current joint state.

## RViz-only simulation

On Ubuntu 20.04 with ROS 2 Foxy:

```bash
cd upper_computer_ws
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --packages-select \
  humanoid_arm_description humanoid_arm_kinematics
source install/setup.bash
ros2 launch humanoid_arm_kinematics rviz_kinematics.launch.py
```

Publish a reachable target position (the quaternion is ignored):

```bash
ros2 topic pub --once /kinematics/target geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: base_link}, pose: {position: {x: 0.10, y: 0.05, z: -0.30}, orientation: {w: 1.0}}}"
```

The launch file publishes simulated `joint_states`, so RViz updates without
hardware, Gazebo, or the communication package.

RViz also displays:

- a red sphere for the shaped target position;
- a green sphere for the achieved `tip_frame` position;
- a yellow error line and a numerical error label in millimetres.

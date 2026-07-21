# 机械臂运动学参数说明

本项目不再维护独立的 DH 参数表。

`upper_computer_ws/src/humanoid_arm_description/urdf/humanoid_arm.urdf`
是机械臂运动学几何的唯一数据源。`humanoid_arm_kinematics` 会在运行时直接
解析 `base_link` 到 `tip_frame` 的串联链，包括：

- `joint_1` 至 `joint_4` 的顺序；
- 每个关节的 `origin` 和 `axis`；
- `joint_4` 至 `tip_frame` 的固定杆长和变换。

此前的 Modified DH 表与 URDF 的关节轴及零位几何不一致，已经移除，禁止
继续复制到代码、配置或测试中。机械结构变化时只修改并验证 URDF。

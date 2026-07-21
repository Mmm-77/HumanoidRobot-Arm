# Gazebo 仿真方案

## 1. 目标

使用真实外置相机识别 AprilTag，并在 Gazebo 窗口中显示机械臂跟随相机位姿运动。

本方案只验证以下链路：

```text
外置相机 → AprilTag 位姿 → 位姿映射 → 逆运动学 → Gazebo 机械臂运动
```

不验证电机、STM32、串口、PID、力矩或其他底层控制。

## 2. 系统架构

```text
外置 RealSense 相机
        ↓
vision_node
发布 /vision/camera_pose
        ↓
runtime_node
计算相机相对运动，发布 /kinematics/target
        ↓
kinematics_node
执行 IK，发布 /kinematics/joint_command
        ↓
Gazebo 关节驱动插件
更新 joint_1～joint_4
        ↓
Gazebo 窗口显示机械臂运动
```

Gazebo 关节驱动插件同时发布 `/kinematics/joint_state`，供运动学和 runtime 节点获取当前关节状态。

## 3. 软件环境

当前项目面向 ROS 2 Foxy，建议使用：

- Ubuntu 20.04
- ROS 2 Foxy
- Gazebo Classic 11
- `gazebo_ros_pkgs`
- RealSense SDK

ROS 2、Gazebo、RealSense 和项目节点应尽量运行在同一个 Linux 环境中。

## 4. 需要新增的内容

### 4.1 机器人描述包

新增 `humanoid_arm_description` 包，建议结构如下：

```text
humanoid_arm_description/
├── urdf/       # URDF/Xacro 模型
├── meshes/     # 可选的外观模型
├── worlds/     # Gazebo 场景
├── launch/     # 仿真启动文件
└── config/     # 仿真参数
```

机器人模型至少包含：

- `base_link`
- 四段机械臂连杆
- `joint_1`～`joint_4`
- 末端 `tool_link`
- 关节上下限
- Gazebo 关节驱动插件

URDF 是运动学几何的唯一数据源。`humanoid_arm_kinematics` 直接解析关节位置、旋转轴以及到 `tip_frame` 的固定变换，不再维护独立 DH 表。当前 RViz 几何验证阶段不应用真实关节限位。

### 4.2 Gazebo 关节驱动插件

插件只负责运动显示，不模拟电机控制。

主要职责：

1. 订阅 `/kinematics/joint_command`，消息类型为 `trajectory_msgs/JointTrajectory`。
2. 读取 `joint_1`～`joint_4` 的目标角度。
3. 将 Gazebo 关节平滑移动到目标角度。
4. 发布 `/kinematics/joint_state`，消息类型为 `sensor_msgs/JointState`。
5. 拒绝非法数值、缺失关节和超限命令。

建议支持两种模式：

- 直接模式：立即设置目标角度，用于几何验证。
- 插值模式：按设定速度平滑移动，用于正常演示。

默认使用插值模式。

### 4.3 当前末端位姿反馈

`kinematics_node` 根据 `/kinematics/joint_state` 执行正运动学，并新增发布：

```text
/kinematics/end_effector_pose
类型：geometry_msgs/PoseStamped
坐标系：base
```

`runtime_node` 进入 FOLLOW 时记录：

- 当前相机位姿；
- 当前机械臂末端位姿。

之后将相机相对于基准的位姿变化叠加到机械臂末端基准上，再发送给 IK。这样可以避免启动 FOLLOW 时机械臂突然跳动。

现有 `runtime_node` 中的末端基准计算仍是占位实现，接入 Gazebo 前需要替换为上述真实 FK 结果。

### 4.4 坐标系映射

视觉输出位于 AprilTag/相机坐标系，IK 目标位于机械臂 `base` 坐标系，因此必须定义两者之间的固定坐标变换。

需要明确：

- `/vision/camera_pose` 表示相机相对 Tag，还是 Tag 相对相机；
- 相机 X/Y/Z 分别对应机械臂 base 坐标系的哪个方向；
- 相机旋转如何映射到机械臂末端 yaw。

建议使用完整的旋转矩阵或 TF 进行转换。第一版也可以固定相机和 AprilTag 的安装方向，使坐标轴基本对齐，再通过参数调整方向符号和比例。

## 5. 仿真启动方案

新增统一启动文件，例如：

```text
gazebo_camera_follow.launch.py
```

启动顺序：

1. 启动 Gazebo 空场景。
2. 加载机器人 URDF。
3. 在 Gazebo 中生成机械臂模型。
4. 加载 Gazebo 关节驱动插件。
5. 启动 `kinematics_node` 并加载运动学配置。
6. 启动 `runtime_node` 并加载运行参数。
7. 启动 `vision_node`，连接真实外置相机。
8. 等待相机位姿和 Gazebo 关节状态稳定。
9. 系统进入 READY 后，由用户调用 start 服务进入 FOLLOW。

仿真模式中不启动 `communication_node`。

## 6. 使用流程

1. 将 AprilTag 固定在现实环境中。
2. 保证外置相机能够稳定看到 AprilTag。
3. 启动 Gazebo 联合仿真。
4. 确认以下话题持续有数据：

   ```text
   /vision/camera_pose
   /kinematics/joint_state
   /kinematics/end_effector_pose
   ```

5. 确认 `/runtime/state` 为 `READY`。
6. 调用 `/runtime_node/start`，进入 `FOLLOW`。
7. 移动现实中的外置相机。
8. 观察 Gazebo 窗口中的机械臂是否按照相机的相对位姿运动。

保留以下服务，用于测试和操作：

- `/runtime_node/start`
- `/runtime_node/hold`
- `/runtime_node/unhold`
- `/runtime_node/reset`

## 7. 验证内容

### 7.1 基本运动

- 相机静止时，Gazebo 机械臂保持稳定。
- 相机沿 X、Y、Z 分别移动时，机械臂末端沿配置的对应方向移动。
- 相机绕指定轴旋转时，机械臂末端 yaw 跟随变化。
- 连续移动相机时，机械臂运动连续，无明显跳变。

### 7.2 安全和异常

- AprilTag 短暂丢失时进入 HOLD，机械臂保持当前位置。
- AprilTag 长时间丢失时进入 SAFE，不再执行新目标。
- 目标超出工作空间时，IK 拒绝目标，机械臂不跳变。
- 关节命令不得超过角度和速度限制。

### 7.3 建议验收指标

| 指标 | 建议标准 |
|---|---:|
| AprilTag 有效位姿输出率 | ≥ 95% |
| 位姿输入到关节命令的延迟 | ≤ 100 ms |
| 相机静止时末端位置抖动 | ≤ 5 mm |
| 相机静止时末端姿态抖动 | ≤ 1° |
| IK 正运动学回代位置误差 | ≤ 5 mm |
| IK 正运动学回代 yaw 误差 | ≤ 0.02 rad |
| 关节越限次数 | 0 |

## 8. 实施顺序

1. 建立 URDF，并验证 Gazebo 模型的关节轴和尺寸。
2. 完成 Gazebo 关节驱动插件。
3. 手动发布四关节命令，确认 Gazebo 模型可以运动。
4. 接入 `kinematics_node`，用手动末端目标验证 IK。
5. 增加末端 FK 话题，修复 runtime 的 FOLLOW 基准计算。
6. 接入真实 `vision_node`。
7. 标定相机坐标系到机械臂 base 坐标系的映射。
8. 测试正常跟随、Tag 丢失、目标越界和快速移动场景。

## 9. 最终效果

启动联合仿真后，电脑上显示 Gazebo 窗口和四轴机械臂模型。现实中的外置相机检测固定 AprilTag；用户移动相机后，项目计算相对位姿和关节角，Gazebo 中的机械臂随之运动。

整个过程不依赖 STM32、串口或真实电机，只验证 AprilTag 到机械臂姿态控制的上层链路。

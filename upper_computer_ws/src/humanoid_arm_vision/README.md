# humanoid_arm_vision

`humanoid_arm_vision` 是上位机的独立 ROS 2 视觉定位软件包。它接收 D435i
彩色图像，检测指定 AprilTag，解算并过滤相机在固定 `tag` 坐标系中的位姿。
软件包不建立跟随基准、不生成机械臂目标，也不访问串口。

## 实现选择

- 默认复用官方 `realsense2_camera` ROS 2 驱动。驱动负责 D435i 枚举、彩色流、
  出厂 `CameraInfo`、硬件时间戳和重连；视觉节点订阅
  `/camera/camera/color/image_raw` 与
  `/camera/camera/color/camera_info`。
- AprilTag 检测复用 OpenCV ArUco 模块的 AprilTag 字典。
- 位姿使用 `SOLVEPNP_IPPE_SQUARE` 的平面双解，并以迭代 PnP 处理正视退化；
  最终按重投影误差、正深度和标签正面法向选择物理解。
- OpenCV 通用相机采集保留为 `input.mode: opencv` 回退，便于无 ROS 离线调试。

## 必须先填写的实机参数

默认配置有意将 `tag.size_m` 设为 `0.0`，节点会拒绝启动。实机使用前必须在
`config/vision.yaml` 中填写：

1. `tag.family`：`tag16h5`、`tag25h9`、`tag36h10` 或 `tag36h11`；
2. `tag.id`：目标标签编号；
3. `tag.size_m`：标签外侧黑色边框的实测边长，单位为米。

D435i 模式直接使用驱动发布的彩色相机 `CameraInfo`，不应再填写虚构内参。
只有 `input.mode: opencv` 时才读取 `calibration.file` 或内联标定参数；未标定的
OpenCV 回退会安全失败。

## 坐标约定

- `tag`：x 向标签右侧，y 向标签上方，z 从印刷正面垂直向外；
- 相机姿态表示 RealSense 彩色光学坐标系（x 右、y 下、z 前）在 `tag` 中的姿态；
- `/vision/camera_pose` 使用 `geometry_msgs/PoseStamped`，其
  `header.frame_id` 固定为 `tag`，位置单位为米，四元数顺序遵循 ROS 的 xyzw。

OpenCV PnP 原始结果是“tag 到 camera”的变换，代码会显式求逆后再发布，避免
把标签位姿误当成相机位姿。

## ROS 2 接口

| 方向 | 主题 | 类型 | 说明 |
| --- | --- | --- | --- |
| 输入 | `/camera/camera/color/image_raw` | `sensor_msgs/Image` | D435i 彩色图像 |
| 输入 | `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 同一彩色流的内参与畸变 |
| 输出 | `/vision/camera_pose` | `geometry_msgs/PoseStamped` | 仅有效帧发布 |
| 输出 | `/vision/valid` | `std_msgs/Bool` | 每个处理帧的有效标志 |
| 输出 | `/vision/camera_info` | `sensor_msgs/CameraInfo` | 本次解算使用的相机参数 |
| 输出 | `/vision/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 原因、误差、面积、距离和时效 |
| 可选输出 | `/vision/raw_image` | `sensor_msgs/Image` | 原始彩色图像 |
| 可选输出 | `/vision/debug_image` | `sensor_msgs/Image` | 标签轮廓及有效性叠加图 |

无标签、图像过期、`CameraInfo` 缺失、标签过小、重投影误差过大、距离越界或
位姿跳变时，不发布新位姿，只发布 `valid=false` 和诊断原因。OpenCV 后端不暴露
AprilTag decision margin，因此默认以面积、重投影误差和连续性门控；若把
`quality.min_decision_margin` 设为正数，节点会以
`decision_margin_unavailable` 拒绝帧。

## 构建与启动

目标 Linux/ROS 2 环境需要 `realsense2_camera`、`cv_bridge`、OpenCV（含 aruco）、
NumPy 和 PyYAML。将必填 Tag 参数写入配置后，在 `upper_computer_ws` 中执行：

```bash
colcon build --packages-select humanoid_arm_vision
source install/setup.bash
ros2 launch humanoid_arm_vision vision.launch.py
```

默认 launch 只开启 D435i 彩色流，关闭本任务不使用的深度、红外、点云和 IMU。
若驱动已由其他 launch 启动，或要回放 rosbag：

```bash
ros2 launch humanoid_arm_vision vision.launch.py launch_realsense:=false
```

指定多相机环境中的序列号或调整彩色流配置：

```bash
ros2 launch humanoid_arm_vision vision.launch.py \
  serial_no:="'_设备序列号_'" color_profile:=1280,720,30
```

## 测试范围

纯 Python 测试覆盖相机回退、标定校验、合成 AprilTag 检测、已知位姿反解、
四元数滤波、时效和跳变门控。在当前 Windows 环境中未运行 ROS 2、RealSense
硬件或 launch 测试；这些项目需在目标 Linux/ROS 2 主机与 D435i 实机上验收。

# humanoid_arm_vision

`humanoid_arm_vision` 是上位机的独立 ROS 2 视觉定位软件包。节点通过
`pyrealsense2` 直接控制 D435i 彩色相机，不依赖 `realsense2_camera` ROS 包。
它检测指定 AprilTag，解算并过滤相机在固定 `tag` 坐标系中的位姿。

## 相机实现

- 节点使用 RealSense SDK 启动指定的 BGR8 彩色流；
- 图像通过 SDK frame 零额外解码地转换为 NumPy 数组；
- 相机矩阵和畸变参数来自实际启动后的彩色 stream profile；
- SDK 取帧具有超时和重连退避，不会无限阻塞 ROS executor；
- 不提供 ROS RealSense 驱动订阅或通用 OpenCV 相机回退模式。

当前 PnP 流水线支持 SDK 报告的无畸变和 Brown-Conrady 彩色内参。遇到其他
畸变模型时节点会安全拒绝启动相机，避免把不兼容的系数传给 OpenCV。

## 必填实机参数

默认配置有意将 `tag.size_m` 设为 `0.0`，节点会拒绝启动。实机使用前必须在
`config/vision.yaml` 中填写：

1. `tag.family`：`tag16h5`、`tag25h9`、`tag36h10` 或 `tag36h11`；
2. `tag.id`：目标标签编号；
3. `tag.size_m`：标签外侧黑色边框的实测边长，单位为米。

多相机环境还必须填写 `camera.serial_number`。单相机环境可以留空。

## 坐标约定

- `tag`：x 向标签右侧，y 向标签上方，z 从印刷正面垂直向外；
- 相机姿态表示 RealSense 彩色光学坐标系（x 右、y 下、z 前）在 `tag` 中的姿态；
- `/vision/camera_pose` 的 `header.frame_id` 为 `tag`，位置单位为米，四元数为 xyzw。

## ROS 2 输出

| 主题 | 类型 | 说明 |
| --- | --- | --- |
| `/vision/camera_pose` | `geometry_msgs/PoseStamped` | 仅有效帧发布 |
| `/vision/valid` | `std_msgs/Bool` | 每个处理帧的有效标志 |
| `/vision/camera_info` | `sensor_msgs/CameraInfo` | SDK 出厂彩色内参 |
| `/vision/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 设备、错误、面积、距离和质量指标 |
| `/vision/raw_image` | `sensor_msgs/Image` | 可选原始 BGR 图像 |
| `/vision/debug_image` | `sensor_msgs/Image` | 可选标签轮廓与有效性叠加图 |

## Ubuntu 20.04 / ROS 2 Foxy 部署

Foxy 默认使用 Python 3.8。RealSense 官方只在 `pyrealsense2 2.55.x` 及更早版本提供
Python 3.8 发行包，因此 SDK 与 Python binding 应固定在同一兼容版本。若机上
SDK 是源码安装，应在同一 SDK 源码版本中启用 `BUILD_PYTHON_BINDINGS`，避免加载
到另一版本的 `librealsense2.so`。

先验证 SDK 和 Python binding：

```bash
rs-enumerate-devices
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
```

安装 Python binding、填写 Tag 参数后构建启动：

```bash
python3 -m pip install 'pyrealsense2<2.56'
cd upper_computer_ws
colcon build --packages-select humanoid_arm_vision
source install/setup.bash
ros2 launch humanoid_arm_vision vision.launch.py
```

## 测试范围

纯 Python 测试使用模拟 SDK 覆盖彩色流配置、序列号选择、SDK 内参转换、取帧
超时和断线关闭，同时覆盖标定、AprilTag 检测、位姿反解、滤波和质量门控。
目标机仍需使用真实 D435i 验证 USB 带宽、持续运行和拔插恢复。

完整的目标机操作步骤和验收标准见 [TESTING.md](TESTING.md)。

# RealSense SDK 视觉节点实机测试指南

本文档用于在 Ubuntu 20.04、ROS 2 Foxy 和 Intel RealSense D435i 实机环境中，
验证 `humanoid_arm_vision` 的 SDK 直连采集、内参、AprilTag 位姿和异常恢复能力。

## 1. 测试前准备

准备以下设备和数据：

- 安装 Ubuntu 20.04 和 ROS 2 Foxy 的上位机；
- 一台 D435i，使用 USB 3 数据线直接连接上位机；
- 一个已知 family、ID 和黑色外边框实测边长的 AprilTag；
- 已安装且能操作该相机的 librealsense SDK；
- 与系统 Python 3.8 兼容的 `pyrealsense2 2.55.x`。

除非测试多相机选择，否则同一时间只连接一台 RealSense。测试前关闭
RealSense Viewer 和其他可能占用相机的进程。

## 2. 环境检查

加载 ROS 环境并确认 Python 版本：

```bash
source /opt/ros/foxy/setup.bash
python3 --version
which python3
```

预期 Python 为 3.8。确认 SDK、Python binding 和相机均可见：

```bash
rs-enumerate-devices
python3 -c "import importlib.metadata as m; print(m.version('pyrealsense2'))"
python3 -c "import pyrealsense2 as rs; print(rs.context().query_devices().size())"
lsusb -t
```

验收标准：

- `rs-enumerate-devices` 能显示 D435i 型号、序列号和固件版本；
- Python 成功导入 `pyrealsense2`，设备数至少为 1；
- `lsusb -t` 中相机连接速率为 `5000M` 或更高，而不是 `480M`。

如果普通用户看不到相机，但 `sudo rs-enumerate-devices` 可以看到，应先修复
RealSense udev 规则，不要使用 root 身份运行 ROS 节点。

## 3. 配置测试参数

编辑 `upper_computer_ws/src/humanoid_arm_vision/config/vision.yaml`：

```yaml
camera.serial_number: ""  # 单相机留空；多相机填写 rs-enumerate-devices 输出的序列号
camera.width: 640
camera.height: 480
camera.fps: 30

tag.family: "tag36h11"
tag.id: 0
tag.size_m: 0.100  # 示例值；必须替换为黑色外边框的实际边长，单位为米
```

不要测量白纸尺寸，也不要使用标签内部编码区域的边长。错误的 `tag.size_m`
会使位置结果按相同比例整体缩放。

若需要检查图像，将以下参数设为 `true` 后重新启动节点：

```yaml
publish_raw_image: true
publish_debug_image: true
```

这些发布器在节点启动时创建，运行后临时修改参数不会创建图像主题。

## 4. 构建和静态测试

在工作空间执行：

```bash
cd upper_computer_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-select humanoid_arm_vision
source install/setup.bash
colcon test --packages-select humanoid_arm_vision
colcon test-result --verbose
```

验收标准：构建成功，测试结果中没有 failure 或 error。

另外检查实际运行节点所用的 Python 能导入 SDK：

```bash
head -n 1 install/humanoid_arm_vision/lib/humanoid_arm_vision/vision_node
python3 -c "import pyrealsense2, rclpy, cv2, numpy; print('imports OK')"
```

## 5. SDK 独立取帧测试

先绕开 ROS 验证 SDK 彩色流和出厂内参。以下命令会读取 100 帧，不会打开窗口：

```bash
python3 - <<'PY'
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
try:
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = stream.get_intrinsics()
    print("profile:", intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    print("distortion:", intr.model, list(intr.coeffs))
    for index in range(100):
        frame = pipeline.wait_for_frames(1000).get_color_frame()
        if not frame:
            raise RuntimeError("missing color frame")
        image = np.asanyarray(frame.get_data())
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise RuntimeError(f"unexpected frame: {image.shape}, {image.dtype}")
    print("100 BGR frames OK")
finally:
    pipeline.stop()
PY
```

验收标准：输出 `100 BGR frames OK`，图像形状为 `(480, 640, 3)`，且内参中的
`fx`、`fy` 为正数。若此步骤失败，应先解决 SDK、权限、USB 或固件问题，再测试 ROS。

## 6. 启动 ROS 节点

```bash
cd upper_computer_ws
source /opt/ros/foxy/setup.bash
source install/setup.bash
ros2 launch humanoid_arm_vision vision.launch.py
```

另开终端并加载相同环境：

```bash
ros2 node list
ros2 topic list | sort
ros2 topic hz /vision/valid
ros2 topic hz /vision/camera_info
ros2 topic echo /vision/diagnostics --once
```

验收标准：

- 节点列表包含 `/vision_node`；
- `/vision/valid` 和 `/vision/camera_info` 持续发布，频率接近配置值；
- `CameraInfo.width/height` 与采集分辨率一致；
- `CameraInfo.k` 中的 `fx`、`fy` 与第 5 步 SDK 输出一致；
- diagnostics 的 `hardware_id` 包含 D435i 名称和序列号；
- 终端没有持续出现 frame timeout、USB 或 distortion model 错误。

启用图像发布后，可以执行：

```bash
rqt_image_view /vision/raw_image
rqt_image_view /vision/debug_image
```

确认图像颜色正常、无明显撕裂，debug 图像能在目标标签外沿绘制轮廓。

## 7. AprilTag 位姿测试

将标签固定在平整表面，先正对相机并保持静止：

```bash
ros2 topic echo /vision/valid
ros2 topic echo /vision/camera_pose
ros2 topic echo /vision/diagnostics
```

按以下顺序测试：

1. 标签移出画面：`valid` 应为 `false`，且不发布新的 pose；
2. 标签回到画面并保持静止：`valid` 应稳定变为 `true`；
3. 用卷尺测量标签到彩色相机的大致距离，对比 pose 的距离量级；
4. 将相机缓慢向标签右侧、上方和前后移动，确认坐标变化方向符合 README 约定；
5. 缓慢旋转相机，确认四元数连续变化且没有频繁正反解跳变；
6. 快速遮挡并重新露出标签，确认滤波能恢复，不继续发布遮挡前的旧位姿。

建议记录三个已知距离点，例如 0.3 m、0.5 m、0.8 m。距离误差如果按固定比例
变化，优先重新测量 `tag.size_m`；如果图像边缘误差显著增大，应检查 SDK 内参、
畸变模型和标签平整度。

## 8. 断线和重连测试

保持节点运行，拔掉 D435i USB 线：

```bash
ros2 topic echo /vision/diagnostics
```

预期行为：

- 最多经过 `camera.reopen_after_failures` 次读取失败后关闭旧 pipeline；
- `/vision/valid` 持续发布 `false`；
- diagnostics 原因为 `camera_error`，进程不退出、不永久卡死。

等待数秒后重新插入同一个 USB 3 端口。经过 `camera.reopen_delay_s` 退避后，节点
应重新枚举相机并恢复 `/vision/camera_info`；标签可见时 pose 应再次发布。

多相机场景必须填写 `camera.serial_number`，然后重复断线测试，确认节点不会连接到
另一台 RealSense。

## 9. 长时间稳定性测试

建议至少连续运行 2 小时：

```bash
mkdir -p ~/humanoid_vision_test
timeout --signal=INT 2h ros2 launch humanoid_arm_vision vision.launch.py \
  2>&1 | tee ~/humanoid_vision_test/vision.log
```

测试期间每隔 15 分钟记录一次：

```bash
date
ros2 topic hz /vision/valid --window 100
ps -C vision_node -o pid,%cpu,%mem,rss,etime,cmd
```

验收标准：

- 节点不崩溃、不失去响应；
- 没有持续 frame timeout 或 USB reset；
- 发布频率没有持续下降；
- RSS 内存没有随时间持续单调增长；
- 静止相机和标签的 pose 不出现周期性大跳变。

## 10. 测试记录模板

每次实机验收建议保存以下信息：

```text
测试日期：
主机型号：
Ubuntu / kernel：
ROS 2 版本：
D435i 序列号：
相机固件版本：
librealsense 版本：
pyrealsense2 版本：
分辨率 / FPS：
AprilTag family / ID / size_m：
SDK 100 帧测试：通过 / 失败
ROS 主题与内参：通过 / 失败
位姿方向与距离：通过 / 失败
断线恢复：通过 / 失败
2 小时稳定性：通过 / 失败
日志路径：
遗留问题：
```

只有 SDK 独立取帧、ROS 输出、位姿、断线恢复和长时间稳定性均通过后，才建议将
该视觉节点用于机械臂闭环控制。

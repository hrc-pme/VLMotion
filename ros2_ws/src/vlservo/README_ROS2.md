# VLServo ROS2 視覺伺服系統

## 概述

這個套件提供了基於 ROS2 的視覺伺服功能,使用 RealSense 相機和 Stretch 機器人的 TF 系統,而不是直接使用 pyrealsense2。

## 主要改進

- ✅ 使用 ROS2 話題接收相機數據 (不再使用 pyrealsense2)
- ✅ 使用 TF2 進行座標變換
- ✅ 整合 stretch_core 驅動程式
- ✅ 提供完整的 launch 文件

## 系統架構

```
┌─────────────────┐
│ Stretch Driver  │ (stretch_core)
│  - Joint States │
│  - TF Tree      │
└────────┬────────┘
         │
┌────────▼────────┐      ┌──────────────────┐
│ RealSense Node  │◄─────┤ realsense2_camera│
│  - Color Image  │      └──────────────────┘
│  - Depth Image  │
│  - Camera Info  │
└────────┬────────┘
         │
┌────────▼─────────────┐
│ Camera Subscriber    │
│  - Subscribe topics  │
│  - Convert to CV     │
└────────┬─────────────┘
         │
┌────────▼─────────────┐
│ Visual Servo Node    │
│  - Get target point  │
│  - Use TF for coords │
│  - Send commands     │
└──────────────────────┘
```

## 安裝

套件已經安裝在 `/workspace/ros2_ws` 中。

## 使用方法

### 1. 啟動完整系統 (Stretch + RealSense)

```bash
# Terminal 1: 啟動 Stretch 驅動和相機
cd /workspace/ros2_ws
source install/setup.bash
ros2 launch vlservo vlservoing_ros2.launch.py
```

### 2. 單獨啟動 Stretch 驅動

```bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch stretch_launch stretch_driver.launch.py
```

### 3. 單獨啟動 RealSense 相機

```bash
# D435i 頭部相機
ros2 run realsense2_camera realsense2_camera_node \
  --ros-args \
  -p camera_name:=camera \
  -p enable_color:=true \
  -p enable_depth:=true \
  -p align_depth.enable:=true
```

### 4. 測試系統組件

```bash
source /workspace/ros2_ws/install/setup.bash

# 測試相機訂閱
python3 /workspace/ros2_ws/src/vlservo/VLServo/test_ros2_system.py --camera

# 測試 TF 框架
python3 /workspace/ros2_ws/src/vlservo/VLServo/test_ros2_system.py --tf

# 測試目標點發布
python3 /workspace/ros2_ws/src/vlservo/VLServo/test_ros2_system.py --target

# 運行所有測試
python3 /workspace/ros2_ws/src/vlservo/VLServo/test_ros2_system.py --all
```

### 5. 啟動視覺伺服節點

```bash
source /workspace/ros2_ws/install/setup.bash
ros2 run vlservo visual_servoing_ros2_node
```

## 可用的 ROS2 話題

### 相機話題 (來自 realsense2_camera)
- `/camera/color/image_raw` - RGB 彩色影像
- `/camera/depth/image_rect_raw` - 深度影像
- `/camera/aligned_depth_to_color/image_raw` - 對齊到彩色的深度影像
- `/camera/color/camera_info` - 彩色相機參數
- `/camera/depth/camera_info` - 深度相機參數

### Stretch 話題 (來自 stretch_core)
- `/stretch/joint_states` - 關節狀態
- `/stretch/cmd_vel` - 速度命令
- `/joint_pose_cmd` - 關節位置命令
- `/tf` - TF 變換
- `/tf_static` - 靜態 TF 變換

### 視覺伺服話題
- `/visual_servo/target_point` - 目標點 (PointStamped)
- `/visual_servo/status` - 狀態訊息

## TF 框架

系統使用以下主要框架:

- `base_link` - 機器人底座
- `camera_link` - 相機連接點
- `camera_color_optical_frame` - 相機光學框架
- `link_gripper_fingertip_left` - 左夾爪指尖
- `link_gripper_fingertip_right` - 右夾爪指尖

查看所有可用框架:
```bash
ros2 run tf2_tools view_frames
```

## 發送目標點示例

```bash
# 在 base_link 座標系中發送目標點 (x=0.5m, y=0.0m, z=0.5m)
ros2 topic pub /visual_servo/target_point geometry_msgs/msg/PointStamped \
  "{header: {frame_id: 'base_link'}, point: {x: 0.5, y: 0.0, z: 0.5}}" --once
```

## Python API 使用示例

### 使用相機訂閱器

```python
import rclpy
from VLServo.camera_ros2_subscriber import CameraSubscriber

rclpy.init()

# 創建相機訂閱器
camera_sub = CameraSubscriber(camera_name='camera')

# 等待數據
if camera_sub.wait_for_frames(timeout=10.0):
    # 獲取影像
    color, depth, camera_info = camera_sub.get_frames()
    
    # 使用影像...
    print(f"Color shape: {color.shape}")
    print(f"Depth shape: {depth.shape}")

camera_sub.destroy_node()
rclpy.shutdown()
```

### 使用 TF 進行座標轉換

```python
import rclpy
from rclpy.node import Node
import tf2_ros

rclpy.init()

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
    
    def get_transform(self):
        try:
            # 從 gripper 到 base_link 的變換
            trans = self.tf_buffer.lookup_transform(
                'base_link',
                'link_gripper_fingertip_left',
                rclpy.time.Time()
            )
            print(f"Gripper position: {trans.transform.translation}")
        except Exception as e:
            print(f"Transform failed: {e}")

node = MyNode()
rclpy.spin(node)
```

## 故障排除

### 相機無法啟動

檢查 RealSense 設備:
```bash
rs-enumerate-devices
```

### 沒有影像話題

檢查話題列表:
```bash
ros2 topic list | grep camera
```

### TF 框架缺失

檢查 TF 樹:
```bash
ros2 run tf2_tools view_frames
evince frames.pdf
```

確保 stretch_driver 正在運行:
```bash
ros2 node list | grep stretch
```

### 深度數據單位

RealSense 深度影像通常以毫米為單位。使用 `get_depth_scale()` 函數轉換為米:
```python
from VLServo.camera_ros2_subscriber import get_depth_scale

depth_scale = get_depth_scale()  # 返回 0.001 (mm to m)
depth_in_meters = depth_image * depth_scale
```

## 主要文件

- `camera_ros2_subscriber.py` - ROS2 相機訂閱器
- `visual_servoing_ros2_node.py` - 視覺伺服控制節點
- `test_ros2_system.py` - 系統測試腳本
- `vlservoing_ros2.launch.py` - 完整系統啟動文件

## 下一步

1. 測試相機連接: `--camera`
2. 驗證 TF 框架: `--tf`
3. 啟動視覺伺服節點
4. 發送測試目標點
5. 集成您的視覺算法 (YOLO, ArUco, etc.)

## 參考

- [ROS2 Humble 文檔](https://docs.ros.org/en/humble/)
- [realsense2_camera](https://github.com/IntelRealSense/realsense-ros)
- [stretch_ros2](https://github.com/hello-robot/stretch_ros2)

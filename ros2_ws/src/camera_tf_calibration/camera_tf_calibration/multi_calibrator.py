#!/usr/bin/env python3
"""
多相機 TF 互動式校準工具

功能：
1. 同時校準多個相機 (D435i, D405 等)
2. 按 TAB 切換相機
3. 即時在 RViz 中看到調整效果
4. 分別保存每個相機的校準值

使用方式：
    ros2 run camera_tf_calibration multi_calibrator

鍵盤控制：
    TAB     - 切換相機
    1/2/3   - 直接選擇相機
    
    位移 (米):
        W/S - X 軸 前/後
        A/D - Y 軸 左/右  
        Q/E - Z 軸 上/下
    
    旋轉 (度):
        I/K - Pitch 上/下傾斜
        J/L - Yaw 左/右轉
        U/O - Roll 左/右滾
    
    其他:
        +/- - 調整步進大小
        R   - 重置當前相機為零
        P   - 打印所有相機當前值
        C   - 保存所有相機校準值
        H   - 顯示幫助
        ESC - 退出
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math
import sys
import os
import yaml
import termios
import tty
import select
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CameraConfig:
    """相機配置"""
    name: str
    parent_frame: str
    child_frame: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0  # 弧度
    pitch: float = 0.0
    yaw: float = 0.0


class MultiCameraCalibrator(Node):
    """多相機 TF 校準器"""
    
    # 預設相機配置
    DEFAULT_CAMERAS = [
        {
            'name': 'd435i',
            'parent_frame': 'camera_bottom_screw_frame',
            'child_frame': 'd435i_link',
            'description': 'D435i 頭部相機'
        },
        {
            'name': 'd405',
            'parent_frame': 'gripper_camera_bottom_screw_frame',
            'child_frame': 'd405_link',
            'description': 'D405 手腕相機'
        },
    ]
    
    def __init__(self):
        super().__init__('multi_camera_calibrator')
        
        # TF 廣播器
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 配置目錄
        self.config_dir = os.path.expanduser('~/.config/camera_tf_calibration')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 載入相機配置
        self.cameras: List[CameraConfig] = []
        self.load_cameras_config()
        
        # 當前選中的相機索引
        self.current_camera_idx = 0
        
        # 步進大小
        self.translation_step = 0.01  # 1 cm
        self.rotation_step = 1.0      # 1 度
        
        # 載入所有相機的校準值
        for cam in self.cameras:
            self.load_camera_calibration(cam)
        
        # 定時發布所有相機的 TF
        self.timer = self.create_timer(0.05, self.publish_all_tf)  # 20 Hz
        
    def load_cameras_config(self):
        """載入相機配置"""
        config_file = os.path.join(self.config_dir, 'cameras_config.yaml')
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                camera_list = config.get('cameras', self.DEFAULT_CAMERAS)
            except Exception as e:
                self.get_logger().warn(f'載入相機配置失敗: {e}')
                camera_list = self.DEFAULT_CAMERAS
        else:
            camera_list = self.DEFAULT_CAMERAS
            # 保存預設配置
            self.save_cameras_config()
        
        for cam_config in camera_list:
            self.cameras.append(CameraConfig(
                name=cam_config['name'],
                parent_frame=cam_config['parent_frame'],
                child_frame=cam_config['child_frame'],
            ))
    
    def save_cameras_config(self):
        """保存相機配置"""
        config_file = os.path.join(self.config_dir, 'cameras_config.yaml')
        config = {
            'cameras': self.DEFAULT_CAMERAS,
            'description': '相機配置文件，可以添加更多相機'
        }
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def load_camera_calibration(self, camera: CameraConfig):
        """載入單個相機的校準值"""
        cal_file = os.path.join(self.config_dir, f'{camera.name}_calibration.yaml')
        
        if os.path.exists(cal_file):
            try:
                with open(cal_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                camera.x = config.get('x', 0.0)
                camera.y = config.get('y', 0.0)
                camera.z = config.get('z', 0.0)
                camera.roll = math.radians(config.get('roll_deg', 0.0))
                camera.pitch = math.radians(config.get('pitch_deg', 0.0))
                camera.yaw = math.radians(config.get('yaw_deg', 0.0))
                self.get_logger().info(f'已載入 {camera.name} 校準值')
            except Exception as e:
                self.get_logger().warn(f'載入 {camera.name} 校準值失敗: {e}')
    
    def save_camera_calibration(self, camera: CameraConfig):
        """保存單個相機的校準值"""
        cal_file = os.path.join(self.config_dir, f'{camera.name}_calibration.yaml')
        
        config = {
            'camera_name': camera.name,
            'parent_frame': camera.parent_frame,
            'child_frame': camera.child_frame,
            'x': round(camera.x, 4),
            'y': round(camera.y, 4),
            'z': round(camera.z, 4),
            'roll_deg': round(math.degrees(camera.roll), 2),
            'pitch_deg': round(math.degrees(camera.pitch), 2),
            'yaw_deg': round(math.degrees(camera.yaw), 2),
            'roll_rad': round(camera.roll, 4),
            'pitch_rad': round(camera.pitch, 4),
            'yaw_rad': round(camera.yaw, 4),
            'saved_at': datetime.now().isoformat(),
        }
        
        with open(cal_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        return config
    
    def save_all_calibrations(self):
        """保存所有相機的校準值"""
        print(f'\n💾 保存所有相機校準值...')
        for camera in self.cameras:
            config = self.save_camera_calibration(camera)
            print(f'   ✅ {camera.name}: {self.config_dir}/{camera.name}_calibration.yaml')
        
        # 生成 launch 參數摘要
        print(f'\n📋 Launch 參數摘要:')
        for camera in self.cameras:
            print(f'\n   [{camera.name}]')
            print(f'   {camera.name}_tf_x:={camera.x:.4f} \\')
            print(f'   {camera.name}_tf_y:={camera.y:.4f} \\')
            print(f'   {camera.name}_tf_z:={camera.z:.4f} \\')
            print(f'   {camera.name}_tf_roll:={camera.roll:.4f} \\')
            print(f'   {camera.name}_tf_pitch:={camera.pitch:.4f} \\')
            print(f'   {camera.name}_tf_yaw:={camera.yaw:.4f}')
        print()
    
    def publish_all_tf(self):
        """發布所有相機的 TF"""
        for camera in self.cameras:
            self.publish_camera_tf(camera)
    
    def publish_camera_tf(self, camera: CameraConfig):
        """發布單個相機的 TF"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = camera.parent_frame
        t.child_frame_id = camera.child_frame
        
        t.transform.translation.x = camera.x
        t.transform.translation.y = camera.y
        t.transform.translation.z = camera.z
        
        # RPY -> Quaternion
        cy, sy = math.cos(camera.yaw * 0.5), math.sin(camera.yaw * 0.5)
        cp, sp = math.cos(camera.pitch * 0.5), math.sin(camera.pitch * 0.5)
        cr, sr = math.cos(camera.roll * 0.5), math.sin(camera.roll * 0.5)
        
        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        
        self.tf_broadcaster.sendTransform(t)
    
    @property
    def current_camera(self) -> CameraConfig:
        """當前選中的相機"""
        return self.cameras[self.current_camera_idx]
    
    def select_camera(self, idx: int):
        """選擇相機"""
        if 0 <= idx < len(self.cameras):
            self.current_camera_idx = idx
            self.print_camera_status()
    
    def next_camera(self):
        """切換到下一個相機"""
        self.current_camera_idx = (self.current_camera_idx + 1) % len(self.cameras)
        self.print_camera_status()
    
    def print_help(self):
        """打印幫助"""
        help_text = """
╔══════════════════════════════════════════════════════════════════════╗
║                    多相機 TF 互動式校準工具                             ║
╠══════════════════════════════════════════════════════════════════════╣
║  相機切換:                                                            ║
║    TAB     - 切換到下一個相機                                          ║
║    1/2/3   - 直接選擇相機 (按編號)                                     ║
║                                                                      ║
║  位移控制 (米):                旋轉控制 (度):                           ║
║    W/S - X 軸 (前/後)           I/K - Pitch (上/下傾斜)                ║
║    A/D - Y 軸 (左/右)           J/L - Yaw (左/右轉)                    ║
║    Q/E - Z 軸 (上/下)           U/O - Roll (左/右滾)                   ║
║                                                                      ║
║  其他:                                                                ║
║    +/- - 調整步進大小             R - 重置當前相機為零                  ║
║    P   - 打印所有相機值           C - 保存所有校準值                    ║
║    H   - 顯示此幫助              ESC - 退出                            ║
╚══════════════════════════════════════════════════════════════════════╝
"""
        print(help_text)
    
    def print_camera_status(self):
        """打印當前相機狀態"""
        cam = self.current_camera
        print(f'\n🎥 當前相機: [{self.current_camera_idx + 1}/{len(self.cameras)}] {cam.name}')
        print(f'   Frame: {cam.parent_frame} → {cam.child_frame}')
        print(f'   位移: X={cam.x:+.3f}m  Y={cam.y:+.3f}m  Z={cam.z:+.3f}m')
        print(f'   旋轉: R={math.degrees(cam.roll):+.1f}°  P={math.degrees(cam.pitch):+.1f}°  Y={math.degrees(cam.yaw):+.1f}°')
    
    def print_all_cameras(self):
        """打印所有相機狀態"""
        print(f'\n📊 所有相機校準值:')
        print(f'   步進: 位移={self.translation_step*100:.1f}cm  旋轉={self.rotation_step:.1f}°')
        print()
        for i, cam in enumerate(self.cameras):
            marker = '→' if i == self.current_camera_idx else ' '
            print(f'   {marker} [{i+1}] {cam.name}')
            print(f'       Frame: {cam.parent_frame} → {cam.child_frame}')
            print(f'       位移: X={cam.x:+.3f}m  Y={cam.y:+.3f}m  Z={cam.z:+.3f}m')
            print(f'       旋轉: R={math.degrees(cam.roll):+.1f}°  P={math.degrees(cam.pitch):+.1f}°  Y={math.degrees(cam.yaw):+.1f}°')
            print()
    
    def handle_key(self, key: str):
        """處理鍵盤輸入"""
        cam = self.current_camera
        changed = False
        
        # 相機切換
        if key == '\t':  # TAB
            self.next_camera()
            return
        elif key in '123456789':
            idx = int(key) - 1
            if idx < len(self.cameras):
                self.select_camera(idx)
            return
        
        # 位移控制
        if key == 'w':
            cam.x += self.translation_step
            changed = True
        elif key == 's':
            cam.x -= self.translation_step
            changed = True
        elif key == 'a':
            cam.y += self.translation_step
            changed = True
        elif key == 'd':
            cam.y -= self.translation_step
            changed = True
        elif key == 'q':
            cam.z += self.translation_step
            changed = True
        elif key == 'e':
            cam.z -= self.translation_step
            changed = True
        
        # 旋轉控制
        elif key == 'i':
            cam.pitch += math.radians(self.rotation_step)
            changed = True
        elif key == 'k':
            cam.pitch -= math.radians(self.rotation_step)
            changed = True
        elif key == 'j':
            cam.yaw += math.radians(self.rotation_step)
            changed = True
        elif key == 'l':
            cam.yaw -= math.radians(self.rotation_step)
            changed = True
        elif key == 'u':
            cam.roll += math.radians(self.rotation_step)
            changed = True
        elif key == 'o':
            cam.roll -= math.radians(self.rotation_step)
            changed = True
        
        # 步進調整
        elif key == '+' or key == '=':
            self.translation_step *= 2
            self.rotation_step *= 2
            print(f'   步進增大: 位移={self.translation_step*100:.1f}cm  旋轉={self.rotation_step:.1f}°')
        elif key == '-':
            self.translation_step /= 2
            self.rotation_step /= 2
            print(f'   步進減小: 位移={self.translation_step*100:.1f}cm  旋轉={self.rotation_step:.1f}°')
        
        # 其他功能
        elif key == 'r':
            cam.x = cam.y = cam.z = 0.0
            cam.roll = cam.pitch = cam.yaw = 0.0
            print(f'   ⚠️ 已重置 {cam.name} 為零')
            changed = True
        elif key == 'p':
            self.print_all_cameras()
        elif key == 'c':
            self.save_all_calibrations()
        elif key == 'h':
            self.print_help()
        
        if changed:
            # 即時顯示更新
            sys.stdout.write(f'\r   [{cam.name}] X={cam.x:+.3f}  Y={cam.y:+.3f}  Z={cam.z:+.3f}  |  R={math.degrees(cam.roll):+.1f}°  P={math.degrees(cam.pitch):+.1f}°  Y={math.degrees(cam.yaw):+.1f}°     ')
            sys.stdout.flush()


def get_key(timeout=0.1):
    """非阻塞讀取鍵盤輸入"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
            return key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return None


def main(args=None):
    rclpy.init(args=args)
    
    calibrator = MultiCameraCalibrator()
    
    calibrator.print_help()
    calibrator.print_all_cameras()
    
    print('🎮 開始校準... (按 H 顯示幫助, TAB 切換相機, ESC 退出)\n')
    
    try:
        while rclpy.ok():
            # 處理 ROS 回調
            rclpy.spin_once(calibrator, timeout_sec=0.01)
            
            # 讀取鍵盤
            key = get_key(timeout=0.05)
            if key:
                if key == '\x1b':  # ESC
                    print('\n\n👋 退出校準工具')
                    break
                calibrator.handle_key(key.lower() if key.isalpha() else key)
    
    except KeyboardInterrupt:
        print('\n\n👋 退出校準工具')
    finally:
        calibrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

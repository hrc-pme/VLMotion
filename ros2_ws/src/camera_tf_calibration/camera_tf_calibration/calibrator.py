#!/usr/bin/env python3
"""
單相機 TF 校準工具

用於校準單個相機的 TF 變換
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


class CameraTFCalibrator(Node):
    """單相機 TF 校準器"""
    
    def __init__(self, camera_name='d435i', parent_frame='camera_bottom_screw_frame', 
                 child_frame='d435i_link'):
        super().__init__('camera_tf_calibrator')
        
        # 從參數獲取設定
        self.declare_parameter('camera_name', camera_name)
        self.declare_parameter('parent_frame', parent_frame)
        self.declare_parameter('child_frame', child_frame)
        
        self.camera_name = self.get_parameter('camera_name').value
        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value
        
        # TF 廣播器
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 當前校準值
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # 步進大小
        self.translation_step = 0.01
        self.rotation_step = 1.0
        
        # 配置文件
        self.config_dir = os.path.expanduser('~/.config/camera_tf_calibration')
        self.config_file = os.path.join(self.config_dir, f'{self.camera_name}_calibration.yaml')
        
        # 載入校準值
        self.load_calibration()
        
        # 定時發布 TF
        self.timer = self.create_timer(0.05, self.publish_tf)
        
    def load_calibration(self):
        """載入校準值"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                self.x = config.get('x', 0.0)
                self.y = config.get('y', 0.0)
                self.z = config.get('z', 0.0)
                self.roll = math.radians(config.get('roll_deg', 0.0))
                self.pitch = math.radians(config.get('pitch_deg', 0.0))
                self.yaw = math.radians(config.get('yaw_deg', 0.0))
                self.get_logger().info(f'已載入 {self.camera_name} 校準值')
            except Exception as e:
                self.get_logger().warn(f'載入校準值失敗: {e}')
    
    def save_calibration(self):
        """保存校準值"""
        os.makedirs(self.config_dir, exist_ok=True)
        
        config = {
            'camera_name': self.camera_name,
            'parent_frame': self.parent_frame,
            'child_frame': self.child_frame,
            'x': round(self.x, 4),
            'y': round(self.y, 4),
            'z': round(self.z, 4),
            'roll_deg': round(math.degrees(self.roll), 2),
            'pitch_deg': round(math.degrees(self.pitch), 2),
            'yaw_deg': round(math.degrees(self.yaw), 2),
            'roll_rad': round(self.roll, 4),
            'pitch_rad': round(self.pitch, 4),
            'yaw_rad': round(self.yaw, 4),
            'saved_at': datetime.now().isoformat(),
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        self.get_logger().info(f'校準值已保存: {self.config_file}')
        return config
    
    def publish_tf(self):
        """發布 TF"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z
        
        # RPY -> Quaternion
        cy, sy = math.cos(self.yaw * 0.5), math.sin(self.yaw * 0.5)
        cp, sp = math.cos(self.pitch * 0.5), math.sin(self.pitch * 0.5)
        cr, sr = math.cos(self.roll * 0.5), math.sin(self.roll * 0.5)
        
        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        
        self.tf_broadcaster.sendTransform(t)
    
    def get_values(self):
        """獲取當前值"""
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw
        }
    
    def set_values(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None):
        """設置值"""
        if x is not None: self.x = x
        if y is not None: self.y = y
        if z is not None: self.z = z
        if roll is not None: self.roll = roll
        if pitch is not None: self.pitch = pitch
        if yaw is not None: self.yaw = yaw


def main(args=None):
    """單相機校準主程式"""
    rclpy.init(args=args)
    calibrator = CameraTFCalibrator()
    
    print(f'\n單相機校準工具已啟動')
    print(f'相機: {calibrator.camera_name}')
    print(f'Frame: {calibrator.parent_frame} → {calibrator.child_frame}')
    print(f'\n請使用 multi_calibrator 進行互動式校準')
    
    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        pass
    finally:
        calibrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

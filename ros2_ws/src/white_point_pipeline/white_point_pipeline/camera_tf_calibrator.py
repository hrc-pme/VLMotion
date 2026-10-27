#!/usr/bin/env python3
"""
互動式相機 TF 校準工具

功能：
1. 即時調整 D435i 相機的 TF 變換 (XYZ + RPY)
2. 位移和旋轉分開調整，避免互相影響
3. 可保存校準值到 YAML 配置文件
4. 支持鍵盤快捷鍵快速調整

使用方式：
    ros2 run white_point_pipeline camera_tf_calibrator

鍵盤控制：
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
        R   - 重置為零
        P   - 打印當前值
        C   - 保存到配置文件
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


class CameraTFCalibrator(Node):
    def __init__(self):
        super().__init__('camera_tf_calibrator')
        
        # TF 廣播器
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 當前校準值
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0    # 弧度
        self.pitch = 0.0   # 弧度
        self.yaw = 0.0     # 弧度
        
        # 步進大小
        self.translation_step = 0.01  # 1 cm
        self.rotation_step = 1.0      # 1 度
        
        # Frame 設定
        self.parent_frame = 'camera_bottom_screw_frame'
        self.child_frame = 'd435i_link'
        
        # 配置文件路徑
        self.config_dir = os.path.expanduser('~/.config/white_point_pipeline')
        self.config_file = os.path.join(self.config_dir, 'camera_tf_calibration.yaml')
        
        # 嘗試載入已有的校準值
        self.load_calibration()
        
        # 定時發布 TF
        self.timer = self.create_timer(0.05, self.publish_tf)  # 20 Hz
        
        # 打印說明
        self.print_help()
        self.print_current_values()
        
    def load_calibration(self):
        """載入已保存的校準值"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        self.x = config.get('x', 0.0)
                        self.y = config.get('y', 0.0)
                        self.z = config.get('z', 0.0)
                        self.roll = math.radians(config.get('roll_deg', 0.0))
                        self.pitch = math.radians(config.get('pitch_deg', 0.0))
                        self.yaw = math.radians(config.get('yaw_deg', 0.0))
                        self.get_logger().info(f'已載入校準值: {self.config_file}')
            except Exception as e:
                self.get_logger().warn(f'載入校準值失敗: {e}')
    
    def save_calibration(self):
        """保存校準值到配置文件"""
        os.makedirs(self.config_dir, exist_ok=True)
        
        config = {
            'x': round(self.x, 4),
            'y': round(self.y, 4),
            'z': round(self.z, 4),
            'roll_deg': round(math.degrees(self.roll), 2),
            'pitch_deg': round(math.degrees(self.pitch), 2),
            'yaw_deg': round(math.degrees(self.yaw), 2),
            'roll_rad': round(self.roll, 4),
            'pitch_rad': round(self.pitch, 4),
            'yaw_rad': round(self.yaw, 4),
            'parent_frame': self.parent_frame,
            'child_frame': self.child_frame,
            'saved_at': datetime.now().isoformat(),
        }
        
        # 生成 launch 文件參數
        config['launch_args'] = (
            f"camera_tf_x:={config['x']} "
            f"camera_tf_y:={config['y']} "
            f"camera_tf_z:={config['z']} "
            f"camera_tf_roll:={config['roll_rad']} "
            f"camera_tf_pitch:={config['pitch_rad']} "
            f"camera_tf_yaw:={config['yaw_rad']}"
        )
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        self.get_logger().info(f'校準值已保存: {self.config_file}')
        print(f'\n✅ 校準值已保存到: {self.config_file}')
        print(f'\n📋 Launch 參數 (複製到命令行):\n')
        print(f'   {config["launch_args"]}')
        print()
        
    def publish_tf(self):
        """發布當前的 TF 變換"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        
        # 位置
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z
        
        # 旋轉 (RPY -> Quaternion)
        cy = math.cos(self.yaw * 0.5)
        sy = math.sin(self.yaw * 0.5)
        cp = math.cos(self.pitch * 0.5)
        sp = math.sin(self.pitch * 0.5)
        cr = math.cos(self.roll * 0.5)
        sr = math.sin(self.roll * 0.5)
        
        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        
        self.tf_broadcaster.sendTransform(t)
    
    def print_help(self):
        """打印幫助信息"""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║             相機 TF 互動式校準工具                              ║
╠══════════════════════════════════════════════════════════════╣
║  位移控制 (米):                旋轉控制 (度):                   ║
║    W/S - X 軸 (前/後)           I/K - Pitch (上/下傾斜)        ║
║    A/D - Y 軸 (左/右)           J/L - Yaw (左/右轉)            ║
║    Q/E - Z 軸 (上/下)           U/O - Roll (左/右滾)           ║
║                                                              ║
║  其他:                                                        ║
║    +/- - 調整步進大小                                          ║
║    R   - 重置所有值為零                                        ║
║    P   - 打印當前值                                            ║
║    C   - 保存到配置文件                                        ║
║    H   - 顯示此幫助                                            ║
║    ESC - 退出程式                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(help_text)
    
    def print_current_values(self):
        """打印當前校準值"""
        print(f'\n📐 當前校準值:')
        print(f'   位移: X={self.x:.3f}m  Y={self.y:.3f}m  Z={self.z:.3f}m')
        print(f'   旋轉: R={math.degrees(self.roll):.1f}°  P={math.degrees(self.pitch):.1f}°  Y={math.degrees(self.yaw):.1f}°')
        print(f'   步進: 位移={self.translation_step*100:.1f}cm  旋轉={self.rotation_step:.1f}°')
        print(f'   Frame: {self.parent_frame} → {self.child_frame}')
    
    def handle_key(self, key):
        """處理鍵盤輸入"""
        changed = False
        
        # 位移控制
        if key == 'w':
            self.x += self.translation_step
            changed = True
        elif key == 's':
            self.x -= self.translation_step
            changed = True
        elif key == 'a':
            self.y += self.translation_step
            changed = True
        elif key == 'd':
            self.y -= self.translation_step
            changed = True
        elif key == 'q':
            self.z += self.translation_step
            changed = True
        elif key == 'e':
            self.z -= self.translation_step
            changed = True
        
        # 旋轉控制
        elif key == 'i':
            self.pitch += math.radians(self.rotation_step)
            changed = True
        elif key == 'k':
            self.pitch -= math.radians(self.rotation_step)
            changed = True
        elif key == 'j':
            self.yaw += math.radians(self.rotation_step)
            changed = True
        elif key == 'l':
            self.yaw -= math.radians(self.rotation_step)
            changed = True
        elif key == 'u':
            self.roll += math.radians(self.rotation_step)
            changed = True
        elif key == 'o':
            self.roll -= math.radians(self.rotation_step)
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
            self.x = self.y = self.z = 0.0
            self.roll = self.pitch = self.yaw = 0.0
            print('   ⚠️ 已重置所有值為零')
            changed = True
        elif key == 'p':
            self.print_current_values()
        elif key == 'c':
            self.save_calibration()
        elif key == 'h':
            self.print_help()
        
        if changed:
            # 即時顯示更新
            sys.stdout.write(f'\r   X={self.x:+.3f}  Y={self.y:+.3f}  Z={self.z:+.3f}  |  R={math.degrees(self.roll):+.1f}°  P={math.degrees(self.pitch):+.1f}°  Y={math.degrees(self.yaw):+.1f}°     ')
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
    
    calibrator = CameraTFCalibrator()
    
    print('\n🎮 開始校準... (按 H 顯示幫助, ESC 退出)\n')
    
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
                calibrator.handle_key(key.lower())
    
    except KeyboardInterrupt:
        print('\n\n👋 退出校準工具')
    finally:
        calibrator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

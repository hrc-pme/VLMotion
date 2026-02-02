#!/usr/bin/env python3
"""
相機標定偏移補償工具

用途：如果紅色點在真實世界中有系統性偏移，可以在這裡設置補償值
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

class CalibrationOffset(Node):
    def __init__(self):
        super().__init__('calibration_offset')
        
        # 聲明參數：允許從 launch 文件或命令行設置
        self.declare_parameter('offset_x', 0.0)  # base_link X 軸偏移（米）
        self.declare_parameter('offset_y', 0.0)  # base_link Y 軸偏移（米）
        self.declare_parameter('offset_z', 0.0)  # base_link Z 軸偏移（米）
        
        # 讀取參數
        self.offset_x = self.get_parameter('offset_x').value
        self.offset_y = self.get_parameter('offset_y').value
        self.offset_z = self.get_parameter('offset_z').value
        
        # 訂閱原始白點（從 white_point_to_3d 發布）
        self.raw_point_sub = self.create_subscription(
            PointStamped,
            '/white_point_base_raw',  # 重命名原始 topic
            self.point_callback,
            10
        )
        
        # 發布校正後的白點
        self.corrected_point_pub = self.create_publisher(
            PointStamped,
            '/white_point_base',  # 使用原來的名稱
            10
        )
        
        self.get_logger().info(
            f'Calibration offset initialized: '
            f'X={self.offset_x:.3f}m, Y={self.offset_y:.3f}m, Z={self.offset_z:.3f}m'
        )
    
    def point_callback(self, msg: PointStamped):
        """應用校正偏移"""
        corrected = PointStamped()
        corrected.header = msg.header
        corrected.point.x = msg.point.x + self.offset_x
        corrected.point.y = msg.point.y + self.offset_y
        corrected.point.z = msg.point.z + self.offset_z
        
        self.corrected_point_pub.publish(corrected)
        
        self.get_logger().info(
            f'Original: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f}) -> '
            f'Corrected: ({corrected.point.x:.3f}, {corrected.point.y:.3f}, {corrected.point.z:.3f})',
            throttle_duration_sec=1.0
        )

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationOffset()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

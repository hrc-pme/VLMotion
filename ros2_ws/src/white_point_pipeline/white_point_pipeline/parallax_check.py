#!/usr/bin/env python3
"""
診斷工具：驗證深度影像計算的 3D 點座標
目的：確認 TF、相機內參、深度影像計算是否正確

Usage:
    ros2 run white_point_pipeline parallax_check
    
說明：
    - 從深度影像 + 相機內參計算 3D 點（光學座標系）
    - 通過 TF 轉換到 base_link
    - 驗證計算流程是否正確
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point


class ParallaxCheck(Node):
    def __init__(self):
        super().__init__('parallax_check')
        self.bridge = CvBridge()
        
        # 訂閱
        self.depth_sub = self.create_subscription(
            Image, '/d435i/aligned_depth_to_color/image_raw',
            self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/d435i/color/camera_info',
            self.info_callback, 10)
        self.pixel_sub = self.create_subscription(
            Point, '/white_point_pixel',
            self.pixel_callback, 10)
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 資料
        self.cam_K = None
        self.depth_image = None
        
        self.get_logger().info('ParallaxCheck started - click in GUI to check 3D point calculation')
    
    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg)
    
    def info_callback(self, msg):
        self.cam_K = np.array(msg.k).reshape(3, 3)
    
    def pixel_callback(self, msg):
        if self.cam_K is None or self.depth_image is None:
            self.get_logger().warn('Missing camera info or depth image')
            return
        
        H, W = self.depth_image.shape[:2]
        
        # GUI 旋轉後的座標 → 逆旋轉
        u_rot = int(msg.x)
        v_rot = int(msg.y)
        u = v_rot
        v = H - 1 - u_rot
        
        self.get_logger().info(f'\n{"="*60}')
        self.get_logger().info(f'INPUT: GUI pixel ({u_rot}, {v_rot}) → Original ({u}, {v})')
        
        # ========================================
        # 從深度影像計算 3D 點
        # ========================================
        depth_raw = float(self.depth_image[v, u])
        if depth_raw <= 0:
            self.get_logger().warn(f'Invalid depth at ({u}, {v})')
            return
        
        depth_m = depth_raw / 1000.0 if self.depth_image.dtype == np.uint16 else depth_raw
        
        fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
        cx, cy = self.cam_K[0, 2], self.cam_K[1, 2]
        
        # 光學座標系（相機原點為中心，Z 軸向前）
        Xo = (u - cx) * depth_m / fx
        Yo = (v - cy) * depth_m / fy
        Zo = depth_m
        
        self.get_logger().info(f'CALCULATED (optical frame): X={Xo:.4f}, Y={Yo:.4f}, Z={Zo:.4f}')
        self.get_logger().info(f'  → 深度: {depth_m:.3f}m ({depth_raw:.0f} raw)')
        self.get_logger().info(f'  → 內參: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}')
        
        # ========================================
        # 轉換到 base_link（機器人座標系）
        # ========================================
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', 'd435i_color_optical_frame',
                rclpy.time.Time())
            
            pt_cam = PointStamped()
            pt_cam.header.frame_id = 'd435i_color_optical_frame'
            pt_cam.point.x = Xo
            pt_cam.point.y = Yo
            pt_cam.point.z = Zo
            
            pt_base = do_transform_point(pt_cam, tf)
            
            self.get_logger().info(
                f'BASE_LINK: X={pt_base.point.x:.4f}, Y={pt_base.point.y:.4f}, Z={pt_base.point.z:.4f}'
            )
            
            # 顯示 TF 資訊
            t = tf.transform.translation
            r = tf.transform.rotation
            self.get_logger().info(
                f'  → TF (optical→base): t=[{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]'
            )
            self.get_logger().info(
                f'                       r=[{r.x:.3f}, {r.y:.3f}, {r.z:.3f}, {r.w:.3f}]'
            )
            
            # 檢查合理性
            if pt_base.point.z < 0:
                self.get_logger().warn('⚠️ 點在地面以下！檢查 TF 或深度數據')
            elif pt_base.point.x < 0.3:
                self.get_logger().warn('⚠️ 點太靠近機器人！可能超出工作範圍')
            else:
                self.get_logger().info('✅ 3D 座標看起來合理')
                
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
        
        self.get_logger().info('='*60)


def main(args=None):
    rclpy.init(args=args)
    node = ParallaxCheck()
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

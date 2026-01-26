#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time as RclpyTime

# ----------------------------------------------------------
#   9 點補插深度（在「未旋轉」的 depth image 上）
# ----------------------------------------------------------
def get_valid_depth(depth_img, cx, cy):
    h, w = depth_img.shape[:2]
    cx_i = int(np.clip(round(cx), 0, w - 1))
    cy_i = int(np.clip(round(cy), 0, h - 1))

    z = float(depth_img[cy_i, cx_i])
    if z > 0:
        return z

    zs = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx = int(np.clip(cx_i + dx, 0, w - 1))
            ny = int(np.clip(cy_i + dy, 0, h - 1))
            v = float(depth_img[ny, nx])
            if v > 0:
                zs.append(v)

    return float(np.mean(zs)) if zs else 0.0


class WhitePointTo3D(Node):
    def __init__(self):
        super().__init__('white_point_to_3d')
        self.bridge = CvBridge()

        # 對齊到 color 的深度 + color 內參
        self.depth_sub = self.create_subscription(
            Image,
            '/d435i/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/d435i/color/camera_info',
            self.info_callback,
            10
        )

        # GUI 點的 pixel（已經是旋轉 90° 後的座標）
        self.pixel_sub = self.create_subscription(
            Point,
            '/white_point_pixel',
            self.pixel_callback,
            10
        )

        # 發布 3D 點（base_link frame）
        self.point_pub = self.create_publisher(
            PointStamped,
            '/white_point_base',
            10
        )

        # TF Buffer + Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cam_K = None
        self.depth_image = None
        self.depth_msg_header = None

    # ------------------------------------------------------
    # CameraInfo callback：只存 K matrix
    # ------------------------------------------------------
    def info_callback(self, msg: CameraInfo):
        self.cam_K = np.array(msg.k).reshape(3, 3)
        # msg.header.frame_id 是 d435i_color_optical_frame
        # 但 TF tree 用的是 camera_color_optical_frame
        # 所以等一下我們會手動指定 frame_id = "camera_color_optical_frame"

    # ------------------------------------------------------
    # Depth image callback：未旋轉的 aligned depth
    # ------------------------------------------------------
    def depth_callback(self, msg: Image):
        self.depth_msg_header = msg.header
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        self.depth_image = cv_image

    # ------------------------------------------------------
    # Pixel callback：把 GUI 旋轉後的 (u_rot, v_rot) 轉回原始 (u, v)
    # ------------------------------------------------------
    def pixel_callback(self, msg: Point):
        if self.cam_K is None or self.depth_image is None:
            self.get_logger().warn('No camera intrinsics or depth image yet.')
            return

        # 未旋轉 depth 圖大小（例如 720x1280）
        H = self.depth_image.shape[0]
        W = self.depth_image.shape[1]

        # GUI 顯示的 pixel（已旋轉：cv2.ROTATE_90_CLOCKWISE）
        u_rot = int(msg.x)
        v_rot = int(msg.y)

        # 逆旋轉對應：原圖 (u, v) -> 旋轉後 (u_r, v_r) = (H-1-v, u)
        # 反推：u = v_r, v = H-1-u_r
        u = v_rot
        v = H - 1 - u_rot

        if v < 0 or v >= H or u < 0 or u >= W:
            self.get_logger().warn(f'Pixel out of range after unrotate: u={u}, v={v}')
            return

        # 9 點補插深度（回傳的是原始 depth buffer 數值）
        depth_raw = get_valid_depth(self.depth_image, u, v)
        if depth_raw <= 0.0 or np.isnan(depth_raw) or np.isinf(depth_raw):
            self.get_logger().warn(f'Invalid depth at ({u},{v}): {depth_raw}')
            return

        # 依據 encoding 決定單位：16UC1 (mm) 或 32FC1 (m)
        if self.depth_image.dtype == np.uint16:
            depth_m = float(depth_raw) / 1000.0
        else:
            depth_m = float(depth_raw)

        fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
        cx, cy = self.cam_K[0, 2], self.cam_K[1, 2]

        # ---------------- 光學座標系 (optical frame) ----------------
        # ROS optical frame: x 向右, y 向下, z 向前
        Xo = (u - cx) * depth_m / fx   # right
        Yo = (v - cy) * depth_m / fy   # down
        Zo = depth_m                   # forward

        self.get_logger().info(
            f'OPTICAL frame: Xo={Xo:.3f}, Yo={Yo:.3f}, Zo={Zo:.3f}'
        )

        # 這個點放在 TF 裡有的 frame：camera_color_optical_frame
        pt_cam = PointStamped()
        pt_cam.header.stamp = self.get_clock().now().to_msg()
        pt_cam.header.frame_id = 'camera_color_optical_frame'
        pt_cam.point.x = Xo
        pt_cam.point.y = Yo
        pt_cam.point.z = Zo

        try:
            # 從 optical frame 轉到 base_link
            tf = self.tf_buffer.lookup_transform(
                'base_link',                 # 目標座標系
                'camera_color_optical_frame',# 來源座標系
                RclpyTime()
            )
            pt_base = do_transform_point(pt_cam, tf)

            Xb = pt_base.point.x
            Yb = pt_base.point.y
            Zb = pt_base.point.z

            self.point_pub.publish(pt_base)
            self.get_logger().info(
                f'BASE frame: Xb={Xb:.3f}, Yb={Yb:.3f}, Zb={Zb:.3f}'
            )
        except Exception as e:
            self.get_logger().warn(f'Failed to transform point: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = WhitePointTo3D()
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

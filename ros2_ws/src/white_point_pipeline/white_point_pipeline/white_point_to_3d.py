#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Float32, Header
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import struct
import numpy as np
import math
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time as RclpyTime
from sklearn.cluster import DBSCAN

# ----------------------------------------------------------
#   擴展搜尋深度補插（在「未旋轉」的 depth image 上）
#
#   策略：以目標像素為中心，由內而外逐環擴大搜尋（最多 max_r 像素）。
#   找到第一個含有效深度的環時，取該環所有有效值的中位數回傳。
#   這解決了目標位於深度空洞（反射面、遮擋邊緣等）時無法取得深度的問題。
# ----------------------------------------------------------
def get_valid_depth(depth_img, cx, cy, max_r=20):
    h, w = depth_img.shape[:2]
    cx_i = int(np.clip(round(cx), 0, w - 1))
    cy_i = int(np.clip(round(cy), 0, h - 1))

    # 先試中心點
    z = float(depth_img[cy_i, cx_i])
    if z > 0:
        return z

    # 由內而外逐環擴展搜尋
    for r in range(1, max_r + 1):
        zs = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                # 只取當前環的邊界像素，跳過已搜尋過的內圈
                if abs(dx) < r and abs(dy) < r:
                    continue
                nx = int(np.clip(cx_i + dx, 0, w - 1))
                ny = int(np.clip(cy_i + dy, 0, h - 1))
                v = float(depth_img[ny, nx])
                if v > 0:
                    zs.append(v)
        if zs:
            return float(np.median(zs))

    return 0.0


class WhitePointTo3D(Node):
    def __init__(self):
        super().__init__('white_point_to_3d')
        self.bridge = CvBridge()

        # 可配置相機 topic 和 frame（由 launch 檔根據 CAMERA 變數自動設定）
        self.declare_parameter('depth_topic', '')
        self.declare_parameter('camera_info_topic', '')
        self.declare_parameter('camera_frame', '')

        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        self.get_logger().info(f'Camera depth topic: {depth_topic}')
        self.get_logger().info(f'Camera info topic: {camera_info_topic}')
        self.get_logger().info(f'Camera frame: {self.camera_frame}')

        # 對齊到 color 的深度 + color 內參
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
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

        # 發布面板軸向（base_link 下的 yaw 角度）
        self.axis_pub = self.create_publisher(
            Float32,
            '/panel_axis_base',
            10
        )

        # 發布面板座標軸可視化
        self.axis_marker_pub = self.create_publisher(
            Marker,
            '/panel_axis_marker',
            10
        )
        
        # 發布 DBSCAN 處理前的點雲（藍色）
        self.raw_points_pub = self.create_publisher(
            PointCloud2,
            '/dbscan_raw_points',
            10
        )
        
        # 發布 DBSCAN 聚類後的牆面點雲（綠色）
        self.wall_points_pub = self.create_publisher(
            PointCloud2,
            '/dbscan_wall_points',
            10
        )
        
        # 發布牆面法向量 marker（紅色箭頭）
        self.normal_marker_pub = self.create_publisher(
            Marker,
            '/wall_normal_marker',
            10
        )

        # TF Buffer + Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cam_K = None
        self.depth_image = None
        self.depth_msg_header = None

        # 面板朝向計算參數
        self.patch_r = 30  # patch 半徑（像素），61x61 區域（擴大範圍以獲得更準確的法線）
        self.min_plane_pts = 100  # 平面擬合最少點數
        
        # DBSCAN 參數
        self.dbscan_eps = 0.02  # 鄰域半徑（米），2cm
        self.dbscan_min_samples = 20  # 核心點最少鄰居數

    # ------------------------------------------------------
    # CameraInfo callback：只存 K matrix
    # ------------------------------------------------------
    def info_callback(self, msg: CameraInfo):
        self.cam_K = np.array(msg.k).reshape(3, 3)
        # 使用 launch 設定的 camera_frame 以確保跟著相機旋轉

    # ------------------------------------------------------
    # Depth image callback：未旋轉的 aligned depth
    # ------------------------------------------------------
    def depth_callback(self, msg: Image):
        self.depth_msg_header = msg.header
        cv_image = self.bridge.imgmsg_to_cv2(msg)
        self.depth_image = cv_image

    def backproject_optical(self, u, v, depth_m):
        """將像素座標和深度回投影到光學座標系"""
        fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
        cx, cy = self.cam_K[0, 2], self.cam_K[1, 2]
        Xo = (u - cx) * depth_m / fx
        Yo = (v - cy) * depth_m / fy
        Zo = depth_m
        return np.array([Xo, Yo, Zo], dtype=np.float64)

    def fit_plane_normal_pca(self, pts_xyz):
        """
        用 PCA 擬合垂直牆面的法向量
        假設：牆面是垂直的，法向量在 XY 平面上
        pts_xyz: (N,3) numpy array in base_link
        return: unit normal (3,) 在 XY 平面上
        """
        if len(pts_xyz) < 3:
            return np.array([1.0, 0.0, 0.0])
        
        # 垂直牆面假設：只使用 XY 座標進行 PCA
        # 這樣可以避免因為相機旋轉導致的 Z 方向梯度影響法向量計算
        pts_xy = pts_xyz[:, :2]  # 只使用 X, Y 座標
        
        # 計算 XY 平面的點雲範圍
        xy_min = pts_xy.min(axis=0)
        xy_max = pts_xy.max(axis=0)
        xy_range = xy_max - xy_min
        
        self.get_logger().info(
            f'Point cloud XY range: X=[{xy_min[0]:.3f}, {xy_max[0]:.3f}] ({xy_range[0]:.3f}m), '
            f'Y=[{xy_min[1]:.3f}, {xy_max[1]:.3f}] ({xy_range[1]:.3f}m)'
        )
        
        c_xy = pts_xy.mean(axis=0)
        Q_xy = pts_xy - c_xy
        C_xy = Q_xy.T @ Q_xy
        w, V = np.linalg.eigh(C_xy)
        
        # 在 2D PCA 中：
        # - 最大特徵值（w[1]）對應的方向（V[:, 1]）是點雲的主方向（沿著牆面）
        # - 最小特徵值（w[0]）對應的方向（V[:, 0]）是垂直於主方向的（牆面法向量）
        
        main_direction = V[:, 1]  # 最大特徵值對應的方向（沿牆面）
        n_xy = V[:, 0]  # 最小特徵值對應的方向（法向量）
        
        self.get_logger().info(
            f'PCA eigenvalues: λ1={w[0]:.6f} (normal), λ2={w[1]:.6f} (tangent), ratio={w[1]/max(w[0], 1e-9):.2f}'
        )
        self.get_logger().info(
            f'Main direction (along wall): [{main_direction[0]:.3f}, {main_direction[1]:.3f}]'
        )
        
        n_xy = n_xy / (np.linalg.norm(n_xy) + 1e-9)
        
        # 確保法向量指向機器人（負 X 方向，約 180°）
        if n_xy[0] > 0:
            n_xy = -n_xy
        
        # 記錄法向量角度
        pca_angle = math.degrees(math.atan2(n_xy[1], n_xy[0]))
        robot_forward_angle = 180.0  # 機器人前方 (-X) 對應 180°
        angle_deviation = pca_angle - robot_forward_angle
        # 標準化到 [-180, 180]
        if angle_deviation > 180.0:
            angle_deviation -= 360.0
        elif angle_deviation < -180.0:
            angle_deviation += 360.0
        
        self.get_logger().info(
            f'Wall normal angle: {pca_angle:.1f}° (deviation from robot forward: {angle_deviation:.1f}°)'
        )
        
        # 構建 3D 法向量，Z 分量為 0（完全水平）
        n = np.array([n_xy[0], n_xy[1], 0.0], dtype=np.float64)
        
        self.get_logger().info(
            f'Final wall normal (XY-plane): [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]'
        )
            
        return n
    
    def cluster_wall_points_dbscan(self, pts_xyz):
        """
        使用 DBSCAN 聚類找出最大的牆面點雲群
        假設：牆面是垂直的，只在 XY 平面進行聚類
        pts_xyz: (N,3) numpy array in base_link
        return: filtered points (M,3) 或 None
        """
        if len(pts_xyz) < self.dbscan_min_samples:
            self.get_logger().warn(f'Too few points for DBSCAN: {len(pts_xyz)}')
            return None
        
        # 垂直牆面假設：只在 XY 平面進行聚類（忽略 Z 方向）
        # 這樣可以避免因為相機角度導致的 Z 方向梯度影響聚類
        pts_xy = pts_xyz[:, :2]  # 只使用 X, Y 座標
        
        # 使用 DBSCAN 聚類（只在 XY 平面）
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = clustering.fit_predict(pts_xy)
        
        # 找出最大的群（排除噪聲點 label=-1）
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        if len(unique_labels) == 0:
            self.get_logger().warn('DBSCAN found no clusters')
            return None
        
        # 選擇點數最多的群
        largest_cluster_label = None
        largest_cluster_size = 0
        
        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            if cluster_size > largest_cluster_size:
                largest_cluster_size = cluster_size
                largest_cluster_label = label
        
        # 提取最大群的點（保留完整的 XYZ 座標）
        cluster_mask = labels == largest_cluster_label
        cluster_points = pts_xyz[cluster_mask]
        
        self.get_logger().info(
            f'DBSCAN (XY-plane): {len(unique_labels)} clusters, '
            f'largest has {largest_cluster_size} points '
            f'({100.0*largest_cluster_size/len(pts_xyz):.1f}%)'
        )
        
        return cluster_points

    def wrap_pi(self, a):
        """將角度包裹到 [-π, π]"""
        return (a + np.pi) % (2 * np.pi) - np.pi
    
    def project_point_to_plane(self, point, plane_normal, plane_point):
        """
        將點投影到平面上（深度校正）
        
        平面方程：n·(r - r0) = 0
        投影公式：p_proj = p - [(p - r0)·n] × n
        
        這個函數用於校正深度相機的角度相關誤差：
        - 深度相機在斜角度觀察時，測量值會偏離真實表面
        - 通過將點投影到擬合的牆面平面上，可以獲得更準確的位置
        
        Args:
            point: 要投影的點 (x, y, z)
            plane_normal: 平面法向量 (nx, ny, nz)，應已歸一化
            plane_point: 平面上的參考點 (x0, y0, z0)
        
        Returns:
            投影後的點 (x, y, z) numpy array
        """
        p = np.array(point, dtype=np.float64)
        n = np.array(plane_normal, dtype=np.float64)
        r0 = np.array(plane_point, dtype=np.float64)
        
        # 確保法向量歸一化
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            self.get_logger().warn('Plane normal has zero magnitude, cannot project')
            return p
        n = n / n_norm
        
        # 計算點到平面的有向距離
        dist = np.dot(p - r0, n)
        
        # 投影到平面
        p_proj = p - dist * n
        
        return p_proj

    def create_pointcloud2(self, points_xyz, frame_id, stamp):
        """
        將 numpy array (N,3) 轉換為 PointCloud2 訊息
        points_xyz: (N,3) numpy array
        """
        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        cloud_data = []
        for point in points_xyz:
            cloud_data.append(struct.pack('fff', float(point[0]), float(point[1]), float(point[2])))
        
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points_xyz)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True
        cloud_msg.data = b''.join(cloud_data)
        
        return cloud_msg
    
    def create_normal_marker(self, position, normal_vector, marker_id=0):
        """
        創建法向量的可視化 marker（紅色箭頭）
        position: (x, y, z) 起點位置
        normal_vector: (nx, ny, nz) 法向量方向（已歸一化）
        """
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'wall_normal'
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 設定箭頭的起點和終點
        start = Point()
        start.x = float(position[0])
        start.y = float(position[1])
        start.z = float(position[2])
        
        # 法向量長度 0.3 米
        arrow_length = 0.3
        end = Point()
        end.x = start.x + float(normal_vector[0]) * arrow_length
        end.y = start.y + float(normal_vector[1]) * arrow_length
        end.z = start.z + float(normal_vector[2]) * arrow_length
        
        marker.points.append(start)
        marker.points.append(end)
        
        # 箭頭大小
        marker.scale.x = 0.02  # 箭頭軸直徑
        marker.scale.y = 0.04  # 箭頭頭部直徑
        marker.scale.z = 0.06  # 箭頭頭部長度
        
        # 紅色
        from std_msgs.msg import ColorRGBA
        marker.color = ColorRGBA()
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        return marker

    def create_axis_marker(self, position, theta_axis):
        """
        創建面板座標軸的可視化 marker
        position: (x, y, z) 按鈕位置
        theta_axis: 面板軸向角度（弧度）
        """
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'panel_axes'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD

        marker.scale.x = 0.01
        marker.pose.orientation.w = 1.0

        axis_length = 0.2

        origin = Point()
        origin.x = float(position[0])
        origin.y = float(position[1])
        origin.z = float(position[2])

        x_dir = np.array([
            math.cos(theta_axis),
            math.sin(theta_axis),
            0.0
        ])

        y_dir = np.array([
            -math.sin(theta_axis),
            math.cos(theta_axis),
            0.0
        ])

        z_dir = np.array([0.0, 0.0, 1.0])

        x_end = Point()
        x_end.x = origin.x + x_dir[0] * axis_length
        x_end.y = origin.y + x_dir[1] * axis_length
        x_end.z = origin.z + x_dir[2] * axis_length

        y_end = Point()
        y_end.x = origin.x + y_dir[0] * axis_length
        y_end.y = origin.y + y_dir[1] * axis_length
        y_end.z = origin.z + y_dir[2] * axis_length

        z_end = Point()
        z_end.x = origin.x + z_dir[0] * axis_length
        z_end.y = origin.y + z_dir[1] * axis_length
        z_end.z = origin.z + z_dir[2] * axis_length

        from std_msgs.msg import ColorRGBA

        marker.points.append(origin)
        marker.points.append(x_end)
        red = ColorRGBA()
        red.r = 1.0
        red.a = 1.0
        marker.colors.append(red)
        marker.colors.append(red)

        marker.points.append(origin)
        marker.points.append(y_end)
        green = ColorRGBA()
        green.g = 1.0
        green.a = 1.0
        marker.colors.append(green)
        marker.colors.append(green)

        marker.points.append(origin)
        marker.points.append(z_end)
        blue = ColorRGBA()
        blue.b = 1.0
        blue.a = 1.0
        marker.colors.append(blue)
        marker.colors.append(blue)

        return marker

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

        # 使用 launch 設定的 color optical frame（會跟著相機旋轉）
        pt_cam = PointStamped()
        pt_cam.header.stamp = self.get_clock().now().to_msg()
        pt_cam.header.frame_id = self.camera_frame
        pt_cam.point.x = Xo
        pt_cam.point.y = Yo
        pt_cam.point.z = Zo

        try:
            # 直接從相機光學座標系轉到 base_link
            # TF 系統會自動處理所有中間的轉換
            tf = self.tf_buffer.lookup_transform(
                'base_link',
                self.camera_frame,
                RclpyTime()
            )
            pt_base = do_transform_point(pt_cam, tf)

            self.point_pub.publish(pt_base)
            self.get_logger().info(
                f'BASE frame: Xb={pt_base.point.x:.3f}, '
                f'Yb={pt_base.point.y:.3f}, Zb={pt_base.point.z:.3f}'
            )

            # =============================
            # 計算面板朝向（平面法向）
            # 策略：採樣大範圍像素，用 DBSCAN 過濾出牆面主體
            # =============================
            pts_base = []
            r = self.patch_r
            
            for dv in range(-r, r + 1):
                for du in range(-r, r + 1):
                    uu = u + du
                    vv = v + dv
                    if uu < 0 or uu >= W or vv < 0 or vv >= H:
                        continue

                    d_raw = float(self.depth_image[vv, uu])
                    if d_raw <= 0:
                        continue

                    if self.depth_image.dtype == np.uint16:
                        dm = d_raw / 1000.0
                    else:
                        dm = d_raw

                    if dm < 0.2 or dm > 3.0:
                        continue

                    p_opt = self.backproject_optical(uu, vv, dm)

                    tmp = PointStamped()
                    tmp.header.frame_id = self.camera_frame
                    tmp.point.x = float(p_opt[0])
                    tmp.point.y = float(p_opt[1])
                    tmp.point.z = float(p_opt[2])
                    tmp_base = do_transform_point(tmp, tf)
                    
                    pts_base.append([tmp_base.point.x, tmp_base.point.y, tmp_base.point.z])

            if len(pts_base) >= self.min_plane_pts:
                pts_base = np.array(pts_base, dtype=np.float64)
                
                # 發布 DBSCAN 處理前的原始點雲（藍色）
                raw_cloud_msg = self.create_pointcloud2(
                    pts_base,
                    'base_link',
                    self.get_clock().now().to_msg()
                )
                self.raw_points_pub.publish(raw_cloud_msg)
                self.get_logger().info(f'Published {len(pts_base)} raw points for visualization')
                
                # 使用 DBSCAN 聚類找出牆面主體
                wall_points = self.cluster_wall_points_dbscan(pts_base)
                
                if wall_points is not None and wall_points.shape[0] >= self.min_plane_pts:
                    # 發布 DBSCAN 聚類後的牆面點雲（綠色）
                    wall_cloud_msg = self.create_pointcloud2(
                        wall_points,
                        'base_link',
                        self.get_clock().now().to_msg()
                    )
                    self.wall_points_pub.publish(wall_cloud_msg)
                    self.get_logger().info(f'Published {len(wall_points)} wall points for visualization')
                    
                    # 對聚類後的點進行 PCA 擬合平面（XY 平面，垂直牆面假設）
                    n = self.fit_plane_normal_pca(wall_points)
                    nx, ny = float(n[0]), float(n[1])
                    
                    # ========================================
                    # 深度校正：將原始點投影到擬合的平面上
                    # ========================================
                    # 計算平面中心點（作為參考點）
                    plane_center = wall_points.mean(axis=0)
                    
                    # 原始點（可能因深度測量誤差偏離真實牆面）
                    original_point = np.array([pt_base.point.x, pt_base.point.y, pt_base.point.z])
                    
                    # 投影到擬合的平面上（校正深度誤差）
                    corrected_point = self.project_point_to_plane(original_point, n, plane_center)
                    
                    # 計算校正量
                    correction_distance = np.linalg.norm(corrected_point - original_point)
                    
                    # 更新發布的點為校正後的點
                    pt_base.point.x = float(corrected_point[0])
                    pt_base.point.y = float(corrected_point[1])
                    pt_base.point.z = float(corrected_point[2])
                    
                    self.get_logger().info(
                        f'Depth correction applied:\n'
                        f'  Original:  ({original_point[0]:.3f}, {original_point[1]:.3f}, {original_point[2]:.3f})\n'
                        f'  Corrected: ({corrected_point[0]:.3f}, {corrected_point[1]:.3f}, {corrected_point[2]:.3f})\n'
                        f'  Correction distance: {correction_distance*100:.1f} cm'
                    )
                    
                    # 發布修正後的牆面法向量 marker（紅色箭頭）
                    # 使用校正後的點位置
                    normal_marker = self.create_normal_marker(
                        [pt_base.point.x, pt_base.point.y, pt_base.point.z],
                        n
                    )
                    self.normal_marker_pub.publish(normal_marker)
                    
                    if abs(nx) + abs(ny) > 1e-6:
                        # 法向量的 XY 平面投影方向
                        theta_face = math.atan2(ny, nx)
                        # 切向量垂直於法向量（逆時針旋轉 90°）
                        theta_axis = self.wrap_pi(theta_face + math.pi / 2)

                        msg_axis = Float32()
                        msg_axis.data = float(theta_axis)
                        self.axis_pub.publish(msg_axis)

                        axis_marker = self.create_axis_marker(
                            [pt_base.point.x, pt_base.point.y, pt_base.point.z],
                            theta_axis
                        )
                        self.axis_marker_pub.publish(axis_marker)

                        # 計算牆面的傾斜角度
                        # 理想垂直牆面的法向量 Z 分量應該接近 0
                        tilt_angle_rad = math.asin(min(1.0, max(-1.0, abs(n[2]))))
                        tilt_angle_deg = math.degrees(tilt_angle_rad)

                        self.get_logger().info(
                            f'Panel normal: [{nx:.3f}, {ny:.3f}, {n[2]:.3f}], '
                            f'axis: {theta_axis:.3f} rad ({math.degrees(theta_axis):.1f}°), '
                            f'tilt from vertical: {tilt_angle_deg:.1f}°, '
                            f'wall_pts={wall_points.shape[0]}/{pts_base.shape[0]}'
                        )
                    else:
                        self.get_logger().warn(f'Normal vector too small in XY plane')
                else:
                    self.get_logger().warn(f'DBSCAN clustering failed or too few points')
            else:
                self.get_logger().warn(f'Patch points too few: {len(pts_base)} < {self.min_plane_pts}')
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

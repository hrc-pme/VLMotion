#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped, Twist
from std_msgs.msg import Float32, ColorRGBA
from sensor_msgs.msg import Image, CameraInfo
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.action import ActionClient
from rclpy.duration import Duration
from cv_bridge import CvBridge
import math
import numpy as np

import tf2_ros
from tf2_geometry_msgs import do_transform_point


class WhitePointFullMotion(Node):

    def __init__(self):
        super().__init__('white_point_full_motion')

        # 訂閱白點（目前是 base_link，但我們會轉成 world 再用）
        self.target_sub = self.create_subscription(
            PointStamped,
            '/white_point_base',
            self.target_callback,
            10
        )

        # 訂閱面板軸向（base_link yaw）
        self.axis_sub = self.create_subscription(
            Float32,
            '/panel_axis_base',
            self.axis_callback,
            10
        )
        
        # 發布底盤速度
        self.cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        
        # 發布視覺化 Marker
        self.marker_pub = self.create_publisher(MarkerArray, '/white_point_markers', 10)

        # Joint trajectory client
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/stretch_controller/follow_joint_trajectory'
        )

        # TF buffer + listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 軌跡記錄
        self.trajectory_points = []  # 存儲機器人軌跡點
        self.max_trajectory_points = 500  # 最多記錄 500 個點

        # Control parameters
        self.base_desired_dist = 0.6
        self.approach_dist = 0.6
        self.k_lin = 0.4
        self.k_ang = 1.0
        self.max_lin = 0.2
        self.max_ang = 0.5
        self.angle_thresh = 5.0 * math.pi / 180.0
        self.dist_thresh = 0.03

        # Lift/Arm range
        self.lift_min = 0.0
        self.lift_max = 1.1
        self.arm_min = 0.0
        self.arm_max = 0.5
        self.gripper_z_offset = 0.1
        self.gripper_y_offset_base = -0.1  # 夾爪中心在 base_link 的 Y 偏移（左側為負）

        # Target control state
        self.target_world = None         # 原始目標的世界座標（紅色點）
        self.compensated_target_world = None  # 補償後目標的世界座標（粉紅色點）- 鎖定後不變
        self.current_target = None       # 當前 base_link 中的目標點
        self.base_aligned = False
        self.joints_moved = False
        self.arm_extended = False
        self.target_locked = False
        self.wrist_reset_done = False
        self.final_forward_done = False
        self.forward_target_dist = None
        self.panel_axis_base = None
        self.desired_yaw_world = None
        self.approach_world = None
        self.orange_point_world = None   # 橘色點（世界座標系）- 從黃色點沿切線偏移 15cm
        self.tangent_vec_world = None    # 切線方向向量（世界座標系）- 鎖定後不變
        self.normal_vec_world = None     # 法線方向向量（世界座標系）- 鎖定後不變
        self.base_pos_at_lock = None
        self.is_close_range_mode = False # 近距離修正模式標記
        self.close_range_locked_yaw = None  # 近距離模式下鎖定的目標方向
        self.close_range_phase = None  # 近距離模式階段: 'backup', 'rotate', 'approach'
        self.close_range_backup_start = None  # 後退開始時的位置
        self.close_range_backup_dist = 0.20  # 後退距離（20cm，確保有足夠空間旋轉和重新對齊）
        self.tangent_offset_distance = 0.15  # 橘色點沿切線偏移距離（15 公分）
        self.final_gripper_dist_thresh = 0.04  # 4cm 到達閾值
        self.final_max_lin = 0.10  # 10 cm/s，穩定移動
        self.final_max_ang = 0.4
        self.final_angle_allow = 0.3  # 降低到 17°，先對齊角度再前進
        self.compensation_gain = 2.0  # 補償放大倍率，避免補償量過小
        
        # 近距離模式專用參數
        self.close_range_min_dist = float('inf')  # 追蹤最小距離
        self.close_range_approach_start_dist = None  # approach 開始時的距離

        # Action goal handles
        self.wrist_goal_handle = None
        self.wrist_sending = False
        self.joints_goal_handle = None
        self.joints_sending = False  # 防止重複發送
        self.joints_result = None    # 保存執行結果
        self.arm_goal_handle = None
        self.arm_sending = False
        self.arm_result = None

        # ================================================
        # 動態目標追蹤：利用深度影像持續校正目標位置
        # ================================================
        self.cv_bridge = CvBridge()
        
        # 訂閱深度影像（對齊到 color）
        self.depth_sub = self.create_subscription(
            Image,
            '/d435i/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        # 訂閱相機內參
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/d435i/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # 深度影像和相機內參
        self.depth_image = None
        self.depth_header = None
        self.cam_K = None  # 3x3 內參矩陣
        
        # 動態追蹤參數
        self.target_tracking_enabled = True  # 是否啟用動態追蹤
        self.tracking_update_interval = 0.2  # 更新間隔（秒）
        self.last_tracking_update = None
        self.tracking_correction_gain = 0.3  # 校正增益（0~1），避免過度修正
        self.tracking_max_correction = 0.05  # 單次最大校正距離（米）
        self.last_target_pixel = None  # 上次投影的像素位置（用於調試）

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("WhitePointFullMotion initialized with dynamic target tracking.")

    # ================================================
    # 深度影像和相機內參 callback
    # ================================================
    def depth_callback(self, msg: Image):
        """接收深度影像"""
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg)
            self.depth_header = msg.header
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth image: {e}", throttle_duration_sec=5.0)
    
    def camera_info_callback(self, msg: CameraInfo):
        """接收相機內參"""
        self.cam_K = np.array(msg.k).reshape(3, 3)

    # ================================================
    # 動態目標追蹤核心函數
    # ================================================
    def project_point_to_pixel(self, point_odom):
        """
        將 odom 座標系中的 3D 點投影到相機影像像素座標
        
        Args:
            point_odom: 目標點在 odom 座標系中的 Point
        
        Returns:
            (u, v): 像素座標 (未旋轉的原始影像)，失敗則返回 None
        """
        if self.cam_K is None:
            return None
        
        try:
            # 獲取 odom → camera optical frame 的轉換
            tf = self.tf_buffer.lookup_transform(
                'd435i_color_optical_frame',
                'odom',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # 將點從 odom 轉換到 camera optical frame
            ps = PointStamped()
            ps.header.frame_id = 'odom'
            ps.point = point_odom
            pt_cam = do_transform_point(ps, tf)
            
            # 相機座標 (X_cam 向右, Y_cam 向下, Z_cam 向前)
            Xc = pt_cam.point.x
            Yc = pt_cam.point.y
            Zc = pt_cam.point.z
            
            # 如果在相機後方，無法投影
            if Zc <= 0.1:
                return None
            
            # 投影到像素座標
            fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
            cx, cy = self.cam_K[0, 2], self.cam_K[1, 2]
            
            u = int(round(fx * Xc / Zc + cx))
            v = int(round(fy * Yc / Zc + cy))
            
            return (u, v)
        
        except Exception as e:
            self.get_logger().debug(f"Failed to project point to pixel: {e}")
            return None

    def backproject_pixel_to_point(self, u, v):
        """
        將像素座標和深度反投影回 odom 座標系中的 3D 點
        
        Args:
            u, v: 像素座標 (未旋轉的原始影像)
        
        Returns:
            Point: odom 座標系中的 3D 點，失敗則返回 None
        """
        if self.cam_K is None or self.depth_image is None:
            return None
        
        try:
            H, W = self.depth_image.shape[:2]
            
            # 邊界檢查
            if u < 0 or u >= W or v < 0 or v >= H:
                return None
            
            # 獲取深度（使用 9 點補插）
            depth_raw = self.get_valid_depth(u, v)
            if depth_raw <= 0.0 or np.isnan(depth_raw) or np.isinf(depth_raw):
                return None
            
            # 依據 encoding 決定單位：16UC1 (mm) 或 32FC1 (m)
            if self.depth_image.dtype == np.uint16:
                depth_m = float(depth_raw) / 1000.0
            else:
                depth_m = float(depth_raw)
            
            # 反投影到光學座標系
            fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
            cx, cy = self.cam_K[0, 2], self.cam_K[1, 2]
            
            Xo = (u - cx) * depth_m / fx
            Yo = (v - cy) * depth_m / fy
            Zo = depth_m
            
            # 轉換到 odom 座標系
            tf = self.tf_buffer.lookup_transform(
                'odom',
                'd435i_color_optical_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            pt_cam = PointStamped()
            pt_cam.header.frame_id = 'd435i_color_optical_frame'
            pt_cam.point.x = Xo
            pt_cam.point.y = Yo
            pt_cam.point.z = Zo
            
            pt_odom = do_transform_point(pt_cam, tf)
            return pt_odom.point
        
        except Exception as e:
            self.get_logger().debug(f"Failed to backproject pixel: {e}")
            return None

    def get_valid_depth(self, cx, cy):
        """9 點補插深度（在未旋轉的 depth image 上）"""
        if self.depth_image is None:
            return 0.0
        
        h, w = self.depth_image.shape[:2]
        cx_i = int(np.clip(round(cx), 0, w - 1))
        cy_i = int(np.clip(round(cy), 0, h - 1))
        
        z = float(self.depth_image[cy_i, cx_i])
        if z > 0:
            return z
        
        zs = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx = int(np.clip(cx_i + dx, 0, w - 1))
                ny = int(np.clip(cy_i + dy, 0, h - 1))
                v = float(self.depth_image[ny, nx])
                if v > 0:
                    zs.append(v)
        
        return float(np.mean(zs)) if zs else 0.0

    def update_target_from_depth(self):
        """
        利用深度影像動態校正目標位置
        
        改進策略：
        1. 只在「移動到橘色點」階段校正（不在前進到目標階段）
        2. 只校正 XY 平面位置，不改變 Z（高度較穩定）
        3. 使用距離變化檢測：如果校正會讓夾爪到目標的距離突然增加，則忽略
        4. 更保守的校正增益
        """
        if not self.target_tracking_enabled:
            return
        
        if self.target_world is None:
            return
        
        # 只在移動到橘色點階段校正
        # 一旦 base_aligned=True（到達橘色點），就停止校正
        # 這樣可以避免在最後接近階段目標飄移
        if self.base_aligned:
            return
        
        if self.final_forward_done:
            return
        
        # 如果還沒開始移動（wrist/lift 還在調整），也不校正
        if not self.wrist_reset_done or not self.joints_moved:
            return
        
        # 檢查更新間隔
        current_time = self.get_clock().now()
        if self.last_tracking_update is not None:
            elapsed = (current_time - self.last_tracking_update).nanoseconds / 1e9
            if elapsed < self.tracking_update_interval:
                return
        
        self.last_tracking_update = current_time
        
        # Step 1: 投影目標點到像素
        pixel = self.project_point_to_pixel(self.target_world)
        if pixel is None:
            self.get_logger().debug("Target not visible in camera")
            return
        
        u, v = pixel
        self.last_target_pixel = (u, v)
        
        # Step 2: 反投影得到新的 3D 點
        new_point = self.backproject_pixel_to_point(u, v)
        if new_point is None:
            self.get_logger().debug(f"Failed to backproject pixel ({u}, {v})")
            return
        
        # Step 3: 計算校正量（只在 XY 平面）
        dx = new_point.x - self.target_world.x
        dy = new_point.y - self.target_world.y
        # 不校正 Z（高度），因為深度測量在接近時較不穩定
        dz = 0.0
        
        correction_dist_xy = math.sqrt(dx*dx + dy*dy)
        
        # 只有當 XY 偏差超過閾值時才校正（避免噪聲）
        min_correction_threshold = 0.02  # 2cm 以下的偏差忽略
        if correction_dist_xy < min_correction_threshold:
            return
        
        # Step 4: 檢查校正是否合理
        # 獲取當前夾爪到目標的距離
        gripper_center = self.get_gripper_center_in_odom()
        if gripper_center is not None:
            # 計算校正前後的距離
            old_dx = self.target_world.x - gripper_center.x
            old_dy = self.target_world.y - gripper_center.y
            old_dist = math.sqrt(old_dx*old_dx + old_dy*old_dy)
            
            new_target_x = self.target_world.x + dx * self.tracking_correction_gain
            new_target_y = self.target_world.y + dy * self.tracking_correction_gain
            new_dx = new_target_x - gripper_center.x
            new_dy = new_target_y - gripper_center.y
            new_dist = math.sqrt(new_dx*new_dx + new_dy*new_dy)
            
            # 如果校正會讓距離大幅增加（超過 5cm），則忽略此次校正
            # 這表示深度測量可能有誤
            if new_dist > old_dist + 0.05:
                self.get_logger().debug(
                    f"Skipping correction: would increase distance from {old_dist*100:.1f}cm to {new_dist*100:.1f}cm"
                )
                return
        
        # 限制單次校正量
        if correction_dist_xy > self.tracking_max_correction:
            scale = self.tracking_max_correction / correction_dist_xy
            dx *= scale
            dy *= scale
            correction_dist_xy = self.tracking_max_correction
        
        # 應用校正增益
        dx *= self.tracking_correction_gain
        dy *= self.tracking_correction_gain
        
        # 更新目標位置
        old_x, old_y, old_z = self.target_world.x, self.target_world.y, self.target_world.z
        self.target_world.x += dx
        self.target_world.y += dy
        # self.target_world.z 不改變
        
        # 同步更新補償目標（如果存在）
        if self.compensated_target_world is not None:
            self.compensated_target_world.x += dx
            self.compensated_target_world.y += dy
        
        # 同步更新準備點和橘色點（如果存在）
        if self.approach_world is not None:
            self.approach_world.x += dx
            self.approach_world.y += dy
        
        if self.orange_point_world is not None:
            self.orange_point_world.x += dx
            self.orange_point_world.y += dy
        
        self.get_logger().info(
            f"🎯 Target corrected (XY only): pixel=({u},{v}), "
            f"ΔX={dx*100:.1f}cm, ΔY={dy*100:.1f}cm"
        )

    # ================================================
    # 收到白點 → 儲存到 world（odom）座標
    # ================================================
    def get_gripper_center_in_base(self):
        """取得夾爪中心在 base_link 的位置"""
        try:
            tf_left = self.tf_buffer.lookup_transform(
                'base_link',
                'link_gripper_fingertip_left',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            tf_right = self.tf_buffer.lookup_transform(
                'base_link',
                'link_gripper_fingertip_right',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            center = Point()
            center.x = (tf_left.transform.translation.x + tf_right.transform.translation.x) / 2.0
            center.y = (tf_left.transform.translation.y + tf_right.transform.translation.y) / 2.0
            center.z = (tf_left.transform.translation.z + tf_right.transform.translation.z) / 2.0
            return center
        except Exception as e:
            self.get_logger().warn(f"Failed to get gripper center TF: {e}")
            return None

    def get_gripper_center_in_odom(self):
        """取得夾爪中心在 odom 的位置"""
        try:
            tf_left = self.tf_buffer.lookup_transform(
                'odom',
                'link_gripper_fingertip_left',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            tf_right = self.tf_buffer.lookup_transform(
                'odom',
                'link_gripper_fingertip_right',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )

            center = Point()
            center.x = (tf_left.transform.translation.x + tf_right.transform.translation.x) / 2.0
            center.y = (tf_left.transform.translation.y + tf_right.transform.translation.y) / 2.0
            center.z = (tf_left.transform.translation.z + tf_right.transform.translation.z) / 2.0
            return center
        except Exception as e:
            self.get_logger().warn(f"Failed to get gripper center TF (odom): {e}")
            return None
    
    def target_callback(self, msg: PointStamped):

        if self.target_locked:
            return

        try:
            # base_link → odom 的 transform
            transform = self.tf_buffer.lookup_transform(
                'odom',      # 目標座標系
                'base_link', # 原始座標系
                rclpy.time.Time()
            )

            # 把白點從 base_link 轉到 odom（原始目標 - 紅色點）
            world_point = do_transform_point(msg, transform)
            new_target = world_point.point
            
            # 檢查是否為近距離修正模式
            # 如果夾爪已經靠近新目標點，直接進入微調模式
            gripper_center = self.get_gripper_center_in_odom()
            is_close_range_adjustment = False
            
            if gripper_center is not None:
                dx = gripper_center.x - new_target.x
                dy = gripper_center.y - new_target.y
                dz = gripper_center.z - new_target.z
                dist_to_new_target = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # 如果夾爪距離新目標點小於 30cm，進入近距離修正模式
                if dist_to_new_target < 0.30:
                    is_close_range_adjustment = True
                    self.get_logger().info(
                        f"🎯 Close-range adjustment mode! Gripper is {dist_to_new_target*100:.1f}cm from new target"
                    )
            
            self.target_world = new_target
            self.base_pos_at_lock = Point()
            self.base_pos_at_lock.x = transform.transform.translation.x
            self.base_pos_at_lock.y = transform.transform.translation.y
            self.base_pos_at_lock.z = transform.transform.translation.z
            
            self.target_locked = True
            
            if is_close_range_adjustment:
                # 近距離修正模式：跳過準備點，直接進行微調
                self.is_close_range_mode = True  # 設置近距離模式標記
                self.close_range_locked_yaw = None  # 重置鎖定方向，讓下次進入時重新計算
                self.close_range_phase = 'backup'  # 從後退階段開始
                self.close_range_backup_start = None  # 後退開始位置會在控制時設定
                self.base_aligned = True  # 跳過移動到橘色點
                self.joints_moved = True  # 假設 lift 已在正確高度（或稍後微調）
                self.arm_extended = False
                self.wrist_reset_done = True  # 跳過 wrist reset
                self.final_forward_done = False  # 需要微調前進
                self.forward_target_dist = None
                
                # 近距離模式不使用法向量，清除相關變數
                self.tangent_vec_world = None
                self.normal_vec_world = None
                self.approach_world = None
                self.orange_point_world = None
                self.compensated_target_world = None  # 不使用補償
                self.desired_yaw_world = None
                
                self.get_logger().info(
                    f"Close-range target (red): "
                    f"Xw={self.target_world.x:.3f}, Yw={self.target_world.y:.3f}, Zw={self.target_world.z:.3f}"
                )
            else:
                # 完整模式：從頭開始
                self.is_close_range_mode = False  # 清除近距離模式標記
                self.close_range_locked_yaw = None  # 重置鎖定方向
                self.close_range_phase = None  # 清除近距離階段
                self.close_range_backup_start = None
                self.base_aligned = False
                self.joints_moved = False
                self.arm_extended = False
                self.wrist_reset_done = False
                self.final_forward_done = False
                self.forward_target_dist = None
                self.desired_yaw_world = None
                self.approach_world = None
                self.orange_point_world = None
                self.tangent_vec_world = None
                self.normal_vec_world = None
                self.compensated_target_world = None
                
                self.get_logger().info(
                    f"Locked WORLD target (red): "
                    f"Xw={self.target_world.x:.3f}, Yw={self.target_world.y:.3f}, Zw={self.target_world.z:.3f}"
                )
                
                if self.panel_axis_base is not None:
                    self.update_desired_yaw_world()
                    self.calculate_vectors_world(transform)
                    self.update_approach_world()
                    self.update_orange_point_world()
                    self.update_compensated_target_world()
            
            # 重置 goal handles（兩種模式都需要）
            self.wrist_goal_handle = None
            self.wrist_sending = False
            self.joints_goal_handle = None
            self.joints_sending = False
            self.joints_result = None
            self.arm_goal_handle = None
            self.arm_sending = False
            self.arm_result = None

        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
    
    def calculate_vectors_world(self, transform_to_odom):
        """計算切線和法線向量的世界座標，鎖定後不再改變"""
        if self.panel_axis_base is None:
            return
        
        try:
            # 在 base_link 中計算切線和法線方向
            tangent_angle = self.panel_axis_base
            tangent_vec_base = [math.cos(tangent_angle), math.sin(tangent_angle), 0.0]
            
            # 法線方向垂直於切線方向（逆時針旋轉90度）
            normal_angle = self.panel_axis_base + math.pi / 2.0
            normal_vec_base = [math.cos(normal_angle), math.sin(normal_angle), 0.0]
            
            # 獲取旋轉四元數
            qx = transform_to_odom.transform.rotation.x
            qy = transform_to_odom.transform.rotation.y
            qz = transform_to_odom.transform.rotation.z
            qw = transform_to_odom.transform.rotation.w
            
            # 四元數旋轉向量函數
            def rotate_vector_by_quaternion(v, q):
                """用四元數旋轉向量"""
                qx, qy, qz, qw = q
                x, y, z = v
                
                # q * v * q^-1
                ix = qw * x + qy * z - qz * y
                iy = qw * y + qz * x - qx * z
                iz = qw * z + qx * y - qy * x
                iw = -qx * x - qy * y - qz * z
                
                rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
                ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
                rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
                
                return [rx, ry, rz]
            
            # 轉換到世界座標系並保存（鎖定，不再改變）
            self.tangent_vec_world = rotate_vector_by_quaternion(
                tangent_vec_base, 
                [qx, qy, qz, qw]
            )
            self.normal_vec_world = rotate_vector_by_quaternion(
                normal_vec_base, 
                [qx, qy, qz, qw]
            )
            
            self.get_logger().info(
                f"Locked tangent vector (world): [{self.tangent_vec_world[0]:.3f}, "
                f"{self.tangent_vec_world[1]:.3f}, {self.tangent_vec_world[2]:.3f}]"
            )
            self.get_logger().info(
                f"Locked normal vector (world): [{self.normal_vec_world[0]:.3f}, "
                f"{self.normal_vec_world[1]:.3f}, {self.normal_vec_world[2]:.3f}]"
            )
            
        except Exception as e:
            self.get_logger().warn(f"Failed to calculate vectors in world frame: {e}")
    
    def snap_point_to_pointcloud(self, point_odom, search_radius=5):
        """
        將 3D 點投影到相機像素後，使用附近像素的深度重新反投影到點雲上
        
        Args:
            point_odom: 原始 Point (odom 座標系)
            search_radius: 搜尋附近像素的半徑（預設 5 像素）
        
        Returns:
            Point: 在點雲上的 3D 點 (odom 座標系)，失敗則返回原始點
        """
        if self.cam_K is None or self.depth_image is None:
            self.get_logger().debug(
                f"Cannot snap: cam_K={self.cam_K is not None}, depth={self.depth_image is not None}"
            )
            return point_odom
        
        try:
            # Step 1: 將點投影到像素座標
            pixel = self.project_point_to_pixel(point_odom)
            if pixel is None:
                self.get_logger().info("Cannot project compensated point to pixel for snapping")
                return point_odom
            
            u, v = pixel
            
            # Step 2: 在附近像素中尋找有效深度
            H, W = self.depth_image.shape[:2]
            
            # 先嘗試中心點
            depth_raw = self.get_valid_depth(u, v)
            best_u, best_v = u, v
            
            # 如果中心點深度無效，搜尋附近像素
            if depth_raw <= 0.0 or np.isnan(depth_raw) or np.isinf(depth_raw):
                best_depth = 0.0
                min_dist_sq = float('inf')
                
                for du in range(-search_radius, search_radius + 1):
                    for dv in range(-search_radius, search_radius + 1):
                        nu = u + du
                        nv = v + dv
                        
                        # 邊界檢查
                        if nu < 0 or nu >= W or nv < 0 or nv >= H:
                            continue
                        
                        d = self.get_valid_depth(nu, nv)
                        if d > 0.0 and not np.isnan(d) and not np.isinf(d):
                            # 選擇距離中心最近的有效深度像素
                            dist_sq = du * du + dv * dv
                            if dist_sq < min_dist_sq:
                                min_dist_sq = dist_sq
                                best_depth = d
                                best_u, best_v = nu, nv
                
                if best_depth <= 0.0:
                    self.get_logger().info(f"No valid depth found near pixel ({u}, {v}) for snapping")
                    return point_odom
                
                depth_raw = best_depth
            
            # Step 3: 使用找到的深度反投影回 3D 點
            if self.depth_image.dtype == np.uint16:
                depth_m = float(depth_raw) / 1000.0
            else:
                depth_m = float(depth_raw)
            
            fx, fy = self.cam_K[0, 0], self.cam_K[1, 1]
            cx, cy = self.cam_K[0, 2], self.cam_K[1, 2]
            
            Xo = (best_u - cx) * depth_m / fx
            Yo = (best_v - cy) * depth_m / fy
            Zo = depth_m
            
            # 轉換到 odom 座標系
            tf = self.tf_buffer.lookup_transform(
                'odom',
                'd435i_color_optical_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            pt_cam = PointStamped()
            pt_cam.header.frame_id = 'd435i_color_optical_frame'
            pt_cam.point.x = Xo
            pt_cam.point.y = Yo
            pt_cam.point.z = Zo
            
            pt_odom = do_transform_point(pt_cam, tf)
            
            # 只使用 XY 座標，保留原始 Z（高度）
            # 因為原始高度是正確的，snap 只是為了讓 XY 對齊到點雲表面
            result = Point()
            result.x = pt_odom.point.x
            result.y = pt_odom.point.y
            result.z = point_odom.z  # 保留原始高度
            
            self.get_logger().info(
                f"📍 Snapped to pointcloud (XY only): pixel ({u},{v})->({best_u},{best_v}), "
                f"XY ({point_odom.x:.3f},{point_odom.y:.3f})->({result.x:.3f},{result.y:.3f}), "
                f"Z kept at {result.z:.3f}"
            )
            
            return result
            
        except Exception as e:
            self.get_logger().warn(f"Failed to snap point to pointcloud: {e}")
            return point_odom

    def update_compensated_target_world(self):
        """依據「起始→準備」與「準備→目標」方向計算補償量，並將結果貼合到點雲上"""
        if self.target_world is None:
            return False

        if self.approach_world is None or self.base_pos_at_lock is None:
            return False

        try:
            offset = -self.gripper_y_offset_base  # 夾爪在 base 的 y 偏移 → 目標反向補償

            # 方向1：起始位置 → 準備點
            dir1_x = self.approach_world.x - self.base_pos_at_lock.x
            dir1_y = self.approach_world.y - self.base_pos_at_lock.y
            dir1_norm = math.hypot(dir1_x, dir1_y)

            # 方向2：準備點 → 目標點（沿法向量前進方向）
            dir2_x = self.target_world.x - self.approach_world.x
            dir2_y = self.target_world.y - self.approach_world.y
            dir2_norm = math.hypot(dir2_x, dir2_y)

            if dir1_norm < 1e-6 or dir2_norm < 1e-6:
                return False

            # 單位化
            dir1_x /= dir1_norm
            dir1_y /= dir1_norm
            dir2_x /= dir2_norm
            dir2_y /= dir2_norm

            # 轉向角度（兩方向夾角）
            cos_theta = max(-1.0, min(1.0, dir1_x * dir2_x + dir1_y * dir2_y))
            theta = math.acos(cos_theta)

            # 方向決定（左/右轉）
            cross_z = dir1_x * dir2_y - dir1_y * dir2_x
            sign = 1.0 if cross_z >= 0.0 else -1.0

            # 以「準備→目標」為前進方向，補償沿其垂直方向
            lateral_x = -dir2_y
            lateral_y = dir2_x

            # 補償量：與轉向角度相關（轉向越大越補）
            mag = abs(offset) * math.sin(theta) * self.compensation_gain
            delta_x = lateral_x * sign * mag
            delta_y = lateral_y * sign * mag

            # 先計算幾何補償後的目標位置
            compensated_point = Point()
            compensated_point.x = self.target_world.x + delta_x
            compensated_point.y = self.target_world.y + delta_y
            compensated_point.z = self.target_world.z
            
            # 將補償後的目標點貼合到點雲上（使用附近像素的深度）
            snapped_point = self.snap_point_to_pointcloud(compensated_point, search_radius=5)
            
            self.compensated_target_world = Point()
            self.compensated_target_world.x = snapped_point.x
            self.compensated_target_world.y = snapped_point.y
            self.compensated_target_world.z = snapped_point.z

            self.get_logger().info(
                f"Updated COMPENSATED target (snapped to pointcloud): "
                f"Xw={self.compensated_target_world.x:.3f}, Yw={self.compensated_target_world.y:.3f}, "
                f"Zw={self.compensated_target_world.z:.3f}"
            )
            return True
        except Exception as e:
            self.get_logger().warn(f"Failed to update compensated target: {e}")
            return False

    def axis_callback(self, msg: Float32):
        """接收面板軸向角度"""
        self.panel_axis_base = float(msg.data)
        
        # 近距離修正模式下不重新計算向量和準備點
        if self.is_close_range_mode:
            return
        
        # 如果目標已鎖定但向量還沒計算，現在計算
        if self.target_locked and self.tangent_vec_world is None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'odom',
                    'base_link',
                    rclpy.time.Time()
                )
                self.calculate_vectors_world(transform)
                if self.desired_yaw_world is not None:
                    self.update_approach_world()
                    self.update_orange_point_world()
                    self.update_compensated_target_world()
            except Exception as e:
                self.get_logger().warn(f"Failed to calculate vectors in axis_callback: {e}")
        
        if self.target_locked and self.desired_yaw_world is None:
            self.update_desired_yaw_world()
            self.update_approach_world()
            self.update_orange_point_world()
            self.update_compensated_target_world()

    def get_effective_target_world(self):
        """控制用的目標點：優先補償後的目標點"""
        if self.compensated_target_world is not None:
            return self.compensated_target_world
        return self.target_world

    # ================================================
    # 視覺化函數
    # ================================================
    def publish_visualization_markers(self):
        """發布視覺化標記到 RViz"""
        marker_array = MarkerArray()
        
        # 1. 原始目標點（紅色球體）
        if self.target_world is not None:
            target_marker = Marker()
            target_marker.header.frame_id = "odom"
            target_marker.header.stamp = self.get_clock().now().to_msg()
            target_marker.ns = "target"
            target_marker.id = 0
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            target_marker.pose.position.x = self.target_world.x
            target_marker.pose.position.y = self.target_world.y
            target_marker.pose.position.z = self.target_world.z
            target_marker.scale.x = 0.05
            target_marker.scale.y = 0.05
            target_marker.scale.z = 0.05
            target_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # 紅色
            marker_array.markers.append(target_marker)
        
        # 2. 準備點（黃色球體）
        if self.approach_world is not None:
            approach_marker = Marker()
            approach_marker.header.frame_id = "odom"
            approach_marker.header.stamp = self.get_clock().now().to_msg()
            approach_marker.ns = "approach"
            approach_marker.id = 1
            approach_marker.type = Marker.SPHERE
            approach_marker.action = Marker.ADD
            approach_marker.pose.position.x = self.approach_world.x
            approach_marker.pose.position.y = self.approach_world.y
            approach_marker.pose.position.z = self.approach_world.z
            approach_marker.scale.x = 0.04
            approach_marker.scale.y = 0.04
            approach_marker.scale.z = 0.04
            approach_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)  # 黃色
            marker_array.markers.append(approach_marker)
        
        # 2.5. 橘色對齊點（橘色球體）
        if self.orange_point_world is not None:
            orange_marker = Marker()
            orange_marker.header.frame_id = "odom"
            orange_marker.header.stamp = self.get_clock().now().to_msg()
            orange_marker.ns = "orange_point"
            orange_marker.id = 8
            orange_marker.type = Marker.SPHERE
            orange_marker.action = Marker.ADD
            orange_marker.pose.position.x = self.orange_point_world.x
            orange_marker.pose.position.y = self.orange_point_world.y
            orange_marker.pose.position.z = self.orange_point_world.z
            orange_marker.scale.x = 0.045
            orange_marker.scale.y = 0.045
            orange_marker.scale.z = 0.045
            orange_marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)  # 橘色
            marker_array.markers.append(orange_marker)
        
        # 3. 機器人軌跡（藍色線條）
        if len(self.trajectory_points) > 1:
            trajectory_marker = Marker()
            trajectory_marker.header.frame_id = "odom"
            trajectory_marker.header.stamp = self.get_clock().now().to_msg()
            trajectory_marker.ns = "trajectory"
            trajectory_marker.id = 2
            trajectory_marker.type = Marker.LINE_STRIP
            trajectory_marker.action = Marker.ADD
            trajectory_marker.scale.x = 0.01  # 線寬
            trajectory_marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)  # 藍色
            trajectory_marker.points = self.trajectory_points.copy()
            marker_array.markers.append(trajectory_marker)
        
        # 4. 當前夾爪位置（白色球體）
        try:
            gripper_tf = self.tf_buffer.lookup_transform(
                'odom',
                'link_gripper_fingertip_left',
                rclpy.time.Time()
            )
            gripper_marker = Marker()
            gripper_marker.header.frame_id = "odom"
            gripper_marker.header.stamp = self.get_clock().now().to_msg()
            gripper_marker.ns = "gripper"
            gripper_marker.id = 3
            gripper_marker.type = Marker.SPHERE
            gripper_marker.action = Marker.ADD
            gripper_marker.pose.position.x = gripper_tf.transform.translation.x
            gripper_marker.pose.position.y = gripper_tf.transform.translation.y
            gripper_marker.pose.position.z = gripper_tf.transform.translation.z
            gripper_marker.scale.x = 0.03
            gripper_marker.scale.y = 0.03
            gripper_marker.scale.z = 0.03
            gripper_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # 白色
            marker_array.markers.append(gripper_marker)
        except:
            pass
        
        # 5. 目標點的切線方向向量（綠色箭頭）- 使用鎖定的世界座標向量
        if self.target_world is not None and self.tangent_vec_world is not None:
            tangent_marker = Marker()
            tangent_marker.header.frame_id = "odom"
            tangent_marker.header.stamp = self.get_clock().now().to_msg()
            tangent_marker.ns = "tangent_vector"
            tangent_marker.id = 4
            tangent_marker.type = Marker.ARROW
            tangent_marker.action = Marker.ADD
            
            # 設置箭頭起點和終點
            start_point = Point()
            start_point.x = self.target_world.x
            start_point.y = self.target_world.y
            start_point.z = self.target_world.z
            
            arrow_length = 0.2  # 20cm
            end_point = Point()
            end_point.x = self.target_world.x + self.tangent_vec_world[0] * arrow_length
            end_point.y = self.target_world.y + self.tangent_vec_world[1] * arrow_length
            end_point.z = self.target_world.z + self.tangent_vec_world[2] * arrow_length
            
            tangent_marker.points = [start_point, end_point]
            tangent_marker.scale.x = 0.01  # 箭桿直徑
            tangent_marker.scale.y = 0.02  # 箭頭直徑
            tangent_marker.scale.z = 0.03  # 箭頭長度
            tangent_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # 綠色
            marker_array.markers.append(tangent_marker)
        elif self.target_world is not None:
            # 調試信息：如果目標點存在但向量不存在
            if self.tangent_vec_world is None:
                self.get_logger().warn("Tangent vector is None", throttle_duration_sec=2.0)
        
        # 6. 目標點的法線方向向量（青色箭頭）- 使用鎖定的世界座標向量
        if self.target_world is not None and self.normal_vec_world is not None:
            normal_marker = Marker()
            normal_marker.header.frame_id = "odom"
            normal_marker.header.stamp = self.get_clock().now().to_msg()
            normal_marker.ns = "normal_vector"
            normal_marker.id = 5
            normal_marker.type = Marker.ARROW
            normal_marker.action = Marker.ADD
            
            # 設置箭頭起點和終點
            start_point = Point()
            start_point.x = self.target_world.x
            start_point.y = self.target_world.y
            start_point.z = self.target_world.z
            
            arrow_length = 0.2  # 20cm
            end_point = Point()
            end_point.x = self.target_world.x + self.normal_vec_world[0] * arrow_length
            end_point.y = self.target_world.y + self.normal_vec_world[1] * arrow_length
            end_point.z = self.target_world.z + self.normal_vec_world[2] * arrow_length
            
            normal_marker.points = [start_point, end_point]
            normal_marker.scale.x = 0.01  # 箭桿直徑
            normal_marker.scale.y = 0.02  # 箭頭直徑
            normal_marker.scale.z = 0.03  # 箭頭長度
            normal_marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)  # 青色
            marker_array.markers.append(normal_marker)
        elif self.target_world is not None:
            # 調試信息：如果目標點存在但向量不存在
            if self.normal_vec_world is None:
                self.get_logger().warn("Normal vector is None", throttle_duration_sec=2.0)
        
        # 7. 補償後的目標點（洋紅色球體）- 使用鎖定時計算的固定世界座標
        if self.compensated_target_world is not None:
            compensated_marker = Marker()
            compensated_marker.header.frame_id = "odom"
            compensated_marker.header.stamp = self.get_clock().now().to_msg()
            compensated_marker.ns = "compensated_target"
            compensated_marker.id = 6
            compensated_marker.type = Marker.SPHERE
            compensated_marker.action = Marker.ADD
            compensated_marker.pose.position.x = self.compensated_target_world.x
            compensated_marker.pose.position.y = self.compensated_target_world.y
            compensated_marker.pose.position.z = self.compensated_target_world.z
            compensated_marker.scale.x = 0.04
            compensated_marker.scale.y = 0.04
            compensated_marker.scale.z = 0.04
            compensated_marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.8)  # 洋紅色
            marker_array.markers.append(compensated_marker)
        
        # 8. 顯示夾爪到目標的距離（TEXT_VIEW_FACING）
        if self.target_world is not None:
            try:
                gripper_center = self.get_gripper_center_in_odom()
                if gripper_center is None:
                    return
                
                # 計算距離
                dx = gripper_center.x - self.target_world.x
                dy = gripper_center.y - self.target_world.y
                dz = gripper_center.z - self.target_world.z
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # 創建文本標記
                text_marker = Marker()
                text_marker.header.frame_id = "odom"
                text_marker.header.stamp = self.get_clock().now().to_msg()
                text_marker.ns = "distance_text"
                text_marker.id = 7
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                # 文本位置在目標點上方
                text_marker.pose.position.x = self.target_world.x
                text_marker.pose.position.y = self.target_world.y
                text_marker.pose.position.z = self.target_world.z + 0.15  # 上方 15cm
                text_marker.scale.z = 0.05  # 文字高度
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)  # 白色
                text_marker.text = f"Distance: {distance*100:.1f}cm"
                marker_array.markers.append(text_marker)
            except:
                pass
        
        self.marker_pub.publish(marker_array)
    
    def update_trajectory(self):
        """更新機器人軌跡記錄"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time()
            )
            
            current_point = Point()
            current_point.x = transform.transform.translation.x
            current_point.y = transform.transform.translation.y
            current_point.z = 0.0  # 地面軌跡
            
            # 避免重複記錄相同位置
            if len(self.trajectory_points) == 0:
                self.trajectory_points.append(current_point)
            else:
                last_point = self.trajectory_points[-1]
                dist = math.sqrt(
                    (current_point.x - last_point.x)**2 + 
                    (current_point.y - last_point.y)**2
                )
                # 只有移動超過 2cm 才記錄新點
                if dist > 0.02:
                    self.trajectory_points.append(current_point)
                    # 限制軌跡點數量
                    if len(self.trajectory_points) > self.max_trajectory_points:
                        self.trajectory_points.pop(0)
        except:
            pass

    # ================================================
    # 主循環
    # ================================================
    def control_loop(self):
        # 更新軌跡和視覺化
        self.update_trajectory()
        self.publish_visualization_markers()
        
        # 持續追蹤目標點（即使還沒開始移動）
        if self.target_world is not None:
            self.update_head_tracking()
        
        # 動態目標追蹤：利用深度影像持續校正目標位置
        # 這可以補償 odom 累積誤差
        self.update_target_from_depth()
        
        if self.target_world is None:
            return

        # Step 1: 世界座標 → base_link（每迴圈更新）
        # 注意：底盤對齊階段使用原始目標點，不做偏移補償
        base_target = self.get_point_in_base(self.target_world)
        if base_target is None:
            return

        base_approach = None
        if self.approach_world is not None:
            # 準備點：直接轉換
            base_approach = self.get_point_in_base(self.approach_world)
        desired_dist = 0.0
        if base_approach is None:
            base_approach = base_target
            desired_dist = self.base_desired_dist

        self.current_target = base_target

        # Step 1: 重置手腕、頭部，並抬高 lift（但不伸 arm）
        if not self.wrist_reset_done:
            if not self.wrist_sending:
                success = self.reset_wrist_and_head()
                # wrist_reset_done 會在 callback 中設定
            return
        
        # Step 2: 抬高 lift 到目標高度（但保持 arm 收起）
        if not self.joints_moved:
            if self.joints_goal_handle is None and not self.joints_sending:
                success = self.move_lift_only()
                if not success:
                    return
            elif self.joints_result is not None:
                if self.joints_result.status == 4:  # STATUS_SUCCEEDED
                    self.joints_moved = True
                    self.joints_goal_handle = None
                    self.joints_result = None
                    self.get_logger().info("Lift raised successfully!")
                else:
                    self.get_logger().warn(f'Lift goal failed with status: {self.joints_result.status}')
                    self.joints_goal_handle = None
                    self.joints_result = None
                return
            else:
                return

        # Step 3: 使用 base_link 移動到橘色對齊點
        if not self.base_aligned:
            self.move_base_to_orange_point()
            return

        # Step 4: 沿著法向量前進到目標點（不需要旋轉對齊，直接前進）
        if not self.final_forward_done:
            self.move_forward_to_target()
            return
                
        # Step 5: 到達目標點後，伸出 arm
        if not self.arm_extended:
            if self.arm_goal_handle is None and not self.arm_sending:
                success = self.extend_arm()
                if not success:
                    return
            elif self.arm_result is not None:
                if self.arm_result.status == 4:  # STATUS_SUCCEEDED
                    self.arm_extended = True
                    self.arm_goal_handle = None
                    self.arm_result = None
                    # arm_sending 已在 callback 中清除
                    self.get_logger().info("Arm extended successfully! Task finished.")
                else:
                    self.get_logger().warn(f'Arm goal failed with status: {self.arm_result.status}')
                    self.arm_goal_handle = None
                    self.arm_result = None
                    # arm_sending 已在 callback 中清除
                return
            else:
                # 還在等待結果
                return

    # ================================================
    # 將 world 座標轉成 base_link（不再需要動態補償）
    # ================================================
    def get_point_in_base(self, world_point, apply_offset=False):
        """
        將世界座標轉換到 base_link
        
        注意：偏移已經在 target_world 中一次性應用了，
        這裡不再需要動態補償（apply_offset 參數已廢棄）
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'odom',
                rclpy.time.Time()
            )

            # 建立 PointStamped for TF
            ps = PointStamped()
            ps.header.frame_id = 'odom'
            ps.point = world_point

            base_point = do_transform_point(ps, transform)
            return base_point.point

        except Exception as e:
            self.get_logger().warn(f"TF (odom→base) failed: {e}")
            return None

    # ================================================
    # 底盤控制
    # ================================================
    def move_base_to_orange_point(self):
        """使用 base_link 精確移動到橘色對齊點"""
        if self.orange_point_world is None:
            self.get_logger().warn("No orange point available!")
            return
        
        # 獲取 base_link 位置（世界座標）
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time()
            )
            base_pos = transform.transform.translation
        except Exception as e:
            self.get_logger().warn(f"Failed to get base_link position: {e}")
            return
        
        # 計算 base_link 到橘色點的距離（世界座標）
        to_orange_x = self.orange_point_world.x - base_pos.x
        to_orange_y = self.orange_point_world.y - base_pos.y
        dist_world = math.sqrt(to_orange_x*to_orange_x + to_orange_y*to_orange_y)
        
        # 將橘色點轉換到 base_link 進行控制
        try:
            orange_base = self.get_point_in_base(self.orange_point_world)
            if orange_base is None:
                return
        except Exception as e:
            self.get_logger().warn(f"Failed to transform orange point: {e}")
            return
        
        # 在 base_link 中計算到橘色點的方向
        dx = orange_base.x
        dy = orange_base.y
        dist_base = math.sqrt(dx*dx + dy*dy)
        angle_to_orange = math.atan2(dy, dx)
        
        # 檢查是否到達橘色點
        if dist_world <= self.dist_thresh:
            stop = Twist()
            self.cmd_vel_pub.publish(stop)
            self.base_aligned = True
            self.get_logger().info(
                f"✓ Base_link reached orange point! Final distance: {dist_world*100:.1f}cm"
            )
            return
        
        # 控制策略：先對齊橘色點方向，再前進
        twist = Twist()
        
        # 如果角度誤差太大，先轉向
        if abs(angle_to_orange) > self.angle_thresh:
            ang_cmd = max(-self.max_ang, min(self.max_ang, self.k_ang * angle_to_orange))
            twist.angular.z = ang_cmd
            twist.linear.x = 0.0
            self.get_logger().info(
                f"Aligning to orange point: angle_error={math.degrees(angle_to_orange):.1f}°, "
                f"dist={dist_world:.3f}m"
            )
        else:
            # 角度對齊後，直接朝橘色點前進
            # 速度依距離縮放，接近時自動降速
            lin_cmd = min(self.max_lin, self.k_lin * dist_world)
            twist.linear.x = max(0.0, lin_cmd)
            
            # 同時保持小幅度角度修正
            ang_cmd = max(-self.max_ang * 0.3, min(self.max_ang * 0.3, self.k_ang * angle_to_orange))
            twist.angular.z = ang_cmd
            
            self.get_logger().info(
                f"Moving to orange point: dist={dist_world:.3f}m, lin_cmd={lin_cmd:.3f}, "
                f"angle_error={math.degrees(angle_to_orange):.1f}°"
            )
        
        self.cmd_vel_pub.publish(twist)

    def update_desired_yaw_world(self):
        """將面板法向量轉成世界座標 yaw"""
        if self.panel_axis_base is None or self.target_world is None:
            return False

        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time()
            )

            q = transform.transform.rotation
            base_yaw_world = self.quat_to_yaw(q.x, q.y, q.z, q.w)

            # 由 panel axis 推回 panel normal (face)
            theta_face = self.wrap_pi(self.panel_axis_base - math.pi / 2.0)

            # 判斷面板朝向：使用原始目標點（不需要補償）
            base_target = self.get_point_in_base(self.target_world)
            if base_target is not None:
                angle_to_target = math.atan2(base_target.y, base_target.x)
                theta_face_flipped = self.wrap_pi(theta_face + math.pi)
                if abs(self.wrap_pi(angle_to_target - theta_face)) > abs(self.wrap_pi(angle_to_target - theta_face_flipped)):
                    theta_face = theta_face_flipped

            self.desired_yaw_world = self.wrap_pi(base_yaw_world + theta_face)

            self.get_logger().info(
                f"Desired yaw (world): {self.desired_yaw_world:.3f} rad "
                f"({math.degrees(self.desired_yaw_world):.1f}°)"
            )
            return True
        except Exception as e:
            self.get_logger().warn(f"Failed to compute desired yaw: {e}")
            return False

    def update_approach_world(self):
        """在面板前方建立準備點（靠近機器人一側）"""
        if self.desired_yaw_world is None or self.target_world is None:
            return False

        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time()
            )
            base_pos = transform.transform.translation

            # 法向量（世界座標）指向面板正面
            nx = math.cos(self.desired_yaw_world)
            ny = math.sin(self.desired_yaw_world)

            # 讓法向量指向機器人（在面板前方）
            to_robot_x = base_pos.x - self.target_world.x
            to_robot_y = base_pos.y - self.target_world.y
            if (nx * to_robot_x + ny * to_robot_y) < 0.0:
                nx = -nx
                ny = -ny

            self.approach_world = Point()
            self.approach_world.x = self.target_world.x + nx * self.approach_dist
            self.approach_world.y = self.target_world.y + ny * self.approach_dist
            self.approach_world.z = self.target_world.z

            self.get_logger().info(
                f"Approach point (world): X={self.approach_world.x:.3f}, "
                f"Y={self.approach_world.y:.3f}"
            )
            return True
        except Exception as e:
            self.get_logger().warn(f"Failed to compute approach point: {e}")
            return False

    def update_orange_point_world(self):
        """從黃色準備點沿著切線反方向偏移 15 公分，建立橘色對齊點"""
        if self.approach_world is None or self.tangent_vec_world is None:
            return False

        try:
            # 正規化切線向量
            tx = self.tangent_vec_world[0]
            ty = self.tangent_vec_world[1]
            t_norm = math.sqrt(tx*tx + ty*ty)
            
            if t_norm < 1e-6:
                self.get_logger().warn("Tangent vector is too small!")
                return False
            
            tx /= t_norm
            ty /= t_norm
            
            # 橘色點 = 黃色點 - 切線方向 * 15cm（反向）
            self.orange_point_world = Point()
            self.orange_point_world.x = self.approach_world.x - tx * self.tangent_offset_distance
            self.orange_point_world.y = self.approach_world.y - ty * self.tangent_offset_distance
            self.orange_point_world.z = self.approach_world.z
            
            self.get_logger().info(
                f"Orange point (world): X={self.orange_point_world.x:.3f}, "
                f"Y={self.orange_point_world.y:.3f} (offset {self.tangent_offset_distance*100:.0f}cm along reverse tangent)"
            )
            return True
        except Exception as e:
            self.get_logger().warn(f"Failed to compute orange point: {e}")
            return False

    def wrap_pi(self, a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def quat_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ================================================
    # 控制 lift 和 arm
    # ================================================
    def reset_wrist_and_head(self):
        """只重置 wrist（head 會由 update_head_tracking 持續追蹤目標）"""
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Joint controller not ready for wrist reset.')
            return False
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'joint_wrist_yaw', 
            'joint_wrist_pitch',
            'joint_wrist_roll'
        ]

        point = JointTrajectoryPoint()
        point.positions = [
            math.pi/2,           # wrist yaw
            0.0,                 # wrist pitch
            0.0,                 # wrist roll
        ]
        point.time_from_start = Duration(seconds=2.0).to_msg()

        goal.trajectory.points = [point]
        
        # 標記正在發送
        self.wrist_sending = True
        
        # 使用 future 保存 goal handle
        send_goal_future = self.trajectory_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.wrist_goal_response_callback)

        self.get_logger().info("Wrist reset command sent.")
        return True
    
    def update_head_tracking(self):
        """持續更新 head 朝向目標點"""
        if self.target_world is None:
            return
        
        try:
            # 獲取 head_pan_link 在 odom 中的位置
            head_tf = self.tf_buffer.lookup_transform(
                'odom',
                'link_head_pan',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            head_pos = head_tf.transform.translation
            
            # 計算從 head 到目標點的向量
            dx = self.target_world.x - head_pos.x
            dy = self.target_world.y - head_pos.y
            dz = self.target_world.z - head_pos.z
            
            # 計算水平距離
            horizontal_dist = math.sqrt(dx*dx + dy*dy)
            
            # 計算需要的 head_pan（水平旋轉）- 在 odom 座標系中
            target_yaw_odom = math.atan2(dy, dx)
            
            # 獲取 base_link 的 yaw
            base_tf = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time()
            )
            q = base_tf.transform.rotation
            base_yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
            
            # head_pan 是相對於 base_link 的角度
            # Stretch 的 head_pan 正值向左，負值向右
            head_pan = self.wrap_pi(target_yaw_odom - base_yaw)
            
            # 計算需要的 head_tilt（垂直旋轉）
            # head_tilt 負值向下看，正值向上看
            head_tilt = -math.atan2(-dz, horizontal_dist)  # 目標在下方時 dz 為負
            
            # 限制 head 的運動範圍
            head_pan = max(-math.radians(180), min(math.radians(180), head_pan))
            head_tilt = max(math.radians(-90), min(math.radians(20), head_tilt))
            
            # 發送 head 控制指令
            self.send_head_command(head_pan, head_tilt)
            
        except Exception as e:
            self.get_logger().warn(f"Head tracking failed: {e}", throttle_duration_sec=2.0)
    
    def send_head_command(self, head_pan, head_tilt):
        """發送 head 控制指令（非阻塞）"""
        # 避免頻繁發送相同指令
        if not hasattr(self, '_last_head_pan'):
            self._last_head_pan = None
            self._last_head_tilt = None
            self._head_sending = False
        
        # 如果正在發送或變化太小，跳過
        if self._head_sending:
            return
        
        if self._last_head_pan is not None and self._last_head_tilt is not None:
            pan_diff = abs(head_pan - self._last_head_pan)
            tilt_diff = abs(head_tilt - self._last_head_tilt)
            if pan_diff < math.radians(2) and tilt_diff < math.radians(2):
                return
        
        if not self.trajectory_client.wait_for_server(timeout_sec=0.1):
            return
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_head_pan', 'joint_head_tilt']
        
        point = JointTrajectoryPoint()
        point.positions = [head_pan, head_tilt]
        point.time_from_start = Duration(seconds=0.3).to_msg()  # 快速響應
        
        goal.trajectory.points = [point]
        
        self._head_sending = True
        self._last_head_pan = head_pan
        self._last_head_tilt = head_tilt
        
        send_goal_future = self.trajectory_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self._head_goal_response_callback)
    
    def _head_goal_response_callback(self, future):
        """處理 head goal 的回應"""
        self._head_sending = False
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().debug('Head tracking goal rejected')
    
    def wrist_goal_response_callback(self, future):
        """處理 wrist reset goal 的回應
        
        注意: wrist_reset_done=True 代表 goal 被 accepted,
        不代表動作已完成。目前設計是只要 accepted 就開始 base 對齊,
        不等待 wrist/head 實際到位。
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Wrist reset goal rejected!')
            self.wrist_goal_handle = None
            self.wrist_sending = False
            return
        
        self.wrist_goal_handle = goal_handle
        self.wrist_reset_done = True  # accepted 後標記(不是完成)
        self.wrist_sending = False
        self.get_logger().info('Wrist reset goal accepted (not yet completed).')

    def move_lift_only(self):
        """只抬高 lift 到目標高度，arm 保持收起（0）"""
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Joint controller not ready.')
            return False

        target_z = self.current_target.z
        lift_target = max(self.lift_min, min(self.lift_max, target_z - self.gripper_z_offset))

        self.get_logger().info(
            f"Raising lift to: {lift_target:.3f}m (arm stays at 0)"
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_lift', 'wrist_extension', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll']

        point = JointTrajectoryPoint()
        point.positions = [lift_target, 0.0, math.pi/2, 0.0, 0.0]  # arm = 0.0 保持收起
        point.time_from_start = Duration(seconds=2.0).to_msg()

        goal.trajectory.points = [point]
        
        self.joints_sending = True
        send_goal_future = self.trajectory_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.joints_goal_response_callback)

        self.get_logger().info("Lift raising command sent.")
        return True

    def extend_arm(self):
        """到達目標點後，伸出 arm"""
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Joint controller not ready.')
            return False

        target_x = self.current_target.x
        arm_target = max(self.arm_min, min(self.arm_max, target_x - self.base_desired_dist))

        self.get_logger().info(
            f"Extending arm to: {arm_target:.3f}m"
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['wrist_extension']

        point = JointTrajectoryPoint()
        point.positions = [arm_target]
        point.time_from_start = Duration(seconds=2.0).to_msg()

        goal.trajectory.points = [point]
        
        self.arm_sending = True
        send_goal_future = self.trajectory_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.arm_goal_response_callback)

        self.get_logger().info("Arm extension command sent.")
        return True
    
    def arm_goal_response_callback(self, future):
        """處理 arm goal 的回應"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Arm goal rejected!')
            self.arm_goal_handle = None
            self.arm_sending = False
            return
        
        self.arm_goal_handle = goal_handle
        self.get_logger().info('Arm goal accepted, waiting for completion...')
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.arm_result_callback)
    
    def arm_result_callback(self, future):
        """處理 arm goal 的執行結果"""
        result = future.result()
        self.arm_result = result
        self.arm_sending = False
        
        if result.status == 4:  # STATUS_SUCCEEDED
            self.get_logger().info('✓ Task completed! Arm extended successfully!')
            
            # 重置所有狀態，準備接收下一個目標
            self.target_world = None
            self.compensated_target_world = None
            self.target_locked = False
            self.base_aligned = False
            self.joints_moved = False
            self.arm_extended = False
            self.wrist_reset_done = False
            self.desired_yaw_world = None
            self.forward_target_dist = None
            self.final_forward_done = False
            self.trajectory_points = []
            self.tangent_vec_world = None
            self.normal_vec_world = None
            self.approach_world = None
            self.orange_point_world = None
            self.base_pos_at_lock = None
            self._normal_flipped = False  # 重置法向量翻轉標記
            self._min_dist_achieved = float('inf')  # 重置最小距離記錄
            self._use_direct_mode = False  # 重置直接模式標記
            self.is_close_range_mode = False  # 重置近距離模式標記
            self.close_range_locked_yaw = None  # 重置近距離鎖定方向
            self.close_range_phase = None  # 重置近距離階段
            self.close_range_backup_start = None
            self.close_range_min_dist = float('inf')  # 重置近距離最小距離追蹤
            self.close_range_approach_start_dist = None
            
            # 重置 goal handles
            self.wrist_goal_handle = None
            self.wrist_sending = False
            self.joints_goal_handle = None
            self.joints_sending = False
            self.joints_result = None
            self.arm_goal_handle = None
            self.arm_result = None
            
            self.get_logger().info("Ready for next target.")
        else:
            self.get_logger().warn(f'Arm action finished with status: {result.status}')

    def joints_goal_response_callback(self, future):
        """處理 joints goal 的回應"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Joints goal rejected!')
            self.joints_goal_handle = None
            self.joints_sending = False
            return
        
        self.joints_goal_handle = goal_handle
        self.get_logger().info('Joints goal accepted, waiting for completion...')
        
        # 取得結果 - 這才是正確的做法!
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.joints_result_callback)
    
    def joints_result_callback(self, future):
        """處理 joints goal 的執行結果（lift 抬高）"""
        result = future.result()
        self.joints_result = result
        
        # 不論成功或失敗,都要清除 sending 標記
        self.joints_sending = False
        
        if result.status == 4:  # STATUS_SUCCEEDED
            self.get_logger().info('Lift raised successfully!')
        else:
            self.get_logger().warn(f'Lift action finished with status: {result.status}')

    def move_forward_to_target(self):
        """沿著法向量方向前進到目標點（或近距離直接朝向目標）"""
        
        # 近距離修正模式：直接使用原始目標點，不用補償
        # 完整模式：使用補償後的目標點
        is_close_range = (self.approach_world is None)
        
        if is_close_range:
            target = self.target_world  # 近距離直接用原始目標
        else:
            target = self.get_effective_target_world()
        
        if target is None:
            self.get_logger().warn("No target available!")
            return
        
        # 獲取夾爪中心位置（世界座標）
        gripper_center_odom = self.get_gripper_center_in_odom()
        if gripper_center_odom is None:
            return
        
        # 計算夾爪到目標的距離（世界座標）
        to_target_x = target.x - gripper_center_odom.x
        to_target_y = target.y - gripper_center_odom.y
        dist_world = math.sqrt(to_target_x*to_target_x + to_target_y*to_target_y)
        
        # 計算直接朝向目標的 yaw（這是最可靠的方向）
        direct_yaw_world = math.atan2(to_target_y, to_target_x)
        
        # 獲取當前機器人的 yaw 和位置
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time()
            )
            q = transform.transform.rotation
            current_yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
            current_pos = transform.transform.translation
        except Exception as e:
            self.get_logger().warn(f"Failed to get current pose: {e}")
            return
        
        # ========================================
        # 近距離模式：三階段控制（後退→旋轉→前進）
        # ========================================
        if is_close_range:
            # 計算從 base_link 到目標的方向（這是機器人應該面對的方向）
            to_target_from_base_x = target.x - current_pos.x
            to_target_from_base_y = target.y - current_pos.y
            base_to_target_yaw = math.atan2(to_target_from_base_y, to_target_from_base_x)
            yaw_error = self.wrap_pi(base_to_target_yaw - current_yaw)
            
            # 第一次進入時記錄初始狀態
            if self.forward_target_dist is None:
                self.forward_target_dist = dist_world
                # 使用 base_link 到目標的方向，而不是夾爪到目標
                self.close_range_locked_yaw = base_to_target_yaw
                self.get_logger().info(
                    f"🎯 Close-range adjustment started:"
                )
                self.get_logger().info(
                    f"  Target (world): (X:{target.x:.3f}m, Y:{target.y:.3f}m, Z:{target.z:.3f}m)"
                )
                self.get_logger().info(
                    f"  Base position: (X:{current_pos.x:.3f}m, Y:{current_pos.y:.3f}m)"
                )
                self.get_logger().info(
                    f"  Gripper center: (X:{gripper_center_odom.x:.3f}m, Y:{gripper_center_odom.y:.3f}m)"
                )
                self.get_logger().info(
                    f"  Initial distance: {dist_world*100:.1f}cm, yaw_error: {math.degrees(yaw_error):.1f}°"
                )
                self.get_logger().info(
                    f"  Phase: {self.close_range_phase} (will backup {self.close_range_backup_dist*100:.0f}cm first)"
                )
            
            twist = Twist()
            
            # 階段 1: 直接後退（沿機器人當前 -X 方向）
            # 簡化版：不旋轉，直接後退，之後在旋轉階段再對齊正確方向
            if self.close_range_phase == 'backup':
                # 記錄後退起始位置
                if self.close_range_backup_start is None:
                    self.close_range_backup_start = Point()
                    self.close_range_backup_start.x = current_pos.x
                    self.close_range_backup_start.y = current_pos.y
                    self.get_logger().info(f"📍 Starting backup from ({current_pos.x:.3f}, {current_pos.y:.3f})")
                
                # 計算已後退距離
                backup_dx = current_pos.x - self.close_range_backup_start.x
                backup_dy = current_pos.y - self.close_range_backup_start.y
                backup_dist = math.sqrt(backup_dx*backup_dx + backup_dy*backup_dy)
                
                if backup_dist >= self.close_range_backup_dist:
                    # 後退完成，進入旋轉階段
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(twist)
                    self.close_range_phase = 'rotate'
                    
                    # 計算並鎖定目標 yaw（世界座標系）
                    # 這個值在整個旋轉過程中保持不變
                    gripper_in_base = self.get_gripper_center_in_base()
                    if gripper_in_base is not None:
                        gripper_offset_angle = math.atan2(gripper_in_base.y, gripper_in_base.x)
                    else:
                        gripper_offset_angle = 0.0
                    
                    # 鎖定目標方向（機器人應該轉到的世界座標系 yaw）
                    self.close_range_locked_yaw = self.wrap_pi(direct_yaw_world - gripper_offset_angle)
                    
                    self.get_logger().info(
                        f"✓ Backup complete ({backup_dist*100:.1f}cm). "
                        f"Locked target yaw: {math.degrees(self.close_range_locked_yaw):.1f}°. Now rotating..."
                    )
                    return
                else:
                    # 直接後退，不需要旋轉對齊
                    twist.linear.x = -0.08  # 8 cm/s 後退
                    twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(twist)
                    self.get_logger().info(
                        f"⏪ Backing up: {backup_dist*100:.1f}/{self.close_range_backup_dist*100:.0f}cm",
                        throttle_duration_sec=0.5
                    )
                    return
            
            # 階段 2: 旋轉對齊
            # 使用在後退完成時鎖定的目標 yaw
            elif self.close_range_phase == 'rotate':
                # 使用鎖定的目標 yaw（在後退完成時計算並儲存）
                if self.close_range_locked_yaw is not None:
                    desired_robot_yaw = self.close_range_locked_yaw
                else:
                    # 備用方案：直接朝向目標
                    gripper_in_base = self.get_gripper_center_in_base()
                    if gripper_in_base is not None:
                        gripper_offset_angle = math.atan2(gripper_in_base.y, gripper_in_base.x)
                    else:
                        gripper_offset_angle = 0.0
                    desired_robot_yaw = self.wrap_pi(direct_yaw_world - gripper_offset_angle)
                
                yaw_error = self.wrap_pi(desired_robot_yaw - current_yaw)
                
                # 旋轉對齊閾值
                rotate_thresh = math.radians(5)  # 5度
                
                if abs(yaw_error) <= rotate_thresh:
                    # 旋轉完成，進入前進階段
                    twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(twist)
                    self.close_range_phase = 'approach'
                    # 重置追蹤變數
                    self.close_range_min_dist = dist_world
                    self.close_range_approach_start_dist = dist_world
                    self.get_logger().info(
                        f"✓ Rotation complete (error: {math.degrees(yaw_error):.1f}°). "
                        f"Now approaching from {dist_world*100:.1f}cm..."
                    )
                    return
                else:
                    # 繼續旋轉（原地轉）
                    ang_cmd = max(-self.final_max_ang, min(self.final_max_ang, self.k_ang * yaw_error))
                    twist.linear.x = 0.0
                    twist.angular.z = ang_cmd
                    self.cmd_vel_pub.publish(twist)
                    self.get_logger().info(
                        f"🔄 Rotating: yaw_error={math.degrees(yaw_error):.1f}°, target_yaw={math.degrees(desired_robot_yaw):.1f}°, current_yaw={math.degrees(current_yaw):.1f}°",
                        throttle_duration_sec=0.5
                    )
                    return
            
            # 階段 3: 前進到目標
            # 使用 gripper 到目標的即時方向（不是鎖定的方向，因為需要微調）
            elif self.close_range_phase == 'approach':
                # 計算 gripper 相對於 base_link 的偏移角度
                gripper_in_base = self.get_gripper_center_in_base()
                if gripper_in_base is not None:
                    gripper_offset_angle = math.atan2(gripper_in_base.y, gripper_in_base.x)
                else:
                    gripper_offset_angle = 0.0
                
                # 在 approach 階段：使用即時的 gripper->target 方向來微調
                # 這樣可以補償任何偏移
                desired_robot_yaw = self.wrap_pi(direct_yaw_world - gripper_offset_angle)
                
                yaw_error = self.wrap_pi(desired_robot_yaw - current_yaw)
                
                # 更新最小距離追蹤
                if dist_world < self.close_range_min_dist:
                    self.close_range_min_dist = dist_world
                
                # 放寬距離增加的容許範圍（從 5cm 改成 8cm）
                # 因為在轉向過程中距離可能會稍微增加
                dist_increase_tolerance = 0.08
                
                # 檢查是否距離開始大幅增加（走錯方向）
                if dist_world > self.close_range_min_dist + dist_increase_tolerance:
                    # 增加重試計數器，避免無限循環
                    if not hasattr(self, '_close_range_retry_count'):
                        self._close_range_retry_count = 0
                    self._close_range_retry_count += 1
                    
                    if self._close_range_retry_count >= 3:
                        # 重試太多次，可能已經到達極限，視為完成
                        self.get_logger().warn(
                            f"⚠️ Reached minimum distance {self.close_range_min_dist*100:.1f}cm after {self._close_range_retry_count} retries. "
                            f"Considering task complete."
                        )
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                        self.cmd_vel_pub.publish(twist)
                        self.final_forward_done = True
                        self._close_range_retry_count = 0
                        return
                    
                    self.get_logger().warn(
                        f"⚠️ Distance increasing! min={self.close_range_min_dist*100:.1f}cm, "
                        f"current={dist_world*100:.1f}cm. Retry {self._close_range_retry_count}/3"
                    )
                    # 重新開始：後退再來一次
                    self.close_range_phase = 'backup'
                    self.close_range_backup_start = None
                    self.close_range_min_dist = float('inf')
                    return
                
                # 如果角度偏差太大（超過30度），先停下來旋轉
                if abs(yaw_error) > math.radians(30):
                    twist.linear.x = 0.0
                    twist.angular.z = max(-self.final_max_ang, min(self.final_max_ang, self.k_ang * yaw_error))
                    self.cmd_vel_pub.publish(twist)
                    self.get_logger().info(
                        f"🔄 Adjusting direction: yaw_error={math.degrees(yaw_error):.1f}°, dist={dist_world*100:.1f}cm",
                        throttle_duration_sec=0.5
                    )
                    return
                
                # 檢查是否到達最終目標
                if dist_world <= self.final_gripper_dist_thresh:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.cmd_vel_pub.publish(twist)
                    self.final_forward_done = True
                    self._close_range_retry_count = 0
                    self.get_logger().info(
                        f"✓ Close-range adjustment complete! Final distance: {dist_world*100:.1f}cm"
                    )
                    return
                
                # 正常前進：邊走邊微調方向
                # 速度根據距離調整，接近時減速
                lin_cmd = min(self.final_max_lin, self.k_lin * dist_world)
                lin_cmd = max(0.02, lin_cmd)  # 最小 2cm/s
                
                # 同時微調角度
                ang_cmd = max(-self.final_max_ang * 0.5, min(self.final_max_ang * 0.5, self.k_ang * yaw_error))
                
                twist.linear.x = lin_cmd
                twist.angular.z = ang_cmd
                self.cmd_vel_pub.publish(twist)
                
                self.get_logger().info(
                    f"➡️ Approaching: dist={dist_world*100:.1f}cm (min={self.close_range_min_dist*100:.1f}cm), "
                    f"yaw_error={math.degrees(yaw_error):.1f}°, lin={lin_cmd:.3f}",
                    throttle_duration_sec=0.5
                )
                return
            
            else:
                # 未知階段，重置為後退
                self.close_range_phase = 'backup'
                self.get_logger().warn("Unknown close-range phase, resetting to backup")
                return
        
        # ========================================
        # 完整模式：使用法向量前進，但監控距離變化
        # ========================================
        # 使用法向量方向，但要確保法向量指向目標方向
        normal_yaw = math.atan2(self.normal_vec_world[1], self.normal_vec_world[0])
        
        # 第一次進入時決定是否需要翻轉，並鎖定這個決定
        if not hasattr(self, '_normal_flipped'):
            self._normal_flipped = False
        if not hasattr(self, '_min_dist_achieved'):
            self._min_dist_achieved = float('inf')
        if not hasattr(self, '_use_direct_mode'):
            self._use_direct_mode = False
        
        if self.forward_target_dist is None:
            # 第一次進入：檢查法向量是否指向目標（與直接方向的夾角應小於 90°）
            angle_diff = abs(self.wrap_pi(normal_yaw - direct_yaw_world))
            if angle_diff > math.pi / 2:
                # 法向量指向相反方向，需要翻轉
                self._normal_flipped = True
            self._min_dist_achieved = dist_world
            self._use_direct_mode = False
        
        # 更新最小距離
        if dist_world < self._min_dist_achieved:
            self._min_dist_achieved = dist_world
        
        # 檢測是否距離開始增加（走錯方向了）
        # 如果當前距離比最小距離大超過 3cm，切換到直接模式
        if not self._use_direct_mode and dist_world > self._min_dist_achieved + 0.03:
            self._use_direct_mode = True
            self.get_logger().warn(
                f"⚠️ Distance increasing! Switching to direct mode. "
                f"min_dist={self._min_dist_achieved*100:.1f}cm, current={dist_world*100:.1f}cm"
            )
        
        # 根據模式選擇目標方向
        if self._use_direct_mode:
            # 直接模式：朝向目標
            target_yaw_world = direct_yaw_world
            mode_str = "direct"
        else:
            # 法向量模式
            if self._normal_flipped:
                normal_yaw = self.wrap_pi(normal_yaw + math.pi)
                mode_str = "normal(flipped)"
            else:
                mode_str = "normal"
            target_yaw_world = normal_yaw
        
        # 計算需要旋轉的角度
        yaw_error = self.wrap_pi(target_yaw_world - current_yaw)
        
        # 第一次進入時，記錄初始距離
        if self.forward_target_dist is None:
            self.forward_target_dist = dist_world
            self.get_logger().info(
                f"Moving along normal vector to target:"
            )
            self.get_logger().info(
                f"  Target (world): (X:{target.x:.3f}m, "
                f"Y:{target.y:.3f}m, Z:{target.z:.3f}m)"
            )
            self.get_logger().info(
                f"  Gripper center (world): (X:{gripper_center_odom.x:.3f}m, "
                f"Y:{gripper_center_odom.y:.3f}m, Z:{gripper_center_odom.z:.3f}m)"
            )
            self.get_logger().info(
                f"  Initial distance: {self.forward_target_dist:.3f}m"
            )
            self.get_logger().info(
                f"  Target yaw ({mode_str}): {math.degrees(target_yaw_world):.1f}°, "
                f"current yaw: {math.degrees(current_yaw):.1f}°, "
                f"yaw error: {math.degrees(yaw_error):.1f}°"
            )
        
        # 檢查是否到達目標
        if dist_world <= self.final_gripper_dist_thresh:
            stop = Twist()
            self.cmd_vel_pub.publish(stop)
            self.final_forward_done = True
            
            # 計算最終精度
            try:
                gripper_tf = self.tf_buffer.lookup_transform(
                    'odom',
                    'link_gripper_fingertip_left',
                    rclpy.time.Time()
                )
                if target is not None:
                    dx_final = gripper_tf.transform.translation.x - target.x
                    dy_final = gripper_tf.transform.translation.y - target.y
                    dz_final = gripper_tf.transform.translation.z - target.z
                    final_distance = math.sqrt(dx_final*dx_final + dy_final*dy_final + dz_final*dz_final)
                    self.get_logger().info(
                        f"✓ Reached target! Distance: {final_distance*100:.1f}cm "
                        f"(X:{dx_final*100:.1f}cm, Y:{dy_final*100:.1f}cm, Z:{dz_final*100:.1f}cm)"
                    )
            except:
                pass
            
            self.get_logger().info("Now adjusting lift and arm to final position...")
            return
        
        # 控制策略（完整模式）
        twist = Twist()
        
        if abs(yaw_error) > self.final_angle_allow:
            # 完整模式：角度誤差太大，先轉向
            ang_cmd = max(-self.final_max_ang, min(self.final_max_ang, self.k_ang * yaw_error))
            twist.angular.z = ang_cmd
            twist.linear.x = 0.0
            self.get_logger().info(
                f"Aligning ({mode_str}): yaw_error={math.degrees(yaw_error):.1f}°, "
                f"dist={dist_world:.3f}m"
            )
        else:
            # 完整模式：角度對齊後，直線前進
            # 速度依距離縮放，接近時自動降速
            lin_cmd = min(self.final_max_lin, self.k_lin * dist_world)
            twist.linear.x = max(0.0, lin_cmd)
            
            # 保持方向，小幅度修正
            ang_cmd = max(-self.final_max_ang * 0.3, min(self.final_max_ang * 0.3, self.k_ang * yaw_error))
            twist.angular.z = ang_cmd
            
            self.get_logger().info(
                f"Moving ({mode_str}): dist={dist_world:.3f}m, lin_cmd={lin_cmd:.3f}, "
                f"yaw_error={math.degrees(yaw_error):.1f}°"
            )
        
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = WhitePointFullMotion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

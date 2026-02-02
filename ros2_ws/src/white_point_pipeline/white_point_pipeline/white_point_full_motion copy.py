#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped, Twist
from std_msgs.msg import Float32
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.action import ActionClient
from rclpy.duration import Duration
import math

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

        # Joint trajectory client
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/stretch_controller/follow_joint_trajectory'
        )

        # TF buffer + listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        self.gripper_y_trim = 0.12  # 夾爪側向微調(公尺)，正值=往右補償
        self.gripper_offset_y = None

        # Target control state
        self.target_world = None         # <<<<<<<<< 目標改成 world 座標
        self.base_aligned = False
        self.joints_moved = False
        self.target_locked = False
        self.wrist_reset_done = False
        self.final_forward_done = False
        self.final_forward_dist = 0.36
        self.forward_target_dist = None
        self.orientation_aligned = False
        self.panel_axis_base = None
        self.desired_yaw_world = None
        self.approach_world = None
        
        # Action goal handles
        self.wrist_goal_handle = None
        self.wrist_sending = False
        self.joints_goal_handle = None
        self.joints_sending = False  # 防止重複發送
        self.joints_result = None    # 保存執行結果

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("WhitePointFullMotion initialized.")

    # ================================================
    # 收到白點 → 儲存到 world（odom）座標
    # ================================================
    def get_gripper_lateral_offset_in_base(self):
        """計算夾爪中心相對於 base_link 的側向偏移 (y)"""
        try:
            # 獲取左右指尖的位置
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
            
            # 計算夾爪中心的側向偏移 (y)
            left_y = tf_left.transform.translation.y
            right_y = tf_right.transform.translation.y
            center_y = (left_y + right_y) / 2.0

            self.get_logger().info(
                f"Gripper lateral offset in base_link: Y={center_y:.3f}"
            )
            
            # 轉成「目標點補償方向」：若夾爪在 base_link 左側(負Y)，
            # 目標點需要往右移動，所以回傳相反號
            return -center_y
            
        except Exception as e:
            self.get_logger().warn(f"Failed to get gripper TF: {e}")
            return None
    
    def target_callback(self, msg: PointStamped):

        if self.target_locked:
            return

        try:
            # 計算夾爪中心相對於 base_link 的偏移
            gripper_offset_y = self.get_gripper_lateral_offset_in_base()
            if gripper_offset_y is None:
                self.get_logger().warn("Cannot get gripper TF, using default offset")
                gripper_offset_y = -0.18  # 預設值（夾爪在 base_link 左側為負）
            # 使用手動微調修正固定偏差
            gripper_offset_y += self.gripper_y_trim
            self.gripper_offset_y = gripper_offset_y
            
            # base_link → odom 的 transform
            transform = self.tf_buffer.lookup_transform(
                'odom',      # 目標座標系
                'base_link', # 原始座標系
                rclpy.time.Time()
            )

            # 把白點從 base_link 轉到 odom
            world_point = do_transform_point(msg, transform)

            self.target_world = world_point.point  # 鎖住 world 座標
            self.target_locked = True
            self.base_aligned = False
            self.joints_moved = False
            self.wrist_reset_done = False
            self.final_forward_done = False  # 重置前進狀態
            self.forward_target_dist = None
            self.orientation_aligned = False
            self.desired_yaw_world = None
            self.approach_world = None
            
            # 重置 goal handles
            self.wrist_goal_handle = None
            self.wrist_sending = False
            self.joints_goal_handle = None
            self.joints_sending = False
            self.joints_result = None

            self.get_logger().info(
                f"Locked WORLD target (gripper offset Y={gripper_offset_y:.3f}): "
                f"Xw={self.target_world.x:.3f}, Yw={self.target_world.y:.3f}, Zw={self.target_world.z:.3f}"
            )

            if self.panel_axis_base is not None:
                self.update_desired_yaw_world()
                self.update_approach_world()

        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")

    def axis_callback(self, msg: Float32):
        """接收面板軸向角度"""
        self.panel_axis_base = float(msg.data)
        if self.target_locked and self.desired_yaw_world is None:
            self.update_desired_yaw_world()
            self.update_approach_world()

    # ================================================
    # 主循環
    # ================================================
    def control_loop(self):
        if self.target_world is None:
            return

        # Step 1: 世界座標 → base_link（每迴圈更新）
        base_target_no_offset = self.get_point_in_base(self.target_world, apply_offset=False)
        if base_target_no_offset is None:
            return

        base_target = base_target_no_offset
        if self.gripper_offset_y is not None:
            base_target = Point()
            base_target.x = base_target_no_offset.x
            base_target.y = base_target_no_offset.y + self.gripper_offset_y
            base_target.z = base_target_no_offset.z

        base_approach = None
        if self.panel_axis_base is not None:
            # 在 base_link 內直接算準備點，避免偏移造成世界座標轉換誤差
            theta_face = self.wrap_pi(self.panel_axis_base - math.pi / 2.0)
            angle_to_robot = math.atan2(-base_target_no_offset.y, -base_target_no_offset.x)
            theta_face_flipped = self.wrap_pi(theta_face + math.pi)
            if abs(self.wrap_pi(angle_to_robot - theta_face)) > abs(self.wrap_pi(angle_to_robot - theta_face_flipped)):
                theta_face = theta_face_flipped

            approach_x = base_target_no_offset.x + math.cos(theta_face) * self.approach_dist
            approach_y = base_target_no_offset.y + math.sin(theta_face) * self.approach_dist
            if self.gripper_offset_y is not None:
                approach_y += self.gripper_offset_y
            base_approach = Point()
            base_approach.x = approach_x
            base_approach.y = approach_y
            base_approach.z = base_target_no_offset.z
        desired_dist = 0.0
        if base_approach is None:
            base_approach = base_target
            desired_dist = self.base_desired_dist

        self.current_target = base_target

        # Step 2: 底盤對齊
        if not self.base_aligned:
            if not self.wrist_reset_done:
                # 如果還沒發送且沒有在發送中
                if not self.wrist_sending:
                    success = self.reset_wrist_and_head()
                    # wrist_reset_done 會在 callback 中設定
            self.control_base(base_approach, desired_dist)
            return
        
        # Step 2.5: 面板朝向對齊
        if not self.orientation_aligned:
            if self.desired_yaw_world is None:
                self.orientation_aligned = True
            else:
                if self.align_orientation():
                    self.orientation_aligned = True
            return

        # Step 3: 控制 lift 和 arm
        if not self.joints_moved:
            # 如果還沒發送 goal 且沒有在發送中
            if self.joints_goal_handle is None and not self.joints_sending:
                success = self.move_lift_and_arm()
                if not success:
                    return  # Server 未準備好,等下次
            # 檢查 goal 是否完成
            elif self.joints_result is not None:
                if self.joints_result.status == 4:  # STATUS_SUCCEEDED
                    self.joints_moved = True
                    self.joints_goal_handle = None
                    self.joints_result = None
                    # joints_sending 已在 callback 中清除
                    self.get_logger().info("Joints movement completed successfully!")
                else:
                    self.get_logger().warn(f'Joints goal failed with status: {self.joints_result.status}')
                    # 失敗了,重置以便重試
                    self.joints_goal_handle = None
                    self.joints_result = None
                    # joints_sending 已在 callback 中清除
                return  # 等待完成,不進入下一步
            else:
                # 還在等待結果
                return
                
        # Step 4: 完成 lift 後再往前 30 公分
        if not self.final_forward_done:
            self.move_forward_after_lift()
            return

    # ================================================
    # 將 world 座標轉成 base_link（這才會收斂）
    # ================================================
    def get_point_in_base(self, world_point, apply_offset=True):
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
            if apply_offset and self.gripper_offset_y is not None:
                base_point.point.y += self.gripper_offset_y
            return base_point.point

        except Exception as e:
            self.get_logger().warn(f"TF (odom→base) failed: {e}")
            return None

    # ================================================
    # 底盤控制
    # ================================================
    def control_base(self, target_point, desired_dist):
        x = target_point.x
        y = target_point.y

        distance = math.sqrt(x*x + y*y)
        angle = math.atan2(y, x)
        
        if abs(angle) <= self.angle_thresh and abs(distance - desired_dist) <= self.dist_thresh:
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            self.base_aligned = True
            self.get_logger().info("Base aligned!")
            return

        twist = Twist()

        # Rotation
        if abs(angle) > self.angle_thresh:
            ang_cmd = max(-self.max_ang, min(self.max_ang, self.k_ang * angle))
            twist.angular.z = ang_cmd
            self.get_logger().info(f"Rotating: angle={angle:.3f}, cmd={ang_cmd:.3f}")

        # Translation
        if abs(angle) < (20 * math.pi / 180.0):
            dist_error = distance - desired_dist
            lin_cmd = max(-self.max_lin, min(self.max_lin, self.k_lin * dist_error))
            twist.linear.x = lin_cmd
            self.get_logger().info(f"Moving forward: dist={distance:.3f}, cmd={lin_cmd:.3f}")

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

            base_target = self.get_point_in_base(self.target_world, apply_offset=False)
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

    def align_orientation(self):
        """旋轉底盤到面板朝向"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time()
            )
            q = transform.transform.rotation
            current_yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
        except Exception as e:
            self.get_logger().warn(f"TF yaw lookup failed: {e}")
            return False

        yaw_err = self.wrap_pi(self.desired_yaw_world - current_yaw)
        if abs(yaw_err) <= self.angle_thresh:
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("Orientation aligned!")
            return True

        twist = Twist()
        ang_cmd = max(-self.max_ang, min(self.max_ang, self.k_ang * yaw_err))
        twist.angular.z = ang_cmd
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"Orienting: yaw_err={yaw_err:.3f}, cmd={ang_cmd:.3f}")
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
        """同時重置 wrist 和調整 head 位置"""
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Joint controller not ready for wrist/head reset.')
            return False
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'joint_wrist_yaw', 
            'joint_wrist_pitch',
            'joint_wrist_roll',
            'joint_head_pan',
            'joint_head_tilt'
        ]

        point = JointTrajectoryPoint()
        point.positions = [
            math.pi/2,           # wrist yaw
            0.0,                 # wrist pitch
            0.0,                 # wrist roll
            math.radians(-30),     # head pan 保持中間
            math.radians(-60)    # head tilt 往下 60°
        ]
        point.time_from_start = Duration(seconds=2.0).to_msg()

        goal.trajectory.points = [point]
        
        # 標記正在發送
        self.wrist_sending = True
        
        # 使用 future 保存 goal handle
        send_goal_future = self.trajectory_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.wrist_goal_response_callback)

        self.get_logger().info("Wrist and head reset command sent.")
        return True
    
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

    def move_lift_and_arm(self):
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Joint controller not ready.')
            return False

        target_z = self.current_target.z
        target_x = self.current_target.x

        lift_target = max(self.lift_min, min(self.lift_max, target_z - self.gripper_z_offset))
        arm_target = max(self.arm_min, min(self.arm_max, target_x - self.base_desired_dist))

        self.get_logger().info(
            f"Command joints: lift={lift_target:.3f}, arm={arm_target:.3f}"
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_lift', 'wrist_extension', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll']

        point = JointTrajectoryPoint()
        point.positions = [lift_target, arm_target, math.pi/2, 0.0, 0.0]
        point.time_from_start = Duration(seconds=3.0).to_msg()

        goal.trajectory.points = [point]
        
        # 標記正在發送,防止重複
        self.joints_sending = True
        
        # 使用 future 保存 goal handle
        send_goal_future = self.trajectory_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.joints_goal_response_callback)

        self.get_logger().info("Lift and arm command sent.")
        return True
    
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
        """處理 joints goal 的執行結果"""
        result = future.result()  # 這是 GetResult 回應物件
        self.joints_result = result
        
        # 不論成功或失敗,都要清除 sending 標記
        self.joints_sending = False
        
        if result.status == 4:  # STATUS_SUCCEEDED
            self.get_logger().info('Joints action SUCCEEDED!')
        else:
            self.get_logger().warn(f'Joints action finished with status: {result.status}')

    def move_forward_after_lift(self):
        # 用現在的 target 與 base_desired_dist 推算剩餘距離
        x = self.current_target.x
        y = self.current_target.y
        dist = math.sqrt(x*x + y*y)

        # 第一次進入時，鎖定「前進前距離」避免過衝
        if self.forward_target_dist is None:
            self.forward_target_dist = max(0.05, dist - self.final_forward_dist)
            self.get_logger().info(
                f"Final forward target distance: {self.forward_target_dist:.3f}m (start {dist:.3f}m)"
            )

        error = dist - self.forward_target_dist

        if error <= 0.01:
            stop = Twist()
            self.cmd_vel_pub.publish(stop)
            self.final_forward_done = True  # 先標記完成,避免重複執行
            
            # 讓下一次可以重新鎖定 - 重置所有狀態
            self.target_world = None
            self.target_locked = False
            self.base_aligned = False
            self.joints_moved = False
            self.wrist_reset_done = False
            self.orientation_aligned = False
            self.desired_yaw_world = None
            self.forward_target_dist = None
            # final_forward_done 不重置,因為當前循環還在檢查
            
            # 重置 goal handles
            self.wrist_goal_handle = None
            self.wrist_sending = False
            self.joints_goal_handle = None
            self.joints_sending = False
            self.joints_result = None
            
            self.get_logger().info("Final forward completed. Ready for next target.")
            return

        # 速度依誤差縮放，接近時自動降速避免過衝
        twist = Twist()
        lin_cmd = min(0.1, self.k_lin * error)
        twist.linear.x = max(0.0, lin_cmd)
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = WhitePointFullMotion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

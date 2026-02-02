#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
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

        # Target control state
        self.target_world = None         # <<<<<<<<< 目標改成 world 座標
        self.base_aligned = False
        self.joints_moved = False
        self.target_locked = False
        self.wrist_reset_done = False
        self.final_forward_done = False
        self.final_forward_dist = 0.36

        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("WhitePointFullMotion initialized.")

    # ================================================
    # 收到白點 → 儲存到 world（odom）座標
    # ================================================
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

            # 把白點從 base_link 轉到 odom
            world_point = do_transform_point(msg, transform)

            self.target_world = world_point.point  # 鎖住 world 座標

            # 加入 gripper 的水平偏移（右邊 → 負的 y）
            GRIPPER_Y_OFFSET = 0.18   # 10 公分，你可調成 -0.07 到 -0.12
            self.target_world.y += GRIPPER_Y_OFFSET
            self.target_locked = True
            self.base_aligned = False
            self.joints_moved = False
            self.wrist_reset_done = False

            self.get_logger().info(
                f"Locked WORLD target: Xw={self.target_world.x:.3f}, "
                f"Yw={self.target_world.y:.3f}, Zw={self.target_world.z:.3f}"
            )

        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")

    # ================================================
    # 主循環
    # ================================================
    def control_loop(self):
        if self.target_world is None:
            return

        # Step 1: 世界座標 → base_link（每迴圈更新）
        base_p = self.get_target_in_base()

        if base_p is None:
            return

        self.current_target = base_p

        # Step 2: 底盤對齊
        if not self.base_aligned:
            if not self.wrist_reset_done:
                self.reset_wrist_and_head()
                self.wrist_reset_done = True
            self.control_base()
            return

        # Step 3: 控制 lift 和 arm
        if not self.joints_moved:
            self.move_lift_and_arm()
        # Step 4: 完成 lift 後再往前 30 公分
        if not self.final_forward_done:
            self.move_forward_after_lift()
            return

    # ================================================
    # 將 world 座標轉成 base_link（這才會收斂）
    # ================================================
    def get_target_in_base(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                'odom',
                rclpy.time.Time()
            )

            # 建立 PointStamped for TF
            ps = PointStamped()
            ps.header.frame_id = 'odom'
            ps.point = self.target_world

            base_point = do_transform_point(ps, transform)
            return base_point.point

        except Exception as e:
            self.get_logger().warn(f"TF (odom→base) failed: {e}")
            return None

    # ================================================
    # 底盤控制
    # ================================================
    def control_base(self):
        x = self.current_target.x
        y = self.current_target.y

        distance = math.sqrt(x*x + y*y)
        angle = math.atan2(y, x)
        

        if abs(angle) <= self.angle_thresh and abs(distance - self.base_desired_dist) <= self.dist_thresh:
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
            dist_error = distance - self.base_desired_dist
            lin_cmd = max(-self.max_lin, min(self.max_lin, self.k_lin * dist_error))
            twist.linear.x = lin_cmd
            self.get_logger().info(f"Moving forward: dist={distance:.3f}, cmd={lin_cmd:.3f}")

        self.cmd_vel_pub.publish(twist)

    # ================================================
    # 控制 lift 和 arm
    # ================================================
    def reset_wrist_and_head(self):
        """同時重置 wrist 和調整 head 位置"""
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
        self.trajectory_client.send_goal_async(goal)

        self.get_logger().info("Wrist and head reset: wrist yaw=90°, head tilt=-60°")

    def move_lift_and_arm(self):
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Joint controller not ready.')
            return

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
        self.trajectory_client.send_goal_async(goal)

        self.joints_moved = True
        self.target_locked = False

    def move_forward_after_lift(self):
        twist = Twist()
        twist.linear.x = 0.1  # 速度 0.1 m/s
        self.cmd_vel_pub.publish(twist)

        # 每次 timer 是 0.05s → 推進 0.3m 大約需要 3 秒
        # 但不要用 sleep，所以用距離判定

        # 用現在的 target 與 base_desired_dist 推算剩餘距離
        x = self.current_target.x
        y = self.current_target.y
        dist = math.sqrt(x*x + y*y)

        # 想往前 30 cm → 新目標距離 = 0.5 - 0.3 = 0.2 m
        target_dist = self.base_desired_dist - self.final_forward_dist

        if dist <= target_dist + 0.01:
            stop = Twist()
            self.cmd_vel_pub.publish(stop)
            self.final_forward_done = True
            self.get_logger().info("Final forward 20 cm completed.")

def main(args=None):
    rclpy.init(args=args)
    node = WhitePointFullMotion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
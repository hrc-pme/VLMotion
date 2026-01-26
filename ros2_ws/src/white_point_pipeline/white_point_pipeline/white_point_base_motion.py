#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
import math
import time

class WhitePointBaseMotion(Node):
    def __init__(self):
        super().__init__('white_point_base_motion')

        self.target_sub = self.create_subscription(
            PointStamped,
            '/white_point_base',
            self.target_callback,
            10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 目標距離與控制參數
        self.desired_dist = 0.5     # 希望按鈕距離 base_link 0.5 m
        self.k_lin = 0.6
        self.k_ang = 1.2
        self.max_lin = 0.3
        self.max_ang = 0.6
        self.angle_thresh = 3.0 * math.pi / 180.0  # 3 deg
        self.dist_thresh = 0.02                    # 2 cm

        self.current_target = None
        self.last_cmd_time = time.time()

        # 用 timer 做閉迴路控制
        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

    def target_callback(self, msg: PointStamped):
        # 直接把最新的目標點存起來
        self.current_target = msg.point
        self.get_logger().info(
            f'New target in base_link: x={msg.point.x:.3f}, y={msg.point.y:.3f}, z={msg.point.z:.3f}'
        )

    def control_loop(self):
        if self.current_target is None:
            return

        x = self.current_target.x
        y = self.current_target.y

        # 極座標：前方距離與偏角
        distance = math.sqrt(x*x + y*y)
        angle = math.atan2(y, x)

        twist = Twist()

        # 先轉正方向
        if abs(angle) > self.angle_thresh:
            ang_cmd = self.k_ang * angle
            ang_cmd = max(-self.max_ang, min(self.max_ang, ang_cmd))
            twist.angular.z = ang_cmd

        # 當角度差不大時再往前走
        if abs(angle) < (20.0 * math.pi / 180.0):
            dist_error = distance - self.desired_dist
            if abs(dist_error) > self.dist_thresh:
                lin_cmd = self.k_lin * dist_error
                lin_cmd = max(-self.max_lin, min(self.max_lin, lin_cmd))
                twist.linear.x = lin_cmd

        # 若角度與距離都在容許範圍內 → 停止
        if (abs(angle) <= self.angle_thresh and
                abs(distance - self.desired_dist) <= self.dist_thresh):
            twist = Twist()
            self.current_target = None
            self.get_logger().info('Reached target position. Stopping.')

        self.cmd_pub.publish(twist)
        self.last_cmd_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = WhitePointBaseMotion()
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

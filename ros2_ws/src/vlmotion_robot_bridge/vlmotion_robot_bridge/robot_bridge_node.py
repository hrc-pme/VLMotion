#!/usr/bin/env python3
"""Gate host velocity onto the Stretch driver.

Service `vlmotion` sets /vlmotion/enable_base_motion and this node copies
/vlmotion/cmd_vel to /stretch/cmd_vel. While the flag is false, the base
receives a zero Twist.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


class RobotBridge(Node):
    def __init__(self):
        super().__init__('vlmotion_robot_bridge')
        self.declare_parameter('odom_topic', '/stretch3/odom')
        self.declare_parameter('cmd_vel_topic', '/stretch/cmd_vel')
        odom_topic = self.get_parameter('odom_topic').value
        cmd_topic = self.get_parameter('cmd_vel_topic').value

        self.enable_base = False
        self.command = Twist()
        self.create_subscription(Bool, '/vlmotion/enable_base_motion', self._on_enable, 10)
        self.create_subscription(Twist, '/vlmotion/cmd_vel', self._on_cmd, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(f'gating {cmd_topic} from /vlmotion/cmd_vel (odom {odom_topic})')

    def _on_enable(self, msg: Bool):
        self.enable_base = bool(msg.data)

    def _on_cmd(self, msg: Twist):
        self.command = msg

    def _on_odom(self, _msg: Odometry):
        return

    def _tick(self):
        out = Twist()
        if self.enable_base:
            out.linear.x = float(self.command.linear.x)
            out.angular.z = float(self.command.angular.z)
        self.cmd_pub.publish(out)


def main():
    rclpy.init()
    node = RobotBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

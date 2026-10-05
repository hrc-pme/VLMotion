#!/usr/bin/env python3
"""Gate host velocity onto the Stretch driver.

Service `vlmotion` sets /vlmotion/enable_base_motion and this node copies
/vlmotion/cmd_vel to /stretch/cmd_vel while the driver is in navigation mode.
Publishing twists in position mode makes stretch_driver log errors every tick.
"""

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String


class RobotBridge(Node):
    def __init__(self):
        super().__init__('vlmotion_robot_bridge')
        self.declare_parameter('odom_topic', '/stretch3/odom')
        self.declare_parameter('cmd_vel_topic', '/stretch/cmd_vel')
        self.declare_parameter('mode_topic', '/stretch3/mode')
        self.declare_parameter('switch_to_navigation_on_start', True)
        odom_topic = self.get_parameter('odom_topic').value
        cmd_topic = self.get_parameter('cmd_vel_topic').value
        mode_topic = self.get_parameter('mode_topic').value

        self.enable_base = False
        self.command = Twist()
        self._robot_mode = ''
        self._forwarding = False
        self._nav_mode_done = False
        self._nav_mode_future = None

        self.create_subscription(Bool, '/vlmotion/enable_base_motion', self._on_enable, 10)
        self.create_subscription(Twist, '/vlmotion/cmd_vel', self._on_cmd, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(String, mode_topic, self._on_mode, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f'gating {cmd_topic} from /vlmotion/cmd_vel '
            f'(odom {odom_topic}, mode {mode_topic})'
        )

        if bool(self.get_parameter('switch_to_navigation_on_start').value):
            from std_srvs.srv import Trigger

            self._nav_mode_client = self.create_client(
                Trigger, '/switch_to_navigation_mode',
            )
            self.create_timer(1.0, self._ensure_navigation_mode)

    def _on_enable(self, msg: Bool):
        self.enable_base = bool(msg.data)

    def _on_cmd(self, msg: Twist):
        self.command = msg

    def _on_mode(self, msg: String):
        self._robot_mode = (msg.data or '').strip()

    def _on_odom(self, _msg: Odometry):
        return

    def _navigation_ready(self) -> bool:
        return self._robot_mode == 'navigation'

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def _tick(self):
        if self.enable_base and self._navigation_ready():
            out = Twist()
            out.linear.x = float(self.command.linear.x)
            out.angular.z = float(self.command.angular.z)
            self.cmd_pub.publish(out)
            self._forwarding = True
            return
        if self._forwarding and self._navigation_ready():
            self._publish_stop()
        self._forwarding = False

    def _ensure_navigation_mode(self):
        if self._nav_mode_done or not hasattr(self, '_nav_mode_client'):
            return
        if self._nav_mode_future is not None and not self._nav_mode_future.done():
            return
        if not self._nav_mode_client.service_is_ready():
            self.get_logger().info('waiting for /switch_to_navigation_mode…')
            return
        from std_srvs.srv import Trigger

        self._nav_mode_future = self._nav_mode_client.call_async(Trigger.Request())
        self._nav_mode_future.add_done_callback(self._on_navigation_mode_result)

    def _on_navigation_mode_result(self, future):
        self._nav_mode_done = True
        self._nav_mode_future = None
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'stretch driver: {response.message}')
            else:
                self.get_logger().warn(
                    f'switch_to_navigation_mode failed: {response.message}'
                )
        except Exception as exc:
            self.get_logger().warn(f'switch_to_navigation_mode call failed: {exc}')


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

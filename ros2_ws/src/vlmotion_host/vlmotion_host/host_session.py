#!/usr/bin/env python3
"""Turn a GUI target pixel into a base Twist.

The robot bridge is the only writer of /stretch/cmd_vel. This node publishes
/vlmotion/cmd_vel and /vlmotion/enable_base_motion. When base motion is off,
the twist is zero.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

# Same aliases as VLServo.ros_link.CAMERAS.
DEPTH_TOPICS = {
    'top camera': '/camera_top/camera_top/aligned_depth_to_color/image_raw',
    'head camera': '/head_camera/head_camera/aligned_depth_to_color/image_raw',
    'gripper camera': '/gripper_camera/gripper_camera/aligned_depth_to_color/image_raw',
}


def _sensor_qos():
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


class HostSession(Node):
    def __init__(self):
        super().__init__('vlmotion_host_session')
        self.declare_parameter('enable_base_motion', '0')
        self.declare_parameter('depth_image_topic', DEPTH_TOPICS['top camera'])
        self.declare_parameter('stop_dist_m', 0.8)
        self.declare_parameter('k_lin', 0.4)
        self.declare_parameter('k_ang', 0.8)

        raw = str(self.get_parameter('enable_base_motion').value).strip().lower()
        self.enable_base = raw in ('1', 'true', 'yes')
        self.stop_dist_m = float(self.get_parameter('stop_dist_m').value)
        self.k_lin = float(self.get_parameter('k_lin').value)
        self.k_ang = float(self.get_parameter('k_ang').value)

        self.running = False
        self.px = None
        self.py = None
        self.depth = None
        self.depth_w = 0
        self.depth_h = 0
        self._depth_sub = None
        self._camera = 'top camera'

        self.create_subscription(Bool, '/vlmotion/run', self._on_run, 10)
        self.create_subscription(Point, '/vlmotion/target_pixel', self._on_pixel, 10)
        self.create_subscription(String, '/vlmotion/user_input', self._on_text, 10)
        self.create_subscription(String, '/vlmotion/camera_select', self._on_camera, 10)
        self._subscribe_depth(self.get_parameter('depth_image_topic').value)

        self.enable_pub = self.create_publisher(Bool, '/vlmotion/enable_base_motion', 10)
        self.cmd_pub = self.create_publisher(Twist, '/vlmotion/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/vlmotion/status', 10)
        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f'enable_base_motion={self.enable_base} camera={self._camera}'
        )

    def _subscribe_depth(self, topic: str):
        if self._depth_sub is not None:
            self.destroy_subscription(self._depth_sub)
            self._depth_sub = None
        self.depth = None
        self.depth_w = 0
        self.depth_h = 0
        self._depth_sub = self.create_subscription(Image, topic, self._on_depth, _sensor_qos())
        self.get_logger().info(f'depth topic {topic}')

    def _on_camera(self, msg: String):
        name = msg.data.strip()
        topic = DEPTH_TOPICS.get(name)
        if topic is None or name == self._camera:
            return
        self._camera = name
        self._subscribe_depth(topic)

    def _on_run(self, msg: Bool):
        self.running = bool(msg.data)
        if not self.running:
            self.px = None
            self.py = None

    def _on_pixel(self, msg: Point):
        self.px = float(msg.x)
        self.py = float(msg.y)

    def _on_text(self, msg: String):
        self.get_logger().info(f'user_input: {msg.data}')

    def _on_depth(self, msg: Image):
        self.depth = msg
        self.depth_w = int(msg.width)
        self.depth_h = int(msg.height)

    def _depth_at(self, x: int, y: int):
        msg = self.depth
        if msg is None or msg.width == 0 or msg.height == 0:
            return None
        if msg.encoding not in ('16UC1', 'mono16'):
            return None
        x = max(0, min(int(msg.width) - 1, x))
        y = max(0, min(int(msg.height) - 1, y))
        step = msg.step if msg.step else msg.width * 2
        idx = y * step + x * 2
        raw = msg.data
        if idx + 1 >= len(raw):
            return None
        millimeters = raw[idx] | (raw[idx + 1] << 8)
        if millimeters <= 0:
            return None
        return millimeters / 1000.0

    def _tick(self):
        flag = Bool()
        flag.data = self.enable_base
        self.enable_pub.publish(flag)

        twist = Twist()
        status = 'idle'
        if self.running and self.px is not None and self.enable_base and self.depth_w > 0:
            cx = (self.depth_w - 1) / 2.0
            err = (self.px - cx) / max(1.0, float(self.depth_w))
            twist.angular.z = float(max(-0.4, min(0.4, -self.k_ang * err)))
            z_m = self._depth_at(int(self.px), int(self.py if self.py is not None else 0))
            if z_m is not None and z_m > self.stop_dist_m:
                twist.linear.x = float(max(0.0, min(0.25, self.k_lin * (z_m - self.stop_dist_m))))
            status = f'approach z={z_m}'
        elif self.running and not self.enable_base:
            status = 'base held'
        elif self.running and self.px is not None and self.enable_base:
            status = f'waiting for depth ({self._camera})'

        self.cmd_pub.publish(twist)
        text = String()
        text.data = status
        self.status_pub.publish(text)


def main():
    rclpy.init()
    node = HostSession()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

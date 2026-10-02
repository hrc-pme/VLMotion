#!/usr/bin/env python3
"""Turn a GUI target pixel into a base Twist.

The robot bridge is the only writer of /stretch/cmd_vel. This node publishes
/vlmotion/cmd_vel and /vlmotion/enable_base_motion. When base motion is off,
the twist is zero.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

# Pixels around the white dot. Zeros are holes. Values more than
# _DEPTH_GAP_M apart are a depth discontinuity and stay out of the mean.
_DEPTH_OFFSETS = (
    (0, 0),
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
)
_DEPTH_GAP_M = 0.30
_DEPTH_MIN_SAMPLES = 3


def decode_compressed_depth(data):
    """compressedDepth is a 12-byte header plus a PNG of 16-bit millimeters."""
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size <= 16 or raw[12:16].tobytes() != b'\x89PNG':
        return None
    img = cv2.imdecode(raw[12:], cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2 or img.dtype != np.uint16:
        return None
    return img


def _robust_depth(samples):
    """Mean of the largest cluster of valid depths within _DEPTH_GAP_M."""
    ordered = sorted(z for z in samples if z is not None and z > 0.0)
    best_mean = None
    best_count = 0
    count = len(ordered)
    end = 0
    for start in range(count):
        if end < start:
            end = start
        while end + 1 < count and ordered[end + 1] - ordered[start] <= _DEPTH_GAP_M:
            end += 1
        cluster = end - start + 1
        if cluster < _DEPTH_MIN_SAMPLES or cluster < best_count:
            continue
        mean = sum(ordered[start:end + 1]) / float(cluster)
        if cluster > best_count or best_mean is None or mean < best_mean:
            best_count = cluster
            best_mean = mean
    return best_mean


# Same aliases as VLServo.ros_link.CAMERAS.
DEPTH_TOPICS = {
    'top camera': '/camera_top/camera_top/aligned_depth_to_color/image_raw/compressedDepth',
    'head camera': '/head_camera/head_camera/aligned_depth_to_color/image_raw/compressedDepth',
    'gripper camera': '/gripper_camera/gripper_camera/aligned_depth_to_color/image_raw/compressedDepth',
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
        depth_topic = self.get_parameter('depth_image_topic').value
        self._camera = next(
            (name for name, topic in DEPTH_TOPICS.items() if topic == depth_topic),
            'top camera',
        )

        self.create_subscription(Bool, '/vlmotion/run', self._on_run, 10)
        self.create_subscription(Point, '/vlmotion/target_pixel', self._on_pixel, 10)
        self.create_subscription(String, '/vlmotion/user_input', self._on_text, 10)
        self.create_subscription(String, '/vlmotion/camera_select', self._on_camera, 10)
        self._subscribe_depth(depth_topic)

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
        self._depth_sub = self.create_subscription(
            CompressedImage, topic, self._on_depth, _sensor_qos(),
        )
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

    def _on_depth(self, msg: CompressedImage):
        img = decode_compressed_depth(msg.data)
        if img is None:
            return
        self.depth = img
        self.depth_h, self.depth_w = img.shape

    def _depth_mm(self, x: int, y: int):
        img = self.depth
        if img is None or x < 0 or y < 0 or y >= img.shape[0] or x >= img.shape[1]:
            return None
        millimeters = int(img[y, x])
        if millimeters <= 0:
            return None
        return millimeters / 1000.0

    def _depth_at(self, x: int, y: int):
        if self.depth is None:
            return None
        samples = [self._depth_mm(int(x) + dx, int(y) + dy) for dx, dy in _DEPTH_OFFSETS]
        return _robust_depth(samples)

    def _yaw(self):
        # Head camera is drawn rotated 90° clockwise. Left/right on that GUI
        # is the raw image's vertical axis: a dot left of center has a larger
        # raw y. Positive angular.z turns the base left. Up/down on the GUI
        # is raw x and is not used for steering.
        if self._camera == 'head camera' and self.py is not None and self.depth_h > 0:
            cy = (self.depth_h - 1) / 2.0
            err = (float(self.py) - cy) / max(1.0, float(self.depth_h))
            return float(max(-0.2, min(0.2, self.k_ang * err)))
        cx = (self.depth_w - 1) / 2.0
        err = (float(self.px) - cx) / max(1.0, float(self.depth_w))
        return float(max(-0.2, min(0.2, -self.k_ang * err)))

    def _tick(self):
        flag = Bool()
        flag.data = self.enable_base
        self.enable_pub.publish(flag)

        twist = Twist()
        status = 'idle'
        if self.running and self.px is not None and self.enable_base and self.depth_w > 0:
            twist.angular.z = self._yaw()
            z_m = self._depth_at(int(self.px), int(self.py if self.py is not None else 0))
            if z_m is not None and z_m > self.stop_dist_m:
                twist.linear.x = float(max(0.0, min(0.125, self.k_lin * (z_m - self.stop_dist_m))))
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

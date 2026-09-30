"""ROS 2 link used when the GUI runs on the 4060ti over Zenoh.

Enabled by VLMOTION_ROS_CAMERA=1. One node owns the camera subscription and the
outbound VLMotion topics so the Qt process does not spin two executors.
"""

import os
import threading

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


class RosLink(QObject):
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.enabled = os.environ.get('VLMOTION_ROS_CAMERA') == '1'
        self.node = None
        self._spinning = False
        self._thread = None

    def start(self):
        if not self.enabled or self.node is not None:
            return
        import rclpy
        from geometry_msgs.msg import Point
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import Bool, String

        if not rclpy.ok():
            rclpy.init()
        topic = os.environ.get(
            'CAMERA_IMAGE_TOPIC',
            '/camera/camera/color/image_raw/compressed',
        )
        self.node = rclpy.create_node('vlmotion_gui_link')
        self.user_pub = self.node.create_publisher(String, '/vlmotion/user_input', 10)
        self.pixel_pub = self.node.create_publisher(Point, '/vlmotion/target_pixel', 10)
        self.run_pub = self.node.create_publisher(Bool, '/vlmotion/run', 10)
        self.node.create_subscription(CompressedImage, topic, self._on_image, 10)
        self._spinning = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.status_changed.emit(f'ROS camera {topic}')

    def _loop(self):
        import rclpy
        while self._spinning and rclpy.ok() and self.node is not None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def _on_image(self, msg):
        import cv2
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        self.frame_received.emit({'color_image': bgr})

    def publish_user_input(self, text: str):
        if self.node is None or not text:
            return
        from std_msgs.msg import String
        msg = String()
        msg.data = text
        self.user_pub.publish(msg)

    def publish_run(self, running: bool):
        if self.node is None:
            return
        from std_msgs.msg import Bool
        msg = Bool()
        msg.data = bool(running)
        self.run_pub.publish(msg)

    def publish_pixel(self, x: int, y: int):
        if self.node is None:
            return
        from geometry_msgs.msg import Point
        msg = Point()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = 0.0
        self.pixel_pub.publish(msg)

    def stop(self):
        self._spinning = False
        self.publish_run(False)

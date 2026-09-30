"""ROS 2 link used when the GUI runs on the 4060ti over Zenoh.

The GUI picks a camera by alias. Those aliases match hellorobot_stretch3:

- ``head camera`` is the D435i at ``/head_camera/head_camera/...``
- ``top camera`` is the current D415 at ``/camera_top/camera_top/...``
"""

import threading

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

HEAD_CAMERA = 'head camera'
TOP_CAMERA = 'top camera'

CAMERAS = {
    TOP_CAMERA: {
        'color': '/camera_top/camera_top/color/image_raw/compressed',
        'depth': '/camera_top/camera_top/aligned_depth_to_color/image_raw',
    },
    HEAD_CAMERA: {
        'color': '/head_camera/head_camera/color/image_raw/compressed',
        'depth': '/head_camera/head_camera/aligned_depth_to_color/image_raw',
    },
}


def _sensor_qos():
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


class RosLink(QObject):
    frame_received = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.enabled = __import__('os').environ.get('VLMOTION_ROS_CAMERA') == '1'
        self.node = None
        self._spinning = False
        self._thread = None
        self._image_sub = None
        self._camera = TOP_CAMERA

    def start(self):
        if not self.enabled or self.node is not None:
            return
        import rclpy
        from geometry_msgs.msg import Point
        from std_msgs.msg import Bool, String

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('vlmotion_gui_link')
        self.user_pub = self.node.create_publisher(String, '/vlmotion/user_input', 10)
        self.pixel_pub = self.node.create_publisher(Point, '/vlmotion/target_pixel', 10)
        self.run_pub = self.node.create_publisher(Bool, '/vlmotion/run', 10)
        self.camera_pub = self.node.create_publisher(String, '/vlmotion/camera_select', 10)
        self._spinning = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.select(self._camera)

    def select(self, name: str):
        if name not in CAMERAS:
            name = TOP_CAMERA
        self._camera = name
        if self.node is None:
            return
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        if self._image_sub is not None:
            self.node.destroy_subscription(self._image_sub)
            self._image_sub = None
        topic = CAMERAS[name]['color']
        self._image_sub = self.node.create_subscription(
            CompressedImage, topic, self._on_image, _sensor_qos(),
        )
        msg = String()
        msg.data = name
        self.camera_pub.publish(msg)
        self.status_changed.emit(f'ROS camera {name}: {topic}')

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

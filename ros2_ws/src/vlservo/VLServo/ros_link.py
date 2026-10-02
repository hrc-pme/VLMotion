"""ROS 2 link used when the GUI runs on the 4060ti over Zenoh.

The GUI picks a camera by alias. Those aliases match hellorobot_stretch3:

- ``head camera`` is the D435i at ``/head_camera/head_camera/...``
- ``top camera`` is the current D415 at ``/camera_top/camera_top/...``
- ``gripper camera`` is the D405 at ``/gripper_camera/gripper_camera/...`` (color:
  ``color/image_rect_raw/compressed``)
"""

import os
import threading

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

HEAD_CAMERA = 'head camera'
TOP_CAMERA = 'top camera'
GRIPPER_CAMERA = 'gripper camera'

CAMERAS = {
    TOP_CAMERA: {
        'color': '/camera_top/camera_top/color/image_raw/compressed',
        'depth': '/camera_top/camera_top/aligned_depth_to_color/image_raw/compressedDepth',
    },
    HEAD_CAMERA: {
        'color': '/head_camera/head_camera/color/image_raw/compressed',
        'depth': '/head_camera/head_camera/aligned_depth_to_color/image_raw/compressedDepth',
    },
    GRIPPER_CAMERA: {
        # Stretch3 D405 publishes rectified color on image_rect_raw, not image_raw.
        'color': '/gripper_camera/gripper_camera/color/image_rect_raw/compressed',
        'depth': '/gripper_camera/gripper_camera/aligned_depth_to_color/image_raw/compressedDepth',
    },
}


def _resolve_camera_topics(name: str) -> dict:
    """Return compressed color/depth topics, with optional env overrides per alias."""
    spec = dict(CAMERAS.get(name, CAMERAS[TOP_CAMERA]))
    key = name.upper().replace(' ', '_')
    spec['color'] = os.environ.get(f'VLMOTION_{key}_COLOR_TOPIC', spec['color'])
    spec['depth'] = os.environ.get(f'VLMOTION_{key}_DEPTH_TOPIC', spec['depth'])
    return spec


def decode_compressed_depth(data):
    """compressedDepth is a 12-byte header plus a PNG of 16-bit millimeters."""
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size <= 16 or raw[12:16].tobytes() != b'\x89PNG':
        return None
    import cv2
    img = cv2.imdecode(raw[12:], cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 2 or img.dtype != np.uint16:
        return None
    return img


# Keep in step with vlmotion_host.host_session: same neighborhood and gap.
_DEPTH_OFFSETS = (
    (0, 0),
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
)
_DEPTH_GAP_M = 0.30
_DEPTH_MIN_SAMPLES = 3


def _robust_depth(samples):
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
        self._depth_sub = None
        self._depth_lock = threading.Lock()
        self._depth = None
        self._camera = TOP_CAMERA
        self._joints = {}
        self._grasp_wanted = False
        self._grasp_phase = 'idle'
        self._grasp_future = None
        self._grasp_px = None
        self._grasp_py = None
        self._grasp_log_t = 0.0
        self._tf = None
        self._camera_info = None
        self._camera_info_sub = None
        self._align_wanted = False
        self._align_phase = 'idle'
        self._align_future = None
        self._align_log_t = 0.0
        self._align_role = None
        self._gripper_cam_aligned = False
        self._color_decode_warned = False

    def start(self):
        if not self.enabled or self.node is not None:
            return
        import rclpy
        from geometry_msgs.msg import Point
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Float64MultiArray, String
        from std_srvs.srv import Trigger

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('vlmotion_gui_link')
        self.user_pub = self.node.create_publisher(String, '/vlmotion/user_input', 10)
        self.pixel_pub = self.node.create_publisher(Point, '/vlmotion/target_pixel', 10)
        self.run_pub = self.node.create_publisher(Bool, '/vlmotion/run', 10)
        self.camera_pub = self.node.create_publisher(String, '/vlmotion/camera_select', 10)
        self.pose_pub = self.node.create_publisher(Float64MultiArray, '/joint_pose_cmd', 1)
        self.node.create_subscription(JointState, '/stretch/joint_states', self._on_joints, 10)
        self._pos_client = self.node.create_client(Trigger, '/switch_to_position_mode')
        self._nav_client = self.node.create_client(Trigger, '/switch_to_navigation_mode')
        self._act_client = self.node.create_client(Trigger, '/activate_streaming_position')
        self._deact_client = self.node.create_client(Trigger, '/deactivate_streaming_position')
        from .stretch3_tf_grasp import Stretch3TfBuffer

        self._tf = Stretch3TfBuffer(self.node)
        self.node.create_timer(0.1, self._grasp_tick)
        self._spinning = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.select(self._camera)

    def _drop_camera_subscriptions(self):
        if self.node is None:
            return
        for attr in ('_image_sub', '_depth_sub', '_camera_info_sub'):
            sub = getattr(self, attr, None)
            if sub is not None:
                self.node.destroy_subscription(sub)
                setattr(self, attr, None)

    def select(self, name: str):
        if name not in CAMERAS:
            name = TOP_CAMERA
        self._camera = name
        if self.node is None:
            return
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String

        self._drop_camera_subscriptions()
        with self._depth_lock:
            self._depth = None
        self._camera_info = None
        self._color_decode_warned = False

        topics = _resolve_camera_topics(name)
        color_topic = topics['color']
        self._image_sub = self.node.create_subscription(
            CompressedImage, color_topic, self._on_image, _sensor_qos(),
        )
        self._depth_sub = self.node.create_subscription(
            CompressedImage, topics['depth'], self._on_depth, _sensor_qos(),
        )
        if name == GRIPPER_CAMERA:
            from sensor_msgs.msg import CameraInfo

            info_topic = os.environ.get(
                'VLMOTION_GRIPPER_CAMERA_INFO_TOPIC',
                '/gripper_camera/gripper_camera/color/camera_info',
            )
            self._camera_info_sub = self.node.create_subscription(
                CameraInfo, info_topic, self._on_camera_info, _sensor_qos(),
            )
        msg = String()
        msg.data = name
        self.camera_pub.publish(msg)
        self.status_changed.emit(f'ROS camera {name}')

    def _loop(self):
        import rclpy
        while self._spinning and rclpy.ok() and self.node is not None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def _emit_color_frame(self, bgr):
        if bgr is None:
            return
        self.frame_received.emit({'color_image': bgr})

    def _on_image(self, msg):
        import cv2
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            if self.node is not None and not self._color_decode_warned:
                self._color_decode_warned = True
                self.node.get_logger().warning(
                    f'failed to decode compressed color on {self._camera}'
                )
            return
        self._emit_color_frame(bgr)

    def _on_depth(self, msg):
        img = decode_compressed_depth(msg.data)
        if img is None:
            return
        with self._depth_lock:
            self._depth = img

    def _on_camera_info(self, msg):
        k = list(msg.k)
        self._camera_info = {
            'camera_matrix': np.array([
                [k[0], k[1], k[2]],
                [k[3], k[4], k[5]],
                [k[6], k[7], k[8]],
            ], dtype=np.float64),
            'distortion_coefficients': np.array(list(msg.d), dtype=np.float64),
            'width': int(msg.width),
            'height': int(msg.height),
        }

    def depth_meters(self, x, y):
        """Mean of nearby valid depths, matching host_session._depth_at."""
        with self._depth_lock:
            img = None if self._depth is None else self._depth
        if img is None:
            return None
        height, width = img.shape[:2]
        samples = []
        for dx, dy in _DEPTH_OFFSETS:
            xi = int(x) + dx
            yi = int(y) + dy
            if xi < 0 or yi < 0 or xi >= width or yi >= height:
                continue
            millimeters = int(img[yi, xi])
            if millimeters > 0:
                samples.append(millimeters / 1000.0)
        return _robust_depth(samples)

    def _on_joints(self, msg):
        self._joints = {name: position for name, position in zip(msg.name, msg.position)}

    def start_grasp(self, px, py):
        self._grasp_px = None if px is None else int(px)
        self._grasp_py = None if py is None else int(py)
        self._grasp_wanted = True

    def update_grasp_pixel(self, px, py):
        if px is None or py is None:
            return
        self._grasp_px = int(px)
        self._grasp_py = int(py)

    def stop_grasp(self):
        self._grasp_wanted = False

    def is_gripper_cam_aligned(self) -> bool:
        return bool(self._gripper_cam_aligned)

    def _begin_cam_alignment(self, role: str):
        if role == 'gripper' and self._camera != GRIPPER_CAMERA:
            return
        if role == 'head' and self._camera != HEAD_CAMERA:
            return
        self._align_role = role
        self._align_wanted = True
        self._align_phase = 'idle'
        self._align_future = None
        self._align_start_t = 0.0
        if role == 'gripper':
            self._gripper_cam_aligned = False

    def request_gripper_cam_alignment(self):
        """Drive wrist joints so gripper cam view matches arm extend direction."""
        self._begin_cam_alignment('gripper')

    def request_head_cam_alignment(self):
        """Drive head pan/tilt so head cam view matches arm extend direction."""
        self._begin_cam_alignment('head')

    def _qpos_hold(self):
        joints = self._joints
        names = (
            'joint_lift', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll',
            'joint_head_pan', 'joint_head_tilt', 'joint_gripper_finger_left',
        )
        if any(name not in joints for name in names):
            return None
        if 'wrist_extension' in joints:
            arm = float(joints['wrist_extension'])
        else:
            parts = [joints.get(name) for name in (
                'joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3',
            )]
            if any(part is None for part in parts):
                return None
            arm = float(sum(parts))
        return [
            arm,
            float(joints['joint_lift']),
            float(joints['joint_wrist_yaw']),
            float(joints['joint_wrist_pitch']),
            float(joints['joint_wrist_roll']),
            float(joints['joint_head_pan']),
            float(joints['joint_head_tilt']),
            float(joints['joint_gripper_finger_left']),
            0.0,
            0.0,
        ]

    def _call_trigger(self, client):
        from std_srvs.srv import Trigger
        if not client.service_is_ready():
            return None
        return client.call_async(Trigger.Request())

    def _grasp_tick(self):
        if self._align_wanted:
            self._align_tick()
        phase = self._grasp_phase
        future = self._grasp_future
        if self._grasp_wanted and phase == 'idle':
            pending = self._call_trigger(self._pos_client)
            if pending is None:
                return
            self._grasp_future = pending
            self._grasp_phase = 'to_position'
            self.node.get_logger().info('grasp: switching to position mode')
            return
        if phase in ('to_position', 'to_stream', 'to_deact', 'to_nav'):
            if future is None or not future.done():
                return
            if phase == 'to_position':
                if not self._grasp_wanted:
                    self._grasp_phase = 'idle'
                    self._grasp_future = None
                    return
                pending = self._call_trigger(self._act_client)
                if pending is None:
                    return
                self._grasp_future = pending
                self._grasp_phase = 'to_stream'
                self.node.get_logger().info('grasp: activating streaming position')
                return
            if phase == 'to_stream':
                self._grasp_future = None
                self._grasp_phase = 'run'
                if self._grasp_wanted:
                    self.node.get_logger().info('grasp: publishing /joint_pose_cmd')
                return
            if phase == 'to_deact':
                pending = self._call_trigger(self._nav_client)
                if pending is None:
                    return
                self._grasp_future = pending
                self._grasp_phase = 'to_nav'
                return
            self._grasp_future = None
            self._grasp_phase = 'idle'
            self.node.get_logger().info('grasp: streaming off, navigation mode')
            return
        if phase == 'run' and not self._grasp_wanted:
            pending = self._call_trigger(self._deact_client)
            if pending is None:
                return
            self._grasp_future = pending
            self._grasp_phase = 'to_deact'
            return
        if phase == 'run':
            self._publish_grasp_pose()

    def _alignment_qpos(self, qpos):
        from .stretch3_tf_grasp import (
            GRASP_WRIST_PITCH_RAD,
            GRASP_WRIST_ROLL_RAD,
            GRASP_WRIST_YAW_RAD,
        )

        qpos[2] = GRASP_WRIST_YAW_RAD
        qpos[3] = GRASP_WRIST_PITCH_RAD
        qpos[4] = GRASP_WRIST_ROLL_RAD
        return qpos

    def _align_tick(self):
        from .stretch3_tf_grasp import head_pan_tilt_step, optical_z_alignment_with_arm

        role = self._align_role or 'gripper'
        phase = self._align_phase
        future = self._align_future
        if phase == 'idle':
            pending = self._call_trigger(self._pos_client)
            if pending is None:
                return
            self._align_future = pending
            self._align_phase = 'to_position'
            return
        if phase == 'to_position':
            if future is None or not future.done():
                return
            pending = self._call_trigger(self._act_client)
            if pending is None:
                return
            self._align_future = pending
            self._align_phase = 'to_stream'
            return
        if phase == 'to_stream':
            if future is not None and not future.done():
                return
            self._align_future = None
            self._align_phase = 'run'
            return
        if phase == 'run':
            qpos = self._qpos_hold()
            if qpos is None:
                return
            if role == 'head':
                qpos = head_pan_tilt_step(qpos, self._tf)
            else:
                qpos = self._alignment_qpos(qpos)
            from std_msgs.msg import Float64MultiArray

            msg = Float64MultiArray()
            msg.data = [float(value) for value in qpos]
            self.pose_pub.publish(msg)
            now = self.node.get_clock().now().nanoseconds * 1e-9
            if self._align_start_t <= 0.0:
                self._align_start_t = now
            aligned = False
            if self._tf is not None:
                dot = optical_z_alignment_with_arm(self._tf, role)
                aligned = dot is not None and dot >= 0.92
            timeout_sec = 12.0 if role == 'head' else 8.0
            if not aligned and (now - self._align_start_t) >= timeout_sec:
                aligned = True
                self.node.get_logger().warn(
                    f'align {role} cam: TF alignment timeout; stopping alignment loop'
                )
            if aligned:
                self._align_wanted = False
                self._align_phase = 'idle'
                if role == 'gripper':
                    self._gripper_cam_aligned = True
                    self.status_changed.emit('Gripper cam aligned with arm axis')
                else:
                    self.status_changed.emit('Head cam aligned with arm axis')
                pending = self._call_trigger(self._deact_client)
                if pending is not None:
                    self._align_future = pending
                    self._align_phase = 'to_deact'
                return
            if now - self._align_log_t > 2.0:
                self._align_log_t = now
                self.node.get_logger().info(f'align {role} cam: adjusting pose')
            return
        if phase == 'to_deact':
            if future is None or not future.done():
                return
            pending = self._call_trigger(self._nav_client)
            if pending is None:
                return
            self._align_future = pending
            self._align_phase = 'to_nav'
            return
        if phase == 'to_nav':
            if future is not None and not future.done():
                return
            self._align_future = None
            self._align_phase = 'idle'
            if not self._gripper_cam_aligned:
                self._align_wanted = False

    def _publish_grasp_pose(self):
        from std_msgs.msg import Float64MultiArray
        qpos = self._qpos_hold()
        if qpos is None:
            return
        px, py = self._grasp_px, self._grasp_py
        with self._depth_lock:
            depth = None if self._depth is None else self._depth
        if px is not None and py is not None and depth is not None:
            z_m = self.depth_meters(px, py)
            height, width = depth.shape[:2]
            cx = (width - 1) / 2.0
            cy = (height - 1) / 2.0
            if self._camera == HEAD_CAMERA:
                yaw_err = (float(py) - cy) / max(height, 1)
                lift_err = (cx - float(px)) / max(width, 1)
            else:
                yaw_err = -(float(px) - cx) / max(width, 1)
                lift_err = (cy - float(py)) / max(height, 1)

            arm_err_m = None
            lift_err_m = None
            if (
                self._camera == GRIPPER_CAMERA
                and self._tf is not None
                and self._camera_info is not None
                and z_m is not None
            ):
                from .stretch3_tf_grasp import arm_lift_errors_m

                tf_err = arm_lift_errors_m(self._tf, px, py, z_m, self._camera_info)
                if tf_err is not None:
                    arm_err_m, lift_err_m = tf_err

            centered = abs(yaw_err) < 0.05 and abs(lift_err) < 0.05
            closing = z_m is not None and z_m <= 0.28 and centered
            if closing:
                v_arm = 0.0
                v_lift = 0.0
                rotate = 0.0
                grip_target = 0.0
            elif arm_err_m is not None and lift_err_m is not None:
                v_arm = max(-0.10, min(0.18, 0.55 * arm_err_m))
                v_lift = max(-0.06, min(0.06, 0.45 * lift_err_m))
                rotate = max(-0.015, min(0.015, 0.8 * yaw_err))
                grip_target = 0.45
            else:
                dist_err = 0.0 if z_m is None else (z_m - 0.38)
                v_arm = 0.0 if z_m is None else max(-0.10, min(0.18, 0.8 * dist_err))
                v_lift = max(-0.06, min(0.06, 0.6 * lift_err))
                rotate = max(-0.015, min(0.015, 0.8 * yaw_err))
                grip_target = 0.45
            qpos[0] = max(0.0, min(0.52, qpos[0] + v_arm * 0.1))
            qpos[1] = max(0.2, min(1.10, qpos[1] + v_lift * 0.1))
            qpos[7] = qpos[7] + max(-0.05, min(0.05, grip_target - qpos[7]))
            qpos[9] = rotate
            now = self.node.get_clock().now().nanoseconds * 1e-9
            if now - self._grasp_log_t > 2.0:
                self._grasp_log_t = now
                z_text = '--' if z_m is None else f'{z_m:.2f}'
                if arm_err_m is not None:
                    self.node.get_logger().info(
                        f'grasp tf arm_err={arm_err_m:+.3f} lift_err={lift_err_m:+.3f} '
                        f'arm={qpos[0]:.3f} lift={qpos[1]:.3f} z={z_text}'
                    )
                else:
                    self.node.get_logger().info(
                        f'grasp cmd arm={qpos[0]:.3f} lift={qpos[1]:.3f} '
                        f'yaw={qpos[9]:+.3f} z={z_text}'
                    )
        msg = Float64MultiArray()
        msg.data = [float(value) for value in qpos]
        self.pose_pub.publish(msg)

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

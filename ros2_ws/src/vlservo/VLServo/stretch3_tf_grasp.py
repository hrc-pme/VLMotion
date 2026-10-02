"""TF helpers for gripper-camera LLM grasping on Stretch3.

Uses /stretch3/tf (and /stretch3/tf_static) plus gripper camera intrinsics to
map a white-dot pixel + depth into arm extension and lift adjustments in
base_link.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np

from .d405_helpers_without_pyrealsense import pixel_to_3d

# Same nominal pose as pose_utils.go_to_start_pose (gripper / camera forward).
GRASP_WRIST_YAW_RAD = math.pi / 2.0
GRASP_WRIST_PITCH_RAD = 0.0
GRASP_WRIST_ROLL_RAD = 0.0

# stretch_driver.launch.py tf_prefix:=stretch3 → robot_state_publisher frame_prefix.
BASE_FRAME = os.environ.get('VLMOTION_BASE_FRAME', 'stretch3/base_link')
# joint_arm_l0 prismatic axis (stretch-se3-3092/exported_urdf/stretch.urdf).
ARM_FRAME = os.environ.get('VLMOTION_ARM_FRAME', 'stretch3/link_arm_l0')
# joint_lift prismatic axis (same URDF; not base_link +Z).
LIFT_FRAME = os.environ.get('VLMOTION_LIFT_FRAME', 'stretch3/link_lift')
# custom_test uses stretch3/camera_top_color_optical_frame; gripper URDF link is
# gripper_camera_color_optical_frame. Try prefixed name first, then RealSense node name.
_OPTICAL_FRAME_CANDIDATES = (
    'stretch3/gripper_camera_color_optical_frame',
    'gripper_camera_color_optical_frame',
)

TF_TOPIC = os.environ.get('VLMOTION_TF_TOPIC', '/stretch3/tf')
TF_STATIC_TOPIC = os.environ.get('VLMOTION_TF_STATIC_TOPIC', '/stretch3/tf_static')

# Standoff along arm axis (meters) before closing gripper.
DEFAULT_STANDOFF_M = float(os.environ.get('VLMOTION_GRASP_STANDOFF_M', '0.38'))


def _quat_to_rot_matrix(x, y, z, w):
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def _transform_point(translation, rotation, point):
    rot = _quat_to_rot_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
    t = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
    return rot @ np.asarray(point, dtype=np.float64) + t


class Stretch3TfBuffer:
    """Subscribe to Stretch3 TF topics (not /tf) and expose lookups."""

    def __init__(self, node):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from tf2_msgs.msg import TFMessage
        from tf2_ros import Buffer

        self._node = node
        self._buffer = Buffer()
        volatile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        node.create_subscription(TFMessage, TF_TOPIC, self._on_tf, volatile)
        node.create_subscription(TFMessage, TF_STATIC_TOPIC, self._on_tf_static, static_qos)
        self._optical_frame = None

    def _on_tf(self, msg):
        for transform in msg.transforms:
            self._buffer.set_transform(transform, 'vlmotion')

    def _on_tf_static(self, msg):
        for transform in msg.transforms:
            self._buffer.set_transform_static(transform, 'vlmotion')

    def lookup(self, target_frame: str, source_frame: str, timeout_sec: float = 0.25):
        from rclpy.duration import Duration
        from rclpy.time import Time
        from tf2_ros import TransformException

        try:
            return self._buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=timeout_sec),
            )
        except TransformException:
            return None

    def optical_frame(self) -> Optional[str]:
        override = os.environ.get('VLMOTION_GRIPPER_OPTICAL_FRAME')
        if override:
            return override
        if self._optical_frame is not None:
            return self._optical_frame
        for frame in _OPTICAL_FRAME_CANDIDATES:
            if self.lookup(BASE_FRAME, frame) is not None:
                self._optical_frame = frame
                return frame
        return None


def _link_prismatic_axis_in_base(tf_buffer: Stretch3TfBuffer, link_frame: str) -> Optional[np.ndarray]:
    trans = tf_buffer.lookup(BASE_FRAME, link_frame)
    if trans is None:
        return None
    rot = _quat_to_rot_matrix(
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w,
    )
    axis = rot[:, 2]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None
    return axis / norm


def arm_extension_unit(tf_buffer: Stretch3TfBuffer) -> Optional[np.ndarray]:
    """Unit vector in base_link for telescoping (arm extend/retract) direction."""
    return _link_prismatic_axis_in_base(tf_buffer, ARM_FRAME)


def lift_extension_unit(tf_buffer: Stretch3TfBuffer) -> Optional[np.ndarray]:
    """Unit vector in base_link for joint_lift (prismatic) direction."""
    return _link_prismatic_axis_in_base(tf_buffer, LIFT_FRAME)


def white_dot_in_base(
    tf_buffer: Stretch3TfBuffer,
    px: int,
    py: int,
    depth_m: float,
    camera_info: dict,
) -> Optional[np.ndarray]:
    """3D target point in base_link from gripper cam pixel + depth."""
    if camera_info is None or depth_m is None or depth_m <= 0.0:
        return None
    p_optical = pixel_to_3d(np.array([float(px), float(py)], dtype=np.float32), float(depth_m), camera_info)
    optical = tf_buffer.optical_frame()
    if optical is None:
        return None
    trans = tf_buffer.lookup(BASE_FRAME, optical)
    if trans is None:
        return None
    return _transform_point(trans.transform.translation, trans.transform.rotation, p_optical)


def camera_origin_in_base(tf_buffer: Stretch3TfBuffer) -> Optional[np.ndarray]:
    optical = tf_buffer.optical_frame()
    if optical is None:
        return None
    trans = tf_buffer.lookup(BASE_FRAME, optical)
    if trans is None:
        return None
    t = trans.transform.translation
    return np.array([t.x, t.y, t.z], dtype=np.float64)


def arm_lift_errors_m(
    tf_buffer: Stretch3TfBuffer,
    px: int,
    py: int,
    depth_m: float,
    camera_info: dict,
    standoff_m: float = DEFAULT_STANDOFF_M,
) -> Optional[Tuple[float, float]]:
    """Return (arm_extension_error_m, lift_error_m) in base_link."""
    target = white_dot_in_base(tf_buffer, px, py, depth_m, camera_info)
    origin = camera_origin_in_base(tf_buffer)
    arm_axis = arm_extension_unit(tf_buffer)
    lift_axis = lift_extension_unit(tf_buffer)
    if target is None or origin is None or arm_axis is None or lift_axis is None:
        return None
    delta = target - origin
    arm_err = float(np.dot(delta, arm_axis)) - float(standoff_m)
    lift_err = float(np.dot(delta, lift_axis))
    return arm_err, lift_err


def optical_z_alignment_with_arm(tf_buffer: Stretch3TfBuffer) -> Optional[float]:
    """Dot product between camera optical +Z and arm extension axis (1.0 = aligned)."""
    optical = tf_buffer.optical_frame()
    if optical is None:
        return None
    trans = tf_buffer.lookup(BASE_FRAME, optical)
    arm_axis = arm_extension_unit(tf_buffer)
    if trans is None or arm_axis is None:
        return None
    rot = _quat_to_rot_matrix(
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w,
    )
    optical_z = rot[:, 2]
    return float(np.dot(optical_z, arm_axis))


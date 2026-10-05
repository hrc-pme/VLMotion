"""TF helpers for Stretch3 camera / arm geometry (gripper + head cameras).

Uses /stretch3/tf (and /stretch3/tf_static) plus camera intrinsics to map
pixels to arm/lift motion and to align camera optical axes with arm extension.
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
# RealSense nodes use {camera_name}_color_optical_frame; robot_state_publisher
# prefixes URDF links with stretch3/ (see custom_test reloc.yaml for top cam).
_GRIPPER_OPTICAL_CANDIDATES = (
    'stretch3/gripper_camera_color_optical_frame',
    'gripper_camera_color_optical_frame',
)
_HEAD_OPTICAL_CANDIDATES = (
    'stretch3/head_camera_color_optical_frame',
    'head_camera_color_optical_frame',
    # RealSense node under /head_camera/head_camera/... (often on /tf, not /stretch3/tf).
    'head_camera_head_camera_color_optical_frame',
    'head_camera/head_camera_color_optical_frame',
)
# When head camera optical TF is absent from the Stretch URDF tree, approximate
# view direction from the tilt link (D435 mounts on link_head_tilt in URDF).
_HEAD_VIEW_FALLBACK_LINKS = (
    'stretch3/link_head_tilt',
    'stretch3/link_head_nav_cam',
    'stretch3/link_head',
)

_OPTICAL_ROLE = {
    'gripper': (_GRIPPER_OPTICAL_CANDIDATES, 'VLMOTION_GRIPPER_OPTICAL_FRAME'),
    'head': (_HEAD_OPTICAL_CANDIDATES, 'VLMOTION_HEAD_OPTICAL_FRAME'),
}

# joint_head_pan / joint_head_tilt limits (stretch-se3-3092/exported_urdf/stretch.urdf).
HEAD_PAN_LIMITS = (-3.9, 1.5)
HEAD_TILT_LIMITS = (-1.53, 0.79)
HEAD_ALIGN_GAIN = float(os.environ.get('VLMOTION_HEAD_ALIGN_GAIN', '0.35'))
GRIPPER_ALIGN_GAIN = float(os.environ.get('VLMOTION_GRIPPER_ALIGN_GAIN', '0.35'))
# GUI RYP → same-name wrist joint (yaw→wrist_yaw, roll→wrist_roll, pitch→wrist_pitch).
class GripperViewBias:
    yaw_deg = float(os.environ.get('VLMOTION_GRIPPER_VIEW_YAW_BIAS_DEG', '-90'))
    pitch_deg = float(os.environ.get('VLMOTION_GRIPPER_VIEW_PITCH_BIAS_DEG', '11'))
    roll_deg = float(os.environ.get('VLMOTION_GRIPPER_VIEW_ROLL_BIAS_DEG', '2.5'))


GRIPPER_VIEW_YAW_BIAS_DEG = GripperViewBias.yaw_deg
GRIPPER_VIEW_PITCH_BIAS_DEG = GripperViewBias.pitch_deg
GRIPPER_VIEW_ROLL_BIAS_DEG = GripperViewBias.roll_deg


def gripper_view_bias_deg() -> Tuple[float, float, float]:
    return GripperViewBias.yaw_deg, GripperViewBias.pitch_deg, GripperViewBias.roll_deg


def set_gripper_view_bias_deg(
    yaw_deg: Optional[float] = None,
    pitch_deg: Optional[float] = None,
    roll_deg: Optional[float] = None,
) -> Tuple[float, float, float]:
    if yaw_deg is not None:
        GripperViewBias.yaw_deg = float(yaw_deg)
    if pitch_deg is not None:
        GripperViewBias.pitch_deg = float(pitch_deg)
    if roll_deg is not None:
        GripperViewBias.roll_deg = float(roll_deg)
    return gripper_view_bias_deg()


def nudge_gripper_view_bias_deg(
    dyaw: float = 0.0,
    dpitch: float = 0.0,
    droll: float = 0.0,
) -> Tuple[float, float, float]:
    GripperViewBias.yaw_deg += float(dyaw)
    GripperViewBias.pitch_deg += float(dpitch)
    GripperViewBias.roll_deg += float(droll)
    return gripper_view_bias_deg()
WRIST_YAW_LIMITS = (-1.75, 4.0)
WRIST_PITCH_LIMITS = (-1.57, 0.56)
WRIST_ROLL_LIMITS = (-3.14, 3.14)

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
        if os.environ.get('VLMOTION_TF_INCLUDE_GLOBAL', '1') == '1':
            node.create_subscription(TFMessage, '/tf', self._on_tf, volatile)
            node.create_subscription(TFMessage, '/tf_static', self._on_tf_static, static_qos)
        self._optical_by_role = {}
        self._head_view_fallback_link = None

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

    def optical_frame(self, role: str = 'gripper') -> Optional[str]:
        spec = _OPTICAL_ROLE.get(role)
        if spec is None:
            return None
        candidates, env_key = spec
        override = os.environ.get(env_key)
        if override:
            return override
        cached = self._optical_by_role.get(role)
        if cached is not None:
            return cached
        for frame in candidates:
            if self.lookup(BASE_FRAME, frame) is not None:
                self._optical_by_role[role] = frame
                return frame
        if role == 'head':
            override_link = os.environ.get('VLMOTION_HEAD_VIEW_FALLBACK_LINK')
            links = (override_link,) if override_link else _HEAD_VIEW_FALLBACK_LINKS
            for link in links:
                if link and self.lookup(BASE_FRAME, link) is not None:
                    self._head_view_fallback_link = link
                    return link
        return None


def _axis_in_base(tf_buffer: Stretch3TfBuffer, frame: str, axis_col: int = 2) -> Optional[np.ndarray]:
    trans = tf_buffer.lookup(BASE_FRAME, frame)
    if trans is None:
        return None
    rot = _quat_to_rot_matrix(
        trans.transform.rotation.x,
        trans.transform.rotation.y,
        trans.transform.rotation.z,
        trans.transform.rotation.w,
    )
    col = int(np.clip(axis_col, 0, 2))
    axis = rot[:, col]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None
    return axis / norm


def _head_view_axis_in_base(tf_buffer: Stretch3TfBuffer) -> Optional[np.ndarray]:
    """Unit view axis for head camera in base_link (optical +Z or URDF fallback)."""
    frame = tf_buffer.optical_frame('head')
    if frame is None:
        return None
    if frame in _HEAD_OPTICAL_CANDIDATES or os.environ.get('VLMOTION_HEAD_OPTICAL_FRAME'):
        return _axis_in_base(tf_buffer, frame, 2)
    try:
        col = int(os.environ.get('VLMOTION_HEAD_FALLBACK_AXIS_COL', '0'))
    except ValueError:
        col = 0
    return _axis_in_base(tf_buffer, frame, col)


def _optical_z_in_base(tf_buffer: Stretch3TfBuffer, role: str) -> Optional[np.ndarray]:
    if role == 'head':
        return _head_view_axis_in_base(tf_buffer)
    optical = tf_buffer.optical_frame(role)
    if optical is None:
        return None
    return _axis_in_base(tf_buffer, optical, 2)


def _wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def base_forward_horizontal_unit() -> np.ndarray:
    """Unit forward in base_link (horizontal); Stretch uses +X forward, +Z up."""
    raw = os.environ.get('VLMOTION_BASE_FORWARD_AXIS', '1,0,0')
    try:
        parts = [float(part.strip()) for part in raw.split(',')]
        vec = np.array(parts[:3], dtype=np.float64)
    except (ValueError, IndexError):
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    xy = float(np.hypot(vec[0], vec[1]))
    if xy < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([vec[0] / xy, vec[1] / xy, 0.0], dtype=np.float64)


def gripper_target_view_axis_in_base() -> np.ndarray:
    """Compensated view axis in base_link (level forward in the real world)."""
    horiz = base_forward_horizontal_unit()
    yaw_rad = math.radians(GripperViewBias.yaw_deg)
    pitch_rad = math.radians(GripperViewBias.pitch_deg)
    cx, cy = float(horiz[0]), float(horiz[1])
    dx = cx * math.cos(yaw_rad) - cy * math.sin(yaw_rad)
    dy = cx * math.sin(yaw_rad) + cy * math.cos(yaw_rad)
    horiz_len = math.hypot(dx, dy)
    if horiz_len < 1e-9:
        dx, dy, horiz_len = 1.0, 0.0, 1.0
    c = math.cos(pitch_rad)
    s = math.sin(pitch_rad)
    vec = np.array([dx / horiz_len * c, dy / horiz_len * c, -s], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def _gripper_view_yaw_joint_rad() -> float:
    return GRASP_WRIST_YAW_RAD + math.radians(GripperViewBias.yaw_deg)


def _gripper_view_pitch_joint_rad() -> float:
    return GRASP_WRIST_PITCH_RAD - math.radians(GripperViewBias.pitch_deg)


def _gripper_view_roll_joint_rad() -> float:
    return GRASP_WRIST_ROLL_RAD + math.radians(GripperViewBias.roll_deg)


def _gripper_wrist_from_view_bias(qpos: list) -> list:
    """Open-loop wrist targets: GUI axis name matches wrist joint name."""
    qpos[2] = float(np.clip(
        _gripper_view_yaw_joint_rad(),
        WRIST_YAW_LIMITS[0],
        WRIST_YAW_LIMITS[1],
    ))
    qpos[3] = float(np.clip(
        _gripper_view_pitch_joint_rad(),
        WRIST_PITCH_LIMITS[0],
        WRIST_PITCH_LIMITS[1],
    ))
    qpos[4] = float(np.clip(
        _gripper_view_roll_joint_rad(),
        WRIST_ROLL_LIMITS[0],
        WRIST_ROLL_LIMITS[1],
    ))
    return qpos


def _gripper_nominal_wrist_from_bias(qpos: list) -> list:
    return _gripper_wrist_from_view_bias(qpos)


def gripper_wrist_step(qpos: list, tf_buffer: Stretch3TfBuffer) -> list:
    """Apply view RYP biases directly; TF only used for aligned check (not pan/tilt loop)."""
    qpos = _gripper_wrist_from_view_bias(qpos)
    if os.environ.get('VLMOTION_GRIPPER_TF_WRIST_TRIM', '0').strip() not in ('1', 'true', 'yes'):
        return qpos
    optical_z = _optical_z_in_base(tf_buffer, 'gripper')
    target = gripper_target_view_axis_in_base()
    if optical_z is None:
        return qpos
    pan_err = _wrap_pi(
        math.atan2(float(target[1]), float(target[0]))
        - math.atan2(float(optical_z[1]), float(optical_z[0])),
    )
    target_elev = math.atan2(
        float(target[2]),
        math.hypot(float(target[0]), float(target[1])),
    )
    opt_elev = math.atan2(
        float(optical_z[2]),
        math.hypot(float(optical_z[0]), float(optical_z[1])),
    )
    tilt_err = target_elev - opt_elev
    gain = GRIPPER_ALIGN_GAIN * 0.25
    qpos[2] = float(np.clip(
        qpos[2] + gain * pan_err,
        WRIST_YAW_LIMITS[0],
        WRIST_YAW_LIMITS[1],
    ))
    qpos[3] = float(np.clip(
        qpos[3] + gain * tilt_err,
        WRIST_PITCH_LIMITS[0],
        WRIST_PITCH_LIMITS[1],
    ))
    return qpos


def head_pan_tilt_step(qpos: list, tf_buffer: Stretch3TfBuffer) -> list:
    """Increment head pan/tilt so head cam view axis tracks arm extension."""
    view_axis = _head_view_axis_in_base(tf_buffer)
    arm_axis = arm_extension_unit(tf_buffer)
    if view_axis is None or arm_axis is None:
        return qpos
    optical_z = view_axis
    pan_err = _wrap_pi(
        math.atan2(float(arm_axis[1]), float(arm_axis[0]))
        - math.atan2(float(optical_z[1]), float(optical_z[0])),
    )
    arm_elev = math.atan2(float(arm_axis[2]), math.hypot(float(arm_axis[0]), float(arm_axis[1])))
    opt_elev = math.atan2(float(optical_z[2]), math.hypot(float(optical_z[0]), float(optical_z[1])))
    tilt_err = arm_elev - opt_elev
    qpos[5] = float(np.clip(
        qpos[5] + HEAD_ALIGN_GAIN * pan_err,
        HEAD_PAN_LIMITS[0],
        HEAD_PAN_LIMITS[1],
    ))
    qpos[6] = float(np.clip(
        qpos[6] + HEAD_ALIGN_GAIN * tilt_err,
        HEAD_TILT_LIMITS[0],
        HEAD_TILT_LIMITS[1],
    ))
    return qpos


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
    optical = tf_buffer.optical_frame('gripper')
    if optical is None:
        return None
    trans = tf_buffer.lookup(BASE_FRAME, optical)
    if trans is None:
        return None
    return _transform_point(trans.transform.translation, trans.transform.rotation, p_optical)


def camera_origin_in_base(tf_buffer: Stretch3TfBuffer) -> Optional[np.ndarray]:
    optical = tf_buffer.optical_frame('gripper')
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


def optical_z_alignment_with_arm(tf_buffer: Stretch3TfBuffer, role: str = 'gripper') -> Optional[float]:
    """Dot product between camera optical +Z and arm extension axis (1.0 = aligned)."""
    optical_z = _optical_z_in_base(tf_buffer, role)
    arm_axis = arm_extension_unit(tf_buffer)
    if optical_z is None or arm_axis is None:
        return None
    return float(np.dot(optical_z, arm_axis))


def optical_z_alignment_with_forward(
    tf_buffer: Stretch3TfBuffer,
    role: str = 'gripper',
) -> Optional[float]:
    """Dot product between camera optical +Z and compensated forward (1.0 = aligned)."""
    optical_z = _optical_z_in_base(tf_buffer, role)
    if optical_z is None:
        return None
    if role == 'gripper':
        forward = gripper_target_view_axis_in_base()
    else:
        forward = base_forward_horizontal_unit()
    return float(np.dot(optical_z, forward))


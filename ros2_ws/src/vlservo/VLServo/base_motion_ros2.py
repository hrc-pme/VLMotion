#!/usr/bin/env python3
"""
ROS2 Base Motion - Visual Servoing Navigation
Replaces stretch_body-based base_motion.py with pure ROS2 implementation
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import Image, CameraInfo, JointState
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as TrajectoryDuration
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point
from tf2_msgs.msg import TFMessage
import numpy as np
import cv2
import math
import time
import argparse
import sys
from threading import Lock, Thread
import os
from dataclasses import dataclass

from VLServo.white_point_tracker import WhitePointTracker
from VLServo.stretch_tf import StretchTransforms
from VLServo.gui_rotation import (
    compute_display_offsets,
    normalized_display_position,
    resolve_rotation_degrees,
)


def robust_depth_at_pixel(depth_image, px, py, kernel_sizes=(3, 5, 7, 9, 11)):
    """
    Lookup a reliable depth measurement by taking the median of a small window
    around the requested pixel. Expands the window until a non-zero/non-NaN depth
    is found or the search sizes are exhausted.
    """
    if depth_image is None:
        return None
    h, w = depth_image.shape[:2]
    if h == 0 or w == 0:
        return None
    px = int(np.clip(round(px), 0, w - 1))
    py = int(np.clip(round(py), 0, h - 1))

    for k in kernel_sizes:
        half = k // 2
        x0 = max(0, px - half)
        y0 = max(0, py - half)
        x1 = min(w, px + half + 1)
        y1 = min(h, py + half + 1)
        patch = depth_image[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        vals = patch.reshape(-1)
        if np.issubdtype(vals.dtype, np.integer):
            vals = vals.astype(np.float32)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0]
        if vals.size > 0:
            return float(np.median(vals))
    return None


@dataclass
class PostGraspConfig:
    """Tune the post-navigation gripper alignment stage."""

    base_turn_deg: float = -60.0  # positive = rotate chassis left (gripper is at y=-0.21m, needs large rotation)
    head_pan_deg: float = -60.0  # bias (deg) applied to the computed head pan angle after rotation
    lift_offset_m: float = 0.10  # Height above target (0.1m = 10cm clearance above object)
    min_lift_m: float = 0.05
    max_lift_m: float = 1.05
    arm_offset_m: float = 0.32  # distance from base_link to gripper when retracted
    max_arm_extension_m: float = 0.5
    lift_duration_s: float = 3.0
    arm_duration_s: float = 3.5
    wrist_yaw_deg: float = 0.0  # 0° = gripper points forward, +90° = left (collision), -90° = right (safe)
    wrist_yaw_duration_s: float = 1.5
    wrist_roll_deg: float = 0.0  # rotate gripper fingers to point down/forward (0 = horizontal)
    wrist_roll_duration_s: float = 1.5
    wrist_pitch_deg: float = None  # optional pitch adjustment
    gripper_reference_frame: str = 'link_grasp_center'


class VisualServoNavigator(Node):
    """Navigate robot to target point using visual servoing"""
    
    def __init__(self, target_x, target_y, camera_name='d435i', 
                 stop_distance_m=1.0, tilt_only=False, 
                 tilt_down_negative=True, invert_yaw=False,
                 base_frame='base_link', post_grasp_config=None):
        super().__init__('visual_servo_navigator')
        
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.camera_name = camera_name
        self.stop_distance_m = stop_distance_m
        self.tilt_only = tilt_only
        self.tilt_down_negative = tilt_down_negative
        self.invert_yaw = invert_yaw
        self.base_frame = base_frame
        self.post_grasp_config = post_grasp_config or PostGraspConfig()
        
        self.bridge = CvBridge()
        self.lock = Lock()
        
        # Image and camera info
        self.color_image = None
        self.color_camera_info = None
        self.depth_camera_info = None
        self.image_width = 1280
        self.image_height = 720
        self.depth_image_raw = None
        self.raw_depth_scale = 0.001  # RealSense Z16 publishes millimeters by default
        self.raw_depth_frame_id = None
        self.color_camera_matrix = None
        self.depth_camera_matrix = None
        self.tracker = None
        self.tracker_template_size = 41
        self.tracker_search_radius = 45
        self.tracker_visible = False
        self.tf_buffer = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_timeout = RclpyDuration(seconds=1.0)
        self.urdf_transforms = None
        self.urdf_failed = False
        self.urdf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'stretch_uncalibrated.urdf'
        )
        
        # Control parameters
        self.linear_gain = 0.5
        self.angular_gain = 0.8
        self.max_linear_vel = 0.15  # m/s
        self.max_angular_vel = 0.3  # rad/s
        self.min_linear_vel = 0.03  # ensure forward progress when far
        self.angular_deadband_rad = math.radians(2.0)
        self.angular_deadband_px_frac = 0.02  # fraction of image width considered centered
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.tracked_pixel_pub = self.create_publisher(PointStamped, '/visual_servo/tracked_point', 10)
        
        # Service clients for mode switching
        self.nav_mode_client = self.create_client(Trigger, '/switch_to_navigation_mode')
        self.pos_mode_client = self.create_client(Trigger, '/switch_to_position_mode')
        
        # Subscribers
        # RealSense topics: /d435i/color/image_raw, /d435i/depth/image_rect_raw
        color_topic = f'/{camera_name}/color/image_raw'
        # Use raw depth for better compatibility
        raw_depth_topic = f'/{camera_name}/depth/image_rect_raw'
        color_info_topic = f'/{camera_name}/color/camera_info'
        depth_info_topic = f'/{camera_name}/depth/camera_info'
        
        self.get_logger().info(f'Subscribing to color: {color_topic}')
        self.get_logger().info(f'Subscribing to depth: {raw_depth_topic}')
        
        self.color_sub = self.create_subscription(
            Image, color_topic, self.color_callback, 10)
        # Subscribe to raw depth (primary source)
        self.raw_depth_sub = self.create_subscription(
            Image, raw_depth_topic, self.raw_depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, color_info_topic, self.info_callback, 10)
        self.depth_info_sub = self.create_subscription(
            CameraInfo, depth_info_topic, self.depth_info_callback, 10)
        self.joint_state_lock = Lock()
        self.joint_positions = {}
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/stretch/joint_states',
            self.joint_state_callback,
            10
        )
        self.missing_tf_frames = set()
        self.failed_tf_pairs = set()
        self.urdf_fallback_pairs = set()
        
        # Head control / general joint trajectory client
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/stretch_controller/follow_joint_trajectory'
        )
        self.trajectory_action_ready = False
        self.head_goal_handle = None
        self.last_head_command_time = time.time()
        self.head_command_interval = 0.30  # slow down head updates to avoid sudden swings
        self.head_move_duration = 0.35
        self.head_pan_limits = (-1.5, 1.5)
        self.head_tilt_limits = (-1.4, 0.35)
        self.head_pan_gain = 0.5
        self.head_tilt_gain = 0.9
        self.head_pan_step = 0.10
        self.head_tilt_step = 0.15
        self.head_pan_edge_suppression = 0.2  # only pan head when target is near image center
        
        self.image_rotation_deg = resolve_rotation_degrees(default_deg=-90.0)
        self.get_logger().info(f'Image rotation compensation set to {self.image_rotation_deg:.1f}°')
        self.last_tracker_update = time.time()
        self.tracker_grace_s = 1.0
        self.stop_hold_counter = 0
        self.stop_hold_required = 8
        self.spin_thread = None
        self.spin_running = False
        self.spin_sleep = 0.02
        self.in_navigation_mode = False
        self._last_base_rotation_rad = 0.0  # Track base rotation for wrist yaw compensation
        
        self.get_logger().info(f'Visual Servo Navigator initialized')
        self.get_logger().info(f'Target: ({target_x}, {target_y})')
        self.get_logger().info(f'Stop distance: {stop_distance_m}m')
        self.get_logger().info(f'Camera: {camera_name}')
        self.get_logger().info(f'Base frame: {self.base_frame}')
    
    def color_callback(self, msg):
        """Callback for color image"""
        with self.lock:
            try:
                # cv_bridge automatically handles RGB->BGR conversion when using 'bgr8'
                # No need for manual cvtColor conversion
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.image_height, self.image_width = self.color_image.shape[:2]
            except Exception as e:
                self.get_logger().error(f'Error in color callback: {e}')
    
    def depth_callback(self, msg):
        """Callback for raw depth image (primary source for d435i)"""
        with self.lock:
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Track scale based on dtype (floats are already meters)
                if np.issubdtype(depth.dtype, np.floating):
                    self.raw_depth_scale = 1.0
                else:
                    self.raw_depth_scale = 0.001
                self.depth_image_raw = depth
                normalized_frame = self._normalize_frame_id(msg.header.frame_id)
                resolved_frame = self._ensure_optical_frame(normalized_frame or msg.header.frame_id)
                self.raw_depth_frame_id = resolved_frame or self.raw_depth_frame_id
            except Exception as e:
                self.get_logger().error(f'Error in depth callback: {e}')

    def raw_depth_callback(self, msg):
        """Alias for depth_callback for compatibility"""
        self.depth_callback(msg)

    def info_callback(self, msg):
        """Callback for camera info"""
        first = self.color_camera_info is None
        self.color_camera_info = msg
        self.color_camera_matrix = np.array(msg.k).reshape((3, 3))
        if first:
            self.get_logger().info('Camera info received')

    def depth_info_callback(self, msg):
        """Callback for depth camera info"""
        first = self.depth_camera_info is None
        self.depth_camera_info = msg
        self.depth_camera_matrix = np.array(msg.k).reshape((3, 3))
        if first:
            self.get_logger().info('Depth camera info received')
    
    def joint_state_callback(self, msg):
        """Track head joint positions for alignment control."""
        with self.joint_state_lock:
            for idx, name in enumerate(msg.name):
                if idx < len(msg.position):
                    self.joint_positions[name] = msg.position[idx]
    
    def _color_pixel_to_depth_pixel(self, px, py):
        """Approximate the corresponding depth pixel for a color pixel.

        RealSense publishes color at 1280x720 and depth at 848x480, so this scaling
        keeps the relative UV coordinate even though the imagers are offset. It acts
        as a last-resort depth lookup when aligned depth is unavailable.
        """
        if self.depth_camera_info is None or self.image_width <= 0 or self.image_height <= 0:
            return None, None
        depth_w = int(self.depth_camera_info.width)
        depth_h = int(self.depth_camera_info.height)
        if depth_w <= 0 or depth_h <= 0:
            return None, None
        scale_x = float(depth_w) / float(self.image_width)
        scale_y = float(depth_h) / float(self.image_height)
        px_d = int(np.clip(round(px * scale_x), 0, depth_w - 1))
        py_d = int(np.clip(round(py * scale_y), 0, depth_h - 1))
        return px_d, py_d

    def _sample_aligned_depth(self, x, y):
        """
        Sample depth at color pixel coordinates.
        Since we use raw depth, we need to map color pixels to depth coordinates.
        """
        with self.lock:
            depth_image = self.depth_image_raw
            scale = self.raw_depth_scale
            camera_matrix = self.depth_camera_matrix
            frame_id = self.raw_depth_frame_id
        
        if depth_image is None or camera_matrix is None:
            return None
        
        # Map color pixel to depth pixel
        px_d, py_d = self._color_pixel_to_depth_pixel(x, y)
        if px_d is None or py_d is None:
            return None
        
        depth_raw = robust_depth_at_pixel(depth_image, px_d, py_d)
        if depth_raw is None:
            return None
        
        depth_m = float(depth_raw) * float(scale)
        return {
            'depth_m': depth_m,
            'pixel_x': float(px_d),
            'pixel_y': float(py_d),
            'color_pixel': (float(x), float(y)),
            'camera_matrix': camera_matrix,
            'frame_id': self._ensure_optical_frame(frame_id),
            'source': 'depth'
        }

    def _sample_raw_depth(self, x, y):
        """Fallback: sample raw depth by scaling color pixel into depth frame."""
        with self.lock:
            depth_image = self.depth_image_raw
            scale = self.raw_depth_scale
            camera_matrix = self.depth_camera_matrix
            frame_id = self.raw_depth_frame_id
        if depth_image is None or camera_matrix is None:
            return None
        px_d, py_d = self._color_pixel_to_depth_pixel(x, y)
        if px_d is None or py_d is None:
            return None
        depth_raw = robust_depth_at_pixel(depth_image, px_d, py_d)
        if depth_raw is None:
            return None
        depth_m = float(depth_raw) * float(scale)
        return {
            'depth_m': depth_m,
            'pixel_x': float(px_d),
            'pixel_y': float(py_d),
            'color_pixel': (float(x), float(y)),
            'camera_matrix': camera_matrix,
            'frame_id': self._ensure_optical_frame(frame_id),
            'source': 'raw'
        }

    def get_depth_at_point(self, x, y):
        """
        Get depth sample info at the tracked pixel.
        Now uses raw depth as primary source.
        """
        sample = self._sample_aligned_depth(x, y)
        if sample is not None:
            return sample
        # Fallback to raw depth method
        return self._sample_raw_depth(x, y)
    
    def pixel_to_3d(self, px, py, depth_m, camera_matrix=None):
        """Project pixel coordinate into the camera optical frame using stored intrinsics.
        
        Camera optical frame convention (ROS REP-103):
        - X: right
        - Y: down  
        - Z: forward (depth direction)
        """
        matrix = camera_matrix if camera_matrix is not None else self.color_camera_matrix
        if matrix is None or depth_m is None or depth_m <= 0:
            return None
        fx = matrix[0, 0]
        fy = matrix[1, 1]
        cx = matrix[0, 2]
        cy = matrix[1, 2]
        if fx == 0.0 or fy == 0.0:
            return None
        
        # Standard pinhole camera model
        # Camera optical frame: +X right, +Y down, +Z forward
        x = ((px - cx) * depth_m) / fx
        y = ((py - cy) * depth_m) / fy
        z = depth_m
        
        return np.array([x, y, z], dtype=np.float32)

    def transform_point_to_base(self, point_xyz, source_frame):
        """Transform a camera-frame 3D point into the base frame using TF."""
        if point_xyz is None or source_frame is None:
            return None
        if self.tf_buffer is None:
            return None
        transformed = self._transform_point_between_frames(point_xyz, source_frame, self.base_frame)
        if transformed is not None:
            return transformed
        return self._fallback_transform(point_xyz, source_frame)

    def _load_urdf_transforms(self):
        """Lazy-load the Stretch URDF for TF fallbacks."""
        if self.urdf_transforms is not None or self.urdf_failed:
            return
        try:
            self.urdf_transforms = StretchTransforms(self.urdf_path)
            self.get_logger().info('Loaded Stretch URDF transforms for TF fallback')
        except Exception as exc:
            self.urdf_failed = True
            self.get_logger().warn(f'Failed to load URDF transforms ({self.urdf_path}): {exc}')

    def _frame_to_urdf_link(self, frame_id: str):
        """
        Map a RealSense frame id to the URDF camera link name.
        RealSense publishes: d435i_color_optical_frame, d435i_depth_optical_frame
        URDF has: camera_color_optical_frame, camera_depth_optical_frame
        We override frame_id in launch file to match URDF.
        """
        if not frame_id:
            return None
        frame = frame_id.lstrip('/')
        
        # Frames should match URDF (camera_*)
        if frame.startswith('camera_'):
            return frame
        
        # Fallback: map d435i_* to camera_* 
        if frame.startswith('d435i_'):
            suffix = frame[len('d435i_'):]
            return f'camera_{suffix}'
        
        # Map d405_* to gripper_camera_* (if needed)
        if frame.startswith('d405_'):
            suffix = frame[len('d405_'):]
            return f'gripper_camera_{suffix}'
        
        return None
    
    def _urdf_transform_between_frames(self, point_xyz, source_frame, target_frame):
        """Fallback transform using the URDF kinematic chain."""
        if point_xyz is None or source_frame is None or target_frame is None:
            return None
        self._load_urdf_transforms()
        if self.urdf_transforms is None:
            return None
        source = self._normalize_frame_id(source_frame)
        target = self._normalize_frame_id(target_frame)
        if source is None or target is None:
            return None
        with self.joint_state_lock:
            joint_positions = dict(self.joint_positions)
        try:
            base_T_source = self.urdf_transforms.get_transform(
                source, joint_positions=joint_positions, base_link=self.base_frame
            )
            base_T_target = self.urdf_transforms.get_transform(
                target, joint_positions=joint_positions, base_link=self.base_frame
            )
        except Exception as exc:
            self.get_logger().warn(
                f'URDF fallback failed: {source} -> {target}: {exc}'
            )
            return None
        vec = np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]), 1.0], dtype=float)
        point_in_base = base_T_source @ vec
        try:
            target_T_base = np.linalg.inv(base_T_target)
        except Exception as exc:
            self.get_logger().warn(
                f'URDF fallback inversion failed: {target}: {exc}'
            )
            return None
        converted = target_T_base @ point_in_base
        pair = (source, target)
        if pair not in self.urdf_fallback_pairs:
            self.urdf_fallback_pairs.add(pair)
            self.get_logger().warn(f'Used URDF fallback for {source} -> {target}')
        return converted[:3]
    
    def _lookup_urdf_frame_position(self, frame_id):
        """Fallback lookup of frame pose via URDF."""
        if frame_id is None:
            return None
        return self._urdf_transform_between_frames(
            np.zeros(3, dtype=float), frame_id, self.base_frame
        )

    def _fallback_transform(self, point_xyz, source_frame):
        """Use the URDF to approximate the base transform when TF data is missing."""
        return self._urdf_transform_between_frames(point_xyz, source_frame, self.base_frame)
    
    def _normalize_frame_id(self, frame_id):
        """Map incoming frame IDs to TF-resolvable frames."""
        if not frame_id:
            return None
        frame = frame_id.strip()
        if frame.startswith('/'):
            frame = frame[1:]
        mapped = self._frame_to_urdf_link(frame)
        if mapped:
            return mapped
        return frame

    def _ensure_optical_frame(self, frame_id):
        """Ensure the returned frame represents the optical frame used for projection."""
        normalized = self._normalize_frame_id(frame_id)
        if not normalized:
            return None
        if normalized.endswith('_optical_frame'):
            return normalized
        if normalized.endswith('_frame'):
            base = normalized[:-len('_frame')]
            return f'{base}_optical_frame'
        return normalized

    def _transform_point_between_frames(self, point_xyz, source_frame, target_frame):
        """Generic TF transform helper."""
        if point_xyz is None or source_frame is None or target_frame is None:
            return None
        if self.tf_buffer is None:
            return None
        source = self._normalize_frame_id(source_frame) or source_frame
        target = self._normalize_frame_id(target_frame) or target_frame
        ps = PointStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = source
        ps.point.x = float(point_xyz[0])
        ps.point.y = float(point_xyz[1])
        ps.point.z = float(point_xyz[2])
        try:
            # Use Time(seconds=0) to get latest transform
            transform = self.tf_buffer.lookup_transform(
                target,
                source,
                rclpy.time.Time(),
                timeout=self.tf_timeout
            )
            converted = do_transform_point(ps, transform)
            return np.array([
                converted.point.x,
                converted.point.y,
                converted.point.z,
            ], dtype=np.float32)
        except TransformException as exc:
            pair = (source, target)
            if pair not in self.failed_tf_pairs:
                self.failed_tf_pairs.add(pair)
                self.get_logger().warn(f'Failed to TF {source} -> {target}: {exc}')
            return self._urdf_transform_between_frames(point_xyz, source, target)

    def _lookup_frame_position(self, frame_id, timeout=None):
        """Return the XYZ translation of a frame relative to the base frame."""
        if self.tf_buffer is None or not frame_id:
            return None
        
        normalized_frame = self._normalize_frame_id(frame_id)
        
        try:
            # Use Time(seconds=0) to get the latest available transform
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                normalized_frame,
                rclpy.time.Time(),
                timeout=timeout or self.tf_timeout
            )
            t = transform.transform.translation
            result = np.array([t.x, t.y, t.z], dtype=np.float32)
            
            # Log successful lookup (only first time per frame)
            cache_key = f'tf_success_{frame_id}'
            if not hasattr(self, cache_key):
                setattr(self, cache_key, True)
                self.get_logger().info(
                    f'✓ TF lookup success: {normalized_frame} → {self.base_frame}: '
                    f'({result[0]:.3f}, {result[1]:.3f}, {result[2]:.3f})'
                )
            
            return result
        except TransformException as exc:
            if frame_id not in self.missing_tf_frames:
                self.missing_tf_frames.add(frame_id)
                self.get_logger().warn(
                    f'Failed to lookup {normalized_frame} relative to {self.base_frame}: {exc}'
                )
            return self._lookup_urdf_frame_position(frame_id)
    
    
    def compute_rotated_offsets(self, px, py):
        """
        Return normalized offsets (horizontal, vertical) after compensating for camera rotation.
        Also returns the rotated delta in pixels for horizontal axis, used for angular control.
        """
        return compute_display_offsets(
            px,
            py,
            max(1.0, float(self.image_width)),
            max(1.0, float(self.image_height)),
            rotation_deg=self.image_rotation_deg,
        )

    def publish_tracked_pixel(self, px, py, visible):
        """Publish the tracker output so the GUI can display the active target."""
        if self.tracked_pixel_pub is None:
            return
        try:
            msg = PointStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_pixel'
            msg.point.x = float(px)
            msg.point.y = float(py)
            msg.point.z = 1.0 if visible else 0.0
            self.tracked_pixel_pub.publish(msg)
        except Exception:
            pass
    
    def update_tracker(self):
        """Update tracked pixel using Lucas-Kanade/template tracker."""
        with self.lock:
            if self.color_image is None:
                return False
            frame = self.color_image.copy()
        now = time.time()
        if self.tracker is None:
            self.tracker = WhitePointTracker(
                int(self.target_x),
                int(self.target_y),
                template_size=self.tracker_template_size,
                search_radius=self.tracker_search_radius
            )
            self.tracker.initialize(frame)
            self.get_logger().info(
                f'Initialized tracker at ({int(self.target_x)}, {int(self.target_y)})'
            )
            self.last_tracker_update = now
            self.tracker_visible = True
            return True
        
        px, py, visible = self.tracker.update(frame)
        px = float(np.clip(px, 0, self.image_width - 1))
        py = float(np.clip(py, 0, self.image_height - 1))
        if visible:
            self.target_x = px
            self.target_y = py
            self.last_tracker_update = now
        grace_visible = bool(visible) or ((now - self.last_tracker_update) < self.tracker_grace_s)
        self.tracker_visible = grace_visible
        self.publish_tracked_pixel(self.target_x, self.target_y, grace_visible)
        return grace_visible
    
    def compute_angular_error(self, point_xyz=None, rotated_dx=None):
        """Compute angular error to target point"""
        if point_xyz is not None:
            x_m = float(point_xyz[0])
            z_m = float(point_xyz[2])
            angular_error = math.atan2(-x_m, max(z_m, 1e-6))
        else:
            # Fallback to pixel-based error using horizontal axis (camera X)
            if rotated_dx is None:
                _, _, rotated_dx, _ = self.compute_rotated_offsets(self.target_x, self.target_y)
            pixel_error = float(rotated_dx)
            if self.color_camera_info is not None:
                fx = self.color_camera_info.k[0]
                fy = self.color_camera_info.k[4]
                theta = math.radians(-float(self.image_rotation_deg))
                focal = abs(fx) * abs(math.cos(theta)) + abs(fy) * abs(math.sin(theta))
                if focal != 0.0:
                    angular_error = math.atan2(pixel_error, focal)
                else:
                    angular_error = (pixel_error / max(self.image_width, 1.0)) * math.radians(60.0)
            else:
                angular_error = (pixel_error / max(self.image_width, 1.0)) * math.radians(60.0)
        
        if self.invert_yaw:
            angular_error = -angular_error
        
        return angular_error

    def apply_heading_deadband(self, angular_error, pixel_offset):
        """
        Zero out small angular errors once the white point is near the image center.
        This avoids continuous rotation once the base is aligned.
        """
        if angular_error is None:
            return 0.0
        deadband_rad = max(0.0, float(getattr(self, 'angular_deadband_rad', 0.0)))
        if deadband_rad > 0.0 and abs(angular_error) <= deadband_rad:
            return 0.0
        px_threshold = getattr(self, 'angular_deadband_px_frac', 0.0)
        if px_threshold > 0.0 and abs(pixel_offset) <= px_threshold:
            return 0.0
        return angular_error
    
    def get_head_state(self):
        """Return current (pan, tilt) if known."""
        with self.joint_state_lock:
            pan = self.joint_positions.get('joint_head_pan')
            tilt = self.joint_positions.get('joint_head_tilt')
        if pan is None or tilt is None:
            return None
        return float(pan), float(tilt)
    
    def get_joint_position(self, joint_name):
        """Return the latest position for the requested joint, if known."""
        with self.joint_state_lock:
            pos = self.joint_positions.get(joint_name)
        if pos is None:
            return None
        return float(pos)

    def _get_current_arm_extension(self):
        """Return the total telescoping arm extension if joint states are known."""
        total = 0.0
        found = False
        with self.joint_state_lock:
            for idx in range(4):
                name = f'joint_arm_l{idx}'
                val = self.joint_positions.get(name)
                if val is None:
                    continue
                total += float(val)
                found = True
        if not found:
            return None
        return total
    
    def command_head_alignment(self, target_px, target_py, tracker_visible):
        """Command head pan/tilt to center the tracked pixel."""
        try:
            if not rclpy.ok():
                return
                
            now = time.time()
            if not tracker_visible:
                return
            if (now - self.last_head_command_time) < self.head_command_interval:
                return
            head_state = self.get_head_state()
            if head_state is None:
                return
            head_pan, head_tilt = head_state
            horizontal_offset, vertical_offset, _, _ = self.compute_rotated_offsets(target_px, target_py)
            tilt_err = vertical_offset
            pan_err = horizontal_offset
            tilt_gain = -self.head_tilt_gain if self.tilt_down_negative else self.head_tilt_gain
            tilt_delta = float(np.clip(tilt_gain * tilt_err, -self.head_tilt_step, self.head_tilt_step))
            if self.tilt_only:
                pan_delta = 0.0
            else:
                pan_delta = float(np.clip(-self.head_pan_gain * pan_err, -self.head_pan_step, self.head_pan_step))
                # When the white point drifts far to the horizontal edges, rely on base rotation first
                suppression_threshold = max(0.0, min(0.49, getattr(self, 'head_pan_edge_suppression', 0.0)))
                if suppression_threshold > 0.0:
                    offset_mag = abs(horizontal_offset)
                    if offset_mag >= suppression_threshold:
                        span = max(1e-3, 0.5 - suppression_threshold)
                        suppression = min(1.0, (offset_mag - suppression_threshold) / span)
                        pan_delta *= max(0.0, 1.0 - suppression)
            target_tilt = float(np.clip(head_tilt + tilt_delta, self.head_tilt_limits[0], self.head_tilt_limits[1]))
            target_pan = float(np.clip(head_pan + pan_delta, self.head_pan_limits[0], self.head_pan_limits[1]))
            if abs(target_tilt - head_tilt) < 1e-3 and abs(target_pan - head_pan) < 1e-3:
                return
            self.send_head_goal(target_pan, target_tilt, duration=self.head_move_duration)
            self.last_head_command_time = now
        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f'Error in command_head_alignment: {e}')
            # Silently ignore if ROS context is invalid
    
    def send_head_goal(self, target_pan, target_tilt, duration=0.35):
        """Send a short joint trajectory goal for head pan/tilt."""
        try:
            if not rclpy.ok():
                return
            
            if not self.trajectory_action_ready:
                if self.trajectory_client.wait_for_server(timeout_sec=0.0):
                    self.trajectory_action_ready = True
                else:
                    if not self.trajectory_client.wait_for_server(timeout_sec=0.5):
                        if rclpy.ok():
                            self.get_logger().warn('Head trajectory action server not available')
                        return
                    self.trajectory_action_ready = True
            if self.head_goal_handle is not None:
                try:
                    self.head_goal_handle.cancel_goal_async()
                except Exception:
                    pass
                self.head_goal_handle = None
            
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = JointTrajectory()
            goal.trajectory.joint_names = ['joint_head_pan', 'joint_head_tilt']
            point = JointTrajectoryPoint()
            point.positions = [float(target_pan), float(target_tilt)]
            point.time_from_start = TrajectoryDuration(sec=0, nanosec=int(max(duration, 0.1) * 1e9))
            goal.trajectory.points = [point]
            
            send_future = self.trajectory_client.send_goal_async(goal)
            send_future.add_done_callback(self._handle_head_goal_response)
        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f'Error sending head goal: {e}')
            # Silently ignore if ROS context is invalid
    
    def _handle_head_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'Head goal failed: {exc}')
            return
        if not goal_handle.accepted:
            self.get_logger().warn('Head trajectory goal rejected')
            return
        self.head_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(lambda _: None)
    
    def send_joint_positions(self, joint_names, positions, duration=3.0):
        """Send a joint trajectory command for arbitrary joints."""
        if not joint_names or not positions or len(joint_names) != len(positions):
            self.get_logger().error('Invalid joint command request')
            return False
        try:
            if not rclpy.ok():
                return False
            if not self.trajectory_action_ready:
                if self.trajectory_client.wait_for_server(timeout_sec=0.0):
                    self.trajectory_action_ready = True
                else:
                    if not self.trajectory_client.wait_for_server(timeout_sec=0.5):
                        self.get_logger().error('Joint trajectory action server unavailable')
                        return False
                    self.trajectory_action_ready = True
            goal = FollowJointTrajectory.Goal()
            goal.trajectory = JointTrajectory()
            goal.trajectory.joint_names = list(joint_names)
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in positions]
            duration = max(0.2, float(duration))
            point.time_from_start = TrajectoryDuration(
                sec=int(duration),
                nanosec=int((duration % 1.0) * 1e9)
            )
            goal.trajectory.points = [point]
            send_future = self.trajectory_client.send_goal_async(goal)
            if not self._wait_for_future(send_future, timeout_sec=duration + 2.0):
                self.get_logger().warn('Timed out sending joint trajectory goal')
                return False
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                self.get_logger().warn('Joint trajectory goal rejected')
                return False
            result_future = goal_handle.get_result_async()
            self._wait_for_future(result_future, timeout_sec=duration + 5.0)
            return True
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().error(f'Failed to send joint command: {exc}')
            return False
    
    def _wait_for_depth_ready(self, timeout=5.0):
        """Ensure depth frames, intrinsics, and images are ready before planning."""
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout:
            with self.lock:
                ready = (
                    self.color_image is not None and
                    self.depth_image_raw is not None and
                    self.depth_camera_matrix is not None and
                    self.raw_depth_frame_id is not None
                )
            if ready:
                return True
            time.sleep(0.05)
        return False
    
    def _publish_stop_twist(self, repeats=3):
        if not self.in_navigation_mode:
            return
        twist = Twist()
        for _ in range(max(1, repeats)):
            try:
                self.cmd_vel_pub.publish(twist)
            except Exception:
                break
            time.sleep(0.05)
    
    def rotate_base_in_place(self, angle_rad, max_speed=0.25):
        """Rotate chassis left/right to bias the gripper toward the target."""
        if angle_rad is None or abs(angle_rad) < math.radians(0.5):
            return
        speed_limit = float(np.clip(abs(max_speed), 0.05, self.max_angular_vel))
        angular_vel = math.copysign(speed_limit, angle_rad)
        duration = abs(angle_rad) / speed_limit
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_vel
        end_time = time.time() + duration
        self.get_logger().info(
            f'Rotating base by {math.degrees(angle_rad):.1f}° at {angular_vel:.2f}rad/s (duration {duration:.1f}s)'
        )
        while time.time() < end_time and rclpy.ok():
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)
        self._publish_stop_twist()
    
    def adjust_head_pan_relative(self, delta_rad, duration=0.8):
        """Offset head pan by a relative amount to keep target centered."""
        if delta_rad is None or abs(delta_rad) < math.radians(0.25):
            return False
        head_state = self.get_head_state()
        if head_state is None:
            self.get_logger().warn('Cannot adjust head pan: state unknown')
            return False
        head_pan, head_tilt = head_state
        target_pan = float(np.clip(head_pan + delta_rad, self.head_pan_limits[0], self.head_pan_limits[1]))
        self.get_logger().info(
            f'Adjusting head pan to {math.degrees(target_pan):.1f}° (delta {math.degrees(delta_rad):.1f}°)'
        )
        self.send_head_goal(target_pan, head_tilt, duration=max(0.3, duration))
        return True

    def set_head_pan_absolute(self, target_rad, duration=0.8):
        """Move head pan to an absolute angle so the arm remains in view."""
        if target_rad is None:
            return False
        head_state = self.get_head_state()
        if head_state is None:
            self.get_logger().warn('Cannot set head pan: state unknown')
            return False
        head_pan, head_tilt = head_state
        clamped = float(np.clip(target_rad, self.head_pan_limits[0], self.head_pan_limits[1]))
        if abs(clamped - head_pan) < math.radians(0.1):
            return True
        self.get_logger().info(f'Setting head pan to {math.degrees(clamped):.1f}° (absolute)')
        self.send_head_goal(clamped, head_tilt, duration=max(0.3, duration))
        return True
    
    def command_wrist_yaw(self, target_deg=None, duration=1.5):
        """Rotate the wrist yaw joint to the requested absolute angle (degrees)."""
        if target_deg is None:
            return False
        target_rad = math.radians(target_deg)
        self.get_logger().info(f'Rotating wrist yaw to {target_deg:.1f}°')
        return self.send_joint_positions(['joint_wrist_yaw'], [target_rad], duration=max(0.5, duration))
    
    def command_wrist_roll(self, target_deg=None, duration=1.5):
        """Rotate the wrist roll joint to the requested absolute angle (degrees)."""
        if target_deg is None:
            return False
        target_rad = math.radians(target_deg)
        self.get_logger().info(f'Rotating wrist roll to {target_deg:.1f}°')
        return self.send_joint_positions(['joint_wrist_roll'], [target_rad], duration=max(0.5, duration))
    
    def command_wrist_pitch(self, target_deg=None, duration=1.5):
        """Rotate the wrist pitch joint to the requested absolute angle (degrees)."""
        if target_deg is None:
            return False
        target_rad = math.radians(target_deg)
        self.get_logger().info(f'Rotating wrist pitch to {target_deg:.1f}°')
        return self.send_joint_positions(['joint_wrist_pitch'], [target_rad], duration=max(0.5, duration))
    
    def _estimate_target_pose(self, attempts=20, sleep_s=0.05):
        """Use the latest depth + TF data to recover the white point in base coordinates."""
        for attempt in range(max(1, attempts)):
            tracker_visible = self.update_tracker()
            sample = self.get_depth_at_point(self.target_x, self.target_y)
            if sample is None:
                time.sleep(sleep_s)
                continue
            depth_val = float(sample.get('depth_m', 0.0))
            if depth_val <= 0.0 or not math.isfinite(depth_val):
                time.sleep(sleep_s)
                continue
            camera_matrix = sample.get('camera_matrix')
            point_xyz = self.pixel_to_3d(
                sample.get('pixel_x', self.target_x),
                sample.get('pixel_y', self.target_y),
                depth_val,
                camera_matrix=camera_matrix
            )
            if point_xyz is None:
                time.sleep(sleep_s)
                continue
            point_base = self.transform_point_to_base(point_xyz, sample.get('frame_id'))
            if point_base is None:
                time.sleep(sleep_s)
                continue
            return {
                'base_point': point_base,
                'camera_point': point_xyz,
                'depth_m': depth_val,
                'pixel': (float(self.target_x), float(self.target_y)),
                'tracker_visible': tracker_visible,
            }
        return None
    
    def _plan_gripper_motion(self, base_point, config: PostGraspConfig):
        if base_point is None:
            return None
        planar = float(math.hypot(base_point[0], base_point[1]))
        gripper_frame = getattr(config, 'gripper_reference_frame', None) or 'link_grasp_center'
        
        # Log available TF frames for debugging
        self.get_logger().info(f'🔍 Attempting to lookup gripper frame: {gripper_frame}')
        
        # Try to find a valid gripper frame if the default one is missing
        gripper_pose = self._lookup_frame_position(gripper_frame)
        if gripper_pose is None:
            alternatives = ['link_gripper_fingertip_center', 'link_gripper_finger_left', 'link_lift', 'link_arm_l4']
            self.get_logger().warn(f'Frame {gripper_frame} not found in TF, trying alternatives...')
            for alt in alternatives:
                test_pose = self._lookup_frame_position(alt)
                if test_pose is not None:
                    self.get_logger().info(f'✓ Found alternative frame: {alt}')
                    gripper_frame = alt
                    gripper_pose = test_pose
                    break
        
        if gripper_pose is None:
            self.get_logger().error(
                f'❌ Unable to lookup any gripper frame; cannot plan gripper motion. '
                f'Tried: link_grasp_center, link_gripper_fingertip_center, link_gripper_finger_left, link_lift, link_arm_l4'
            )
            return None
        
        self.get_logger().info(
            f'Using gripper frame: {gripper_frame} at position '
            f'({gripper_pose[0]:.3f}, {gripper_pose[1]:.3f}, {gripper_pose[2]:.3f}) in {self.base_frame}'
        )
        
        target_in_gripper = self._transform_point_between_frames(base_point, self.base_frame, gripper_frame)
        diff_base = base_point - gripper_pose
        forward_delta = float(diff_base[0])
        vertical_delta = float(diff_base[2])
        planar_delta = float(math.hypot(diff_base[0], diff_base[1]))
        
        # Calculate lift height: CORRECT approach for Stretch robot
        # Key insight: base_link is ~0.1m above ground, so floor objects have negative Z!
        # We need to find the HEIGHT of the target relative to the robot's lift mechanism.
        current_lift = self.get_joint_position('joint_lift')
        
        # Try to get gripper Z in base frame
        gripper_z_in_base = None
        if gripper_pose is not None:
            gripper_z_in_base = gripper_pose[2]
        
        if gripper_z_in_base is not None and base_point is not None and current_lift is not None:
            target_z = base_point[2]
            
            # GEOMETRY OF STRETCH ROBOT:
            # - base_link is ~0.1m above ground
            # - Ground objects appear at Z ≈ -0.1m in base_link frame
            # - Gripper (link_grasp_center) Z position = lift_height + fixed_offset
            #   where fixed_offset ≈ 0.191m - current_lift (from TF data)
            # - Camera (D435i) is at Z ≈ 1.28m in base_link frame
            
            # CORRECT LIFT CALCULATION:
            # We want: gripper_z_final = target_z + offset
            # We know: gripper_z_in_base = lift_joint + base_offset
            # Therefore: lift_goal = (target_z + offset) - base_offset
            
            # The relationship: gripper_z = base_to_lift_base + lift_joint
            # From TF: when lift=0.312m, gripper_z=0.191m
            # So: base_to_lift_base ≈ 0.191 - 0.312 = -0.121m
            base_to_lift_base = gripper_z_in_base - current_lift
            
            # Target gripper height: slightly above target (by lift_offset_m)
            desired_gripper_z = target_z + config.lift_offset_m
            
            # Calculate required lift joint value
            lift_goal = desired_gripper_z - base_to_lift_base
            
            # Safety clamp
            lift_goal = float(np.clip(lift_goal, config.min_lift_m, config.max_lift_m))
            
            self.get_logger().info(
                f'Lift calculation: target_z={target_z:.3f}m, '
                f'gripper_z={gripper_z_in_base:.3f}m, '
                f'current_lift={current_lift:.3f}m, '
                f'base_offset={base_to_lift_base:.3f}m, '
                f'desired_gripper_z={desired_gripper_z:.3f}m, '
                f'→ lift_goal={lift_goal:.3f}m'
            )
        elif lift_pose is not None and base_point is not None:
            # Fallback to link_lift logic if gripper frame missing
            target_height_relative_to_lift = base_point[2] - lift_pose[2]
            lift_goal = current_lift + target_height_relative_to_lift + config.lift_offset_m
            self.get_logger().warn(f'Using link_lift fallback for lift calculation')
        elif vertical_delta is not None and current_lift is not None:
            # Fallback: use gripper vertical delta (less accurate)
            lift_goal = current_lift + vertical_delta + config.lift_offset_m
            self.get_logger().warn(
                f'Using fallback lift calculation: vertical_delta={vertical_delta:.3f}m'
            )
        else:
            # Last resort: use absolute target height (likely incorrect for ground objects)
            lift_goal = base_point[2] + config.lift_offset_m
            self.get_logger().warn(
                f'Using absolute lift calculation (may be incorrect): target_z={base_point[2]:.3f}m'
            )
        
        lift_goal = float(np.clip(lift_goal, config.min_lift_m, config.max_lift_m))

        current_extension = self._get_current_arm_extension()
        if forward_delta is not None and current_extension is not None:
            desired_extension = current_extension + forward_delta
        else:
            desired_extension = planar - config.arm_offset_m
        arm_extension = float(np.clip(desired_extension, 0.0, config.max_arm_extension_m))
        return {
            'planar_distance': planar,
            'lift_goal': lift_goal,
            'arm_extension': arm_extension,
            'target_point': base_point,
            'planar_delta': planar_delta,
            'gripper_point': gripper_pose,
            'target_in_gripper': target_in_gripper,
            'forward_delta': forward_delta,
            'vertical_delta': vertical_delta,
        }
    
    def _execute_gripper_plan(self, plan, config: PostGraspConfig):
        if plan is None:
            return False
        success = True
        
        # NO wrist yaw compensation needed!
        # The base has already rotated to point at the target.
        # The wrist should stay neutral (0°) to avoid collision with lift.
        # Only apply the user-configured offset if explicitly set.
        base_rotation_rad = getattr(self, '_last_base_rotation_rad', 0.0)
        config_wrist_yaw = config.wrist_yaw_deg if config.wrist_yaw_deg is not None else 0.0
        
        # Use configured offset directly without base compensation
        final_wrist_yaw_deg = config_wrist_yaw
        
        # Clamp to safe range to avoid collision
        # For Stretch: +90° = left (collision risk), -90° = right (safe)
        final_wrist_yaw_deg = float(np.clip(final_wrist_yaw_deg, -45.0, 45.0))
        
        self.get_logger().info(
            f'Wrist yaw: base_rotation={math.degrees(base_rotation_rad):.1f}°, '
            f'compensation=DISABLED (avoid collision), '
            f'config_offset={config_wrist_yaw:.1f}°, '
            f'final={final_wrist_yaw_deg:.1f}°'
        )
        
        # Step 1: Set wrist orientation first (yaw, roll, pitch)
        # Only adjust wrist yaw if significantly different from neutral (>2°)
        if abs(final_wrist_yaw_deg) > 2.0:
            self.get_logger().info(f'Adjusting wrist yaw to {final_wrist_yaw_deg:.1f}°')
            success &= self.command_wrist_yaw(final_wrist_yaw_deg, config.wrist_yaw_duration_s)
        else:
            self.get_logger().info('Wrist yaw near neutral — skipping adjustment')
        
        # Only adjust roll if explicitly configured and non-zero
        if config.wrist_roll_deg is not None and abs(config.wrist_roll_deg) > 2.0:
            self.get_logger().info(f'Adjusting wrist roll to {config.wrist_roll_deg:.1f}°')
            success &= self.command_wrist_roll(config.wrist_roll_deg, config.wrist_roll_duration_s)
        
        # Only adjust pitch if explicitly configured
        if config.wrist_pitch_deg is not None and abs(config.wrist_pitch_deg) > 2.0:
            self.get_logger().info(f'Adjusting wrist pitch to {config.wrist_pitch_deg:.1f}°')
            success &= self.command_wrist_pitch(config.wrist_pitch_deg, config.wrist_roll_duration_s)
        
        arm_extension = plan.get('arm_extension') or 0.0
        lift_goal = plan.get('lift_goal')
        current_lift = self.get_joint_position('joint_lift')
        
        # Determine if we're lowering or raising the lift
        lowering_move = False
        if lift_goal is not None and current_lift is not None:
            lowering_move = lift_goal < (current_lift - 0.02)  # Lowering by >2cm
        
        def extend_arm():
            nonlocal success
            if success and arm_extension > 1e-3:
                per_joint = arm_extension / 4.0
                joints = ['joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3']
                self.get_logger().info(f'Extending arm to {arm_extension:.3f}m total')
                success &= self.send_joint_positions(joints, [per_joint] * 4, duration=config.arm_duration_s)
        
        # CORRECT SEQUENCING based on lift direction:
        # - Lowering: extend arm first to avoid collision, then lower lift
        # - Raising/neutral: adjust lift first, then extend arm
        if lowering_move:
            self.get_logger().info(
                f'Lowering lift from {current_lift:.2f}m to {lift_goal:.2f}m; '
                'extending arm first for clearance'
            )
            extend_arm()
            if lift_goal is not None and success:
                success &= self.send_joint_positions(['joint_lift'], [lift_goal], duration=config.lift_duration_s)
        else:
            # Raising or staying: lift first, then extend
            if lift_goal is not None:
                self.get_logger().info(f'Adjusting lift to {lift_goal:.2f}m, then extending arm')
                success &= self.send_joint_positions(['joint_lift'], [lift_goal], duration=config.lift_duration_s)
            extend_arm()
        
        return success
    
    def post_navigation_grasp(self, config: PostGraspConfig = None):
        """Align chassis/head and raise the gripper using TF + depth after navigation."""
        cfg = config or self.post_grasp_config
        self.get_logger().info('Starting post-navigation gripper alignment sequence...')
        self._start_spin_thread()
        try:
            if not self._wait_for_depth_ready(timeout=5.0):
                self.get_logger().error('Depth topics not ready; cannot align gripper')
                return False
            # Allow tracker to settle at the stop pose
            settle_until = time.time() + 0.8
            while time.time() < settle_until and rclpy.ok():
                self.update_tracker()
                time.sleep(0.05)
            pose = self._estimate_target_pose()
            if pose is None:
                self.get_logger().error('Unable to recover target pose for gripper alignment')
                return False
            base_point = pose['base_point']
            target_yaw = math.atan2(base_point[1], base_point[0])
            self.get_logger().info(
                f'Target yaw relative to base: {math.degrees(target_yaw):.2f}°'
            )
            # Rotate chassis to bias left of the target and free-up gripper workspace
            applied_base_adjust = 0.0
            base_adjust = target_yaw + math.radians(cfg.base_turn_deg)
            if abs(base_adjust) > math.radians(0.5):
                if self.switch_to_navigation_mode():
                    self.rotate_base_in_place(base_adjust)
                    applied_base_adjust = base_adjust
                    self._publish_stop_twist()
                    self.switch_to_position_mode()
                else:
                    self.get_logger().warn('Failed to enter navigation mode for base adjustment')
            
            # Store the applied rotation for wrist compensation
            self._last_base_rotation_rad = applied_base_adjust
            # Give tracker a moment to settle before sampling depth again
            settle_until = time.time() + 1.0
            while time.time() < settle_until and rclpy.ok():
                self.update_tracker()
                time.sleep(0.05)
            pose = self._estimate_target_pose()
            if pose is None:
                self.get_logger().error('Unable to recover target pose for gripper alignment after base/head move')
                return False
            plan = self._plan_gripper_motion(pose['base_point'], cfg)
            if plan is None:
                self.get_logger().error('Failed to build gripper motion plan')
                return False
            tp = plan['target_point']
            head_bias_deg = getattr(cfg, 'head_pan_deg', 0.0)
            if head_bias_deg is None:
                head_bias_deg = 0.0
            head_target_rad = math.atan2(tp[1], tp[0]) + math.radians(head_bias_deg)
            self.set_head_pan_absolute(head_target_rad)
            time.sleep(0.3)
            self.get_logger().info(
                f"Target in base frame: ({tp[0]:.2f}, {tp[1]:.2f}, {tp[2]:.2f}) m — planar {plan['planar_distance']:.2f} m"
            )
            self.get_logger().info(
                f"Commanding lift to {plan['lift_goal']:.2f} m and arm extension {plan['arm_extension']:.2f} m"
            )
            rel = plan.get('target_in_gripper')
            if rel is not None:
                gripper_frame = getattr(cfg, 'gripper_reference_frame', None) or 'link_grasp_center'
                self.get_logger().info(
                    f"Target relative to gripper ({gripper_frame} frame): ({rel[0]:.2f}, {rel[1]:.2f}, {rel[2]:.2f}) m"
                )
            # Execute the complete gripper motion plan (includes wrist orientation + lift + arm)
            success = self._execute_gripper_plan(plan, cfg)
            if success:
                self.get_logger().info('Post-navigation gripper alignment complete ✅')
            else:
                self.get_logger().warn('Post-navigation gripper alignment failed to execute completely')
            return success
        finally:
            self._publish_stop_twist()
            self._stop_spin_thread()
    
    def switch_to_navigation_mode(self):
        """Switch robot to navigation mode"""
        if not self.nav_mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Navigation mode service not available')
            return False
        
        request = Trigger.Request()
        for attempt in range(3):
            future = self.nav_mode_client.call_async(request)
            ok = self._wait_for_future(future, timeout_sec=2.0)
            result = future.result() if ok else None
            if result is not None and getattr(result, 'success', False):
                self.get_logger().info('Switched to navigation mode')
                self.in_navigation_mode = True
                return True
            self.get_logger().warn('Navigation mode request failed, retrying...')
            time.sleep(0.2)
        
        self.get_logger().error('Failed to switch to navigation mode after retries')
        self.in_navigation_mode = False
        return False
    
    def switch_to_position_mode(self):
        """Switch robot back to position mode"""
        if not self.pos_mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Position mode service not available')
            return False
        
        request = Trigger.Request()
        future = self.pos_mode_client.call_async(request)
        ok = self._wait_for_future(future, timeout_sec=2.0)
        result = future.result() if ok else None
        
        if result is not None and getattr(result, 'success', False):
            self.get_logger().info('Switched back to position mode')
            self.in_navigation_mode = False
            return True
        else:
            self.get_logger().warn('Failed to switch back to position mode')
            self.in_navigation_mode = False
            return False
    
    def _spin_worker(self):
        """Background worker that continuously processes ROS callbacks."""
        while self.spin_running and rclpy.ok():
            try:
                rclpy.spin_once(self, timeout_sec=self.spin_sleep)
            except Exception as exc:
                if self.spin_running and rclpy.ok():
                    self.get_logger().warn(f'Spin worker exception: {exc}')
                break
    
    def _start_spin_thread(self):
        """Start background spinner to keep image/depth callbacks flowing."""
        if self.spin_running:
            return
        self.spin_running = True
        self.spin_thread = Thread(target=self._spin_worker, daemon=True)
        self.spin_thread.start()
    
    def _stop_spin_thread(self):
        """Stop the background spinner thread."""
        self.spin_running = False
        thread = self.spin_thread
        if thread is not None:
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        self.spin_thread = None
    
    def _wait_for_future(self, future, timeout_sec=2.0):
        """Wait for a ROS future without blocking the navigation loop."""
        if future is None:
            return False
        end_time = time.time() + float(max(timeout_sec, 0.0))
        while rclpy.ok():
            if future.done():
                return True
            if time.time() >= end_time:
                return future.done()
            if not self.spin_running:
                try:
                    rclpy.spin_once(self, timeout_sec=0.05)
                except Exception:
                    break
            else:
                time.sleep(0.01)
        return future.done()
    
    def navigate_to_target(self):
        """Main navigation loop"""
        self.get_logger().info('Starting navigation...')
        self._start_spin_thread()
        
        # Switch to navigation mode first
        if not self.switch_to_navigation_mode():
            self.get_logger().error('Cannot switch to navigation mode, aborting')
            self._stop_spin_thread()
            return False
        
        # Wait for camera data
        self.get_logger().info('Waiting for camera data...')
        timeout = 10.0
        start_time = time.time()
        while (self.color_image is None or self.depth_image_raw is None) and \
              (time.time() - start_time) < timeout:
            time.sleep(0.05)

        if self.color_image is None or self.depth_image_raw is None:
            self.get_logger().error('Timeout waiting for camera data')
            self._stop_spin_thread()
            return False
        
        self.get_logger().info('Camera data received, starting servo...')
        self.tracker = None
        
        loop_period = 0.1  # 10 Hz
        iteration = 0
        
        self.get_logger().info('🔄 Starting navigation loop...')
        
        try:
            while rclpy.ok():
                iteration += 1
                loop_start = time.time()
                
                # Update tracker to get current pixel position of white point
                tracker_visible = self.update_tracker()
                
                # Use the UPDATED tracker position (not initial target)
                # The tracker follows the white point in the image
                current_x = float(self.target_x)
                current_y = float(self.target_y)
                horizontal_offset, vertical_offset, rotated_dx, _ = self.compute_rotated_offsets(current_x, current_y)
                disp_u, disp_v = normalized_display_position(
                    current_x,
                    current_y,
                    max(1.0, float(self.image_width)),
                    max(1.0, float(self.image_height)),
                    rotation_deg=self.image_rotation_deg,
                )
                pixel_offset = horizontal_offset
                
                # Command head to keep target centered (visual servoing)
                self.command_head_alignment(current_x, current_y, tracker_visible)
                
                # Get depth at CURRENT tracked pixel position
                # This is re-sampled every iteration, not cached
                depth_sample = self.get_depth_at_point(current_x, current_y)
                point_xyz = None
                depth_frame_id = None
                forward_dist = None
                point_base = None
                depth_source = 'none'
                depth_val = None
                has_valid_depth = False
                if depth_sample is None:
                    msg = 'Tracker lost and no valid depth at target point' if not tracker_visible else 'No valid depth at target point'
                    self.get_logger().warn(msg)
                else:
                    depth_source = depth_sample.get('source', 'aligned')
                    depth_val = float(depth_sample.get('depth_m', 0.0))
                    if not math.isfinite(depth_val) or depth_val <= 0.0:
                        self.get_logger().warn(
                            f'Invalid depth measurement at target point (value={depth_val:.3f})'
                        )
                    else:
                        has_valid_depth = True
                        forward_dist = depth_val
                        depth_frame_id = depth_sample.get('frame_id')
                        px_used = depth_sample.get('pixel_x', current_x)
                        py_used = depth_sample.get('pixel_y', current_y)
                        camera_matrix = depth_sample.get('camera_matrix')
                        
                        # Convert pixel + depth to 3D point in camera frame
                        point_xyz = self.pixel_to_3d(px_used, py_used, depth_val, camera_matrix=camera_matrix)
                        
                        if point_xyz is not None:
                            # Transform to base frame (re-computed every iteration based on current robot pose)
                            point_base = self.transform_point_to_base(point_xyz, depth_frame_id)
                            
                            if point_base is not None:
                                # Use XY plane distance for navigation (horizontal distance to target)
                                # base_link: x=forward, y=left, z=up
                                forward_dist = float(math.hypot(point_base[0], point_base[1]))
                                
                                # Log tracking updates to verify continuous re-computation
                                if iteration <= 10 or iteration % 20 == 0:
                                    self.get_logger().info(
                                        f'📍 Pixel tracking: ({current_x:.0f},{current_y:.0f}) → '
                                        f'depth={depth_val:.2f}m → base=({point_base[0]:.2f},{point_base[1]:.2f}) → dist={forward_dist:.2f}m'
                                    )
                                
                                # Debug: log if distance seems too small initially
                                if iteration <= 5 and forward_dist < 1.0:
                                    self.get_logger().warn(
                                        f'Initial distance seems small: {forward_dist:.2f}m, '
                                        f'depth={depth_val:.2f}m, point_base=({point_base[0]:.2f}, {point_base[1]:.2f}, {point_base[2]:.2f})'
                                    )
                            else:
                                # Fallback: use Z component of camera frame (depth)
                                forward_dist = float(point_xyz[2])
                                if iteration <= 5:
                                    self.get_logger().warn(
                                        f'Using camera depth fallback: {forward_dist:.2f}m (TF failed)'
                                    )
                        else:
                            # Last resort: keep raw depth value in meters
                            forward_dist = depth_val
                            if iteration <= 5:
                                self.get_logger().warn(
                                    f'Using raw depth fallback: {forward_dist:.2f}m (3D projection failed)'
                                )

                # Check if reached target distance
                # Only stop if we're very close AND have confirmed it multiple times
                if has_valid_depth and forward_dist is not None and forward_dist <= self.stop_distance_m:
                    self.stop_hold_counter += 1
                    if iteration % 5 == 0:
                        self.get_logger().info(
                            f'Close to target: {forward_dist:.2f}m <= {self.stop_distance_m:.2f}m, '
                            f'hold counter: {self.stop_hold_counter}/{self.stop_hold_required}'
                        )
                else:
                    self.stop_hold_counter = 0
                
                # Only stop if we've been close for enough consecutive checks
                if self.stop_hold_counter >= self.stop_hold_required:
                    self.get_logger().info(f'Reached target! Distance: {forward_dist:.2f}m')
                    self.stop_robot()
                    self._stop_spin_thread()
                    return True
                
                # Compute control commands
                if point_base is not None:
                    angular_error = math.atan2(point_base[1], point_base[0])
                    if self.invert_yaw:
                        angular_error = -angular_error
                else:
                    angular_error = self.compute_angular_error(point_xyz=point_xyz, rotated_dx=rotated_dx)
                angular_error = self.apply_heading_deadband(angular_error, pixel_offset)

                # Create twist command
                twist = Twist()

                # In tilt_only mode: only rotate to face target, don't move forward
                # In normal mode: move forward and rotate
                if self.tilt_only:
                    # Only rotate to align with target, no forward movement
                    linear_vel = 0.0
                else:
                    # Move forward only when we have a valid depth measurement
                    if not has_valid_depth or forward_dist is None:
                        linear_vel = 0.0
                        if iteration <= 5 or iteration % 15 == 0:
                            self.get_logger().warn('Valid depth not available, holding position')
                    else:
                        distance_error = forward_dist - self.stop_distance_m
                        if distance_error > 0.0:
                            # 使用指數縮放：距離越近，速度越慢
                            # dist_ratio = 距離誤差 / 停止距離，用於計算應有的速度比例
                            dist_ratio = distance_error / max(0.1, self.stop_distance_m)
                            
                            # 指數縮放函數：距離遠時接近1.0（全速），接近時逐漸減小
                            exp_scale = min(1.0 - math.exp(-dist_ratio), 1.0)
                            
                            # 基礎速度使用指數縮放
                            linear_vel = self.max_linear_vel * exp_scale
                            
                            # 確保最小速度閾值，避免太慢而停滯
                            MIN_FORWARD_SPEED = 0.02
                            if distance_error > 0.05:  # 距離大於5cm時才應用最小速度
                                linear_vel = max(MIN_FORWARD_SPEED, linear_vel)
                            else:
                                linear_vel = max(0.0, linear_vel)
                            
                            # Reduce forward speed when turning sharply to improve stability
                            abs_angular_error = abs(angular_error)
                            if abs_angular_error > 0.3:
                                turn_factor = max(0.3, 1.0 - abs_angular_error / 1.57)
                                linear_vel *= turn_factor
                            
                            # If the white point appears low in the GUI or large vertical error exists,
                            # pause forward motion until the head recenters.
                            if (disp_v > (2.0 / 3.0)) or (abs(vertical_offset) > 0.08):
                                linear_vel = 0.0
                            
                            # Clip to safe limits
                            linear_vel = np.clip(linear_vel, 0.0, self.max_linear_vel)
                            
                            # Slow down slightly if tracker is lost but keep moving toward valid depth
                            if not tracker_visible:
                                linear_vel *= 0.5
                                if iteration % 10 == 0:
                                    self.get_logger().warn('Tracker lost, continuing towards valid depth measurement')
                        else:
                            # Already at or inside stop distance
                            linear_vel = 0.0

                twist.linear.x = float(linear_vel)
                
                # Angular velocity to align with target (always active)
                angular_vel = self.angular_gain * angular_error
                angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
                twist.angular.z = float(angular_vel)
                
                # Publish command
                self.cmd_vel_pub.publish(twist)
                
                # Log progress - show every iteration for first 20, then every 10
                should_log = iteration <= 20 or iteration % 10 == 0
                if should_log:
                    base_desc = f" base=({point_base[0]:.2f},{point_base[1]:.2f})" if point_base is not None else ""
                    mode_str = "TILT_ONLY" if self.tilt_only else "NAVIGATE"
                    tracker_str = "VISIBLE" if tracker_visible else "LOST"
                    if has_valid_depth and forward_dist is not None:
                        dist_str = f"{forward_dist:.2f}m"
                        # 計算並顯示距離比例和速度縮放因子
                        distance_error = forward_dist - self.stop_distance_m
                        if distance_error > 0.0:
                            dist_ratio = distance_error / max(0.1, self.stop_distance_m)
                            exp_scale = min(1.0 - math.exp(-dist_ratio), 1.0)
                            dist_str += f" (ratio={dist_ratio:.2f}, scale={exp_scale:.2f})"
                    elif depth_val is not None and math.isfinite(depth_val) and depth_val > 0.0:
                        dist_str = f"{depth_val:.2f}m"
                    else:
                        dist_str = "N/A"
                    stop_str = f" STOP_IN={self.stop_hold_required - self.stop_hold_counter}" if self.stop_hold_counter > 0 else ""
                    
                    # Add pixel position to verify tracker is updating
                    pixel_str = f"px=({int(current_x)},{int(current_y)})"
                    
                    self.get_logger().info(
                        f'[{mode_str}] #{iteration}: {pixel_str}, dist={dist_str} (→{self.stop_distance_m:.2f}m), '
                        f'tracker={tracker_str}, src={depth_source}, ang={math.degrees(angular_error):.1f}°, '
                        f'v_lin={twist.linear.x:.3f}, v_ang={twist.angular.z:.3f}{base_desc}{stop_str}'
                    )
                
                # Sleep to maintain loop rate (callbacks run in background thread)
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, loop_period - elapsed)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                
                # Confirm loop iteration
                if iteration <= 5 or iteration % 50 == 0:
                    self.get_logger().info(f'✓ Loop running: completed {iteration} iterations, rclpy.ok()={rclpy.ok()}')
                
                # Safety timeout
                if iteration > 1000:  # ~100 seconds at 10Hz
                    self.get_logger().warn('Navigation timeout')
                    self.stop_robot()
                    self._stop_spin_thread()
                    return False
        
        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f'❌ Exception in navigation loop at iteration {iteration}: {e}')
            else:
                print(f'❌ Exception in navigation loop (ROS context invalid) at iteration {iteration}: {e}')
            import traceback
            traceback.print_exc()
            try:
                self.stop_robot()
            except Exception as stop_err:
                print(f'Warning: Error during stop_robot: {stop_err}')
            self._stop_spin_thread()
            return False
        
        if rclpy.ok():
            self.get_logger().error(f'🛑 Exited navigation loop (rclpy.ok()={rclpy.ok()}) after {iteration} iterations')
        else:
            print(f'🛑 Exited navigation loop (ROS context invalid) after {iteration} iterations')
        try:
            self.stop_robot()
        except Exception as e:
            print(f'Warning: Error during stop_robot: {e}')
        self._stop_spin_thread()
        return False
    
    def stop_robot(self):
        """Send stop command and switch back to position mode"""
        try:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            
            # Send stop command multiple times to ensure it's received
            for _ in range(5):
                try:
                    if rclpy.ok():
                        self.cmd_vel_pub.publish(twist)
                    time.sleep(0.05)
                except Exception:
                    break
            
            if rclpy.ok():
                self.get_logger().info('Robot stopped')
            else:
                print('Robot stopped (ROS context invalid)')
            
            # Switch back to position mode
            self.switch_to_position_mode()
            self.tracker = None
        except Exception as e:
            print(f'Warning: Error in stop_robot: {e}')
        self.stop_hold_counter = 0
        if self.head_goal_handle is not None:
            try:
                self.head_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.head_goal_handle = None


def main():
    parser = argparse.ArgumentParser(description='ROS2 Visual Servoing Navigation')
    parser.add_argument('-x', '--target-x', type=int, required=True,
                        help='Target pixel X coordinate')
    parser.add_argument('-y', '--target-y', type=int, required=True,
                        help='Target pixel Y coordinate')
    parser.add_argument('--camera', type=str, default='d435i',
                        help='Camera name (d435i or d405)')
    parser.add_argument('--stop-dist-m', type=float, default=1.0,
                        help='Stop distance in meters')
    parser.add_argument('--tilt-only', action='store_true',
                        help='Only adjust heading, do not move forward')
    parser.add_argument('--tilt-down-negative', action='store_true',
                        help='Tilt down is negative (legacy compatibility)')
    parser.add_argument('--invert-yaw', action='store_true',
                        help='Invert yaw direction')
    parser.add_argument('--base-frame', type=str, default='base_link',
                        help='Base frame to use for tf calculations')
    parser.add_argument('--disable-post-grasp', action='store_true',
                        help='Skip the gripper alignment stage after navigation success')
    parser.add_argument('--post-grasp-base-turn', type=float, default=PostGraspConfig.base_turn_deg,
                        help='Degrees to rotate chassis left (>0) once navigation finishes')
    parser.add_argument('--post-grasp-head-pan', type=float, default=PostGraspConfig.head_pan_deg,
                        help='Head pan adjustment in degrees (negative = pan right)')
    parser.add_argument('--post-grasp-lift-offset', type=float, default=PostGraspConfig.lift_offset_m,
                        help='Offset added to the measured target height when commanding lift (meters)')
    parser.add_argument('--post-grasp-arm-offset', type=float, default=PostGraspConfig.arm_offset_m,
                        help='Distance from base_link to gripper when arm is retracted (meters)')
    parser.add_argument('--post-grasp-wrist-yaw-deg', type=float, default=PostGraspConfig.wrist_yaw_deg,
                        help='Absolute wrist yaw angle in degrees for the gripper (0 = forward, 90 = left, -90 = right)')
    parser.add_argument('--post-grasp-wrist-roll-deg', type=float, default=PostGraspConfig.wrist_roll_deg,
                        help='Absolute wrist roll angle in degrees (0 = fingers horizontal, + = rotate counterclockwise)')
    parser.add_argument('--post-grasp-wrist-pitch-deg', type=float, default=None,
                        help='Absolute wrist pitch angle in degrees (optional)')
    
    args = parser.parse_args()
    
    # Initialize ROS2 with fresh context
    try:
        rclpy.init()
    except Exception as e:
        print(f"Warning: rclpy.init() failed: {e}")
        # Might already be initialized, check if context is OK
        if not rclpy.ok():
            print("ERROR: ROS2 context is not OK and cannot initialize")
            sys.exit(1)
    
    navigator = None
    exit_code = 0
    
    try:
        post_cfg = PostGraspConfig(
            base_turn_deg=args.post_grasp_base_turn,
            head_pan_deg=args.post_grasp_head_pan,
            lift_offset_m=args.post_grasp_lift_offset,
            arm_offset_m=args.post_grasp_arm_offset,
            wrist_yaw_deg=args.post_grasp_wrist_yaw_deg,
            wrist_roll_deg=args.post_grasp_wrist_roll_deg,
            wrist_pitch_deg=args.post_grasp_wrist_pitch_deg,
        )

        # Create navigator node
        navigator = VisualServoNavigator(
            target_x=args.target_x,
            target_y=args.target_y,
            camera_name=args.camera,
            stop_distance_m=args.stop_dist_m,
            tilt_only=args.tilt_only,
            tilt_down_negative=args.tilt_down_negative,
            invert_yaw=args.invert_yaw,
            base_frame=args.base_frame,
            post_grasp_config=post_cfg
        )
        
        # Run navigation
        success = navigator.navigate_to_target()
        
        if success:
            print('✅ Navigation completed successfully!')
            if not args.disable_post_grasp:
                print('🔁 Starting gripper alignment using tf/depth data...')
                post_success = navigator.post_navigation_grasp()
                if post_success:
                    print('🤖 Post-navigation gripper alignment finished')
                else:
                    print('⚠️ Post-navigation gripper alignment failed')
            exit_code = 0
        else:
            print('⚠️ Navigation failed or interrupted')
            exit_code = 1
    
    except KeyboardInterrupt:
        print('\n⚠️ Navigation interrupted by user')
        if navigator and rclpy.ok():
            navigator.stop_robot()
        exit_code = 2
    
    except Exception as e:
        print(f'❌ Navigation error: {e}')
        import traceback
        traceback.print_exc()
        if navigator and rclpy.ok():
            navigator.stop_robot()
        exit_code = 3
    
    finally:
        # Clean shutdown
        if navigator:
            try:
                navigator.destroy_node()
            except Exception as e:
                print(f'Warning: Error destroying node: {e}')
        
        # Only shutdown if we initialized it
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f'Warning: Error during shutdown: {e}')
        
        sys.exit(exit_code)


if __name__ == '__main__':
    main()

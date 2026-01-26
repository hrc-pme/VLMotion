#!/usr/bin/env python3
"""
ROS2 Visual Servoing Node
Uses TF2 and ROS2 topics instead of direct hardware access
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PointStamped
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import String
import tf2_ros
from tf2_ros import TransformException
import numpy as np
import math
import time
from threading import Lock
from cv_bridge import CvBridge

from VLServo.gui_rotation import display_to_camera, resolve_rotation_degrees
from VLServo.white_point_tracker import WhitePointTracker


def robust_depth_at_pixel(depth_image, px, py, kernel_sizes=(3, 5, 7, 9, 11)):
    """
    Return a stable depth estimate by taking the median over progressively
    larger windows until a valid (non-zero, finite) value is found.
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


def pixel_to_3d(xy_pix, depth_m, camera_matrix):
    """
    Project a pixel and depth measurement into the camera optical frame.
    Matches the helper used by white_point.py so GUI-selected pixels
    can be converted to metric points.
    """
    if camera_matrix is None or depth_m is None:
        return None
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    x_pix, y_pix = xy_pix
    x = ((x_pix - cx) * depth_m) / fx
    y = ((y_pix - cy) * depth_m) / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float32)


class VisualServoingNode(Node):
    """ROS2 node for visual servoing control"""
    
    def __init__(self):
        super().__init__('visual_servoing_node')
        
        # Parameters
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('gripper_frame', 'link_gripper_fingertip_left')
        self.declare_parameter('control_rate', 30.0)
        self.declare_parameter('camera_name', 'd435i')
        self.declare_parameter('stop_distance', 0.5)
        self.declare_parameter('linear_gain', 0.5)
        self.declare_parameter('angular_gain', 0.8)
        self.declare_parameter('max_linear_vel', 0.15)
        self.declare_parameter('max_angular_vel', 0.3)
        self.declare_parameter('min_linear_vel', 0.02)
        self.declare_parameter('pixel_target_is_display', False)
        self.declare_parameter('pixel_target_rotation_deg', float('nan'))
        self.declare_parameter('tracker_template_size', 41)
        self.declare_parameter('tracker_search_radius', 45)
        self.declare_parameter('tracker_min_score', 0.85)
        self.declare_parameter('tracker_reacquire_score', 0.9)
        self.declare_parameter('tracker_grace_s', 0.75)
        
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.gripper_frame = self.get_parameter('gripper_frame').value
        self.control_rate = self.get_parameter('control_rate').value
        self.camera_name = self.get_parameter('camera_name').value
        self.stop_distance = float(self.get_parameter('stop_distance').value)
        self.linear_gain = float(self.get_parameter('linear_gain').value)
        self.angular_gain = float(self.get_parameter('angular_gain').value)
        self.max_linear_vel = float(self.get_parameter('max_linear_vel').value)
        self.max_angular_vel = float(self.get_parameter('max_angular_vel').value)
        self.min_linear_vel = float(self.get_parameter('min_linear_vel').value)
        self.stop_hold_required = 5
        self.stop_hold_counter = 0
        self.pixel_target_is_display = bool(self.get_parameter('pixel_target_is_display').value)
        rotation_param = self.get_parameter('pixel_target_rotation_deg').value
        if isinstance(rotation_param, float) and math.isnan(rotation_param):
            rotation_param = None
        self.pixel_target_rotation_deg = (
            resolve_rotation_degrees(rotation_param) if self.pixel_target_is_display else None
        )
        self.tracker_template_size = int(max(11, int(self.get_parameter('tracker_template_size').value) | 1))
        self.tracker_search_radius = int(max(8, int(self.get_parameter('tracker_search_radius').value)))
        self.tracker_min_score = float(self.get_parameter('tracker_min_score').value)
        self.tracker_reacquire_score = float(max(self.tracker_min_score, self.get_parameter('tracker_reacquire_score').value))
        self.tracker_grace_s = float(max(0.0, self.get_parameter('tracker_grace_s').value))
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_pose_cmd', 10)
        self.status_pub = self.create_publisher(String, '/visual_servo/status', 10)
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/stretch/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.target_sub = self.create_subscription(
            PointStamped,
            '/visual_servo/target_point',
            self.target_callback,
            10
        )
        
        # Camera/depth data for pixel-to-3D projection
        self.bridge = CvBridge()
        self.depth_lock = Lock()
        self.color_lock = Lock()
        self.tracker_lock = Lock()
        self.depth_image = None
        self.depth_scale = 0.001
        self.camera_matrix = None
        self.camera_info_msg = None
        self.pixel_frame_ids = {'camera_pixel', 'camera_color_pixel', 'pixel'}
        self.color_image = None
        self.color_width = None
        self.color_height = None
        self.tracker = None
        self.tracker_visible = False
        self.tracker_needs_reset = False
        self.tracked_pixel = None

        depth_topic = f'/{self.camera_name}/aligned_depth_to_color/image_raw'
        color_info_topic = f'/{self.camera_name}/color/camera_info'
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            color_info_topic,
            self.camera_info_callback,
            10
        )
        color_topic = f'/{self.camera_name}/color/image_raw'
        self.color_sub = self.create_subscription(
            Image,
            color_topic,
            self.color_callback,
            10
        )

        # State
        self.lock = Lock()
        self.current_joint_state = None
        self.metric_target = None
        self.pixel_target = None
        self.servoing_active = False
        self.tracked_pixel_pub = self.create_publisher(PointStamped, '/visual_servo/tracked_point', 10)
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop
        )
        
        self.get_logger().info('Visual Servoing Node initialized')
        self.get_logger().info(f'Listening for {self.camera_name} depth on {depth_topic}')
        self.get_logger().info(f'Listening for {self.camera_name} color on {color_topic}')
        if self.pixel_target_is_display:
            self.get_logger().info(
                f'Pixel targets interpreted as GUI coordinates (rotation {self.pixel_target_rotation_deg:.1f}°)'
            )
    
    def joint_state_callback(self, msg):
        """Callback for joint states"""
        with self.lock:
            self.current_joint_state = msg
    
    def target_callback(self, msg):
        """Callback for target point"""
        if msg is None:
            return
        frame = msg.header.frame_id or ''
        with self.lock:
            if frame in self.pixel_frame_ids:
                self.pixel_target = {
                    'x': float(msg.point.x),
                    'y': float(msg.point.y),
                    'frame_id': frame
                }
                self._request_tracker_reset()
                self.metric_target = None
                self.get_logger().info(f'Pixel target received: ({msg.point.x:.1f}, {msg.point.y:.1f}) in {frame}')
            else:
                self.metric_target = msg
                self.pixel_target = None
                self._reset_tracker_state()
                self.get_logger().info(
                    f'3D target received: [{msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f}] in {frame}'
                )
            self.servoing_active = True
            self.stop_hold_counter = 0

    def camera_info_callback(self, msg):
        """Store camera intrinsics"""
        with self.depth_lock:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.camera_info_msg = msg

    def depth_callback(self, msg):
        """Cache latest aligned depth frame"""
        with self.depth_lock:
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if np.issubdtype(depth.dtype, np.floating):
                    self.depth_scale = 1.0
                else:
                    self.depth_scale = 0.001
                self.depth_image = depth
            except Exception as exc:
                self.get_logger().error(f'Failed to convert depth image: {exc}')

    def color_callback(self, msg):
        """Cache latest color frame and update the pixel tracker."""
        try:
            color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert color image: {exc}')
            return
        with self.color_lock:
            self.color_image = color
            self.color_width = int(msg.width or color.shape[1])
            self.color_height = int(msg.height or color.shape[0])
        self._update_tracker(color)

    def get_depth_at_pixel(self, px, py):
        """Return depth (m) at the requested pixel using robust averaging."""
        with self.depth_lock:
            if self.depth_image is None:
                return None
            depth_raw = robust_depth_at_pixel(self.depth_image, px, py)
            if depth_raw is None:
                return None
            return float(depth_raw) * float(self.depth_scale)

    def _resolve_pixel_target(self, target):
        if target is None:
            return None
        px = float(target.get('x', 0.0))
        py = float(target.get('y', 0.0))
        if self.pixel_target_is_display:
            if self.camera_info_msg is None:
                return None
            try:
                px, py = display_to_camera(
                    px,
                    py,
                    float(self.camera_info_msg.width),
                    float(self.camera_info_msg.height),
                    rotation_deg=self.pixel_target_rotation_deg,
                )
            except Exception as exc:
                self.get_logger().warn(f'Failed to remap display pixel target: {exc}')
                return None
        return px, py

    def _build_point_from_camera_pixels(self, px, py):
        if self.camera_matrix is None:
            self.get_logger().warn('Pixel target received but camera intrinsics are unavailable')
            return None
        depth = self.get_depth_at_pixel(px, py)
        if depth is None:
            return None
        camera_point = pixel_to_3d((px, py), depth, self.camera_matrix)
        if camera_point is None:
            return None
        converted = PointStamped()
        converted.header.stamp = self.get_clock().now().to_msg()
        converted.header.frame_id = self.camera_frame
        converted.point.x = float(camera_point[0])
        converted.point.y = float(camera_point[1])
        converted.point.z = float(camera_point[2])
        return converted

    def _build_point_from_pixel_target(self, target):
        px_py = self._resolve_pixel_target(target)
        if px_py is None:
            return None
        return self._build_point_from_camera_pixels(*px_py)

    def _reset_tracker_state(self):
        with self.tracker_lock:
            self.tracker = None
            self.tracker_visible = False
            self.tracker_needs_reset = False
            self.tracked_pixel = None
        self._publish_tracked_pixel(None, None, False)

    def _request_tracker_reset(self):
        with self.tracker_lock:
            self.tracker = None
            self.tracked_pixel = None
        self.tracker_needs_reset = True
        self._publish_tracked_pixel(None, None, False)

    def _update_tracker(self, frame):
        with self.lock:
            target = dict(self.pixel_target) if self.pixel_target else None
            needs_reset = self.tracker_needs_reset
            self.tracker_needs_reset = False
        if target is None or frame is None:
            return
        px_py = self._resolve_pixel_target(target)
        if px_py is None:
            return
        px0, py0 = px_py
        h, w = frame.shape[:2]
        with self.tracker_lock:
            if self.tracker is None or needs_reset:
                tracker = WhitePointTracker(
                    int(px0),
                    int(py0),
                    template_size=self.tracker_template_size,
                    search_radius=self.tracker_search_radius,
                    min_match_score=self.tracker_min_score,
                    reacquire_score=self.tracker_reacquire_score,
                    max_lost_frames=5,
                )
                tracker.initialize(frame)
                self.tracker = tracker
                px_est, py_est, visible = tracker.px, tracker.py, True
            else:
                px_est, py_est, visible = self.tracker.update(frame)
            px_est = float(np.clip(px_est, 0, w - 1))
            py_est = float(np.clip(py_est, 0, h - 1))
            self.tracker_visible = bool(visible)
            self.tracked_pixel = {
                'x': px_est,
                'y': py_est,
                'visible': bool(visible),
                'timestamp': time.time()
            }
            publish_px, publish_py = px_est, py_est
            publish_visible = self.tracker_visible
        self._publish_tracked_pixel(publish_px, publish_py, publish_visible)

    def _publish_tracked_pixel(self, px, py, visible):
        if self.tracked_pixel_pub is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_pixel'
        if px is None or py is None:
            msg.point.x = 0.0
            msg.point.y = 0.0
            msg.point.z = 0.0
        else:
            msg.point.x = float(px)
            msg.point.y = float(py)
            msg.point.z = 1.0 if visible else 0.0
        self.tracked_pixel_pub.publish(msg)

    def _get_active_pixel(self):
        with self.tracker_lock:
            tracked = dict(self.tracked_pixel) if self.tracked_pixel else None
        if tracked is not None:
            age = time.time() - tracked.get('timestamp', 0.0)
            if tracked.get('visible') or age <= self.tracker_grace_s:
                return tracked['x'], tracked['y']
        with self.lock:
            target = dict(self.pixel_target) if self.pixel_target else None
        return self._resolve_pixel_target(target)
    
    def get_transform(self, target_frame, source_frame, timeout=1.0):
        """
        Get transform between two frames
        
        Args:
            target_frame: Target frame name
            source_frame: Source frame name
            timeout: Timeout in seconds
            
        Returns:
            TransformStamped or None if failed
        """
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=timeout)
            )
            return trans
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
            return None
    
    def transform_point(self, point_stamped, target_frame):
        """
        Transform a point to target frame
        
        Args:
            point_stamped: PointStamped message
            target_frame: Target frame name
            
        Returns:
            Transformed point as numpy array [x, y, z] or None
        """
        try:
            # Transform point
            transformed = self.tf_buffer.transform(point_stamped, target_frame)
            return np.array([
                transformed.point.x,
                transformed.point.y,
                transformed.point.z
            ])
        except TransformException as ex:
            self.get_logger().warn(f'Could not transform point: {ex}')
            return None
    
    def control_loop(self):
        """Main control loop"""
        with self.lock:
            active = self.servoing_active
            metric_target = self.metric_target
        
        if not active:
            return
        
        active_pixel = self._get_active_pixel()
        if active_pixel is not None:
            point_msg = self._build_point_from_camera_pixels(*active_pixel)
            if point_msg is None:
                self.publish_status('Waiting for depth at pixel target')
                self.send_velocity_command(0.0, 0.0)
                return
        elif metric_target is not None:
            point_msg = metric_target
        else:
            self.send_velocity_command(0.0, 0.0)
            return
        
        target_in_base = self.transform_point(point_msg, self.base_frame)
        if target_in_base is None:
            self.publish_status('TF lookup failed for target')
            self.send_velocity_command(0.0, 0.0)
            return
        
        distance = float(np.linalg.norm(target_in_base[:2]))
        if distance <= self.stop_distance:
            self.stop_hold_counter += 1
        else:
            self.stop_hold_counter = 0
        
        if self.stop_hold_counter >= self.stop_hold_required:
            with self.lock:
                self.servoing_active = False
            self.send_velocity_command(0.0, 0.0)
            self.publish_status('Target reached')
            self.get_logger().info('Target reached')
            return
        
        distance_error = max(0.0, distance - self.stop_distance)
        linear_vel = self.linear_gain * distance_error
        if distance_error > 0.1 and linear_vel > 0.0:
            linear_vel = max(linear_vel, self.min_linear_vel)
        linear_vel = float(np.clip(linear_vel, 0.0, self.max_linear_vel))
        
        angular_error = math.atan2(target_in_base[1], target_in_base[0])
        angular_vel = self.angular_gain * angular_error
        angular_vel = float(np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel))
        
        self.send_velocity_command(linear_vel, angular_vel)
        self.publish_status(
            f'Servoing: dist={distance:.2f}m (→{self.stop_distance:.2f}m), ang={math.degrees(angular_error):.1f}°'
        )
    
    def publish_status(self, status):
        """Publish status message"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
    
    def send_joint_command(self, joint_name, position):
        """
        Send joint position command
        
        Args:
            joint_name: Name of the joint
            position: Desired position
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [joint_name]
        msg.position = [position]
        self.joint_cmd_pub.publish(msg)
    
    def send_velocity_command(self, linear_x=0.0, angular_z=0.0):
        """
        Send velocity command
        
        Args:
            linear_x: Linear velocity in x direction (m/s)
            angular_z: Angular velocity around z axis (rad/s)
        """
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = VisualServoingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot
        node.send_velocity_command(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

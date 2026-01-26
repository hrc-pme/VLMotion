#!/usr/bin/env python3
"""
ROS2 Camera Subscriber for RealSense Cameras
Replaces pyrealsense2 with ROS2 topic subscriptions
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
from threading import Lock
import time

from VLServo.gui_rotation import resolve_rotation_degrees


class CameraSubscriber(Node):
    """Subscribe to RealSense camera topics via ROS2"""
    
    def __init__(self, camera_name='camera', node_name='camera_subscriber'):
        """
        Initialize camera subscriber
        
        Args:
            camera_name: Name of the camera (e.g., 'camera', 'd405', 'd435i')
            node_name: Name of the ROS2 node
        """
        super().__init__(node_name)
        
        self.camera_name = camera_name
        self.bridge = CvBridge()
        self.lock = Lock()
        
        # Image data
        self.color_image = None
        self.depth_image = None
        self.aligned_depth_image = None
        
        # Camera info
        self.color_camera_info = None
        self.depth_camera_info = None
        
        # Timestamps
        self.last_color_time = None
        self.last_depth_time = None
        
        # Subscribe to color image
        # Topics: /d435i/color/image_raw, /d405/color/image_rect_raw
        color_topic = f'/{camera_name}/color/image_raw'
        self.color_sub = self.create_subscription(
            Image,
            color_topic,
            self.color_callback,
            10
        )
        
        # Subscribe to depth image
        self.depth_sub = self.create_subscription(
            Image,
            f'/{camera_name}/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        # Subscribe to aligned depth image (depth aligned to color)
        self.aligned_depth_sub = self.create_subscription(
            Image,
            f'/{camera_name}/aligned_depth_to_color/image_raw',
            self.aligned_depth_callback,
            10
        )
        
        # Subscribe to camera info
        self.color_info_sub = self.create_subscription(
            CameraInfo,
            f'/{camera_name}/color/camera_info',
            self.color_info_callback,
            10
        )
        
        self.depth_info_sub = self.create_subscription(
            CameraInfo,
            f'/{camera_name}/depth/camera_info',
            self.depth_info_callback,
            10
        )
        
        self.get_logger().info(f'Camera subscriber initialized for {camera_name}')
        self.get_logger().info(f'  Color topic: {color_topic}')
        self.get_logger().info(f'  Depth topic: /{camera_name}/depth/image_rect_raw')
        self.gui_rotation_deg = resolve_rotation_degrees()
    
    def color_callback(self, msg):
        """Callback for color image"""
        with self.lock:
            try:
                # Force OpenCV-friendly RGB regardless of published encoding
                self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                self.last_color_time = time.time()
            except Exception as e:
                self.get_logger().error(f'Error converting color image: {e}')
    
    def depth_callback(self, msg):
        """Callback for depth image"""
        with self.lock:
            try:
                # Depth images are typically 16UC1
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self.last_depth_time = time.time()
            except Exception as e:
                self.get_logger().error(f'Error converting depth image: {e}')
    
    def aligned_depth_callback(self, msg):
        """Callback for aligned depth image"""
        with self.lock:
            try:
                self.aligned_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except Exception as e:
                self.get_logger().error(f'Error converting aligned depth image: {e}')
    
    def color_info_callback(self, msg):
        """Callback for color camera info"""
        with self.lock:
            if self.color_camera_info is None:
                self.color_camera_info = self._camera_info_to_dict(msg)
                self.get_logger().info('Color camera info received')
    
    def depth_info_callback(self, msg):
        """Callback for depth camera info"""
        with self.lock:
            if self.depth_camera_info is None:
                self.depth_camera_info = self._camera_info_to_dict(msg)
                self.get_logger().info('Depth camera info received')
    
    def _camera_info_to_dict(self, msg):
        """Convert CameraInfo message to dictionary format"""
        K = np.array(msg.k).reshape((3, 3))
        D = np.array(msg.d)
        
        return {
            'camera_matrix': K,
            'distortion_coefficients': D,
            'width': msg.width,
            'height': msg.height,
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, 2],
            'cy': K[1, 2],
        }
    
    def _rotate_for_gui(self, image):
        """Rotate image according to GUI orientation if needed."""
        if image is None:
            return None
        angle = int(round(self.gui_rotation_deg)) % 360
        if angle in (90, -270):
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if angle in (270, -90):
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle in (180, -180):
            return cv2.rotate(image, cv2.ROTATE_180)
        return image

    def get_frames(self, rotate_gui=False):
        """
        Get current color and depth frames
        
        Args:
            rotate_gui: If True, rotate the color image to match GUI orientation.
        
        Returns:
            tuple: (color_image, depth_image, color_camera_info)
        """
        with self.lock:
            # Use aligned depth if available, otherwise regular depth
            depth = self.aligned_depth_image if self.aligned_depth_image is not None else self.depth_image
            color = self.color_image.copy() if self.color_image is not None else None
            depth_copy = depth.copy() if depth is not None else None
        if rotate_gui and color is not None:
            color = self._rotate_for_gui(color)
        return color, depth_copy, self.color_camera_info
    
    def is_ready(self):
        """Check if camera data is available"""
        with self.lock:
            return (self.color_image is not None and 
                    (self.depth_image is not None or self.aligned_depth_image is not None) and
                    self.color_camera_info is not None)
    
    def wait_for_frames(self, timeout=5.0):
        """
        Wait for camera frames to be available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if frames are ready, False if timeout
        """
        start_time = time.time()
        rate = self.create_rate(10)  # 10 Hz
        
        while rclpy.ok() and not self.is_ready():
            if time.time() - start_time > timeout:
                self.get_logger().warn(f'Timeout waiting for camera frames')
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
            rate.sleep()
        
        return True


def pixel_from_3d(xyz, camera_info):
    """Convert 3D point to pixel coordinates"""
    x_in, y_in, z_in = xyz
    camera_matrix = camera_info['camera_matrix']
    f_x = camera_matrix[0, 0]
    c_x = camera_matrix[0, 2]
    f_y = camera_matrix[1, 1]
    c_y = camera_matrix[1, 2]
    x_pix = ((f_x * x_in) / z_in) + c_x
    y_pix = ((f_y * y_in) / z_in) + c_y
    xy = np.array([x_pix, y_pix])
    return xy


def pixel_to_3d(xy_pix, z_in, camera_info):
    """Convert pixel coordinates and depth to 3D point"""
    x_pix, y_pix = xy_pix
    camera_matrix = camera_info['camera_matrix']
    f_x = camera_matrix[0, 0]
    c_x = camera_matrix[0, 2]
    f_y = camera_matrix[1, 1]
    c_y = camera_matrix[1, 2]
    x_out = ((x_pix - c_x) * z_in) / f_x
    y_out = ((y_pix - c_y) * z_in) / f_y
    xyz_out = np.array([x_out, y_out, z_in])
    return xyz_out


def get_depth_scale():
    """
    Get depth scale for RealSense camera
    RealSense typically uses millimeters, so scale is 0.001 to convert to meters
    """
    return 0.001


# Example usage
if __name__ == '__main__':
    rclpy.init()
    
    # Create camera subscriber
    camera_sub = CameraSubscriber(camera_name='camera')
    
    try:
        # Wait for frames
        if camera_sub.wait_for_frames(timeout=10.0):
            # Get frames
            color, depth, camera_info = camera_sub.get_frames()
            
            if color is not None:
                print(f"Color image shape: {color.shape}")
            if depth is not None:
                print(f"Depth image shape: {depth.shape}")
            if camera_info is not None:
                print(f"Camera info: {camera_info}")
            
            # Keep spinning
            rclpy.spin(camera_sub)
        else:
            print("Failed to get camera frames")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        camera_sub.destroy_node()
        rclpy.shutdown()

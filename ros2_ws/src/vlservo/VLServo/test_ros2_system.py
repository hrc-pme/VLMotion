#!/usr/bin/env python3
"""
Test script for ROS2 visual servoing system
Demonstrates how to use the camera subscriber and TF
"""

import rclpy
from rclpy.node import Node
import sys
import os

# Add VLServo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from VLServo.camera_ros2_subscriber import CameraSubscriber
from geometry_msgs.msg import PointStamped
import tf2_ros
import numpy as np
import cv2


class VisualServoTest(Node):
    """Test node for visual servoing"""
    
    def __init__(self):
        super().__init__('visual_servo_test')
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publisher for target points
        self.target_pub = self.create_publisher(
            PointStamped,
            '/visual_servo/target_point',
            10
        )
        
        self.get_logger().info('Visual Servo Test Node initialized')
        
    def publish_target(self, x, y, z, frame_id='base_link'):
        """Publish a target point for visual servoing"""
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x = x
        msg.point.y = y
        msg.point.z = z
        
        self.target_pub.publish(msg)
        self.get_logger().info(f'Published target: [{x:.3f}, {y:.3f}, {z:.3f}] in {frame_id}')
    
    def get_transform(self, target_frame, source_frame):
        """Get transform between two frames"""
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return trans
        except Exception as ex:
            self.get_logger().warn(f'Could not transform: {ex}')
            return None
    
    def print_available_frames(self):
        """Print all available TF frames"""
        frames = self.tf_buffer.all_frames_as_string()
        self.get_logger().info(f'Available TF frames:\n{frames}')


def test_camera_subscriber():
    """Test camera subscriber"""
    print("Testing Camera Subscriber...")
    
    rclpy.init()
    
    # Create camera subscriber for head camera (D435i)
    camera_sub = CameraSubscriber(camera_name='camera', node_name='test_camera_sub')
    
    try:
        print("Waiting for camera frames...")
        if camera_sub.wait_for_frames(timeout=10.0):
            print("✓ Camera frames received!")
            
            # Get frames
            color, depth, camera_info = camera_sub.get_frames()
            
            if color is not None:
                print(f"✓ Color image: {color.shape}")
                cv2.imwrite('/tmp/test_color.jpg', color)
                print("  Saved to /tmp/test_color.jpg")
            
            if depth is not None:
                print(f"✓ Depth image: {depth.shape}")
                # Normalize depth for visualization
                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite('/tmp/test_depth.jpg', depth_normalized)
                print("  Saved to /tmp/test_depth.jpg")
            
            if camera_info is not None:
                print(f"✓ Camera info:")
                print(f"  Resolution: {camera_info['width']}x{camera_info['height']}")
                print(f"  Focal length: fx={camera_info['fx']:.2f}, fy={camera_info['fy']:.2f}")
                print(f"  Principal point: cx={camera_info['cx']:.2f}, cy={camera_info['cy']:.2f}")
        else:
            print("✗ Failed to get camera frames (timeout)")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        camera_sub.destroy_node()
        rclpy.shutdown()


def test_tf_frames():
    """Test TF frame lookups"""
    print("\nTesting TF Frames...")
    
    rclpy.init()
    
    test_node = VisualServoTest()
    
    try:
        # Wait a bit for TF to populate
        print("Waiting for TF data...")
        import time
        for i in range(5):
            rclpy.spin_once(test_node, timeout_sec=0.5)
            time.sleep(0.5)
        
        # Print available frames
        test_node.print_available_frames()
        
        # Test some common transforms
        print("\nTesting common transforms:")
        
        transforms_to_test = [
            ('base_link', 'camera_link'),
            ('base_link', 'link_gripper_fingertip_left'),
            ('base_link', 'link_gripper_fingertip_right'),
            ('camera_link', 'camera_color_optical_frame'),
        ]
        
        for target, source in transforms_to_test:
            trans = test_node.get_transform(target, source)
            if trans:
                t = trans.transform.translation
                print(f"✓ {source} → {target}: [{t.x:.3f}, {t.y:.3f}, {t.z:.3f}]")
            else:
                print(f"✗ {source} → {target}: Failed")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        test_node.destroy_node()
        rclpy.shutdown()


def test_publish_target():
    """Test publishing target points"""
    print("\nTesting Target Publishing...")
    
    rclpy.init()
    
    test_node = VisualServoTest()
    
    try:
        # Wait for subscribers
        import time
        time.sleep(1.0)
        
        # Publish a test target in front of the robot
        test_node.publish_target(0.5, 0.0, 0.5, frame_id='base_link')
        print("✓ Published test target")
        
        # Spin for a bit to let it publish
        for i in range(10):
            rclpy.spin_once(test_node, timeout_sec=0.1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        test_node.destroy_node()
        rclpy.shutdown()


def main():
    """Run all tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ROS2 visual servoing components')
    parser.add_argument('--camera', action='store_true', help='Test camera subscriber')
    parser.add_argument('--tf', action='store_true', help='Test TF frames')
    parser.add_argument('--target', action='store_true', help='Test target publishing')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.all or not (args.camera or args.tf or args.target):
        # Run all tests
        args.camera = args.tf = args.target = True
    
    if args.camera:
        test_camera_subscriber()
    
    if args.tf:
        test_tf_frames()
    
    if args.target:
        test_publish_target()
    
    print("\nAll tests completed!")


if __name__ == '__main__':
    main()

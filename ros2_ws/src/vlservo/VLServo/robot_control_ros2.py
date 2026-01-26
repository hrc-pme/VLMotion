#!/usr/bin/env python3
"""
ROS2 Robot Control Module
Replaces stretch_body direct control with ROS2 topics/actions
Compatible with stretch_driver node
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
import threading


class StretchROS2Controller(Node):
    """Control Stretch robot via ROS2 topics and actions"""
    
    def __init__(self, node_name='stretch_ros2_controller'):
        super().__init__(node_name)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        
        # Joint trajectory action client
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/stretch_controller/follow_joint_trajectory'
        )
        
        # Joint state subscriber
        self.joint_states = {}
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/stretch/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.get_logger().info('Stretch ROS2 Controller initialized')
    
    def joint_state_callback(self, msg):
        """Store latest joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_states[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0
                }
    
    def move_base(self, linear_x=0.0, angular_z=0.0, duration=1.0):
        """
        Move the mobile base
        
        Args:
            linear_x: Forward/backward velocity (m/s), positive = forward
            angular_z: Rotation velocity (rad/s), positive = counter-clockwise
            duration: How long to apply the command (seconds)
        """
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        
        # Publish for specified duration
        rate = self.create_rate(10)  # 10 Hz
        end_time = time.time() + duration
        
        while time.time() < end_time and rclpy.ok():
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        
        # Stop
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
    
    def set_head_pan(self, angle_rad, duration=2.0):
        """
        Set head pan position
        
        Args:
            angle_rad: Target angle in radians
            duration: Time to complete motion (seconds)
        """
        self._move_joint('joint_head_pan', angle_rad, duration)
    
    def set_head_tilt(self, angle_rad, duration=2.0):
        """
        Set head tilt position
        
        Args:
            angle_rad: Target angle in radians (negative = look down)
            duration: Time to complete motion (seconds)
        """
        self._move_joint('joint_head_tilt', angle_rad, duration)
    
    def set_head_tilt_deg(self, angle_deg, duration=2.0):
        """
        Set head tilt position in degrees
        
        Args:
            angle_deg: Target angle in degrees (negative = look down)
            duration: Time to complete motion (seconds)
        """
        import math
        angle_rad = math.radians(angle_deg)
        self.set_head_tilt(angle_rad, duration)
    
    def set_arm_extension(self, extension_m, duration=3.0):
        """
        Set arm extension
        
        Args:
            extension_m: Target extension in meters (0.0 to ~0.5)
            duration: Time to complete motion (seconds)
        """
        # Stretch arm is composed of 4 prismatic joints
        # They move together to extend the arm
        joints = [
            'joint_arm_l0',
            'joint_arm_l1', 
            'joint_arm_l2',
            'joint_arm_l3'
        ]
        
        # Each joint moves 1/4 of the total extension
        position_per_joint = extension_m / 4.0
        
        self._move_joints(
            joints,
            [position_per_joint] * 4,
            duration
        )
    
    def set_lift_height(self, height_m, duration=3.0):
        """
        Set lift height
        
        Args:
            height_m: Target height in meters (0.0 to ~1.1)
            duration: Time to complete motion (seconds)
        """
        self._move_joint('joint_lift', height_m, duration)
    
    def set_wrist_yaw(self, angle_rad, duration=2.0):
        """
        Set wrist yaw position
        
        Args:
            angle_rad: Target angle in radians
            duration: Time to complete motion (seconds)
        """
        self._move_joint('joint_wrist_yaw', angle_rad, duration)
    
    def set_gripper_aperture(self, aperture, duration=2.0):
        """
        Set gripper opening
        
        Args:
            aperture: Gripper opening (negative = close, positive = open)
                     Typical range: -100 to 0 (stretch_gripper units)
            duration: Time to complete motion (seconds)
        """
        self._move_joint('joint_gripper_finger_left', aperture, duration)
    
    def _move_joint(self, joint_name, position, duration):
        """Move a single joint to target position"""
        self._move_joints([joint_name], [position], duration)
    
    def _move_joints(self, joint_names, positions, duration):
        """
        Move multiple joints to target positions
        
        Args:
            joint_names: List of joint names
            positions: List of target positions
            duration: Time to complete motion (seconds)
        """
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Trajectory action server not available')
            return False
        
        # Create trajectory goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        
        goal.trajectory.points = [point]
        
        # Send goal asynchronously
        self.get_logger().info(f'Moving joints: {joint_names} to {positions}')
        future = self.trajectory_client.send_goal_async(goal)
        
        return True
    
    def get_joint_position(self, joint_name):
        """Get current position of a joint"""
        if joint_name in self.joint_states:
            return self.joint_states[joint_name]['position']
        return None
    
    def stow_robot(self):
        """Move robot to stowed position"""
        self.get_logger().info('Stowing robot...')
        
        # Retract arm
        self.set_arm_extension(0.0, duration=3.0)
        time.sleep(0.5)
        
        # Lower lift
        self.set_lift_height(0.2, duration=3.0)
        time.sleep(0.5)
        
        # Center head
        self.set_head_pan(0.0, duration=2.0)
        self.set_head_tilt(0.0, duration=2.0)
        
        self.get_logger().info('Robot stowed')
    
    def ready_pose(self, head_tilt_deg=-30.0):
        """
        Move robot to ready position for grasping
        
        Args:
            head_tilt_deg: Head tilt angle in degrees (negative = look down)
        """
        self.get_logger().info('Moving to ready pose...')
        
        # Raise lift to comfortable height
        self.set_lift_height(0.6, duration=3.0)
        time.sleep(0.5)
        
        # Extend arm slightly
        self.set_arm_extension(0.02, duration=2.0)
        time.sleep(0.5)
        
        # Set head to look forward/down
        self.set_head_pan(0.0, duration=2.0)
        self.set_head_tilt_deg(head_tilt_deg, duration=2.0)
        
        # Open gripper
        self.set_gripper_aperture(-50, duration=2.0)
        
        self.get_logger().info('Ready pose complete')


# Singleton instance
_controller_instance = None
_controller_lock = threading.Lock()


def get_controller():
    """Get or create the singleton controller instance"""
    global _controller_instance
    
    with _controller_lock:
        if _controller_instance is None:
            if not rclpy.ok():
                rclpy.init()
            _controller_instance = StretchROS2Controller()
        return _controller_instance


def set_head_tilt_deg(angle_deg, duration=2.0):
    """
    Convenience function to set head tilt
    Compatible with pose_utils.set_head_tilt_deg signature
    """
    controller = get_controller()
    controller.set_head_tilt_deg(angle_deg, duration)


def go_to_start_pose(head_tilt_deg=-30.0):
    """
    Convenience function to move to start pose
    Compatible with pose_utils.go_to_start_pose signature
    """
    controller = get_controller()
    controller.ready_pose(head_tilt_deg)


if __name__ == '__main__':
    # Test the controller
    rclpy.init()
    controller = StretchROS2Controller()
    
    try:
        # Spin in background
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(controller)
        
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        
        # Test commands
        print("Testing ROS2 controller...")
        time.sleep(2.0)
        
        print("Moving head...")
        controller.set_head_tilt_deg(-20.0)
        time.sleep(3.0)
        
        print("Moving to ready pose...")
        controller.ready_pose()
        time.sleep(5.0)
        
        print("Test complete!")
        
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

#!/usr/bin/env python3
import math

import rclpy
from rclpy.duration import Duration

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class HeadTargetTracker:
    """Track a world-frame target with Stretch's head pan/tilt joints."""

    def __init__(
        self,
        node,
        tf_buffer,
        trajectory_client,
        *,
        world_frame='odom',
        base_frame='base_link',
        head_frame='link_head_pan',
        enabled=True,
        min_interval_sec=1.0,
        pan_deadband_deg=4.0,
        tilt_deadband_deg=4.0,
        command_duration_sec=0.25,
        pan_min_deg=-175.0,
        pan_max_deg=175.0,
        tilt_min_deg=-90.0,
        tilt_max_deg=20.0,
        motion_active_cb=None,
    ):
        self.node = node
        self.tf_buffer = tf_buffer
        self.trajectory_client = trajectory_client
        self.world_frame = world_frame
        self.base_frame = base_frame
        self.head_frame = head_frame
        self.enabled = enabled
        self.min_interval_sec = min_interval_sec
        self.pan_deadband = math.radians(pan_deadband_deg)
        self.tilt_deadband = math.radians(tilt_deadband_deg)
        self.command_duration_sec = command_duration_sec
        self.pan_min = math.radians(pan_min_deg)
        self.pan_max = math.radians(pan_max_deg)
        self.tilt_min = math.radians(tilt_min_deg)
        self.tilt_max = math.radians(tilt_max_deg)
        self.motion_active_cb = motion_active_cb

        self._last_pan = None
        self._last_tilt = None
        self._last_cmd_time_ns = 0
        self._head_sending = False
        self._head_goal_active = False

    def update(self, target_world):
        if not self.enabled or target_world is None:
            return
        if self.motion_active_cb is not None and self.motion_active_cb():
            return
        if self._head_sending or self._head_goal_active:
            return

        now_ns = self.node.get_clock().now().nanoseconds
        if now_ns - self._last_cmd_time_ns < int(self.min_interval_sec * 1e9):
            return

        command = self.compute_command(target_world)
        if command is None:
            return
        head_pan, head_tilt = command

        if self._last_pan is not None and self._last_tilt is not None:
            if (
                abs(head_pan - self._last_pan) < self.pan_deadband
                and abs(head_tilt - self._last_tilt) < self.tilt_deadband
            ):
                return

        self.send_command(head_pan, head_tilt, now_ns)

    def compute_command(self, target_world):
        try:
            head_tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.head_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            base_tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception as exc:
            self.node.get_logger().warn(
                f'Head tracking skipped; TF lookup failed: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

        head_pos = head_tf.transform.translation
        dx = target_world.x - head_pos.x
        dy = target_world.y - head_pos.y
        dz = target_world.z - head_pos.z
        horizontal_dist = math.hypot(dx, dy)
        if horizontal_dist < 1e-4:
            return None

        q = base_tf.transform.rotation
        base_yaw = self.quat_to_yaw(q.x, q.y, q.z, q.w)
        target_yaw = math.atan2(dy, dx)
        head_pan = self.wrap_pi(target_yaw - base_yaw)
        head_tilt = math.atan2(dz, horizontal_dist)

        return (
            self.clamp(head_pan, self.pan_min, self.pan_max),
            self.clamp(head_tilt, self.tilt_min, self.tilt_max),
        )

    def send_command(self, head_pan, head_tilt, now_ns):
        if not self.trajectory_client.wait_for_server(timeout_sec=0.1):
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_head_pan', 'joint_head_tilt']

        point = JointTrajectoryPoint()
        point.positions = [head_pan, head_tilt]
        point.time_from_start = Duration(seconds=self.command_duration_sec).to_msg()
        goal.trajectory.points = [point]

        self._head_sending = True
        self._last_pan = head_pan
        self._last_tilt = head_tilt
        self._last_cmd_time_ns = now_ns

        future = self.trajectory_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        self._head_sending = False
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.node.get_logger().warn(f'Head tracking goal response failed: {exc}')
            self._head_goal_active = False
            return

        if not goal_handle.accepted:
            self.node.get_logger().debug('Head tracking goal rejected')
            self._head_goal_active = False
            return

        self._head_goal_active = True
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        self._head_goal_active = False
        try:
            future.result()
        except Exception as exc:
            self.node.get_logger().debug(f'Head tracking result failed: {exc}')

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def clamp(value, lo, hi):
        return max(lo, min(hi, value))

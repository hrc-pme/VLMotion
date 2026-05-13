#!/usr/bin/env python3
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, PointStamped, Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String
from tf2_geometry_msgs import do_transform_point
import tf2_ros
from trajectory_msgs.msg import JointTrajectoryPoint

from white_point_pipeline.head_target_tracker import HeadTargetTracker
from white_point_pipeline.side_reach_planner import SideReachPlanner
from white_point_pipeline.wrist_target_planner import WristTargetPlanner


class WhitePointDirectMotion(Node):
    """Two-stage direct motion node for Stretch.

    Strategy:
    1. First /white_point_base: move base_link to the wall-normal approach point.
    2. Rotate toward the first target and wait for a second point selection.
    3. Second /white_point_base: use the closer, more accurate target estimate.
    4. Rotate toward that final target, drive the gripper close, then extend arm.

    This node intentionally does not use the orange tangent offset, target
    compensation, or functions from white_point_full_motion.py.
    """

    def __init__(self):
        super().__init__('white_point_direct_motion')

        self.declare_parameter('target_topic', '/white_point_base')
        self.declare_parameter('axis_topic', '/panel_axis_base')
        self.declare_parameter('cmd_vel_topic', '/stretch/cmd_vel')
        self.declare_parameter('trajectory_action', '/stretch_controller/follow_joint_trajectory')
        self.declare_parameter('selection_phase_topic', '/white_point_selection_phase')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('left_fingertip_frame', 'link_gripper_fingertip_left')
        self.declare_parameter('right_fingertip_frame', 'link_gripper_fingertip_right')

        self.declare_parameter('two_stage_enabled', True)
        self.declare_parameter('approach_distance', 0.60)
        self.declare_parameter('approach_tolerance', 0.035)
        self.declare_parameter('yaw_tolerance_deg', 4.0)
        self.declare_parameter('final_xy_tolerance', 0.035)
        self.declare_parameter('final_y_tolerance', 0.035)
        self.declare_parameter('gripper_center_back_offset', 0.02)
        self.declare_parameter('gripper_z_offset', 0.10)
        self.declare_parameter('arm_contact_margin', 0.025)
        self.declare_parameter('arm_contact_overshoot_enabled', False)
        self.declare_parameter('final_contact_push_enabled', True)
        self.declare_parameter('final_contact_push_distance', 0.02)
        self.declare_parameter('first_lift_adaptive_enabled', True)
        self.declare_parameter('first_lift_far_dist_near', 0.50)
        self.declare_parameter('first_lift_far_dist_far', 1.60)
        self.declare_parameter('first_lift_max_drop', 0.12)
        self.declare_parameter('first_lift_max_cap', 0.90)

        self.declare_parameter('max_linear_speed', 0.18)
        self.declare_parameter('max_angular_speed', 0.45)
        self.declare_parameter('final_max_linear_speed', 0.08)
        self.declare_parameter('k_linear', 0.45)
        self.declare_parameter('k_angular', 1.1)

        self.declare_parameter('lift_min', 0.0)
        self.declare_parameter('lift_max', 1.10)
        self.declare_parameter('arm_min', 0.0)
        self.declare_parameter('arm_max', 0.50)
        self.declare_parameter('head_tracking_enabled', True)
        self.declare_parameter('head_tracking_min_interval_sec', 1.0)
        self.declare_parameter('head_tracking_pan_deadband_deg', 4.0)
        self.declare_parameter('head_tracking_tilt_deadband_deg', 4.0)
        self.declare_parameter('head_tracking_during_base_motion', True)
        self.declare_parameter('side_reach_enabled', True)
        self.declare_parameter('side_reach_desired_arm_extension', 0.20)
        self.declare_parameter('side_reach_min_arm_extension', 0.08)
        self.declare_parameter('side_reach_x_tolerance', 0.025)
        self.declare_parameter('arm_extension_axis', 'y')
        self.declare_parameter('arm_extension_sign', -1.0)
        self.declare_parameter('arm_contact_tolerance', 0.020)
        self.declare_parameter('arm_contact_retry_limit', 2)
        self.declare_parameter('wrist_initial_yaw', math.pi / 2.0)
        self.declare_parameter('wrist_contact_yaw', 0.0)
        self.declare_parameter('wrist_contact_pitch', 0.0)
        self.declare_parameter('wrist_contact_roll', 0.0)
        self.declare_parameter('close_gripper_on_start', True)
        self.declare_parameter('gripper_joint_name', 'joint_gripper_finger_left')
        self.declare_parameter('gripper_closed_position', -0.05)
        self.declare_parameter('dynamic_wrist_yaw_enabled', True)
        self.declare_parameter('dynamic_wrist_yaw_limit_deg', 70.0)
        self.declare_parameter('wrist_lateral_reach', 0.18)
        self.declare_parameter('side_axis_gripper_lateral_tolerance', 0.020)
        self.declare_parameter('lock_base_yaw_after_first_stage', True)
        self.declare_parameter('second_stage_yaw_micro_adjust_enabled', True)
        self.declare_parameter('second_stage_yaw_micro_adjust_limit_deg', 8.0)
        self.declare_parameter('second_stage_yaw_max_angular_speed', 0.12)

        self.target_topic = self.get_parameter('target_topic').value
        self.axis_topic = self.get_parameter('axis_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.trajectory_action = self.get_parameter('trajectory_action').value
        self.selection_phase_topic = self.get_parameter('selection_phase_topic').value
        self.world_frame = self.get_parameter('world_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.left_fingertip_frame = self.get_parameter('left_fingertip_frame').value
        self.right_fingertip_frame = self.get_parameter('right_fingertip_frame').value

        self.two_stage_enabled = bool(self.get_parameter('two_stage_enabled').value)
        self.approach_distance = float(self.get_parameter('approach_distance').value)
        self.approach_tolerance = float(self.get_parameter('approach_tolerance').value)
        self.yaw_tolerance = math.radians(float(self.get_parameter('yaw_tolerance_deg').value))
        self.final_xy_tolerance = float(self.get_parameter('final_xy_tolerance').value)
        self.final_y_tolerance = float(self.get_parameter('final_y_tolerance').value)
        self.gripper_center_back_offset = float(self.get_parameter('gripper_center_back_offset').value)
        self.gripper_z_offset = float(self.get_parameter('gripper_z_offset').value)
        self.arm_contact_margin = float(self.get_parameter('arm_contact_margin').value)
        self.arm_contact_overshoot_enabled = bool(self.get_parameter('arm_contact_overshoot_enabled').value)
        self.final_contact_push_enabled = bool(self.get_parameter('final_contact_push_enabled').value)
        self.final_contact_push_distance = float(self.get_parameter('final_contact_push_distance').value)
        self.first_lift_adaptive_enabled = bool(self.get_parameter('first_lift_adaptive_enabled').value)
        self.first_lift_far_dist_near = float(self.get_parameter('first_lift_far_dist_near').value)
        self.first_lift_far_dist_far = float(self.get_parameter('first_lift_far_dist_far').value)
        self.first_lift_max_drop = float(self.get_parameter('first_lift_max_drop').value)
        self.first_lift_max_cap = float(self.get_parameter('first_lift_max_cap').value)

        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.final_max_linear_speed = float(self.get_parameter('final_max_linear_speed').value)
        self.k_linear = float(self.get_parameter('k_linear').value)
        self.k_angular = float(self.get_parameter('k_angular').value)

        self.lift_min = float(self.get_parameter('lift_min').value)
        self.lift_max = float(self.get_parameter('lift_max').value)
        self.arm_min = float(self.get_parameter('arm_min').value)
        self.arm_max = float(self.get_parameter('arm_max').value)
        self.head_tracking_enabled = bool(self.get_parameter('head_tracking_enabled').value)
        self.head_tracking_during_base_motion = bool(self.get_parameter('head_tracking_during_base_motion').value)
        self.side_reach_enabled = bool(self.get_parameter('side_reach_enabled').value)
        self.arm_extension_axis = str(self.get_parameter('arm_extension_axis').value).lower()
        self.arm_extension_sign = 1.0 if float(self.get_parameter('arm_extension_sign').value) >= 0.0 else -1.0
        self.arm_contact_tolerance = float(self.get_parameter('arm_contact_tolerance').value)
        self.arm_contact_retry_limit = int(self.get_parameter('arm_contact_retry_limit').value)
        self.wrist_initial_yaw = float(self.get_parameter('wrist_initial_yaw').value)
        self.close_gripper_on_start = bool(self.get_parameter('close_gripper_on_start').value)
        self.gripper_joint_name = str(self.get_parameter('gripper_joint_name').value)
        self.gripper_closed_position = float(self.get_parameter('gripper_closed_position').value)
        self.dynamic_wrist_yaw_enabled = bool(self.get_parameter('dynamic_wrist_yaw_enabled').value)
        self.dynamic_wrist_yaw_limit = math.radians(float(self.get_parameter('dynamic_wrist_yaw_limit_deg').value))
        self.wrist_lateral_reach = float(self.get_parameter('wrist_lateral_reach').value)
        self.side_axis_gripper_lateral_tolerance = float(
            self.get_parameter('side_axis_gripper_lateral_tolerance').value
        )
        self.lock_base_yaw_after_first_stage = bool(self.get_parameter('lock_base_yaw_after_first_stage').value)
        self.second_stage_yaw_micro_adjust_enabled = bool(
            self.get_parameter('second_stage_yaw_micro_adjust_enabled').value
        )
        self.second_stage_yaw_micro_adjust_limit = math.radians(
            float(self.get_parameter('second_stage_yaw_micro_adjust_limit_deg').value)
        )
        self.second_stage_yaw_max_angular_speed = float(
            self.get_parameter('second_stage_yaw_max_angular_speed').value
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.selection_phase_pub = self.create_publisher(String, self.selection_phase_topic, 10)
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.trajectory_action,
        )
        self.head_tracker = HeadTargetTracker(
            self,
            self.tf_buffer,
            self.trajectory_client,
            world_frame=self.world_frame,
            base_frame=self.base_frame,
            enabled=self.head_tracking_enabled,
            min_interval_sec=float(self.get_parameter('head_tracking_min_interval_sec').value),
            pan_deadband_deg=float(self.get_parameter('head_tracking_pan_deadband_deg').value),
            tilt_deadband_deg=float(self.get_parameter('head_tracking_tilt_deadband_deg').value),
            motion_active_cb=self.is_non_head_motion_active,
        )
        self.side_reach_planner = SideReachPlanner(
            enabled=self.side_reach_enabled,
            desired_arm_extension=float(self.get_parameter('side_reach_desired_arm_extension').value),
            min_arm_extension=float(self.get_parameter('side_reach_min_arm_extension').value),
            x_tolerance=float(self.get_parameter('side_reach_x_tolerance').value),
            y_tolerance=self.final_y_tolerance,
            final_xy_tolerance=self.final_xy_tolerance,
            yaw_tolerance_deg=float(self.get_parameter('yaw_tolerance_deg').value),
            arm_extension_axis=self.arm_extension_axis,
            arm_extension_sign=self.arm_extension_sign,
            max_arm_extension=self.arm_max,
            max_linear_speed=self.final_max_linear_speed,
            max_angular_speed=0.25,
            k_linear=self.k_linear,
            k_angular=self.k_angular,
        )
        self.wrist_target_planner = WristTargetPlanner(
            desired_arm_extension=float(self.get_parameter('side_reach_desired_arm_extension').value),
            min_arm_extension=float(self.get_parameter('side_reach_min_arm_extension').value),
            max_arm_extension=self.arm_max,
            wrist_yaw=float(self.get_parameter('wrist_contact_yaw').value),
            wrist_pitch=float(self.get_parameter('wrist_contact_pitch').value),
            wrist_roll=float(self.get_parameter('wrist_contact_roll').value),
        )

        self.target_sub = self.create_subscription(
            PointStamped,
            self.target_topic,
            self.target_callback,
            10,
        )
        self.axis_sub = self.create_subscription(
            Float32,
            self.axis_topic,
            self.axis_callback,
            10,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
        )

        self.timer = self.create_timer(0.05, self.control_loop)
        self.phase_timer = self.create_timer(1.0, self.publish_phase)

        self.target_world = None
        self.approach_world = None
        self.side_reach_base_world = None
        self.panel_axis_base = None
        self.panel_axis_yaw_world = None
        self.panel_normal_yaw_world = None
        self.reach_yaw_world = None
        self.arm_world_yaw = None
        self.locked_base_yaw_world = None
        self.end_effector_plan = None
        self.current_lift_pos = None
        self.current_arm_pos = None
        self.last_commanded_arm_pos = 0.0
        self.phase = 'idle'
        self.second_target_locked = False
        self.after_lift_phase = 'move_approach'
        self.final_min_dist_xy = float('inf')
        self.final_dist_increase_count = 0

        self.wrist_goal_handle = None
        self.wrist_result = None
        self.wrist_sending = False
        self.lift_goal_handle = None
        self.lift_result = None
        self.lift_sending = False
        self.arm_goal_handle = None
        self.arm_result = None
        self.arm_sending = False
        self.arm_contact_retry_count = 0
        self.wrist_contact_pose_sent = False

        self.get_logger().info(
            'white_point_direct_motion ready. It uses a direct two-stage approach point flow; '
            'do not run it at the same time as white_point_full_motion.'
        )

    def axis_callback(self, msg):
        self.panel_axis_base = float(msg.data)
        base_pose = self.lookup_base_pose()
        if base_pose is not None:
            _, base_yaw = base_pose
            self.panel_axis_yaw_world = self.wrap_pi(base_yaw + self.panel_axis_base)
            normal_yaw_base = self.wrap_pi(self.panel_axis_base - math.pi / 2.0)
            self.panel_normal_yaw_world = self.wrap_pi(base_yaw + normal_yaw_base)
            locked_yaw = self.get_second_stage_base_yaw_world()
            yaw_locked = (
                self.lock_base_yaw_after_first_stage
                and self.second_target_locked
                and self.locked_base_yaw_world is not None
            )
            if not yaw_locked:
                if self.arm_extension_axis == 'y':
                    if self.arm_extension_sign >= 0.0:
                        self.reach_yaw_world = self.panel_axis_yaw_world
                    else:
                        self.reach_yaw_world = self.wrap_pi(self.panel_axis_yaw_world + math.pi)
                    self.arm_world_yaw = WristTargetPlanner.arm_yaw_from_side_geometry(
                        self.reach_yaw_world,
                        self.arm_extension_sign,
                    )
                else:
                    self.reach_yaw_world = self.wrap_pi(self.panel_normal_yaw_world + math.pi)
                    self.arm_world_yaw = self.reach_yaw_world
            effective_reach_yaw = locked_yaw if yaw_locked else self.reach_yaw_world
            effective_arm_yaw = self.arm_world_yaw if self.arm_world_yaw is not None else effective_reach_yaw
            yaw_source = 'locked_second_stage' if yaw_locked else 'panel_geometry'
            self.get_logger().info(
                f'Panel geometry locked: tangent_yaw={math.degrees(self.panel_axis_yaw_world):.1f}deg, '
                f'normal_yaw={math.degrees(self.panel_normal_yaw_world):.1f}deg, '
                f'effective_base_reach_yaw={self.format_optional_yaw(effective_reach_yaw)}, '
                f'arm_world_yaw={self.format_optional_yaw(effective_arm_yaw)}, '
                f'yaw_source={yaw_source}, '
                f'arm_axis={self.arm_extension_axis}, arm_sign={self.arm_extension_sign:+.0f}, '
                f'second_stage_yaw_locked={yaw_locked}.',
                throttle_duration_sec=1.0,
            )
        if self.target_world is not None and self.phase in ('reset_wrist', 'raise_lift', 'move_approach', 'idle'):
            self.compute_approach_point()
        if self.target_world is not None and self.phase in ('raise_lift', 'move_side_reach_pose', 'align_target'):
            self.compute_side_reach_base_point()

    def joint_state_callback(self, msg):
        if 'joint_lift' in msg.name:
            idx = msg.name.index('joint_lift')
            self.current_lift_pos = float(msg.position[idx])
        if 'wrist_extension' in msg.name:
            idx = msg.name.index('wrist_extension')
            self.current_arm_pos = float(msg.position[idx])

    def target_callback(self, msg):
        retarget_after_done = self.phase == 'done'
        is_second_or_retarget = self.phase in (
            'waiting_second_point',
            'align_target',
            'final_approach',
            'extend_arm',
            'final_push',
            'done',
        )

        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                msg.header.frame_id or self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            target_world = do_transform_point(msg, tf).point
        except Exception as exc:
            self.get_logger().warn(f'Failed to transform target to {self.world_frame}: {exc}')
            return

        self.stop_base()
        self.target_world = target_world
        self.final_min_dist_xy = float('inf')
        self.final_dist_increase_count = 0
        self.arm_contact_retry_count = 0
        self.reset_action_state()

        if self.two_stage_enabled and is_second_or_retarget:
            self.second_target_locked = True
            if self.lock_base_yaw_after_first_stage and self.locked_base_yaw_world is None:
                base_pose = self.lookup_base_pose()
                if base_pose is not None:
                    _, base_yaw = base_pose
                    self.locked_base_yaw_world = base_yaw
                    self.get_logger().info(
                        f'Second stage base yaw locked at current yaw '
                        f'{math.degrees(self.locked_base_yaw_world):.1f}deg.'
                    )
            self.side_reach_base_world = None
            self.end_effector_plan = None
            self.compute_side_reach_base_point()
            if retarget_after_done:
                self.after_lift_phase = 'final_approach'
                self.phase = 'final_approach'
                self.get_logger().info(
                    f'Fine-adjust target locked after completion: x={target_world.x:.3f}, '
                    f'y={target_world.y:.3f}, z={target_world.z:.3f}. '
                    'Keeping current base yaw and adjusting base/arm from current gripper pose.'
                )
                return
            self.after_lift_phase = 'move_side_reach_pose' if self.side_reach_enabled else 'align_target'
            self.phase = 'raise_lift'
            self.get_logger().info(
                f'Second/final target locked: x={target_world.x:.3f}, '
                f'y={target_world.y:.3f}, z={target_world.z:.3f}. '
                'Using this closer target for final approach.'
            )
            return

        self.second_target_locked = not self.two_stage_enabled
        self.locked_base_yaw_world = None
        self.wrist_contact_pose_sent = False
        self.after_lift_phase = 'move_approach' if self.two_stage_enabled else 'align_target'
        self.approach_world = None
        self.end_effector_plan = None
        self.compute_approach_point()
        self.phase = 'reset_wrist'
        self.get_logger().info(
            f'First target locked: x={target_world.x:.3f}, y={target_world.y:.3f}, z={target_world.z:.3f}. '
            f'Moving to approach distance={self.approach_distance:.2f}m, then waiting for second point.'
        )

    def reset_action_state(self):
        self.wrist_goal_handle = None
        self.wrist_result = None
        self.wrist_sending = False
        self.lift_goal_handle = None
        self.lift_result = None
        self.lift_sending = False
        self.arm_goal_handle = None
        self.arm_result = None
        self.arm_sending = False

    def publish_phase(self):
        msg = String()
        msg.data = self.gui_phase()
        self.selection_phase_pub.publish(msg)

    def gui_phase(self):
        """Map this node's internal state to the GUI's existing phase protocol."""
        if self.phase in ('idle', 'done'):
            return 'select_first_point'
        if self.phase in (
            'reset_wrist',
            'raise_lift',
            'move_approach',
            'move_side_reach_pose',
            'prepare_second_point',
        ):
            return 'moving_to_approach'
        if self.phase == 'waiting_second_point':
            return 'waiting_second_point'
        if self.phase in ('align_target', 'final_approach', 'extend_arm', 'final_push'):
            return 'moving_to_target'
        return 'select_first_point'

    def control_loop(self):
        if self.target_world is None:
            return
        self.head_tracker.update(self.target_world)

        if self.phase == 'reset_wrist':
            self.handle_wrist_reset()
            return

        if self.phase == 'raise_lift':
            self.handle_lift_raise()
            return

        if self.phase == 'move_approach':
            if self.approach_world is None:
                self.compute_approach_point()
                if self.approach_world is None:
                    self.get_logger().info(
                        'Waiting for /panel_axis_base or fallback TF to compute approach.',
                        throttle_duration_sec=2.0,
                    )
                    return
            if self.drive_base_to_point(self.approach_world):
                self.phase = 'align_target'
                self.get_logger().info('Reached normal approach point. Aligning to target.')
            return

        if self.phase == 'move_side_reach_pose':
            if self.side_reach_base_world is None:
                self.compute_side_reach_base_point()
                if self.side_reach_base_world is None:
                    self.get_logger().info(
                        'Waiting for panel tangent to compute side-reach base pose.',
                        throttle_duration_sec=2.0,
                    )
                    return
            yaw_locked = self.get_locked_second_stage_base_yaw() is not None
            if self.drive_base_to_point(self.side_reach_base_world, allow_rotation=not yaw_locked):
                if self.second_target_locked and self.lock_base_yaw_after_first_stage:
                    self.phase = 'final_approach'
                    self.get_logger().info(
                        'Reached side-reach base pose. Keeping first-stage base yaw locked; '
                        'starting wrist/arm final approach.'
                    )
                else:
                    self.phase = 'align_target'
                    self.get_logger().info('Reached side-reach base pose. Aligning base tangent for arm reach.')
            return

        if self.phase == 'align_target':
            if self.rotate_base_toward_target():
                if self.two_stage_enabled and not self.second_target_locked:
                    if self.lock_base_yaw_after_first_stage:
                        base_pose = self.lookup_base_pose()
                        if base_pose is not None:
                            _, base_yaw = base_pose
                            self.locked_base_yaw_world = base_yaw
                            self.get_logger().info(
                                f'Locked second-stage base yaw at '
                                f'{math.degrees(self.locked_base_yaw_world):.1f}deg.'
                            )
                    self.phase = 'prepare_second_point'
                    self.get_logger().info(
                        'Base yaw aligned to first target. Setting wrist contact pose before second selection.'
                    )
                else:
                    self.phase = 'final_approach'
                    self.get_logger().info('Base yaw aligned. Starting final forward approach.')
            return

        if self.phase == 'prepare_second_point':
            self.handle_wrist_contact_pose()
            return

        if self.phase == 'waiting_second_point':
            self.stop_base()
            return

        if self.phase == 'final_approach':
            if self.drive_gripper_to_target():
                self.phase = 'extend_arm'
                self.get_logger().info('Base is aligned for arm reach. Extending arm toward target contact.')
            return

        if self.phase == 'extend_arm':
            self.handle_arm_extend()
            return

        if self.phase == 'final_push':
            self.handle_final_contact_push()
            return

    def compute_approach_point(self):
        if self.target_world is None:
            return False

        base_pose = self.lookup_base_pose()
        if base_pose is None:
            return False
        base_pos, base_yaw = base_pose

        if self.panel_axis_base is not None:
            normal_yaw_base = self.wrap_pi(self.panel_axis_base - math.pi / 2.0)
            normal_yaw_world = self.wrap_pi(base_yaw + normal_yaw_base)
            nx = math.cos(normal_yaw_world)
            ny = math.sin(normal_yaw_world)
        else:
            nx = base_pos.x - self.target_world.x
            ny = base_pos.y - self.target_world.y
            norm = math.hypot(nx, ny)
            if norm < 1e-6:
                return False
            nx /= norm
            ny /= norm

        to_robot_x = base_pos.x - self.target_world.x
        to_robot_y = base_pos.y - self.target_world.y
        if nx * to_robot_x + ny * to_robot_y < 0.0:
            nx = -nx
            ny = -ny

        pt = Point()
        pt.x = self.target_world.x + nx * self.approach_distance
        pt.y = self.target_world.y + ny * self.approach_distance
        pt.z = self.target_world.z
        self.approach_world = pt
        self.get_logger().info(
            f'Direct approach point: x={pt.x:.3f}, y={pt.y:.3f}. '
            'No tangent/orange offset is applied.'
        )
        return True

    def drive_base_to_point(self, point_world, allow_rotation=True):
        base_point = self.transform_point_to_base(point_world)
        if base_point is None:
            return False

        if not allow_rotation:
            if abs(base_point.x) <= self.approach_tolerance:
                self.stop_base()
                if abs(base_point.y) > self.final_y_tolerance:
                    self.get_logger().warn(
                        f'Locked-yaw base move reached forward distance with lateral residual '
                        f'{base_point.y * 100:.1f}cm; wrist/arm will handle final contact.'
                    )
                return True

            twist = Twist()
            twist.linear.x = self.clamp(
                self.k_linear * base_point.x,
                -self.final_max_linear_speed,
                self.final_max_linear_speed,
            )
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info(
                f'Locked-yaw base move: forward_error={base_point.x * 100:.1f}cm, '
                f'lateral_residual={base_point.y * 100:.1f}cm, cmd=({twist.linear.x:.3f}m/s, 0.000rad/s)',
                throttle_duration_sec=1.0,
            )
            return False

        dist = math.hypot(base_point.x, base_point.y)
        if dist <= self.approach_tolerance:
            self.stop_base()
            return True

        heading_error = math.atan2(base_point.y, base_point.x)
        target_behind = base_point.x < -0.02 and abs(heading_error) > math.radians(80.0)
        twist = Twist()
        if target_behind:
            if allow_rotation:
                twist.angular.z = self.clamp(self.k_angular * heading_error, -self.max_angular_speed, self.max_angular_speed)
        else:
            max_heading = math.radians(35.0)
            limited_heading = self.clamp(heading_error, -max_heading, max_heading)
            heading_scale = max(0.15, math.cos(abs(limited_heading)))
            twist.linear.x = min(self.max_linear_speed, self.k_linear * dist) * heading_scale
            if allow_rotation:
                twist.angular.z = self.clamp(self.k_angular * limited_heading, -self.max_angular_speed, self.max_angular_speed)

        if dist < 0.15:
            twist.linear.x = min(twist.linear.x, 0.06)
            if allow_rotation:
                twist.angular.z = self.clamp(twist.angular.z, -0.30, 0.30)

        self.cmd_vel_pub.publish(twist)
        self.get_logger().debug(
            f'Move approach: dist={dist:.3f}m heading={math.degrees(heading_error):.1f}deg '
            f'lin={twist.linear.x:.3f} ang={twist.angular.z:.3f} allow_rotation={allow_rotation}',
            throttle_duration_sec=0.5,
        )
        return False

    def rotate_base_toward_target(self):
        base_pose = self.lookup_base_pose()
        if base_pose is None or self.target_world is None:
            return False
        base_pos, base_yaw = base_pose
        target_yaw = self.get_base_reach_yaw_world(base_pos)
        yaw_error = self.wrap_pi(target_yaw - base_yaw)
        if abs(yaw_error) <= self.yaw_tolerance:
            self.stop_base()
            return True

        twist = Twist()
        twist.angular.z = self.clamp(self.k_angular * yaw_error, -self.max_angular_speed, self.max_angular_speed)
        self.cmd_vel_pub.publish(twist)
        self.get_logger().debug(
            f'Align target: yaw_error={math.degrees(yaw_error):.1f}deg',
            throttle_duration_sec=0.5,
        )
        return False

    def get_base_reach_yaw_world(self, base_pos):
        if self.side_reach_enabled:
            if self.end_effector_plan is not None:
                return self.end_effector_plan.base_yaw
            locked_yaw = self.get_second_stage_base_yaw_world()
            if locked_yaw is not None:
                return locked_yaw
            if self.reach_yaw_world is not None:
                return self.reach_yaw_world
        return math.atan2(
            self.target_world.y - base_pos.y,
            self.target_world.x - base_pos.x,
        )

    def get_locked_second_stage_base_yaw(self):
        if (
            self.lock_base_yaw_after_first_stage
            and self.second_target_locked
            and self.locked_base_yaw_world is not None
        ):
            return self.locked_base_yaw_world
        return None

    def get_second_stage_base_yaw_world(self):
        locked_yaw = self.get_locked_second_stage_base_yaw()
        if locked_yaw is None:
            return None
        if (
            not self.second_stage_yaw_micro_adjust_enabled
            or self.panel_axis_yaw_world is None
            or self.arm_extension_axis != 'y'
        ):
            return locked_yaw

        if self.arm_extension_sign >= 0.0:
            candidate_yaw = self.panel_axis_yaw_world
        else:
            candidate_yaw = self.wrap_pi(self.panel_axis_yaw_world + math.pi)
        yaw_delta = self.wrap_pi(candidate_yaw - locked_yaw)
        yaw_delta = self.clamp(
            yaw_delta,
            -self.second_stage_yaw_micro_adjust_limit,
            self.second_stage_yaw_micro_adjust_limit,
        )
        return self.wrap_pi(locked_yaw + yaw_delta)

    def drive_gripper_to_target(self):
        if self.target_world is None:
            return False

        if self.side_reach_enabled and (self.reach_yaw_world is not None or self.get_locked_second_stage_base_yaw() is not None):
            return self.drive_side_reach_base_alignment()

        gripper = self.get_gripper_center_world()
        base_pose = self.lookup_base_pose()
        if gripper is None or base_pose is None:
            return False

        _, base_yaw = base_pose
        dx_world = self.target_world.x - gripper.x
        dy_world = self.target_world.y - gripper.y
        dist_xy = math.hypot(dx_world, dy_world)
        dx_base, dy_base = self.rotate_world_delta_to_base(dx_world, dy_world, base_yaw)

        if dist_xy < self.final_min_dist_xy:
            self.final_min_dist_xy = dist_xy
            self.final_dist_increase_count = 0
        elif dist_xy > self.final_min_dist_xy + 0.04:
            self.final_dist_increase_count += 1
        else:
            self.final_dist_increase_count = 0

        locked_yaw = self.get_locked_second_stage_base_yaw()
        if self.side_reach_enabled and (self.reach_yaw_world is not None or locked_yaw is not None):
            target_yaw = locked_yaw if locked_yaw is not None else self.reach_yaw_world
        else:
            target_yaw = math.atan2(dy_world, dx_world)
        yaw_error = self.wrap_pi(target_yaw - base_yaw)
        if dist_xy < 0.12 and not self.side_reach_enabled:
            yaw_error = 0.0

        target_is_reached, twist = self.side_reach_planner.make_command(
            dx_base,
            dy_base,
            dist_xy,
            yaw_error,
            self.final_min_dist_xy,
            self.final_dist_increase_count,
        )
        if target_is_reached:
            self.stop_base()
            self.get_logger().info(
                f'Ready for arm reach: xy={dist_xy * 100:.1f}cm, '
                f'min_xy={self.final_min_dist_xy * 100:.1f}cm, '
                f'extension={self.arm_extension_distance(dx_base, dy_base) * 100:.1f}cm, '
                f'lateral={self.arm_lateral_error(dx_base, dy_base) * 100:.1f}cm, '
                f'yaw_error={math.degrees(yaw_error):.1f}deg.'
            )
            return True

        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(
            f'Final approach for arm reach: extension={self.arm_extension_distance(dx_base, dy_base) * 100:.1f}cm, '
            f'lateral={self.arm_lateral_error(dx_base, dy_base) * 100:.1f}cm, '
            f'yaw_error={math.degrees(yaw_error):.1f}deg, '
            f'cmd=({twist.linear.x:.3f}m/s, {twist.angular.z:.3f}rad/s)',
            throttle_duration_sec=1.0,
        )
        return False

    def drive_side_reach_base_alignment(self):
        err = self.compute_side_reach_pose_error()
        if err is None:
            return False

        along_base, lateral_base, yaw_error = err
        extension = self.end_effector_plan.arm_extension if self.end_effector_plan is not None else 0.0
        yaw_locked = self.get_locked_second_stage_base_yaw() is not None
        locked_side_axis = self.is_locked_side_axis_plan()
        lateral_reachable = True
        if locked_side_axis:
            gripper_delta = self.compute_gripper_target_error()
            if gripper_delta is not None:
                along_base = self.arm_lateral_error(*gripper_delta)
                extension = self.arm_extension_distance(*gripper_delta)
            else:
                target_base = self.transform_point_to_base(self.target_world)
                if target_base is None:
                    return False
                along_base = self.arm_lateral_error(target_base.x, target_base.y)
                side_distance = self.arm_extension_distance(target_base.x, target_base.y)
                extension = self.side_axis_arm_extension_from_distance(side_distance)
            lateral_base = 0.0
            ignore_lateral = False
        else:
            ignore_lateral = (
                yaw_locked
                and self.end_effector_plan is not None
                and self.end_effector_plan.contact_axis == 'x'
            )
            lateral_reachable = abs(lateral_base) <= self.wrist_lateral_reach

        along_tolerance = self.side_axis_gripper_lateral_tolerance if locked_side_axis else self.final_y_tolerance
        yaw_aligned = (
            abs(yaw_error) <= self.yaw_tolerance
            or (yaw_locked and not self.second_stage_yaw_micro_adjust_enabled)
        )
        aligned = (
            abs(along_base) <= along_tolerance
            and (
                locked_side_axis
                or (ignore_lateral and lateral_reachable)
                or abs(lateral_base) <= self.final_y_tolerance
            )
            and yaw_aligned
        )
        if aligned:
            self.stop_base()
            self.get_logger().info(
                f'Ready for arm reach: base_along={along_base * 100:.1f}cm, '
                f'base_lateral={lateral_base * 100:.1f}cm, extension={extension * 100:.1f}cm, '
                f'yaw_error={math.degrees(yaw_error):.1f}deg, yaw_locked={yaw_locked}, '
                f'contact_axis={self.end_effector_plan.contact_axis if self.end_effector_plan is not None else "none"}.'
            )
            return True

        if ignore_lateral and not lateral_reachable:
            self.stop_base()
            self.get_logger().warn(
                f'Locked-yaw target lateral residual is {lateral_base * 100:.1f}cm, '
                f'outside wrist compensation reach {self.wrist_lateral_reach * 100:.1f}cm. '
                'Select a closer second point or allow base yaw/side motion.'
            )
            return False

        if not ignore_lateral and abs(lateral_base) > self.final_y_tolerance:
            self.stop_base()
            self.compute_side_reach_base_point()
            self.phase = 'move_side_reach_pose'
            self.get_logger().warn(
                f'Side-reach lateral error is {lateral_base * 100:.1f}cm after yaw alignment; '
                'returning to side-reach base pose.'
            )
            return False

        twist = Twist()
        if not yaw_aligned:
            max_angular = self.second_stage_yaw_max_angular_speed if yaw_locked else 0.25
            twist.angular.z = self.clamp(self.k_angular * yaw_error, -max_angular, max_angular)
        if abs(along_base) > along_tolerance:
            twist.linear.x = self.clamp(self.k_linear * along_base, -self.final_max_linear_speed, self.final_max_linear_speed)

        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(
            f'Final side-reach base align: along={along_base * 100:.1f}cm, '
            f'lateral={lateral_base * 100:.1f}cm, yaw_error={math.degrees(yaw_error):.1f}deg, '
            f'yaw_locked={yaw_locked}, lateral_reachable={lateral_reachable}, '
            f'cmd=({twist.linear.x:.3f}m/s, {twist.angular.z:.3f}rad/s)',
            throttle_duration_sec=1.0,
        )
        return False

    def compute_side_reach_pose_error(self):
        if self.target_world is None:
            return None
        base_pose = self.lookup_base_pose()
        if base_pose is None:
            return None

        base_pos, base_yaw = base_pose
        if self.end_effector_plan is None:
            self.compute_side_reach_base_point()
        if self.end_effector_plan is None:
            return None

        target_base_x = self.end_effector_plan.base_point.x
        target_base_y = self.end_effector_plan.base_point.y

        dx_world = target_base_x - base_pos.x
        dy_world = target_base_y - base_pos.y
        along_base, lateral_base = self.rotate_world_delta_to_base(dx_world, dy_world, base_yaw)
        yaw_error = self.wrap_pi(self.end_effector_plan.base_yaw - base_yaw)
        return along_base, lateral_base, yaw_error

    def compute_side_reach_base_point(self):
        if not self.side_reach_enabled or self.target_world is None:
            return False

        base_yaw = self.get_second_stage_base_yaw_world()
        if base_yaw is None:
            base_yaw = self.reach_yaw_world
        if base_yaw is None:
            return False

        if self.get_locked_second_stage_base_yaw() is not None and self.arm_extension_axis != 'y':
            self.end_effector_plan = self.wrist_target_planner.make_plan(
                self.target_world,
                base_yaw,
            )
        elif self.arm_extension_axis == 'y':
            self.end_effector_plan = self.wrist_target_planner.make_side_axis_plan(
                self.target_world,
                base_yaw,
                self.arm_extension_sign,
            )
        else:
            self.end_effector_plan = self.wrist_target_planner.make_plan(
                self.target_world,
                base_yaw,
            )
        pt = self.end_effector_plan.base_point
        self.side_reach_base_world = pt
        self.get_logger().info(
            f'Side-reach base pose: x={pt.x:.3f}, y={pt.y:.3f}, '
            f'arm_extension={self.end_effector_plan.arm_extension * 100:.1f}cm, '
            f'base_yaw={math.degrees(self.end_effector_plan.base_yaw):.1f}deg, '
            f'contact_axis={self.end_effector_plan.contact_axis}'
            f'{self.end_effector_plan.contact_sign:+.0f}, '
            f'contact_yaw={math.degrees(self.end_effector_plan.contact_axis_yaw):.1f}deg, '
            f'wrist=({math.degrees(self.end_effector_plan.wrist_yaw):.1f}, '
            f'{math.degrees(self.end_effector_plan.wrist_pitch):.1f}, '
            f'{math.degrees(self.end_effector_plan.wrist_roll):.1f})deg.',
            throttle_duration_sec=1.0,
        )
        return True

    def arm_extension_distance(self, dx_base, dy_base):
        if self.side_reach_enabled and self.end_effector_plan is not None:
            if self.end_effector_plan.contact_axis == 'y':
                return self.end_effector_plan.contact_sign * dy_base
            return dx_base
        if self.arm_extension_axis == 'y':
            return self.arm_extension_sign * dy_base
        return dx_base

    def side_axis_arm_extension_from_distance(self, side_distance):
        if (
            self.end_effector_plan is not None
            and self.end_effector_plan.contact_axis == 'y'
        ):
            return max(0.0, side_distance - self.end_effector_plan.arm_extension)
        return side_distance

    def arm_lateral_error(self, dx_base, dy_base):
        if self.side_reach_enabled and self.end_effector_plan is not None:
            if self.end_effector_plan.contact_axis == 'y':
                return dx_base
            return dy_base
        if self.arm_extension_axis == 'y':
            return dx_base
        return dy_base

    def is_locked_side_axis_plan(self):
        return (
            self.get_locked_second_stage_base_yaw() is not None
            and self.end_effector_plan is not None
            and self.end_effector_plan.contact_axis == 'y'
        )

    def compute_gripper_target_error(self):
        if self.target_world is None:
            return None
        gripper = self.get_gripper_center_world()
        base_pose = self.lookup_base_pose()
        if gripper is None or base_pose is None:
            return None
        _, base_yaw = base_pose
        dx_world = self.target_world.x - gripper.x
        dy_world = self.target_world.y - gripper.y
        return self.rotate_world_delta_to_base(dx_world, dy_world, base_yaw)

    def get_wrist_contact_positions(self):
        if self.end_effector_plan is not None:
            return (
                self.end_effector_plan.wrist_yaw,
                self.end_effector_plan.wrist_pitch,
                self.end_effector_plan.wrist_roll,
            )
        return (
            self.wrist_target_planner.wrist_yaw,
            self.wrist_target_planner.wrist_pitch,
            self.wrist_target_planner.wrist_roll,
        )

    def get_wrist_initial_positions(self):
        return (
            self.wrist_initial_yaw,
            self.wrist_target_planner.wrist_pitch,
            self.wrist_target_planner.wrist_roll,
        )

    def dynamic_wrist_yaw_for_lateral(self, extension, lateral_error, nominal_yaw):
        forward = max(abs(extension), 0.05)
        correction = math.atan2(lateral_error, forward)
        correction = self.clamp(correction, -self.dynamic_wrist_yaw_limit, self.dynamic_wrist_yaw_limit)
        return self.wrap_pi(nominal_yaw + correction)

    def send_wrist_contact_pose(self):
        if self.wrist_contact_pose_sent:
            return True
        if self.wrist_sending or (self.wrist_goal_handle is not None and self.wrist_result is None):
            return False

        wrist_yaw, wrist_pitch, wrist_roll = self.get_wrist_contact_positions()
        self.clear_goal('wrist')
        self.get_logger().info(
            f'Setting wrist contact pose after first-stage base yaw alignment: '
            f'yaw={math.degrees(wrist_yaw):.1f}deg, '
            f'pitch={math.degrees(wrist_pitch):.1f}deg, roll={math.degrees(wrist_roll):.1f}deg.'
        )
        if self.send_joint_goal(
            ['joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll'],
            [wrist_yaw, wrist_pitch, wrist_roll],
            1.0,
            'wrist',
        ):
            return True
        return False

    def handle_wrist_contact_pose(self):
        self.stop_base()
        if self.wrist_contact_pose_sent:
            self.phase = 'waiting_second_point'
            self.get_logger().info('Wrist contact pose ready. Waiting for second, closer point selection.')
            return

        if self.wrist_goal_handle is None and not self.wrist_sending:
            self.send_wrist_contact_pose()
            return

        if self.wrist_result is None:
            return

        if self.wrist_result.status == 4:
            self.wrist_contact_pose_sent = True
            self.phase = 'waiting_second_point'
            self.get_logger().info('Wrist contact pose ready. Waiting for second, closer point selection.')
        else:
            self.get_logger().warn(
                f'Wrist contact pose failed with status {self.wrist_result.status}; retrying before second point.'
            )
            self.clear_goal('wrist')

    def is_non_head_motion_active(self):
        base_motion_active = (
            not self.head_tracking_during_base_motion
            and self.phase in ('move_approach', 'move_side_reach_pose', 'align_target', 'final_approach')
        )
        return (
            base_motion_active
            or self.phase in ('reset_wrist', 'raise_lift', 'prepare_second_point', 'extend_arm', 'final_push')
            or self.wrist_sending
            or self.lift_sending
            or self.arm_sending
            or (self.wrist_goal_handle is not None and self.wrist_result is None)
            or (self.lift_goal_handle is not None and self.lift_result is None)
            or (self.arm_goal_handle is not None and self.arm_result is None)
        )

    def handle_wrist_reset(self):
        if self.wrist_goal_handle is None and not self.wrist_sending:
            wrist_yaw, wrist_pitch, wrist_roll = self.get_wrist_initial_positions()
            names = ['joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll']
            positions = [wrist_yaw, wrist_pitch, wrist_roll]
            gripper_text = ''
            if self.close_gripper_on_start:
                names.append(self.gripper_joint_name)
                positions.append(self.gripper_closed_position)
                gripper_text = f', gripper={self.gripper_closed_position:.2f}rad'
            self.get_logger().info(
                f'Setting initial wrist pose: yaw={math.degrees(wrist_yaw):.1f}deg, '
                f'pitch={math.degrees(wrist_pitch):.1f}deg, roll={math.degrees(wrist_roll):.1f}deg'
                f'{gripper_text}.'
            )
            self.send_joint_goal(
                names,
                positions,
                1.5,
                'wrist',
            )
            return
        if self.wrist_result is None:
            return
        if self.wrist_result.status == 4:
            self.phase = 'raise_lift'
            self.get_logger().info('Wrist reset complete.')
        else:
            self.get_logger().warn(f'Wrist reset failed with status {self.wrist_result.status}; retrying.')
            self.wrist_goal_handle = None
            self.wrist_result = None

    def handle_lift_raise(self):
        if self.target_world is None:
            return
        if self.lift_goal_handle is None and not self.lift_sending:
            base_target = self.transform_point_to_base(self.target_world)
            if base_target is None:
                return
            lift_target, lift_debug = self.compute_lift_target(base_target)
            if self.two_stage_enabled and not self.second_target_locked and not self.wrist_contact_pose_sent:
                wrist_yaw, wrist_pitch, wrist_roll = self.get_wrist_initial_positions()
            else:
                wrist_yaw, wrist_pitch, wrist_roll = self.get_wrist_contact_positions()
            self.send_joint_goal(
                ['joint_lift', 'wrist_extension', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll'],
                [lift_target, 0.0, wrist_yaw, wrist_pitch, wrist_roll],
                2.0,
                'lift',
            )
            self.last_commanded_arm_pos = 0.0
            self.get_logger().info(
                f'[LiftHeight] base_z={base_target.z:.3f}m, offset={self.gripper_z_offset:.3f}m, '
                f'unclamped_lift={lift_debug["unclamped"]:.3f}m, '
                f'adaptive_drop={lift_debug["adaptive_drop"]:.3f}m, final_lift={lift_target:.3f}m.'
            )
            if lift_debug['guard_applied']:
                self.get_logger().info(
                    f'[LiftGuard:first_stage] dist_xy={lift_debug["dist_xy"]:.3f}m, '
                    f'drop={lift_debug["adaptive_drop"]:.3f}m, cap={self.first_lift_max_cap:.3f}m.'
                )
            self.get_logger().info(f'Raising lift to {lift_target:.3f}m and keeping arm retracted.')
            return
        if self.lift_result is None:
            return
        if self.lift_result.status == 4:
            self.phase = self.after_lift_phase
            if self.phase == 'move_approach':
                self.get_logger().info('Lift raise complete. Moving to direct approach point.')
            else:
                self.get_logger().info('Lift raise complete. Aligning to final target.')
        else:
            self.get_logger().warn(f'Lift goal failed with status {self.lift_result.status}; retrying.')
            self.lift_goal_handle = None
            self.lift_result = None

    def compute_lift_target(self, base_target):
        unclamped_lift = base_target.z - self.gripper_z_offset
        lift_target = self.clamp(unclamped_lift, self.lift_min, self.lift_max)
        adaptive_drop = 0.0
        dist_xy = float('nan')
        guard_applied = (
            self.first_lift_adaptive_enabled
            and self.two_stage_enabled
            and not self.second_target_locked
        )

        if guard_applied:
            dist_xy = math.hypot(base_target.x, base_target.y)
            near_d = self.first_lift_far_dist_near
            far_d = max(near_d + 1e-6, self.first_lift_far_dist_far)
            alpha = self.clamp((dist_xy - near_d) / (far_d - near_d), 0.0, 1.0)
            adaptive_drop = alpha * self.first_lift_max_drop
            lift_target = max(self.lift_min, lift_target - adaptive_drop)
            lift_target = min(lift_target, self.first_lift_max_cap)

        return lift_target, {
            'unclamped': unclamped_lift,
            'adaptive_drop': adaptive_drop,
            'dist_xy': dist_xy,
            'guard_applied': guard_applied,
        }

    def handle_arm_extend(self):
        if self.arm_goal_handle is None and not self.arm_sending:
            self.send_arm_extension_goal()
            return
        if self.arm_result is None:
            return
        if self.arm_result.status == 4:
            if self.confirm_gripper_contact():
                self.clear_goal('arm')
                if self.final_contact_push_enabled and self.final_contact_push_distance > 0.0:
                    self.phase = 'final_push'
                    self.get_logger().info(
                        f'Gripper contact point reached. Applying final push '
                        f'{self.final_contact_push_distance * 100:.1f}cm outward.'
                    )
                    return
                self.phase = 'done'
                self.stop_base()
                self.get_logger().info('Direct target motion complete: gripper contact point reached.')
            elif self.arm_contact_retry_count < self.arm_contact_retry_limit:
                self.arm_contact_retry_count += 1
                self.clear_goal('arm')
                self.final_min_dist_xy = float('inf')
                self.final_dist_increase_count = 0
                self.phase = 'final_approach'
                self.get_logger().warn(
                    f'Arm extension finished but gripper is not at target; retrying final approach '
                    f'({self.arm_contact_retry_count}/{self.arm_contact_retry_limit}).'
                )
            else:
                self.phase = 'failed'
                self.stop_base()
                self.get_logger().warn(
                    'Arm extension retry limit reached before gripper contact; motion failed.'
                )
        else:
            self.get_logger().warn(f'Arm goal failed with status {self.arm_result.status}.')
            self.phase = 'failed'
            self.stop_base()

    def handle_final_contact_push(self):
        if self.arm_goal_handle is None and not self.arm_sending:
            self.send_final_contact_push_goal()
            return
        if self.arm_result is None:
            return
        if self.arm_result.status in (4, 6):
            self.phase = 'done'
            self.stop_base()
            if self.arm_result.status == 6:
                self.get_logger().info(
                    'Final 2cm push stopped by guarded contact; treating push as complete.'
                )
            else:
                self.get_logger().info('Final 2cm push complete.')
            return
        self.get_logger().warn(f'Final contact push failed with status {self.arm_result.status}.')
        self.phase = 'failed'
        self.stop_base()

    def send_final_contact_push_goal(self):
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Joint trajectory action server is not ready.')
            return False

        current_arm = (
            self.current_arm_pos
            if self.current_arm_pos is not None
            else self.last_commanded_arm_pos
        )
        push_distance = max(0.0, self.final_contact_push_distance)
        arm_target = self.clamp(current_arm + push_distance, self.arm_min, self.arm_max)
        actual_push = arm_target - current_arm

        lift_pos = self.current_lift_pos
        if lift_pos is None and self.target_world is not None:
            base_target = self.transform_point_to_base(self.target_world)
            if base_target is None:
                return False
            lift_pos = self.clamp(base_target.z - self.gripper_z_offset, self.lift_min, self.lift_max)
        if lift_pos is None:
            self.get_logger().warn('Cannot apply final push because lift position is unknown.')
            return False

        wrist_yaw, wrist_pitch, wrist_roll = self.get_wrist_contact_positions()
        self.get_logger().info(
            f'Final contact push: current_arm={current_arm:.3f}m, '
            f'target_extension={arm_target:.3f}m, push={actual_push * 100:.1f}cm.'
        )
        if not self.send_joint_goal(
            ['joint_lift', 'wrist_extension', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll'],
            [lift_pos, arm_target, wrist_yaw, wrist_pitch, wrist_roll],
            1.0,
            'arm',
        ):
            return False
        self.last_commanded_arm_pos = arm_target
        return True

    def send_arm_extension_goal(self):
        if self.target_world is None:
            return False
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Joint trajectory action server is not ready.')
            return False

        base_pose = self.lookup_base_pose()
        if base_pose is None:
            return False

        locked_side_axis = False
        if self.side_reach_enabled and (self.reach_yaw_world is not None or self.get_locked_second_stage_base_yaw() is not None):
            err = self.compute_side_reach_pose_error()
            if err is None:
                return False
            along_base, lateral_error, yaw_error = err
            yaw_locked = self.get_locked_second_stage_base_yaw() is not None
            locked_side_axis = self.is_locked_side_axis_plan()
            if locked_side_axis:
                gripper_delta = self.compute_gripper_target_error()
                if gripper_delta is None:
                    return False
                along_base = self.arm_lateral_error(*gripper_delta)
                extension_delta = self.arm_extension_distance(*gripper_delta)
                lateral_error = self.arm_lateral_error(*gripper_delta)
                ignore_lateral = False
                lateral_reachable = True
            else:
                extension = self.end_effector_plan.arm_extension if self.end_effector_plan is not None else 0.0
                ignore_lateral = (
                    yaw_locked
                    and self.end_effector_plan is not None
                    and self.end_effector_plan.contact_axis == 'x'
                )
                lateral_reachable = abs(lateral_error) <= self.wrist_lateral_reach
            yaw_aligned = (
                abs(yaw_error) <= self.yaw_tolerance
                or (yaw_locked and not self.second_stage_yaw_micro_adjust_enabled)
            )
            along_tolerance = (
                self.side_axis_gripper_lateral_tolerance
                if locked_side_axis
                else self.final_y_tolerance
            )
            if (
                abs(along_base) > along_tolerance
                or not yaw_aligned
            ):
                self.get_logger().warn(
                    f'Base pose drift before arm extension: along={along_base * 100:.1f}cm, '
                    f'yaw_error={math.degrees(yaw_error):.1f}deg, yaw_locked={yaw_locked}; '
                    'returning to final approach.'
                )
                self.phase = 'final_approach'
                return False
        else:
            gripper = self.get_gripper_center_world()
            if gripper is None:
                return False
            _, base_yaw = base_pose
            dx_world = self.target_world.x - gripper.x
            dy_world = self.target_world.y - gripper.y
            dx_base, dy_base = self.rotate_world_delta_to_base(dx_world, dy_world, base_yaw)
            extension = self.arm_extension_distance(dx_base, dy_base)
            lateral_error = self.arm_lateral_error(dx_base, dy_base)

        lateral_tolerance = (
            self.side_axis_gripper_lateral_tolerance
            if locked_side_axis
            else self.final_y_tolerance
        )
        if not ('ignore_lateral' in locals() and ignore_lateral) and abs(lateral_error) > lateral_tolerance:
            self.get_logger().warn(
                f'Lateral error before arm extension is {lateral_error * 100:.1f}cm '
                f'(tolerance={lateral_tolerance * 100:.1f}cm); returning to final approach.'
            )
            self.phase = 'final_approach'
            return False
        if 'ignore_lateral' in locals() and ignore_lateral and not lateral_reachable:
            self.get_logger().warn(
                f'Lateral residual before locked-yaw arm extension is {lateral_error * 100:.1f}cm, '
                f'outside wrist compensation reach {self.wrist_lateral_reach * 100:.1f}cm; '
                'not extending because the gripper would miss along the panel tangent.'
            )
            self.phase = 'failed'
            return False
        if 'ignore_lateral' in locals() and ignore_lateral and abs(lateral_error) > self.final_y_tolerance:
            self.get_logger().warn(
                f'Lateral residual before locked-yaw arm extension is {lateral_error * 100:.1f}cm; '
                'using wrist yaw to compensate before arm extension.'
            )

        if 'extension_delta' in locals():
            current_arm = (
                self.current_arm_pos
                if self.current_arm_pos is not None
                else self.last_commanded_arm_pos
            )
            contact_overshoot = self.effective_arm_contact_margin()
            arm_target = self.clamp(
                current_arm + extension_delta + contact_overshoot,
                self.arm_min,
                self.arm_max,
            )
            extension = arm_target - current_arm
        elif extension < self.arm_min or extension > self.arm_max:
            self.get_logger().warn(
                f'Arm extension distance {extension * 100:.1f}cm is outside range '
                f'[{self.arm_min * 100:.1f}, {self.arm_max * 100:.1f}]cm; returning to final approach.'
            )
            self.phase = 'final_approach'
            return False

        lift_pos = self.current_lift_pos
        if lift_pos is None:
            base_target = self.transform_point_to_base(self.target_world)
            if base_target is None:
                return False
            lift_pos = self.clamp(base_target.z - self.gripper_z_offset, self.lift_min, self.lift_max)

        if 'arm_target' not in locals():
            contact_overshoot = self.effective_arm_contact_margin()
            arm_target = self.clamp(extension + contact_overshoot, self.arm_min, self.arm_max)
        else:
            contact_overshoot = self.effective_arm_contact_margin()
        wrist_yaw, wrist_pitch, wrist_roll = self.get_wrist_contact_positions()
        if (
            self.dynamic_wrist_yaw_enabled
            and 'ignore_lateral' in locals()
            and ignore_lateral
            and abs(lateral_error) > self.final_y_tolerance
        ):
            wrist_yaw = self.dynamic_wrist_yaw_for_lateral(extension, lateral_error, wrist_yaw)
            self.get_logger().info(
                f'Dynamic wrist yaw for tangent residual: lateral={lateral_error * 100:.1f}cm, '
                f'extension={extension * 100:.1f}cm, wrist_yaw={math.degrees(wrist_yaw):.1f}deg.'
            )
        remaining_to_target = extension_delta if 'extension_delta' in locals() else extension
        self.get_logger().info(
            f'Extending arm for contact: target_extension={arm_target:.3f}m, '
            f'current_arm={self.current_arm_pos if self.current_arm_pos is not None else float("nan"):.3f}m, '
            f'remaining_along_contact={remaining_to_target * 100:.1f}cm, '
            f'overshoot={contact_overshoot * 100:.1f}cm, '
            f'command_delta={extension * 100:.1f}cm, lateral={lateral_error * 100:.1f}cm.'
        )
        self.send_joint_goal(
            ['joint_lift', 'wrist_extension', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll'],
            [lift_pos, arm_target, wrist_yaw, wrist_pitch, wrist_roll],
            2.0,
            'arm',
        )
        self.last_commanded_arm_pos = arm_target
        return True

    def confirm_gripper_contact(self):
        if self.side_reach_enabled and self.end_effector_plan is not None:
            err = self.compute_side_reach_pose_error()
            if err is None:
                return False
            along_base, lateral_base, yaw_error = err
            yaw_locked = self.get_locked_second_stage_base_yaw() is not None
            ignore_lateral = (
                yaw_locked
                and self.end_effector_plan is not None
                and self.end_effector_plan.contact_axis == 'x'
            )
            locked_side_axis = self.is_locked_side_axis_plan()
            if locked_side_axis:
                gripper_delta = self.compute_gripper_target_error()
                if gripper_delta is None:
                    return False
                extension_error = self.arm_extension_distance(*gripper_delta)
                lateral_error = self.arm_lateral_error(*gripper_delta)
                yaw_aligned = (
                    abs(yaw_error) <= self.yaw_tolerance
                    or (yaw_locked and not self.second_stage_yaw_micro_adjust_enabled)
                )
                reached = (
                    abs(extension_error) <= self.arm_contact_tolerance
                    and abs(lateral_error) <= self.side_axis_gripper_lateral_tolerance
                    and yaw_aligned
                )
                self.get_logger().info(
                    f'Side-axis contact check after arm extension: '
                    f'extension_error={extension_error * 100:.1f}cm, '
                    f'lateral_error={lateral_error * 100:.1f}cm, '
                    f'arm={self.current_arm_pos if self.current_arm_pos is not None else float("nan"):.3f}m, '
                    f'yaw_error={math.degrees(yaw_error):.1f}deg, yaw_locked={yaw_locked}, '
                    f'reached={reached}.'
                )
                return reached
            lateral_reachable = abs(lateral_base) <= self.wrist_lateral_reach
            arm_target = self.clamp(
                self.end_effector_plan.arm_extension + self.effective_arm_contact_margin(),
                self.arm_min,
                self.arm_max,
            )
            arm_error = 0.0
            arm_known = self.current_arm_pos is not None
            if arm_known:
                arm_error = abs(self.current_arm_pos - arm_target)
            yaw_aligned = (
                abs(yaw_error) <= self.yaw_tolerance
                or (yaw_locked and not self.second_stage_yaw_micro_adjust_enabled)
            )
            reached = (
                abs(along_base) <= max(self.final_y_tolerance, self.arm_contact_tolerance)
                and (
                    (ignore_lateral and lateral_reachable)
                    or abs(lateral_base) <= max(self.final_y_tolerance, self.arm_contact_tolerance)
                )
                and yaw_aligned
                and (not arm_known or arm_error <= 0.06)
            )
            self.get_logger().info(
                f'Side-axis contact check after arm extension: '
                f'base_along={along_base * 100:.1f}cm, '
                f'base_lateral={lateral_base * 100:.1f}cm, '
                f'target_extension={arm_target:.3f}m, '
                f'arm={self.current_arm_pos if self.current_arm_pos is not None else float("nan"):.3f}m, '
                f'yaw_error={math.degrees(yaw_error):.1f}deg, yaw_locked={yaw_locked}, '
                f'lateral_reachable={lateral_reachable}, reached={reached}.'
            )
            return reached

        gripper = self.get_gripper_center_world()
        base_pose = self.lookup_base_pose()
        if gripper is None or base_pose is None or self.target_world is None:
            return False

        _, base_yaw = base_pose
        dx_world = self.target_world.x - gripper.x
        dy_world = self.target_world.y - gripper.y
        dist_xy = math.hypot(dx_world, dy_world)
        dx_base, dy_base = self.rotate_world_delta_to_base(dx_world, dy_world, base_yaw)
        extension = self.arm_extension_distance(dx_base, dy_base)
        lateral_error = self.arm_lateral_error(dx_base, dy_base)
        reached = (
            dist_xy <= self.arm_contact_tolerance
            or (
                extension <= self.arm_contact_margin
                and abs(lateral_error) <= max(self.final_y_tolerance, self.arm_contact_tolerance)
            )
        )
        self.get_logger().info(
            f'Contact check after arm extension: xy={dist_xy * 100:.1f}cm, '
            f'extension={extension * 100:.1f}cm, lateral={lateral_error * 100:.1f}cm, '
            f'arm={self.current_arm_pos if self.current_arm_pos is not None else float("nan"):.3f}m, '
            f'reached={reached}.'
        )
        return reached

    def send_joint_goal(self, names, positions, seconds, kind):
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Joint trajectory action server is not ready.')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(seconds=seconds).to_msg()
        goal.trajectory.points = [point]

        if kind == 'wrist':
            self.wrist_sending = True
        elif kind == 'lift':
            self.lift_sending = True
        elif kind == 'arm':
            self.arm_sending = True
        else:
            return False

        future = self.trajectory_client.send_goal_async(goal)
        future.add_done_callback(lambda fut: self.joint_goal_response_callback(fut, kind))
        return True

    def effective_arm_contact_margin(self):
        if not self.arm_contact_overshoot_enabled:
            return 0.0
        return max(0.0, self.arm_contact_margin)

    def joint_goal_response_callback(self, future, kind):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'{kind} goal response failed: {exc}')
            self.clear_sending(kind)
            return

        if not goal_handle.accepted:
            self.get_logger().warn(f'{kind} goal rejected.')
            self.clear_goal(kind)
            return

        self.set_goal_handle(kind, goal_handle)
        self.clear_sending(kind)
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut: self.joint_result_callback(fut, kind))

    def joint_result_callback(self, future, kind):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warn(f'{kind} result failed: {exc}')
            self.clear_goal(kind)
            return
        if kind == 'wrist':
            self.wrist_result = result
            self.wrist_sending = False
        elif kind == 'lift':
            self.lift_result = result
            self.lift_sending = False
        elif kind == 'arm':
            self.arm_result = result
            self.arm_sending = False

    def clear_sending(self, kind):
        if kind == 'wrist':
            self.wrist_sending = False
        elif kind == 'lift':
            self.lift_sending = False
        elif kind == 'arm':
            self.arm_sending = False

    def clear_goal(self, kind):
        if kind == 'wrist':
            self.wrist_goal_handle = None
            self.wrist_result = None
            self.wrist_sending = False
        elif kind == 'lift':
            self.lift_goal_handle = None
            self.lift_result = None
            self.lift_sending = False
        elif kind == 'arm':
            self.arm_goal_handle = None
            self.arm_result = None
            self.arm_sending = False

    def set_goal_handle(self, kind, goal_handle):
        if kind == 'wrist':
            self.wrist_goal_handle = goal_handle
        elif kind == 'lift':
            self.lift_goal_handle = goal_handle
        elif kind == 'arm':
            self.arm_goal_handle = goal_handle

    def lookup_base_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except Exception as exc:
            self.get_logger().warn(f'Failed to lookup base pose: {exc}', throttle_duration_sec=2.0)
            return None
        pos = tf.transform.translation
        q = tf.transform.rotation
        return pos, self.quat_to_yaw(q.x, q.y, q.z, q.w)

    def transform_point_to_base(self, point_world):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.world_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            stamped = PointStamped()
            stamped.header.frame_id = self.world_frame
            stamped.point = point_world
            return do_transform_point(stamped, tf).point
        except Exception as exc:
            self.get_logger().warn(f'Failed to transform point to {self.base_frame}: {exc}', throttle_duration_sec=2.0)
            return None

    def get_gripper_center_world(self):
        try:
            left = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.left_fingertip_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            right = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.right_fingertip_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            center = Point()
            center.x = (left.transform.translation.x + right.transform.translation.x) / 2.0
            center.y = (left.transform.translation.y + right.transform.translation.y) / 2.0
            center.z = (left.transform.translation.z + right.transform.translation.z) / 2.0

            base_pose = self.lookup_base_pose()
            if base_pose is not None:
                _, yaw = base_pose
                center.x -= self.gripper_center_back_offset * math.cos(yaw)
                center.y -= self.gripper_center_back_offset * math.sin(yaw)
            return center
        except Exception as exc:
            self.get_logger().warn(f'Failed to lookup gripper center: {exc}', throttle_duration_sec=2.0)
            return None

    def rotate_world_delta_to_base(self, dx, dy, base_yaw):
        cos_yaw = math.cos(-base_yaw)
        sin_yaw = math.sin(-base_yaw)
        return (
            dx * cos_yaw - dy * sin_yaw,
            dx * sin_yaw + dy * cos_yaw,
        )

    def stop_base(self):
        try:
            if rclpy.ok():
                self.cmd_vel_pub.publish(Twist())
        except Exception:
            pass

    @staticmethod
    def quat_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def format_optional_yaw(angle):
        if angle is None:
            return 'unset'
        return f'{math.degrees(angle):.1f}deg'

    @staticmethod
    def clamp(value, lo, hi):
        return max(lo, min(hi, value))


def main(args=None):
    rclpy.init(args=args)
    node = WhitePointDirectMotion()
    try:
        rclpy.spin(node)
    finally:
        node.stop_base()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

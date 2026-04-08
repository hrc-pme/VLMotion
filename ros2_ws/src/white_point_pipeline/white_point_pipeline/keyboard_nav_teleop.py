#!/usr/bin/env python3

import os
import select
import sys
import termios
import tty

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


class NonBlockingKeyboard:
    """Minimal non-blocking keyboard reader for terminal teleop."""

    def __init__(self):
        self.fd = None
        self.old_settings = None
        self._owns_fd = False

        # In ros2 launch subprocesses, stdin is often not a TTY. Fallback to /dev/tty.
        try:
            stdin_fd = sys.stdin.fileno()
            if os.isatty(stdin_fd):
                self.fd = stdin_fd
                self.old_settings = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
        except Exception:
            pass

        if self.fd is None:
            self.fd = os.open('/dev/tty', os.O_RDONLY)
            self._owns_fd = True
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def restore(self):
        if self.fd is None or self.old_settings is None:
            return
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        if self._owns_fd:
            os.close(self.fd)
            self.fd = None

    def _read_char(self):
        try:
            return os.read(self.fd, 1).decode('utf-8', errors='ignore')
        except Exception:
            return None

    def get_key(self):
        if self.fd is None:
            return None
        rlist, _, _ = select.select([self.fd], [], [], 0.0)
        if self.fd not in rlist:
            return None

        c0 = self._read_char()
        if c0 is None:
            return None

        if c0 != '\x1b':
            return c0

        # Handle arrow keys: ESC [ A/B/C/D
        seq = c0
        rlist, _, _ = select.select([self.fd], [], [], 0.001)
        if rlist:
            c1 = self._read_char()
            if c1 is not None:
                seq += c1
        rlist, _, _ = select.select([self.fd], [], [], 0.001)
        if rlist:
            c2 = self._read_char()
            if c2 is not None:
                seq += c2
        return seq


class KeyboardNavTeleop(Node):

    def __init__(self):
        super().__init__('keyboard_nav_teleop')

        self.declare_parameter('cmd_vel_topic', '/stretch/cmd_vel')
        self.declare_parameter('block_teleop_when_autonomy_active', True)
        self.declare_parameter('autonomy_phase_topic', '/white_point_selection_phase')
        self.declare_parameter('autonomy_idle_phase', 'select_first_point')
        self.declare_parameter('allow_manual_joint_while_autonomy_active', True)
        self.declare_parameter(
            'manual_joint_allowed_phases_csv',
            'moving_to_approach,moving_to_target,waiting_second_point',
        )

        self.declare_parameter('small_linear_speed', 0.05)
        self.declare_parameter('medium_linear_speed', 0.10)
        self.declare_parameter('big_linear_speed', 0.16)
        self.declare_parameter('small_angular_speed', 0.25)
        self.declare_parameter('medium_angular_speed', 0.45)
        self.declare_parameter('big_angular_speed', 0.70)
        self.declare_parameter('command_hold_sec', 0.20)
        self.declare_parameter('small_joint_step', 0.01)
        self.declare_parameter('medium_joint_step', 0.03)
        self.declare_parameter('big_joint_step', 0.05)
        self.declare_parameter('small_gripper_step_rad', 0.05)
        self.declare_parameter('medium_gripper_step_rad', 0.10)
        self.declare_parameter('big_gripper_step_rad', 0.20)

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.block_teleop_when_autonomy_active = self.get_parameter(
            'block_teleop_when_autonomy_active'
        ).value
        self.autonomy_phase_topic = self.get_parameter('autonomy_phase_topic').value
        self.autonomy_idle_phase = self.get_parameter('autonomy_idle_phase').value
        self.allow_manual_joint_while_autonomy_active = self.get_parameter(
            'allow_manual_joint_while_autonomy_active'
        ).value
        phases_csv = self.get_parameter('manual_joint_allowed_phases_csv').value
        self.manual_joint_allowed_phases = set(
            p.strip() for p in str(phases_csv).split(',') if p.strip()
        )

        self.linear_speed_cfg = {
            'small': float(self.get_parameter('small_linear_speed').value),
            'medium': float(self.get_parameter('medium_linear_speed').value),
            'big': float(self.get_parameter('big_linear_speed').value),
        }
        self.angular_speed_cfg = {
            'small': float(self.get_parameter('small_angular_speed').value),
            'medium': float(self.get_parameter('medium_angular_speed').value),
            'big': float(self.get_parameter('big_angular_speed').value),
        }
        self.command_hold_sec = float(self.get_parameter('command_hold_sec').value)
        self.joint_step_cfg = {
            'small': float(self.get_parameter('small_joint_step').value),
            'medium': float(self.get_parameter('medium_joint_step').value),
            'big': float(self.get_parameter('big_joint_step').value),
        }
        self.gripper_step_cfg = {
            'small': float(self.get_parameter('small_gripper_step_rad').value),
            'medium': float(self.get_parameter('medium_gripper_step_rad').value),
            'big': float(self.get_parameter('big_gripper_step_rad').value),
        }

        self.step_size = 'medium'
        self.autonomy_phase = self.autonomy_idle_phase
        self.autonomy_active = False
        self._last_blocked_log_ns = 0

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.joint_state = JointState()
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/stretch/joint_states',
            self.joint_states_callback,
            1,
        )

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/stretch_controller/follow_joint_trajectory',
        )

        self.autonomy_phase_sub = self.create_subscription(
            String,
            self.autonomy_phase_topic,
            self.autonomy_phase_callback,
            10,
        )

        self.keyboard = NonBlockingKeyboard()
        self.active_twist = Twist()
        self.active_until_ns = 0

        self.get_logger().info(
            'Navigation keyboard teleop started. '
            f'cmd_vel={self.cmd_vel_topic}, phase_topic={self.autonomy_phase_topic}, '
            f'idle_phase={self.autonomy_idle_phase}, lock={self.block_teleop_when_autonomy_active}, '
            f'allow_joint_when_active={self.allow_manual_joint_while_autonomy_active}, '
            f'joint_allowed_phases={sorted(self.manual_joint_allowed_phases)}'
        )
        if not self.trajectory_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn(
                'Joint trajectory action server not ready. '
                'Base control works; lift/arm keys will be unavailable until server is up.'
            )
        self.print_help()

        self.timer = self.create_timer(0.02, self.control_loop)

    def print_help(self):
        print('---------- NAV KEYBOARD TELEOP -----------')
        print(' 1 / Left  : base forward')
        print(' 3 / Right : base back')
        print(' 7 / Home  : rotate left')
        print(' 9 / PgUp  : rotate right')
        print(' 8 / Up    : lift up')
        print(' 2 / Down  : lift down')
        print(' 6         : arm out')
        print(' 4         : arm in')
        print(' 5         : gripper close')
        print(' 0         : gripper open')
        print(' b / m / s : step big / medium / small')
        print(' q         : quit')
        print('------------------------------------------')

    def joint_states_callback(self, msg):
        self.joint_state = msg

    def autonomy_phase_callback(self, msg):
        self.autonomy_phase = msg.data
        prev = self.autonomy_active
        self.autonomy_active = (self.autonomy_phase != self.autonomy_idle_phase)
        if prev != self.autonomy_active:
            state = 'ACTIVE' if self.autonomy_active else 'IDLE'
            self.get_logger().info(
                f'Autonomy phase={self.autonomy_phase}. Keyboard lock -> {state}'
            )

    def is_blocked(self):
        return self.block_teleop_when_autonomy_active and self.autonomy_active

    def is_manual_joint_allowed_during_autonomy(self):
        if not self.is_blocked():
            return True
        if not self.allow_manual_joint_while_autonomy_active:
            return False
        return self.autonomy_phase in self.manual_joint_allowed_phases

    def log_blocked_once(self):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_blocked_log_ns > int(2e9):
            self.get_logger().warn(
                'Keyboard input ignored because autonomy is active. '
                'Press q to quit or wait for idle phase.'
            )
            self._last_blocked_log_ns = now_ns

    def stop_base(self):
        self.active_twist = Twist()
        self.active_until_ns = 0
        self.cmd_vel_pub.publish(Twist())

    def make_twist(self, lin_x=0.0, ang_z=0.0):
        t = Twist()
        t.linear.x = float(lin_x)
        t.angular.z = float(ang_z)
        return t

    def send_joint_delta(self, joint_name, delta):
        if not self.trajectory_client.server_is_ready():
            self.get_logger().warn('Joint server not ready, ignoring lift/arm command.')
            return

        if not self.joint_state.name:
            self.get_logger().warn('No joint state yet, ignoring lift/arm command.')
            return

        if joint_name not in self.joint_state.name:
            self.get_logger().warn(f'Joint {joint_name} not found in joint states.')
            return

        idx = self.joint_state.name.index(joint_name)
        current = self.joint_state.position[idx]
        target = current + float(delta)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [joint_name]
        point = JointTrajectoryPoint()
        point.positions = [target]
        point.time_from_start = Duration(seconds=0.6).to_msg()
        goal.trajectory.points = [point]
        self.trajectory_client.send_goal_async(goal)

    def process_key(self, key):
        if key in ('q', 'Q'):
            self.get_logger().info('keyboard_nav_teleop exiting...')
            self.stop_base()
            raise KeyboardInterrupt

        is_joint_key = key in ('8', '\x1b[A', '2', '\x1b[B', '6', '4', '5', '0')

        if self.is_blocked() and not (is_joint_key and self.is_manual_joint_allowed_during_autonomy()):
            self.log_blocked_once()
            return

        if key in ('s', 'S'):
            self.step_size = 'small'
            self.get_logger().info('Step size -> SMALL')
            return
        if key in ('m', 'M'):
            self.step_size = 'medium'
            self.get_logger().info('Step size -> MEDIUM')
            return
        if key in ('b', 'B'):
            self.step_size = 'big'
            self.get_logger().info('Step size -> BIG')
            return

        lin = self.linear_speed_cfg[self.step_size]
        ang = self.angular_speed_cfg[self.step_size]
        joint_step = self.joint_step_cfg[self.step_size]
        gripper_step = self.gripper_step_cfg[self.step_size]

        if key in ('8', '\x1b[A'):
            self.send_joint_delta('joint_lift', joint_step)
            return
        if key in ('2', '\x1b[B'):
            self.send_joint_delta('joint_lift', -joint_step)
            return
        if key in ('6',):
            self.send_joint_delta('wrist_extension', joint_step)
            return
        if key in ('4',):
            self.send_joint_delta('wrist_extension', -joint_step)
            return
        if key in ('5',):
            self.send_joint_delta('joint_gripper_finger_left', -gripper_step)
            return
        if key in ('0',):
            self.send_joint_delta('joint_gripper_finger_left', gripper_step)
            return

        cmd = None
        if key in ('1', '\x1b[D'):
            cmd = self.make_twist(lin_x=lin)
        elif key in ('3', '\x1b[C'):
            cmd = self.make_twist(lin_x=-lin)
        elif key in ('7', '\x1b[H'):
            cmd = self.make_twist(ang_z=ang)
        elif key in ('9', '\x1b[5'):
            cmd = self.make_twist(ang_z=-ang)

        if cmd is not None:
            now_ns = self.get_clock().now().nanoseconds
            self.active_twist = cmd
            self.active_until_ns = now_ns + int(self.command_hold_sec * 1e9)
            self.cmd_vel_pub.publish(self.active_twist)

    def control_loop(self):
        key = self.keyboard.get_key()
        if key is not None:
            self.process_key(key)

        now_ns = self.get_clock().now().nanoseconds

        # If autonomy starts while moving, immediately stop base.
        if self.is_blocked() and self.active_until_ns > 0:
            self.stop_base()
            return

        if self.active_until_ns > 0 and now_ns < self.active_until_ns:
            self.cmd_vel_pub.publish(self.active_twist)
        elif self.active_until_ns > 0 and now_ns >= self.active_until_ns:
            self.stop_base()

    def shutdown(self):
        self.stop_base()
        self.keyboard.restore()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardNavTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

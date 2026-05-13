#!/usr/bin/env python3
import math

from geometry_msgs.msg import Twist


class SideReachPlanner:
    """Final approach policy that leaves distance for arm extension."""

    def __init__(
        self,
        *,
        enabled=True,
        desired_arm_extension=0.20,
        min_arm_extension=0.08,
        x_tolerance=0.025,
        y_tolerance=0.035,
        final_xy_tolerance=0.035,
        yaw_tolerance_deg=8.0,
        arm_extension_axis='y',
        arm_extension_sign=-1.0,
        max_arm_extension=0.50,
        max_linear_speed=0.08,
        max_angular_speed=0.25,
        k_linear=0.45,
        k_angular=1.1,
    ):
        self.enabled = enabled
        self.desired_arm_extension = desired_arm_extension
        self.min_arm_extension = min_arm_extension
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
        self.final_xy_tolerance = final_xy_tolerance
        self.yaw_tolerance = math.radians(yaw_tolerance_deg)
        self.arm_extension_axis = arm_extension_axis
        self.arm_extension_sign = 1.0 if arm_extension_sign >= 0.0 else -1.0
        self.max_arm_extension = max_arm_extension
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.k_linear = k_linear
        self.k_angular = k_angular

    def make_command(self, dx_base, dy_base, dist_xy, yaw_error, min_dist_xy, dist_increase_count):
        if not self.enabled:
            return self._legacy_command(dx_base, dy_base, dist_xy, yaw_error, min_dist_xy, dist_increase_count)

        if self.arm_extension_axis == 'y':
            return self._side_arm_command(dx_base, dy_base, yaw_error)

        desired_x = max(self.min_arm_extension, self.desired_arm_extension)
        x_error = dx_base - desired_x

        ready_to_extend = (
            abs(x_error) <= self.x_tolerance
            and abs(dy_base) <= self.y_tolerance
            and abs(yaw_error) <= self.yaw_tolerance
        )
        if ready_to_extend:
            return True, Twist()

        twist = Twist()
        if abs(yaw_error) > math.radians(25.0) and dist_xy > desired_x:
            twist.angular.z = self.clamp(self.k_angular * yaw_error, -0.35, 0.35)
            return False, twist

        if abs(dy_base) > self.y_tolerance:
            twist.angular.z = self.clamp(self.k_angular * yaw_error + 0.8 * dy_base, -0.30, 0.30)

        if x_error > self.x_tolerance:
            speed = min(self.max_linear_speed, self.k_linear * x_error)
            heading_scale = max(0.2, math.cos(min(abs(yaw_error), math.radians(50.0))))
            twist.linear.x = speed * heading_scale
            twist.angular.z = self.clamp(
                twist.angular.z + self.k_angular * yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed,
            )
        elif x_error < -self.x_tolerance:
            twist.linear.x = max(-0.04, self.k_linear * x_error)
            twist.angular.z = self.clamp(twist.angular.z, -0.20, 0.20)

        return False, twist

    def _side_arm_command(self, dx_base, dy_base, yaw_error):
        extension = self.arm_extension_sign * dy_base
        lateral_error = dx_base
        extension_reachable = self.min_arm_extension <= extension <= self.max_arm_extension
        lateral_aligned = abs(lateral_error) <= self.y_tolerance
        yaw_aligned = abs(yaw_error) <= self.yaw_tolerance

        if extension_reachable and lateral_aligned and yaw_aligned:
            return True, Twist()

        twist = Twist()
        if not yaw_aligned:
            twist.angular.z = self.clamp(
                self.k_angular * yaw_error,
                -self.max_angular_speed,
                self.max_angular_speed,
            )

        if abs(lateral_error) > self.y_tolerance:
            speed = min(self.max_linear_speed, self.k_linear * abs(lateral_error))
            twist.linear.x = math.copysign(speed, lateral_error)

        return False, twist

    def _legacy_command(self, dx_base, dy_base, dist_xy, yaw_error, min_dist_xy, dist_increase_count):
        reached = (
            dist_xy <= self.final_xy_tolerance
            or (
                dx_base <= 0.0
                and abs(dy_base) <= max(self.y_tolerance, 0.05)
                and min_dist_xy <= 0.08
            )
            or dist_increase_count >= 4
        )
        if reached:
            return True, Twist()

        twist = Twist()
        if abs(yaw_error) > math.radians(25.0) and dist_xy > 0.08:
            twist.angular.z = self.clamp(self.k_angular * yaw_error, -0.35, 0.35)
        else:
            twist.linear.x = min(self.max_linear_speed, self.k_linear * max(0.0, dx_base))
            twist.angular.z = self.clamp(self.k_angular * yaw_error, -0.25, 0.25)
            if dx_base <= 0.0:
                twist.angular.z = 0.0
            if abs(dy_base) > self.y_tolerance:
                twist.angular.z = self.clamp(twist.angular.z + 0.8 * dy_base, -0.30, 0.30)
        return False, twist

    @staticmethod
    def clamp(value, lo, hi):
        return max(lo, min(hi, value))

#!/usr/bin/env python3
import math
from dataclasses import dataclass

from geometry_msgs.msg import Point


@dataclass
class EndEffectorPlan:
    base_point: Point
    base_yaw: float
    contact_axis_yaw: float
    contact_axis: str
    contact_sign: float
    arm_extension: float
    wrist_yaw: float
    wrist_pitch: float
    wrist_roll: float


class WristTargetPlanner:
    """Plan Stretch final contact pose using the real arm extension axis.

    Stretch's telescoping arm extends along base +X. The wrist/gripper can yaw,
    pitch, and roll, but it does not make wrist_extension move along base Y.
    This planner keeps the final base pose consistent with that kinematic fact.
    """

    def __init__(
        self,
        *,
        desired_arm_extension=0.20,
        min_arm_extension=0.08,
        max_arm_extension=0.50,
        wrist_yaw=0.0,
        wrist_pitch=0.0,
        wrist_roll=0.0,
    ):
        self.desired_arm_extension = desired_arm_extension
        self.min_arm_extension = min_arm_extension
        self.max_arm_extension = max_arm_extension
        self.wrist_yaw = wrist_yaw
        self.wrist_pitch = wrist_pitch
        self.wrist_roll = wrist_roll

    def make_plan(self, target_world, arm_world_yaw, arm_extension=None):
        requested_extension = (
            self.desired_arm_extension
            if arm_extension is None
            else arm_extension
        )
        extension = self.clamp(
            requested_extension,
            self.min_arm_extension,
            self.max_arm_extension,
        )

        pt = Point()
        pt.x = target_world.x - extension * math.cos(arm_world_yaw)
        pt.y = target_world.y - extension * math.sin(arm_world_yaw)
        pt.z = target_world.z

        return EndEffectorPlan(
            base_point=pt,
            base_yaw=self.wrap_pi(arm_world_yaw),
            contact_axis_yaw=self.wrap_pi(arm_world_yaw),
            contact_axis='x',
            contact_sign=1.0,
            arm_extension=extension,
            wrist_yaw=self.wrist_yaw,
            wrist_pitch=self.wrist_pitch,
            wrist_roll=self.wrist_roll,
        )

    def make_side_axis_plan(
        self,
        target_world,
        base_yaw,
        side_sign,
        arm_extension=None,
    ):
        requested_extension = (
            self.desired_arm_extension
            if arm_extension is None
            else arm_extension
        )
        extension = self.clamp(
            requested_extension,
            self.min_arm_extension,
            self.max_arm_extension,
        )
        sign = 1.0 if side_sign >= 0.0 else -1.0
        contact_axis_yaw = self.wrap_pi(base_yaw + sign * math.pi / 2.0)

        pt = Point()
        pt.x = target_world.x - extension * math.cos(contact_axis_yaw)
        pt.y = target_world.y - extension * math.sin(contact_axis_yaw)
        pt.z = target_world.z

        return EndEffectorPlan(
            base_point=pt,
            base_yaw=self.wrap_pi(base_yaw),
            contact_axis_yaw=contact_axis_yaw,
            contact_axis='y',
            contact_sign=sign,
            arm_extension=extension,
            wrist_yaw=self.wrist_yaw,
            wrist_pitch=self.wrist_pitch,
            wrist_roll=self.wrist_roll,
        )

    @staticmethod
    def arm_yaw_from_side_geometry(base_reach_yaw, arm_extension_sign):
        """Convert the older side-reach tangent yaw into true arm +X yaw."""
        sign = 1.0 if arm_extension_sign >= 0.0 else -1.0
        return WristTargetPlanner.wrap_pi(base_reach_yaw + sign * math.pi / 2.0)

    @staticmethod
    def clamp(value, lo, hi):
        return max(lo, min(hi, value))

    @staticmethod
    def wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

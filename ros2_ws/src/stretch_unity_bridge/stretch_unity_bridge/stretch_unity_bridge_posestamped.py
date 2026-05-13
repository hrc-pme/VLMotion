#!/usr/bin/env python3
import time
from dataclasses import dataclass

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, Twist
from hello_helpers.hello_misc import HelloNode
from rclpy.duration import Duration
from std_msgs.msg import Float64
from trajectory_msgs.msg import JointTrajectoryPoint


@dataclass
class Commands:
    head_pose: PoseStamped
    lift_arm: PoseStamped
    wrist: PoseStamped
    gripper: PoseStamped
    vel: Twist

    def ready(self):
        return all([self.head_pose, self.lift_arm, self.wrist, self.gripper, self.vel])

    def set_from_joint_states(self, joint_state):
        self.head_pose.pose.position.x = joint_state.position[joint_state.name.index("joint_head_pan")]
        self.head_pose.pose.position.y = joint_state.position[joint_state.name.index("joint_head_tilt")]
        self.lift_arm.pose.position.x = joint_state.position[joint_state.name.index("joint_lift")]
        self.lift_arm.pose.position.y = joint_state.position[joint_state.name.index("wrist_extension")]
        self.wrist.pose.position.x = joint_state.position[joint_state.name.index("joint_wrist_yaw")]
        self.wrist.pose.position.y = joint_state.position[joint_state.name.index("joint_wrist_pitch")]
        self.wrist.pose.position.z = joint_state.position[joint_state.name.index("joint_wrist_roll")]
        self.gripper.data = joint_state.position[joint_state.name.index("joint_gripper_finger_left")]
        self.vel.linear.x = 0.0
        self.vel.angular.z = 0.0

    def constrain_position(self, position, min_val, max_val):
        """Clamp the position to be within the specified range."""
        return max(min(position, max_val), min_val)

    def constrain_all_positions(self):
        """Apply constraints to all positions in the commands."""
        self.head_pose.pose.position.x = self.constrain_position(self.head_pose.pose.position.x, -1.5, 1.5)
        self.head_pose.pose.position.y = self.constrain_position(self.head_pose.pose.position.y, -1.0, 1.0)
        self.lift_arm.pose.position.x = self.constrain_position(self.lift_arm.pose.position.x, 0.0, 1.0)
        self.lift_arm.pose.position.y = self.constrain_position(self.lift_arm.pose.position.y, 0.0, 1.0)
        self.wrist.pose.position.x = self.constrain_position(self.wrist.pose.position.x, -1.21, 4.47)
        self.wrist.pose.position.y = self.constrain_position(self.wrist.pose.position.y, -1.0, 1.0)
        self.wrist.pose.position.z = self.constrain_position(self.wrist.pose.position.z, -1.0, 1.0)
        self.gripper.data = self.constrain_position(self.gripper.data, -0.36, 0.64)


class MultiPointCommand(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)
        HelloNode.main(self, "stretch_unity_bridge", "stretch_unity_bridge", wait_for_first_pointcloud=False)
        self.commands = Commands(PoseStamped(), PoseStamped(), PoseStamped(), Float64(), Twist())
        self.create_subscribers()
        while not self.joint_state.position:
            self.get_logger().info("Waiting for joint states message to arrive")
            time.sleep(0.1)
        self.commands.set_from_joint_states(self.joint_state)
        self.create_timer(1.0, self.timer_callback)

    def create_subscribers(self):
        self.cmd_head_pose_subscriber = self.create_subscription(PoseStamped, "/stretch/joint_states_control", self.cmd_head_pose_callback, 10)
        self.cmd_lift_arm_subscriber = self.create_subscription(PoseStamped, "/vr/cmd/lift_arm", self.cmd_lift_arm_callback, 10)
        self.cmd_wrist_subscriber = self.create_subscription(PoseStamped, "/vr/cmd/wrist", self.cmd_wrist_callback, 10)
        self.cmd_gripper_subscriber = self.create_subscription(Float64, "/vr/cmd/gripper", self.cmd_gripper_callback, 10)
        self.cmd_vel_subscriber = self.create_subscription(Twist, "/vr/cmd/vel", self.cmd_vel_callback, 10)

    def cmd_head_pose_callback(self, msg):
        self.commands.head_pose = msg

    def cmd_lift_arm_callback(self, msg):
        self.commands.lift_arm = msg

    def cmd_wrist_callback(self, msg):
        self.commands.wrist = msg

    def cmd_gripper_callback(self, msg):
        self.commands.gripper = msg

    def cmd_vel_callback(self, msg):
        self.commands.vel = msg

    def timer_callback(self):
        if not self.commands.ready():
            return

        if not self.joint_state.position:
            self.get_logger().info("Waiting for joint states message to arrive")
            return
        joint_state = self.joint_state

        # Constrain positions before setting them in the trajectory points
        self.commands.constrain_all_positions()

        head_pan_index = joint_state.name.index("joint_head_pan")
        head_tilt_index = joint_state.name.index("joint_head_tilt")
        lift_index = joint_state.name.index("joint_lift")
        arm_index = joint_state.name.index("wrist_extension")
        wrist_yaw_index = joint_state.name.index("joint_wrist_yaw")
        wrist_pitch_index = joint_state.name.index("joint_wrist_pitch")
        wrist_roll_index = joint_state.name.index("joint_wrist_roll")
        gripper_left_index = joint_state.name.index("joint_gripper_finger_left")

        points = [JointTrajectoryPoint() for _ in range(2)]

        points[0].positions = [
            joint_state.position[head_pan_index],
            joint_state.position[head_tilt_index],
            joint_state.position[lift_index],
            joint_state.position[arm_index],
            joint_state.position[wrist_yaw_index],
            joint_state.position[wrist_pitch_index],
            joint_state.position[wrist_roll_index],
            joint_state.position[gripper_left_index],
        ]
        points[0].velocities = [0.1 for _ in range(len(points[0].positions))]
        points[0].velocities[7] = 2
        points[0].time_from_start = Duration(seconds=0.0).to_msg()

        points[1].positions = [
            self.commands.head_pose.pose.position.x,
            self.commands.head_pose.pose.position.y,
            self.commands.lift_arm.pose.position.x,
            self.commands.lift_arm.pose.position.y,
            self.commands.wrist.pose.position.x,
            self.commands.wrist.pose.position.y,
            self.commands.wrist.pose.position.z,
            self.commands.gripper.data,
        ]
        points[1].velocities = [0.1 for _ in range(len(points[1].positions))]
        points[1].velocities[7] = 2
        points[1].time_from_start = Duration(seconds=0.8).to_msg()

        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.trajectory.joint_names = [
            "joint_head_pan",
            "joint_head_tilt",
            "joint_lift",
            "wrist_extension",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            "joint_gripper_finger_left",
        ]
        trajectory_goal.trajectory.points = points
        # print(list(points[0].positions))
        # print(list(points[1].positions))
        print([round(value, 2) for value in list(points[0].positions)])
        print([round(value, 2) for value in list(points[1].positions)])

        trajectory_goal.trajectory.header.frame_id = "base_link"
        self.trajectory_client.send_goal_async(trajectory_goal)
        # self.get_logger().info("Goal sent")
        # self.get_logger().info(f"Point 0: {points[0].positions}")
        # self.get_logger().info(f"Point 1: {points[1].positions}")

    def main(self):
        pass


def main():
    try:
        node = MultiPointCommand()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        node.get_logger().info("Exiting")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

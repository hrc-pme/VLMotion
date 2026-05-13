#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass

import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, Twist
from hello_helpers.hello_misc import HelloNode
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectoryPoint


@dataclass
class Commands:
    wrist_extension: float
    joint_lift: float
    joint_head_pan: float
    joint_head_tilt: float
    joint_wrist_yaw: float
    joint_wrist_pitch: float
    joint_wrist_roll: float
    joint_gripper_finger_left: float
    vel: Twist

    def ready(self):
        pass

    def set_from_joint_states(self, joint_state):
        self.wrist_extension = joint_state.position[joint_state.name.index("wrist_extension")]
        self.joint_lift = joint_state.position[joint_state.name.index("joint_lift")]
        self.joint_head_pan = joint_state.position[joint_state.name.index("joint_head_pan")]
        self.joint_head_tilt = joint_state.position[joint_state.name.index("joint_head_tilt")]
        self.joint_wrist_yaw = joint_state.position[joint_state.name.index("joint_wrist_yaw")]
        self.joint_wrist_pitch = joint_state.position[joint_state.name.index("joint_wrist_pitch")]
        self.joint_wrist_roll = joint_state.position[joint_state.name.index("joint_wrist_roll")]
        self.joint_gripper_finger_left = joint_state.position[joint_state.name.index("joint_gripper_finger_left")]
        self.vel.linear.x = 0.0
        self.vel.angular.z = 0.0

    def constrain_all_positions(self):
        pass

    def get_value_by_name(self, name):
        return getattr(self, name)

    def get_value_list_by_name_list(self, name):
        return [getattr(self, n) for n in name]


class MultiPointCommand(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)
        HelloNode.main(self, "stretch_unity_bridge", "stretch_unity_bridge", wait_for_first_pointcloud=False)

        self.commands = Commands(0, 0, 0, 0, 0, 0, 0, 0, Twist())
        self.create_subscribers()

        while not self.joint_state.position:
            self.get_logger().info("Waiting for joint states message to arrive")
            time.sleep(0.1)
        self.commands.set_from_joint_states(self.joint_state)

        self.call_self_collision_avoidance(True)

        while not self.joint_state_control:
            self.get_logger().info("Waiting for joint states control message to arrive")
            time.sleep(0.1)

        self.create_timer(0.05, self.timer_callback)

    def create_subscribers(self):
        self.joint_state_control = None
        self.joint_states_control_subscriber = self.create_subscription(
            JointState, "/stretch/joint_states_control", self.joint_states_control_callback, 10
        )
        self.gripper_finger_left_subscriber = self.create_subscription(Float64, "/stretch/gripper_control", self.gripper_finger_left_callback, 10)

    def joint_states_control_callback(self, msg):
        self.joint_state_control = msg
        self.commands.wrist_extension = (
            msg.position[msg.name.index("joint_arm_l3")]
            + msg.position[msg.name.index("joint_arm_l2")]
            + msg.position[msg.name.index("joint_arm_l1")]
            + msg.position[msg.name.index("joint_arm_l0")]
        )
        self.commands.joint_lift = -(msg.position[msg.name.index("joint_lift")] - 0.5)
        self.commands.joint_head_pan = msg.position[msg.name.index("joint_head_pan")]
        self.commands.joint_head_tilt = msg.position[msg.name.index("joint_head_tilt")]
        self.commands.joint_wrist_yaw = msg.position[msg.name.index("joint_wrist_yaw")] + math.pi / 2
        self.commands.joint_wrist_pitch = msg.position[msg.name.index("joint_wrist_pitch")]
        self.commands.joint_wrist_roll = msg.position[msg.name.index("joint_wrist_roll")]
        # self.commands.joint_gripper_finger_left = msg.position[msg.name.index("joint_gripper_finger_left")]

    def gripper_finger_left_callback(self, msg):
        self.commands.joint_gripper_finger_left = msg.data

    def call_self_collision_avoidance(self, enable):
        self.self_collision_client = self.create_client(SetBool, "/self_collision_avoidance")
        while not self.self_collision_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("/self_collision_avoidance service not available, waiting...")

        request = SetBool.Request()
        request.data = enable
        self.future = self.self_collision_client.call_async(request)
        self.future.add_done_callback(self.handle_self_collision_response)

    def handle_self_collision_response(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info("Self collision avoidance enabled")
            else:
                self.get_logger().info("Self collision avoidance not enabled")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def timer_callback(self):
        joint_state = self.joint_state

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

        points = [JointTrajectoryPoint() for _ in range(1)]

        points[0].positions = self.commands.get_value_list_by_name_list(trajectory_goal.trajectory.joint_names)
        points[0].velocities = [1.0 for _ in range(len(points[0].positions))]
        points[0].velocities[2] = 0.1
        points[0].velocities[3] = 0.1
        points[0].velocities[7] = 30
        points[0].time_from_start = Duration(seconds=0.5).to_msg()

        trajectory_goal.trajectory.points = points
        print([round(value, 3) for value in [joint_state.position[joint_state.name.index(name)] for name in trajectory_goal.trajectory.joint_names]])
        print([round(value, 3) for value in list(points[0].positions)])
        print([round(value, 3) for value in list(points[0].velocities)])

        trajectory_goal.trajectory.header.frame_id = "base_link"
        self.trajectory_client.send_goal_async(trajectory_goal)

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

#!/usr/bin/env python3
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, Twist
from hello_helpers.gripper_conversion import GripperConversion
from hello_helpers.hello_misc import HelloNode
from hello_helpers.joint_qpos_conversion import JointStateMapping, get_Idx
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from std_srvs.srv import SetBool, Trigger
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


class JointPoseStreamingNode(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)
        HelloNode.main(self, "stretch_unity_bridge", "stretch_unity_bridge", wait_for_first_pointcloud=False)

        self.commands = Commands(0, 0, 0, 0, 0, 0, 0, 0, Twist())
        self.create_subscribers()

        self.self_col_avoid_client = self.create_client(SetBool, "/self_collision_avoidance", callback_group=self.reentrant_cb)
        self.switch_to_navigation_mode_service = self.create_client(Trigger, "/switch_to_navigation_mode", callback_group=self.reentrant_cb)
        self.activate_streaming_position_service = self.create_client(Trigger, "/activate_streaming_position", callback_group=self.reentrant_cb)

        while not self.self_col_avoid_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting on '/self_collision_avoidance' service...")

        while not self.switch_to_navigation_mode_service.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting on '/switch_to_navigation_mode' service...")

        while not self.activate_streaming_position_service.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting on '/activate_streaming_position' service...")

        while not self.joint_state.position:
            time.sleep(0.1)

    def create_subscribers(self):
        self.joint_state_control = None
        self.joint_states_control_subscriber = self.create_subscription(
            JointState, "/stretch/joint_states_control", self.joint_states_control_callback, 10
        )

    def joint_states_control_callback(self, msg):
        self.joint_state_control = msg
        self.commands.wrist_extension = (
            msg.position[msg.name.index("joint_arm_l3")]
            + msg.position[msg.name.index("joint_arm_l2")]
            + msg.position[msg.name.index("joint_arm_l1")]
            + msg.position[msg.name.index("joint_arm_l0")]
        )
        self.commands.joint_lift = -msg.position[msg.name.index("joint_lift")]
        self.commands.joint_head_pan = msg.position[msg.name.index("joint_head_pan")]
        self.commands.joint_head_tilt = msg.position[msg.name.index("joint_head_tilt")]
        self.commands.joint_wrist_yaw = msg.position[msg.name.index("joint_wrist_yaw")]
        self.commands.joint_wrist_pitch = msg.position[msg.name.index("joint_wrist_pitch")]
        self.commands.joint_wrist_roll = msg.position[msg.name.index("joint_wrist_roll")]
        # self.commands.joint_gripper_finger_left = msg.position[msg.name.index("joint_gripper_finger_left")]

    def call_self_collision_avoidance(self):
        self.self_col_avoid_client.call_async(SetBool.Request(data=True))
        time.sleep(1)
        return

    def activate_streaming_position(self):
        trigger_request = Trigger.Request()
        trigger_result = self.activate_streaming_position_service.call_async(trigger_request)
        time.sleep(1)
        return

    def switch_to_navigation_mode(self):
        trigger_request = Trigger.Request()
        trigger_result = self.switch_to_navigation_mode_service.call_async(trigger_request)
        return


class JointPosePublisher(Node):
    def __init__(self):
        rclpy.init()
        super().__init__("float_array_publisher")
        self.reentrant_cb = ReentrantCallbackGroup()
        self.publisher_ = self.create_publisher(Float64MultiArray, "joint_pose_cmd", 10, callback_group=self.reentrant_cb)
        # subscribe to joint states
        self.joint_state = JointState()
        self.joint_states_subscriber = self.create_subscription(
            JointState, "/stretch/joint_states", self.joint_states_callback, 10, callback_group=self.reentrant_cb
        )
        self.Idx = get_Idx("eoa_wrist_dw3_tool_sg3")
        self.switch_to_navigation_mode_service = self.create_client(Trigger, "/switch_to_navigation_mode", callback_group=self.reentrant_cb)
        self.activate_streaming_position_service = self.create_client(Trigger, "/activate_streaming_position", callback_group=self.reentrant_cb)

        while not self.switch_to_navigation_mode_service.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting on '/switch_to_navigation_mode' service...")
        self.gripper_conversion = GripperConversion()
        self.activate_streaming_position()
        self.create_timer(1 / 10.0, self.timer_callback)

    def joint_states_callback(self, msg):
        self.joint_state = msg

    def get_joint_status(self):
        j_status = self.parse_joint_state(self.joint_state)
        pose = np.zeros(self.Idx.num_joints)
        pose[self.Idx.LIFT] = j_status[JointStateMapping.ROS_LIFT_JOINT]
        pose[self.Idx.ARM] = sum(j_status[joint] for joint in JointStateMapping.ROS_ARM_JOINTS)
        pose[self.Idx.GRIPPER] = j_status[JointStateMapping.ROS_GRIPPER_FINGER]
        pose[self.Idx.WRIST_ROLL] = j_status[JointStateMapping.ROS_WRIST_ROLL]
        pose[self.Idx.WRIST_PITCH] = j_status[JointStateMapping.ROS_WRIST_PITCH]
        pose[self.Idx.WRIST_YAW] = j_status[JointStateMapping.ROS_WRIST_YAW]
        pose[self.Idx.HEAD_PAN] = j_status[JointStateMapping.ROS_HEAD_PAN]
        pose[self.Idx.HEAD_TILT] = j_status[JointStateMapping.ROS_HEAD_TILT]
        return pose

    def parse_joint_state(self, joint_state_msg):
        joint_status = {}
        for name, position in zip(joint_state_msg.name, joint_state_msg.position):
            joint_status[name] = position
        return joint_status

    def publish_joint_pose(self, joint_pose):
        msg = Float64MultiArray()
        msg.data = list(joint_pose)
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

    def wait_until_at_setpoint(self, goal_qpos):
        while abs(goal_qpos[:-2] - joint_pose_publisher.get_joint_status()[:-2]).mean() > 0.01:
            rclpy.spin_once(self)
            time.sleep(0.01)


if __name__ == "__main__":
    joint_pose_publisher = JointPosePublisher()
    rclpy.spin_once(joint_pose_publisher)

    Idx = get_Idx("eoa_wrist_dw3_tool_sg3")

    # joint_pose_publisher.switch_to_navigation_mode()
    joint_pose_publisher.activate_streaming_position()

    qpos = np.zeros(Idx.num_joints)
    qpos[Idx.LIFT] = 0.6
    qpos[Idx.ARM] = 0
    qpos[Idx.WRIST_PITCH] = 0
    qpos[Idx.WRIST_ROLL] = 0
    qpos[Idx.WRIST_YAW] = 0
    qpos[Idx.GRIPPER] = joint_pose_publisher.gripper_conversion.robotis_to_finger(0)
    qpos[Idx.BASE_TRANSLATE] = 0
    qpos[Idx.BASE_ROTATE] = 0
    qpos[Idx.HEAD_PAN] = 0
    qpos[Idx.HEAD_TILT] = 0
    joint_pose_publisher.publish_joint_pose(qpos)
    joint_pose_publisher.wait_until_at_setpoint(qpos)

    joint_pose_publisher.destroy_node()
    rclpy.shutdown()

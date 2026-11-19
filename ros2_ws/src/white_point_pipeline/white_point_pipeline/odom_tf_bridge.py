#!/usr/bin/env python3
"""Publish the Stretch odometry pose on the namespaced TF topic."""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdomTfBridge(Node):
    def __init__(self):
        super().__init__('vlmotion_odom_tf_bridge')
        self.declare_parameter('odom_topic', '/stretch3/odom')
        self.declare_parameter('parent_frame', 'stretch3/odom')
        self.declare_parameter('child_frame', 'stretch3/base_link')

        odom_topic = str(self.get_parameter('odom_topic').value)
        self.parent_frame = str(self.get_parameter('parent_frame').value)
        self.child_frame = str(self.get_parameter('child_frame').value)
        self.broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10,
        )
        self.get_logger().info(
            f'Bridging {odom_topic} to TF {self.parent_frame} -> {self.child_frame}'
        )

    def odom_callback(self, message):
        transform = TransformStamped()
        transform.header.stamp = message.header.stamp
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = message.pose.pose.position.x
        transform.transform.translation.y = message.pose.pose.position.y
        transform.transform.translation.z = message.pose.pose.position.z
        transform.transform.rotation = message.pose.pose.orientation
        self.broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

import rclpy
import tf_transformations
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_ros import (Buffer, ConnectivityException, ExtrapolationException,
                     LookupException, TransformListener)


class TFToPoseStampedPublisher(Node):
    def __init__(self):
        super().__init__("tf_to_posestamped_publisher")

        # Declare and get the parameters
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("source_frame", "map")
        self.declare_parameter("target_frame", "base_link")

        self.publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value
        self.source_frame = self.get_parameter("source_frame").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value

        # Create a publisher for PoseStamped
        self.pose_publisher = self.create_publisher(PoseStamped, "pose", 10)

        # Create a buffer and TransformListener to get TF data
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Create a timer to publish PoseStamped at the specified rate
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_pose)

    def publish_pose(self):
        try:
            # Lookup the transform from source_frame to target_frame
            transform = self.tf_buffer.lookup_transform(self.source_frame, self.target_frame, rclpy.time.Time())

            # Convert the transform to PoseStamped
            pose = PoseStamped()
            pose.header.stamp = transform.header.stamp
            pose.header.frame_id = transform.header.frame_id
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z

            pose.pose.orientation.x = transform.transform.rotation.x
            pose.pose.orientation.y = transform.transform.rotation.y
            pose.pose.orientation.z = transform.transform.rotation.z
            pose.pose.orientation.w = transform.transform.rotation.w

            # Publish the PoseStamped
            self.pose_publisher.publish(pose)

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to get transform: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = TFToPoseStampedPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

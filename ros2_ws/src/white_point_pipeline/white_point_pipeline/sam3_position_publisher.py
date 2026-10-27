#!/usr/bin/env python3
"""Publish one reliable SAM3 instrument position as PointStamped."""

import json
import math

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def extract_single_target(result, expected_type="fork", minimum_depth_samples=20):
    """Extract and validate one camera-frame 3D instrument position."""
    instances = [
        instance
        for instance in result.get("instrument_instances", [])
        if instance.get("status") == "detected"
    ]
    if len(instances) != 1:
        raise ValueError(f"expected exactly 1 detected instance, got {len(instances)}")

    instance = instances[0]
    instrument_type = str(instance.get("instrument_type", "")).lower()
    if expected_type and instrument_type != expected_type.lower():
        raise ValueError(
            f"expected instrument type {expected_type!r}, got {instrument_type!r}"
        )
    if instance.get("depth_status") != "valid":
        raise ValueError(
            f"depth status is {instance.get('depth_status', 'missing')!r}"
        )

    position = instance.get("position", {})
    xyz = position.get("center_camera_m")
    if not isinstance(xyz, list) or len(xyz) != 3:
        raise ValueError("center_camera_m is missing or malformed")
    try:
        xyz = [float(value) for value in xyz]
    except (TypeError, ValueError) as exc:
        raise ValueError("center_camera_m contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in xyz):
        raise ValueError("center_camera_m contains a non-finite value")

    sample_count = int(position.get("depth_valid_samples", 0))
    if sample_count < int(minimum_depth_samples):
        raise ValueError(
            f"only {sample_count} valid depth samples; "
            f"minimum is {minimum_depth_samples}"
        )

    depth_capture = result.get("depth_capture", {})
    frame_id = (
        instance.get("camera_coordinate_frame")
        or depth_capture.get("frame_id")
    )
    if not frame_id:
        raise ValueError("camera coordinate frame is missing")
    stamp = depth_capture.get("stamp", {})
    try:
        stamp_sec = int(stamp["sec"])
        stamp_nanosec = int(stamp["nanosec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("depth timestamp is missing or malformed") from exc

    return {
        "instrument_id": str(instance.get("instrument_id", "1")),
        "instrument_type": instrument_type,
        "confidence": float(instance.get("confidence", 0.0)),
        "frame_id": str(frame_id),
        "stamp_sec": stamp_sec,
        "stamp_nanosec": stamp_nanosec,
        "xyz_m": xyz,
        "depth_m": float(position.get("depth_m", xyz[2])),
        "depth_valid_samples": sample_count,
    }


class Sam3PositionPublisher(Node):
    def __init__(self):
        super().__init__("sam3_position_publisher")
        self.declare_parameter("input_topic", "/sam3_recognition/result")
        self.declare_parameter(
            "output_topic", "/sam3_recognition/target_point_camera"
        )
        self.declare_parameter("expected_instrument_type", "fork")
        self.declare_parameter("minimum_depth_samples", 20)
        self.declare_parameter(
            "output_frame_id", "camera_color_optical_frame"
        )

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.expected_type = str(
            self.get_parameter("expected_instrument_type").value
        ).strip().lower()
        self.minimum_depth_samples = int(
            self.get_parameter("minimum_depth_samples").value
        )
        self.output_frame_id = str(
            self.get_parameter("output_frame_id").value
        ).strip()

        retained_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.point_publisher = self.create_publisher(
            PointStamped, output_topic, retained_qos
        )
        self.result_subscription = self.create_subscription(
            String, input_topic, self._result_callback, retained_qos
        )
        self.get_logger().info(
            f"Listening on {input_topic}; publishing camera-frame points "
            f"to {output_topic}"
        )

    def _result_callback(self, message):
        try:
            result = json.loads(message.data)
            target = extract_single_target(
                result,
                expected_type=self.expected_type,
                minimum_depth_samples=self.minimum_depth_samples,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().warning(f"Target point not published: {exc}")
            return

        point = PointStamped()
        point.header.frame_id = self.output_frame_id or target["frame_id"]
        point.header.stamp.sec = target["stamp_sec"]
        point.header.stamp.nanosec = target["stamp_nanosec"]
        point.point.x, point.point.y, point.point.z = target["xyz_m"]
        self.point_publisher.publish(point)
        self.get_logger().info(
            "Published %s ID %s in %s: x=%.4f, y=%.4f, z=%.4f m"
            % (
                target["instrument_type"],
                target["instrument_id"],
                point.header.frame_id,
                point.point.x,
                point.point.y,
                point.point.z,
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = Sam3PositionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

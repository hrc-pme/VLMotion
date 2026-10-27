#!/usr/bin/env python3
"""Launch a RealSense camera and the SAM3 recognition GUI."""

import os
import stat
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def ensure_realsense_video_devices():
    """Restore missing RealSense V4L2 nodes in a privileged container."""
    restored = []
    for sysfs_path in Path("/sys/class/video4linux").glob("video*"):
        try:
            name = (sysfs_path / "name").read_text(encoding="utf-8").strip()
            if "RealSense" not in name:
                continue
            major_text, minor_text = (
                sysfs_path / "dev"
            ).read_text(encoding="utf-8").strip().split(":", 1)
            device_path = Path("/dev") / sysfs_path.name
            expected_device = os.makedev(int(major_text), int(minor_text))
            if device_path.exists():
                current = device_path.stat()
                if stat.S_ISCHR(current.st_mode) and current.st_rdev == expected_device:
                    continue
                print(
                    f"[WARN] {device_path} exists but does not match "
                    f"RealSense device {major_text}:{minor_text}"
                )
                continue
            os.mknod(device_path, stat.S_IFCHR | 0o666, expected_device)
            os.chmod(device_path, 0o666)
            restored.append(str(device_path))
        except (OSError, ValueError) as exc:
            print(f"[WARN] Could not expose {sysfs_path.name}: {exc}")
    if restored:
        print(f"[INFO] Restored RealSense device nodes: {', '.join(restored)}")


def launch_setup(context):
    ensure_realsense_video_devices()
    package_dir = get_package_share_directory("white_point_pipeline")
    config_path = os.path.join(package_dir, "config", "cameras.yaml")
    with open(config_path, "r", encoding="utf-8") as stream:
        cameras = yaml.safe_load(stream) or {}

    camera_name = LaunchConfiguration("camera").perform(context)
    if camera_name not in cameras:
        raise ValueError(
            f"Unknown camera {camera_name!r}; available: {', '.join(cameras)}"
        )
    camera = cameras[camera_name]
    use_compressed = LaunchConfiguration("use_compressed_color").perform(context)
    compressed = str(use_compressed).lower() in {"1", "true", "yes", "on"}
    color_topic = f"{camera['topic_prefix']}/color/image_raw"
    if compressed:
        color_topic += "/compressed"
    depth_topic = f"{camera['topic_prefix']}/aligned_depth_to_color/image_raw"
    camera_info_topic = f"{camera['topic_prefix']}/color/camera_info"

    camera_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name=camera_name,
        namespace="",
        condition=IfCondition(LaunchConfiguration("start_camera")),
        parameters=[{
            "camera_name": camera_name,
            "serial_no": camera["serial_no"],
            "device_type": camera.get("device_type", ""),
            "enable_color": True,
            "enable_depth": True,
            "align_depth.enable": True,
            "align_depth": True,
            "rgb_camera.profile": "640x480x30",
            "depth_module.profile": "640x480x30",
            "color0.enable_auto_exposure": True,
            "color0.auto_exposure_priority": True,
            "publish_tf": False,
        }],
        output="screen",
    )

    gui_node = Node(
        package="white_point_pipeline",
        executable="sam3_recognition_gui",
        name="sam3_recognition_gui",
        output="screen",
        parameters=[{
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
        }],
        arguments=[
            "--controller-url", LaunchConfiguration("controller_url"),
            "--model-name", LaunchConfiguration("model_name"),
            "--target-object", LaunchConfiguration("target_object"),
            "--overlay-path", LaunchConfiguration("overlay_path"),
            "--output-dir", LaunchConfiguration("output_dir"),
        ],
    )
    position_publisher_node = Node(
        package="white_point_pipeline",
        executable="sam3_position_publisher",
        name="sam3_position_publisher",
        output="screen",
        respawn=True,
        respawn_delay=2.0,
        # Keep high-bandwidth camera/GUI traffic on its normal transports,
        # but force this small cross-container XYZ message path onto UDP.
        # This avoids Fast DDS shared-memory delivery across isolated Docker
        # IPC namespaces and does not depend on the launching shell setup.
        additional_env={
            "ROS_LOCALHOST_ONLY": "0",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "FASTRTPS_DEFAULT_PROFILES_FILE": "/workspace/fastdds_udp_only.xml",
            "FASTDDS_DEFAULT_PROFILES_FILE": "/workspace/fastdds_udp_only.xml",
        },
    )
    return [camera_node, position_publisher_node, gui_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera", default_value="d435"),
        DeclareLaunchArgument("start_camera", default_value="true"),
        DeclareLaunchArgument("use_compressed_color", default_value="false"),
        DeclareLaunchArgument(
            "controller_url", default_value="http://127.0.0.1:11000"
        ),
        DeclareLaunchArgument("model_name", default_value="llava-v1.5-13b"),
        DeclareLaunchArgument("target_object", default_value="scissors"),
        DeclareLaunchArgument(
            "overlay_path", default_value="/workspace/sam3_worker_candidates.jpg"
        ),
        DeclareLaunchArgument(
            "output_dir", default_value="/workspace/Outputs/sam3_recognition"
        ),
        OpaqueFunction(function=launch_setup),
    ])

#!/usr/bin/env python3
"""Host GUI plus the session node that turns a pixel into /vlmotion/cmd_vel."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable = LaunchConfiguration('enable_base_motion')
    model_path = LaunchConfiguration('model_path')
    camera_image_topic = LaunchConfiguration('camera_image_topic')
    depth_image_topic = LaunchConfiguration('depth_image_topic')

    session = Node(
        package='vlmotion_host',
        executable='host_session',
        name='vlmotion_host_session',
        output='screen',
        parameters=[{
            'enable_base_motion': enable,
            'camera_image_topic': camera_image_topic,
            'depth_image_topic': depth_image_topic,
        }],
    )

    gui = ExecuteProcess(
        cmd=[
            'python3', '-m', 'VLServo.vlservoing',
            '--model-path', model_path,
            '--controller-url', 'http://127.0.0.1:11000',
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('enable_base_motion', default_value='0'),
        DeclareLaunchArgument(
            'model_path',
            default_value='wentao-yuan/robopoint-v1-vicuna-v1.5-13b',
        ),
        DeclareLaunchArgument(
            'camera_image_topic',
            default_value='/camera/camera/color/image_raw/compressed',
        ),
        DeclareLaunchArgument(
            'depth_image_topic',
            default_value='/camera/camera/aligned_depth_to_color/image_raw',
        ),
        session,
        gui,
    ])

#!/usr/bin/env python3
"""
white_point_gui_standalone.launch.py
─────────────────────────────────────
單獨啟動獨立測試 GUI，不需要：
  ✗ stretch_driver
  ✗ RealSense 相機
  ✗ RViz

GUI 會從本機檔案載入照片作為測試影像，
仍可透過 ROS2 topic 與其他已啟動的節點溝通。
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    controller_url_arg = DeclareLaunchArgument(
        'controller_url',
        default_value='http://10.0.0.30:11000',
        description='LLM Controller URL'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='PME033541/vla13',
        description='Model path for the LLM worker'
    )

    gui_node = Node(
        package='white_point_pipeline',
        executable='white_point_gui_standalone',
        name='white_point_gui_standalone',
        output='screen',
        arguments=[
            '--controller-url', LaunchConfiguration('controller_url'),
            '--model-path',     LaunchConfiguration('model_path'),
        ],
    )

    return LaunchDescription([
        controller_url_arg,
        model_path_arg,
        gui_node,
    ])

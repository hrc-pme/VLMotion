#!/usr/bin/env python3
"""
Launch the VLPoint Visual Servoing GUI via ros2 launch.

This uses ExecuteProcess to run the non-ROS Qt app module
`VLServo.vlservoing` with a few configurable arguments.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    controller_url_arg = DeclareLaunchArgument(
        'controller_url',
        default_value='http://127.0.0.1:11000',
        description='Controller URL for VLPoint server'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='wentao-yuan/robopoint-v1-vicuna-v1.5-13b',
        description='Model to load in the VLPoint GUI'
    )

    controller_url = LaunchConfiguration('controller_url')
    model_path = LaunchConfiguration('model_path')

    # Run the GUI as a Python module so it works from source or install
    gui = ExecuteProcess(
        cmd=[
            'python3', '-m', 'VLServo.vlservoing',
            '--controller-url', controller_url,
            '--model-path', model_path,
            # Note: --load-4bit defaults to True in the script
            # Note: omit --remote by default (can be added to script later if needed)
        ],
        output='screen'
    )

    return LaunchDescription([
        controller_url_arg,
        model_path_arg,
        gui,
    ])

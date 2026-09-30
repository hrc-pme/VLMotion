#!/usr/bin/env python3
"""Host GUI plus the session node that turns a pixel into /vlmotion/cmd_vel."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    enable = LaunchConfiguration('enable_base_motion')
    model_path = LaunchConfiguration('model_path')

    session = Node(
        package='vlmotion_host',
        executable='host_session',
        name='vlmotion_host_session',
        output='screen',
        parameters=[{
            # "0"/"1" must stay strings. Bare numbers become integers in the
            # params file, which rejects the node's string declaration.
            'enable_base_motion': ParameterValue(enable, value_type=str),
        }],
    )

    gui = ExecuteProcess(
        cmd=[
            'python3', '-m', 'VLServo.vlservoing',
            '--model-path', model_path,
            '--controller-url', 'http://127.0.0.1:11000',
            '--autostart',
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('enable_base_motion', default_value='0'),
        DeclareLaunchArgument(
            'model_path',
            default_value='wentao-yuan/robopoint-v1-vicuna-v1.5-13b',
        ),
        session,
        gui,
    ])

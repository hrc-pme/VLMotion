#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('odom_topic', default_value='/stretch3/odom'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/stretch/cmd_vel'),
        Node(
            package='vlmotion_robot_bridge',
            executable='robot_bridge',
            name='vlmotion_robot_bridge',
            output='screen',
            parameters=[{
                'odom_topic': LaunchConfiguration('odom_topic'),
                'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
            }],
        ),
    ])

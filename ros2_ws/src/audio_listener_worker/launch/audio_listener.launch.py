#!/usr/bin/env python3
"""
Launch file for audio listener worker node
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for audio listener worker"""
    
    audio_listener_node = Node(
        package='audio_listener_worker',
        executable='audio_listener_node',
        name='audio_listener_worker',
        output='screen',
        parameters=[
            {'record_duration': 4.0},
            {'sample_rate': 16000},
            {'chunk_size': 1024},
            {'ws_client_enabled': True},
            {'ws_url': 'ws://10.0.0.3:9090'},
            {'ws_start_listen_topic': '/start_listen'},
            {'ws_speech_to_text_topic': '/speech_to_text'},
            {'ws_reconnect_sec': 2.0},
            {'use_ros_topic_trigger': False},
            {'use_ws_trigger': True},
            {'use_ros_topic_result_publish': False},
            {'use_ws_result_publish': True},
        ]
    )
    
    return LaunchDescription([
        audio_listener_node,
    ])

#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('white_point_pipeline')
    rviz_config = os.path.join(pkg_dir, 'config', 'white_point_visualization.rviz')
    
    # 包含原始的 white_point_pipeline.launch.py
    original_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'white_point_pipeline.launch.py')
        )
    )
    
    # 啟動 RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )
    
    return LaunchDescription([
        original_launch,
        rviz_node
    ])

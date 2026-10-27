#!/usr/bin/env python3
"""
相機 TF 校準 Launch 文件

這個 launch 文件用於即時校準相機 TF：
1. 啟動 stretch_driver 和 D435i 相機
2. 啟動 RViz 可視化點雲
3. 【重要】需要在另一個終端手動啟動校準工具

使用方式：
    # 終端 1: 啟動系統
    ros2 launch white_point_pipeline camera_calibration.launch.py
    
    # 終端 2: 啟動校準工具
    ros2 run white_point_pipeline camera_tf_calibrator

調整後可以在 RViz 中即時看到點雲位置變化！
按 C 保存校準值，下次啟動時會自動載入。
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('white_point_pipeline')
    rviz_config = os.path.join(pkg_dir, 'config', 'white_point_visualization.rviz')
    
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='navigation',
        description='Stretch driver mode'
    )
    mode = LaunchConfiguration('mode')
    
    # -------------------------
    # 1. Stretch Driver
    # -------------------------
    stretch_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('stretch_core'),
                'launch',
                'stretch_driver.launch.py'
            ])
        ]),
        launch_arguments={
            'mode': mode,
            'broadcast_odom_tf': 'True',
        }.items()
    )
    
    # -------------------------
    # 2. D435i 相機 (不發布 d435i_link，由校準工具發布)
    # -------------------------
    d435i_camera = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='d435i',
        namespace='',
        parameters=[{
            'camera_name': 'd435i',
            'serial_no': '239122070936',
            'enable_color': True,
            'enable_depth': True,
            'align_depth.enable': True,
            'align_depth': True,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'publish_tf': True,
            'tf_publish_rate': 0.0,
            'pointcloud.enable': True,
            'pointcloud.stream_filter': 2,
            'pointcloud.allow_no_texture_points': False,
        }],
        output='screen'
    )
    
    # -------------------------
    # 3. RViz 可視化
    # -------------------------
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )
    
    # 提示信息
    info_msg = LogInfo(msg='''
╔══════════════════════════════════════════════════════════════════════╗
║                    相機 TF 校準模式                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  請在另一個終端運行校準工具:                                            ║
║                                                                      ║
║    ros2 run white_point_pipeline camera_tf_calibrator                ║
║                                                                      ║
║  然後用鍵盤調整 TF，RViz 中會即時顯示點雲位置變化                         ║
║  按 C 保存校準值                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
''')
    
    return LaunchDescription([
        mode_arg,
        info_msg,
        stretch_driver,
        d435i_camera,
        rviz_node,
    ])

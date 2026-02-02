#!/usr/bin/env python3
"""
多相機 TF 校準 Launch 文件

啟動 Stretch 機器人驅動、多個 RealSense 相機和 RViz，
然後在另一個終端運行校準工具進行即時校準。

使用方式：
    # 終端 1: 啟動系統
    ros2 launch camera_tf_calibration calibration.launch.py
    
    # 終端 2: 啟動校準工具
    ros2 run camera_tf_calibration multi_calibrator
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('camera_tf_calibration')
    rviz_config = os.path.join(pkg_dir, 'config', 'calibration.rviz')
    
    # Launch 參數
    mode_arg = DeclareLaunchArgument('mode', default_value='navigation')
    enable_d405_arg = DeclareLaunchArgument('enable_d405', default_value='false',
                                             description='是否啟用 D405 手腕相機')
    
    mode = LaunchConfiguration('mode')
    enable_d405 = LaunchConfiguration('enable_d405')
    
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
    # 2. D435i 頭部相機
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
    # 3. D405 手腕相機 (可選)
    # -------------------------
    d405_camera = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='d405',
        namespace='',
        parameters=[{
            'camera_name': 'd405',
            'serial_no': '218622277570',
            'enable_color': True,
            'enable_depth': True,
            'align_depth.enable': True,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'publish_tf': True,
            'tf_publish_rate': 0.0,
            'pointcloud.enable': True,
        }],
        output='screen',
        condition=IfCondition(enable_d405)
    )
    
    # -------------------------
    # 4. RViz
    # -------------------------
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen'
    )
    
    # 提示信息
    info_msg = LogInfo(msg='''
╔══════════════════════════════════════════════════════════════════════════════╗
║                         多相機 TF 校準模式                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  請在另一個終端運行校準工具:                                                    ║
║                                                                              ║
║      ros2 run camera_tf_calibration multi_calibrator                         ║
║                                                                              ║
║  控制說明:                                                                    ║
║    TAB     - 切換相機                                                         ║
║    W/S/A/D - 調整 X/Y 位移                                                    ║
║    Q/E     - 調整 Z 位移                                                      ║
║    I/K/J/L - 調整 Pitch/Yaw                                                   ║
║    U/O     - 調整 Roll                                                        ║
║    C       - 保存校準值                                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
''')
    
    return LaunchDescription([
        mode_arg,
        enable_d405_arg,
        info_msg,
        stretch_driver,
        d435i_camera,
        # d405_camera,  # 需要時取消註釋
        rviz_node,
    ])

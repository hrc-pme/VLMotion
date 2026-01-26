#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # -------------------------
    # Launch Arguments
    # -------------------------
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='navigation',
        description='Stretch driver mode (position, navigation, or manipulation)'
    )

    controller_url_arg = DeclareLaunchArgument(
        'controller_url',
        default_value='http://10.0.0.1:11000',
        description='Controller URL for VLPoint server'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='PME033541/vla2.7',
        description='Model to load in the VLPoint GUI'
    )

    mode = LaunchConfiguration('mode')
    controller_url = LaunchConfiguration('controller_url')
    model_path = LaunchConfiguration('model_path')

    # -------------------------
    # 1. Stretch Driver  ros2 launch stretch_core stretch_driver.launch.py mode:=navigation broadcast_odom_tf:=True
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
            'broadcast_odom_tf': 'True',  # 發布 odom frame 給全方位運動控制使用
        }.items()
    )

    # -------------------------
    # 2. RealSense D435i Head Camera
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
            'color0.enable_auto_exposure': True,
            'color0.auto_exposure_priority': True,
            'publish_tf': False,
        }],
        output='screen'
    )

    # -------------------------
    # 3. RealSense D405 Wrist Camera
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
            'publish_tf': False,
        }],
        output='screen'
    )

    # ===============================================
    # 4. White Point GUI  (顯示相機＋滑鼠點白點)
    # ===============================================
    white_point_gui = Node(
        package='white_point_pipeline',
        executable='white_point_gui',
        name='white_point_gui',
        output='screen',
        parameters=[{
            # 如果你的 GUI 想用 D405 改成 /d405/color/image_raw
            'color_topic': '/d435i/color/image_raw'
        }],
        arguments=[
            '--controller-url', controller_url,
            '--model-path', model_path,
        ]
    )

    # ===============================================
    # 5. Pixel → TF → 3D Base Link
    # ===============================================
    white_point_to_3d = Node(
        package='white_point_pipeline',
        executable='white_point_to_3d',
        name='white_point_to_3d',
        output='screen',
        parameters=[{
            'depth_topic': '/d435i/depth/image_rect_raw',
            'camera_info_topic': '/d435i/depth/camera_info',
            'camera_frame': 'd435i_depth_optical_frame'  # 根據你的相機 frame 設定
        }]
    )

    # ===============================================
    # 6. Full Motion Controller (Base + Lift + Arm)
    # ===============================================
    white_point_full_motion = Node(
        package='white_point_pipeline',
        executable='white_point_full_motion',
        name='white_point_full_motion',
        output='screen'
    )

    # -------------------------
    # Return final launch description
    # -------------------------
    return LaunchDescription([
        mode_arg,
        controller_url_arg,
        model_path_arg,
        stretch_driver,
        d435i_camera,
        d405_camera,
        white_point_gui,
        white_point_to_3d,
        white_point_full_motion,
    ])

#!/usr/bin/env python3
"""
Launch file for Visual Servoing with ROS2
Starts Stretch driver, RealSense cameras, visual servoing node, and GUI
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch import conditions


def generate_launch_description():
    """Generate launch description for visual servoing"""
    
    # Declare arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='position',
        description='Stretch driver mode (position, navigation, trajectory)'
    )
    
    camera_name_arg = DeclareLaunchArgument(
        'camera_name',
        default_value='camera',
        description='Name of the RealSense camera'
    )
    
    controller_url_arg = DeclareLaunchArgument(
        'controller_url',
        default_value='http://10.0.0.1:11000',
        description='Controller URL for VLPoint server'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='wentao-yuan/robopoint-v1-vicuna-v1.5-13b',
        description='Model to load in the VLPoint GUI'
    )
    
    launch_gui_arg = DeclareLaunchArgument(
        'launch_gui',
        default_value='true',
        description='Whether to launch the GUI'
    )
    
    image_rotation_arg = DeclareLaunchArgument(
        'image_rotation_deg',
        default_value='-90.0',
        description='Rotation (deg) applied to the GUI camera view'
    )
    
    # Include Stretch driver
    stretch_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('stretch_core'),
                'launch',
                'stretch_driver.launch.py'
            ])
        ]),
        launch_arguments={
            'mode': LaunchConfiguration('mode'),
        }.items()
    )
    
    # RealSense D435i camera (head camera)
    # camera_name='d435i' produces topics: /d435i/color/image_raw, /d435i/depth/image_rect_raw
    # Frame IDs will be: d435i_color_optical_frame, d435i_depth_optical_frame
    # These are mapped to URDF frames (camera_*) in the code
    d435i_camera = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='d435i',
        namespace='',
        parameters=[{
            'camera_name': 'd435i',
            'serial_no': '239122070936',  # D435i serial number
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'align_depth.enable': True,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'publish_tf': False,  # Disable TF to use URDF frames
        }],
        output='screen'
    )
    
    # RealSense D405 camera (wrist camera)
    d405_camera = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='d405',
        namespace='',
        parameters=[{
            'camera_name': 'd405',
            'serial_no': '218622277570',  # D405 serial number
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'align_depth.enable': True,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'publish_tf': False,  # Disable TF publishing
        }],
        output='screen'
    )
    
    # Visual Servoing Node
    visual_servo_node = Node(
        package='vlservo',
        executable='visual_servoing_ros2_node',
        name='visual_servoing_node',
        output='screen',
        parameters=[{
            'base_frame': 'base_link',
            'camera_frame': 'camera_color_optical_frame',
            'gripper_frame': 'link_gripper_fingertip_left',
            'control_rate': 30.0,
            'pixel_target_is_display': True,
            'pixel_target_rotation_deg': LaunchConfiguration('image_rotation_deg'),
        }]
    )
    
    # Visual Servoing GUI
    gui = ExecuteProcess(
        cmd=[
            'python3', '-m', 'VLServo.vlservoing',
            '--controller-url', LaunchConfiguration('controller_url'),
            '--model-path', LaunchConfiguration('model_path'),
        ],
        output='screen',
        condition=conditions.IfCondition(LaunchConfiguration('launch_gui'))
    )
    
    rotation_env = SetEnvironmentVariable(
        name='VL_IMAGE_ROTATION_DEG',
        value=LaunchConfiguration('image_rotation_deg')
    )
    
    return LaunchDescription([
        mode_arg,
        camera_name_arg,
        controller_url_arg,
        model_path_arg,
        launch_gui_arg,
        image_rotation_arg,
        rotation_env,
        stretch_driver,
        d435i_camera,
        d405_camera,  # Enable D405 wrist camera
        gui,
        # visual_servo_node,  # Uncomment to start visual servoing
    ])

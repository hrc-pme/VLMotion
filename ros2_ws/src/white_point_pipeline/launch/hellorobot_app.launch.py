#!/usr/bin/env python3
"""VLMotion application nodes for the external helloRobot_stretch3 stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


TF_REMAPPINGS = [
    ('/tf', '/stretch3/tf'),
    ('/tf_static', '/stretch3/tf_static'),
]


def generate_launch_description():
    controller_url = LaunchConfiguration('controller_url')
    model_path = LaunchConfiguration('model_path')
    enable_motion = LaunchConfiguration('enable_motion')
    publish_odom_tf = LaunchConfiguration('publish_odom_tf')
    enable_stt = LaunchConfiguration('enable_stt')
    stt_model_path = LaunchConfiguration('stt_model_path')
    stt_model_dir = LaunchConfiguration('stt_model_dir')
    stt_language = LaunchConfiguration('stt_language')
    stt_cpu_threads = LaunchConfiguration('stt_cpu_threads')

    head_color_topic = LaunchConfiguration('head_color_topic')
    head_depth_topic = LaunchConfiguration('head_depth_topic')
    head_camera_info_topic = LaunchConfiguration('head_camera_info_topic')
    head_camera_frame = LaunchConfiguration('head_camera_frame')
    gripper_color_topic = LaunchConfiguration('gripper_color_topic')
    gripper_depth_topic = LaunchConfiguration('gripper_depth_topic')
    gripper_camera_info_topic = LaunchConfiguration('gripper_camera_info_topic')
    gripper_camera_frame = LaunchConfiguration('gripper_camera_frame')
    base_frame = LaunchConfiguration('base_frame')
    world_frame = LaunchConfiguration('world_frame')

    return LaunchDescription([
        DeclareLaunchArgument('controller_url', default_value='http://192.168.0.222:11000'),
        DeclareLaunchArgument('model_path', default_value='PME033541/vla13'),
        DeclareLaunchArgument('enable_motion', default_value='false'),
        DeclareLaunchArgument('publish_odom_tf', default_value='true'),
        DeclareLaunchArgument('enable_stt', default_value='true'),
        DeclareLaunchArgument('stt_model_path', default_value='small'),
        DeclareLaunchArgument('stt_model_dir', default_value='/models/faster-whisper'),
        DeclareLaunchArgument('stt_language', default_value='auto'),
        DeclareLaunchArgument('stt_cpu_threads', default_value='8'),

        DeclareLaunchArgument(
            'head_color_topic',
            default_value='/head_camera/head_camera/color/image_raw/compressed',
        ),
        DeclareLaunchArgument(
            'head_depth_topic',
            default_value='/head_camera/head_camera/aligned_depth_to_color/image_raw',
        ),
        DeclareLaunchArgument(
            'head_camera_info_topic',
            default_value='/head_camera/head_camera/color/camera_info',
        ),
        DeclareLaunchArgument(
            'head_camera_frame',
            default_value='head_camera_color_optical_frame',
        ),
        DeclareLaunchArgument(
            'gripper_color_topic',
            default_value='/gripper_camera/gripper_camera/color/image_rect_raw/compressed',
        ),
        DeclareLaunchArgument(
            'gripper_depth_topic',
            default_value='/gripper_camera/gripper_camera/aligned_depth_to_color/image_raw',
        ),
        DeclareLaunchArgument(
            'gripper_camera_info_topic',
            default_value='/gripper_camera/gripper_camera/color/camera_info',
        ),
        DeclareLaunchArgument(
            'gripper_camera_frame',
            default_value='gripper_camera_color_optical_frame',
        ),
        DeclareLaunchArgument('world_frame', default_value='stretch3/odom'),
        DeclareLaunchArgument('base_frame', default_value='stretch3/base_link'),

        Node(
            package='vlmotion_stt',
            executable='speech_to_text',
            name='vlmotion_speech_to_text',
            condition=IfCondition(enable_stt),
            parameters=[{
                'model_path': stt_model_path,
                'model_dir': stt_model_dir,
                'language': stt_language,
                'cpu_threads': ParameterValue(stt_cpu_threads, value_type=int),
                'input_topic': '/speech_audio',
                'output_topic': '/vlmotion/speech_to_text',
                'sample_rate': 16000,
                'sample_width': 2,
            }],
            output='screen',
        ),

        Node(
            package='white_point_pipeline',
            executable='odom_tf_bridge',
            name='vlmotion_odom_tf_bridge',
            condition=IfCondition(publish_odom_tf),
            parameters=[{
                'odom_topic': '/stretch3/odom',
                'parent_frame': world_frame,
                'child_frame': base_frame,
            }],
            remappings=TF_REMAPPINGS,
            output='screen',
        ),

        Node(
            package='white_point_pipeline',
            executable='white_point_gui',
            name='white_point_gui',
            parameters=[{
                'color_topic': head_color_topic,
                'depth_topic': head_depth_topic,
                'camera_info_topic': head_camera_info_topic,
                'camera_frame': head_camera_frame,
                'secondary_color_topic': gripper_color_topic,
                'secondary_rotate_display': False,
                'secondary_pixel_topic': '/white_point_pixel_d405',
                'speech_to_text_topic': '/vlmotion/speech_to_text',
                'voice_ros_timeout_sec': 20.0,
                'require_point_confirmation': True,
            }],
            arguments=[
                '--controller-url', controller_url,
                '--model-path', model_path,
            ],
            remappings=TF_REMAPPINGS,
            output='screen',
        ),

        Node(
            package='white_point_pipeline',
            executable='white_point_to_3d',
            name='white_point_to_3d_head',
            parameters=[{
                'depth_topic': head_depth_topic,
                'camera_info_topic': head_camera_info_topic,
                'camera_frame': head_camera_frame,
                'target_frame': base_frame,
                'rotate_display': True,
            }],
            remappings=TF_REMAPPINGS,
            output='screen',
        ),

        Node(
            package='white_point_pipeline',
            executable='white_point_to_3d',
            name='white_point_to_3d_gripper',
            parameters=[{
                'depth_topic': gripper_depth_topic,
                'camera_info_topic': gripper_camera_info_topic,
                'camera_frame': gripper_camera_frame,
                'target_frame': base_frame,
                'rotate_display': False,
            }],
            remappings=TF_REMAPPINGS + [
                ('/white_point_pixel', '/white_point_pixel_d405'),
                ('/white_point_base', '/white_point_base_d405'),
            ],
            output='screen',
        ),

        Node(
            package='white_point_pipeline',
            executable='white_point_direct_motion',
            name='white_point_direct_motion',
            condition=IfCondition(enable_motion),
            parameters=[{
                'world_frame': world_frame,
                'base_frame': base_frame,
                'left_fingertip_frame': 'stretch3/link_gripper_fingertip_left',
                'right_fingertip_frame': 'stretch3/link_gripper_fingertip_right',
                'cmd_vel_topic': '/stretch/cmd_vel',
                'trajectory_action': '/stretch_controller/follow_joint_trajectory',
            }],
            remappings=TF_REMAPPINGS,
            output='screen',
        ),
    ])

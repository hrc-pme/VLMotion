#!/usr/bin/env python3
"""
White Point Direct Pipeline Launch
==================================
This launch file mirrors white_point_pipeline.launch.py, but starts the
simpler direct motion controller instead of white_point_full_motion.

It does not modify or import white_point_pipeline.launch.py.
"""

# Change only this line to switch cameras: 'd415' / 'd435i'
CAMERA = 'd435i'

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def load_camera_config(camera_name):
    pkg_dir = get_package_share_directory('white_point_pipeline')
    config_path = os.path.join(pkg_dir, 'config', 'cameras.yaml')

    with open(config_path, 'r') as f:
        all_cameras = yaml.safe_load(f)

    if camera_name not in all_cameras:
        available = ', '.join(all_cameras.keys())
        raise ValueError(
            f"Unknown camera '{camera_name}'. Available options: {available}\n"
            f"Config file: {config_path}"
        )
    return all_cameras[camera_name]


def load_calibration_defaults(camera_name):
    defaults = {
        'x': '0.0', 'y': '0.0', 'z': '0.0',
        'roll': '0.0', 'pitch': '0.0', 'yaw': '0.0',
    }

    calibration_file = os.path.expanduser(
        f'~/.config/camera_tf_calibration/{camera_name}_calibration.yaml'
    )
    if not os.path.exists(calibration_file):
        calibration_file = os.path.expanduser(
            '~/.config/white_point_pipeline/camera_tf_calibration.yaml'
        )

    if os.path.exists(calibration_file):
        try:
            with open(calibration_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            defaults['x'] = str(config.get('x', 0.0))
            defaults['y'] = str(config.get('y', 0.0))
            defaults['z'] = str(config.get('z', 0.0))
            defaults['roll'] = str(config.get('roll_rad', 0.0))
            defaults['pitch'] = str(config.get('pitch_rad', 0.0))
            defaults['yaw'] = str(config.get('yaw_rad', 0.0))
            print(f'[INFO] Loaded {camera_name} calibration: {calibration_file}')
            print(f'       translation: X={defaults["x"]}m Y={defaults["y"]}m Z={defaults["z"]}m')
            print(f'       rotation: R={defaults["roll"]} P={defaults["pitch"]} Y={defaults["yaw"]} rad')
        except Exception as exc:
            print(f'[WARN] Failed to load calibration config: {exc}')

    return defaults


def launch_setup(context, *args, **kwargs):
    camera_name = CAMERA
    controller_url = LaunchConfiguration('controller_url').perform(context)
    model_path = LaunchConfiguration('model_path').perform(context)
    use_compressed_color = LaunchConfiguration('use_compressed_color').perform(context)
    require_point_confirmation = LaunchConfiguration('require_point_confirmation').perform(context)
    point_confirmation_timeout_sec = LaunchConfiguration('point_confirmation_timeout_sec').perform(context)
    enable_d405_view = LaunchConfiguration('enable_d405_view').perform(context)
    d405_camera_name = LaunchConfiguration('d405_camera_name').perform(context)
    d405_serial_no = LaunchConfiguration('d405_serial_no').perform(context)
    d405_topic_prefix = LaunchConfiguration('d405_topic_prefix').perform(context)
    d405_camera_frame = LaunchConfiguration('d405_camera_frame').perform(context)
    d405_profile = LaunchConfiguration('d405_profile').perform(context)
    d405_use_compressed_color = LaunchConfiguration('d405_use_compressed_color').perform(context)
    d405_rotate_display = LaunchConfiguration('d405_rotate_display').perform(context)

    cam = load_camera_config(camera_name)
    cal = load_calibration_defaults(camera_name)

    prefix = cam['topic_prefix']
    serial = cam['serial_no']
    optical_frame = cam['color_optical_frame']
    link_name = cam['link_name']
    link_adjusted = cam['link_adjusted_name']

    use_compressed = str(use_compressed_color).lower() in ('1', 'true', 'yes', 'on')
    color_topic = f'{prefix}/color/image_raw/compressed' if use_compressed else f'{prefix}/color/image_raw'
    depth_topic = f'{prefix}/aligned_depth_to_color/image_raw'
    camera_info_topic = f'{prefix}/color/camera_info'
    d405_enabled = str(enable_d405_view).lower() in ('1', 'true', 'yes', 'on')
    d405_use_compressed = str(d405_use_compressed_color).lower() in ('1', 'true', 'yes', 'on')
    d405_color_topic = (
        f'{d405_topic_prefix}/color/image_raw/compressed'
        if d405_use_compressed
        else f'{d405_topic_prefix}/color/image_raw'
    )
    d405_depth_topic = f'{d405_topic_prefix}/aligned_depth_to_color/image_raw'
    d405_camera_info_topic = f'{d405_topic_prefix}/color/camera_info'

    print(f'[white_point_direct_pipeline] camera: {camera_name}')
    print(f'  color topic:  {color_topic}')
    print(f'  depth topic:  {depth_topic}')
    print(f'  camera frame: {optical_frame}')
    if d405_enabled:
        print('[white_point_direct_pipeline] D405 second-point view enabled')
        print(f'  D405 color topic:  {d405_color_topic}')
        print(f'  D405 depth topic:  {d405_depth_topic}')
        print(f'  D405 camera frame: {d405_camera_frame}')
    print('  motion node:  white_point_pipeline.white_point_direct_motion')

    nodes = []

    nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('stretch_core'),
                'launch',
                'stretch_driver.launch.py',
            ])
        ]),
        launch_arguments={
            'mode': LaunchConfiguration('mode'),
            'broadcast_odom_tf': 'True',
        }.items(),
    ))

    nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('respeaker_ros2'),
                'launch',
                'respeaker.launch.py',
            ])
        ]),
        condition=IfCondition(LaunchConfiguration('enable_respeaker')),
        launch_arguments={
            'language': LaunchConfiguration('speech_language'),
            'self_cancellation': LaunchConfiguration('speech_self_cancellation'),
            'launch_soundplay': 'True',
        }.items(),
    ))

    nodes.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=cam['link_connector_name'],
        arguments=[
            '--x', cal['x'], '--y', cal['y'], '--z', cal['z'],
            '--roll', cal['roll'], '--pitch', cal['pitch'], '--yaw', cal['yaw'],
            '--frame-id', 'camera_bottom_screw_frame',
            '--child-frame-id', link_name,
        ],
    ))

    nodes.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=cam['tf_publisher_name'],
        arguments=[
            '--x', cal['x'], '--y', cal['y'], '--z', cal['z'],
            '--roll', cal['roll'], '--pitch', cal['pitch'], '--yaw', cal['yaw'],
            '--frame-id', 'link_head_tilt',
            '--child-frame-id', link_adjusted,
        ],
    ))

    nodes.append(Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=camera_name,
        namespace='',
        parameters=[{
            'camera_name': camera_name,
            'serial_no': serial,
            'enable_color': True,
            'enable_depth': True,
            'align_depth.enable': True,
            'align_depth': True,
            'depth_module.profile': '640x480x30',
            'rgb_camera.profile': '640x480x30',
            'color0.enable_auto_exposure': True,
            'color0.auto_exposure_priority': True,
            'publish_tf': True,
            'tf_publish_rate': 0.0,
            'pointcloud.enable': True,
            'pointcloud.stream_filter': 2,
            'pointcloud.allow_no_texture_points': False,
        }],
        output='screen',
    ))

    if d405_enabled:
        nodes.append(Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name=d405_camera_name,
            namespace='',
            parameters=[{
                'camera_name': d405_camera_name,
                'serial_no': d405_serial_no,
                'enable_color': True,
                'enable_depth': True,
                'align_depth.enable': True,
                'align_depth': True,
                'depth_module.profile': d405_profile,
                'rgb_camera.profile': d405_profile,
                'color0.enable_auto_exposure': True,
                'color0.auto_exposure_priority': True,
                'publish_tf': True,
                'tf_publish_rate': 0.0,
                'pointcloud.enable': True,
                'pointcloud.stream_filter': 2,
                'pointcloud.allow_no_texture_points': False,
            }],
            output='screen',
        ))

    nodes.append(Node(
        package='white_point_pipeline',
        executable='white_point_gui',
        name='white_point_gui',
        output='screen',
        parameters=[{
            'color_topic': color_topic,
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'camera_frame': optical_frame,
            'secondary_color_topic': d405_color_topic if d405_enabled else '',
            'secondary_rotate_display': str(d405_rotate_display).lower() in ('1', 'true', 'yes', 'on'),
            'secondary_pixel_topic': '/white_point_pixel_d405',
            'require_point_confirmation': str(require_point_confirmation).lower() in ('1', 'true', 'yes', 'on'),
            'point_confirmation_timeout_sec': float(point_confirmation_timeout_sec),
        }],
        arguments=[
            '--controller-url', controller_url,
            '--model-path', model_path,
        ],
    ))

    nodes.append(Node(
        package='white_point_pipeline',
        executable='white_point_to_3d',
        name='white_point_to_3d',
        output='screen',
        parameters=[{
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'camera_frame': optical_frame,
            'rotate_display': True,
        }],
    ))

    if d405_enabled:
        nodes.append(Node(
            package='white_point_pipeline',
            executable='white_point_to_3d',
            name='white_point_to_3d_d405',
            output='screen',
            parameters=[{
                'depth_topic': d405_depth_topic,
                'camera_info_topic': d405_camera_info_topic,
                'camera_frame': d405_camera_frame,
                'rotate_display': str(d405_rotate_display).lower() in ('1', 'true', 'yes', 'on'),
            }],
            remappings=[
                ('/white_point_pixel', '/white_point_pixel_d405'),
                ('/white_point_base', '/white_point_base_d405'),
            ],
        ))

    nodes.append(ExecuteProcess(
        cmd=[
            'python3',
            '-m',
            'white_point_pipeline.white_point_direct_motion',
            '--ros-args',
            '-r',
            '__node:=white_point_direct_motion',
            '-p',
            ['adaptive_workspace_enabled:=', LaunchConfiguration('adaptive_workspace_enabled')],
            '-p',
            ['workspace_height_tolerance:=', LaunchConfiguration('workspace_height_tolerance')],
            '-p',
            ['workspace_horizontal_min:=', LaunchConfiguration('workspace_horizontal_min')],
            '-p',
            ['workspace_horizontal_max:=', LaunchConfiguration('workspace_horizontal_max')],
            '-p',
            ['workspace_horizontal_tolerance:=', LaunchConfiguration('workspace_horizontal_tolerance')],
            '-p',
            ['workspace_reselection_position_tolerance:=', LaunchConfiguration('workspace_reselection_position_tolerance')],
            '-p',
            ['d405_view_alignment_enabled:=', LaunchConfiguration('d405_view_alignment_enabled')],
            '-p',
            ['d405_view_target_base_x:=', LaunchConfiguration('d405_view_target_base_x')],
            '-p',
            ['d405_view_base_x_tolerance:=', LaunchConfiguration('d405_view_base_x_tolerance')],
            '-p',
            ['d405_view_max_linear_adjustment:=', LaunchConfiguration('d405_view_max_linear_adjustment')],
            '-p',
            ['d405_height_tolerance:=', LaunchConfiguration('d405_height_tolerance')],
        ],
        output='screen',
    ))

    nodes.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_publisher',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--roll', '0',
            '--pitch', '0',
            '--yaw', '0',
            '--frame-id', 'map',
            '--child-frame-id', 'odom',
        ],
    ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='navigation',
            description='Stretch driver mode used while this pipeline is running.',
        ),
        DeclareLaunchArgument(
            'adaptive_workspace_enabled',
            default_value='true',
            description='Choose direct/lift/base/two-stage motion from independent height and XY reach tests.',
        ),
        DeclareLaunchArgument(
            'workspace_height_tolerance',
            default_value='0.04',
            description='Allowed target-to-gripper contact height error before moving the lift (metres).',
        ),
        DeclareLaunchArgument(
            'workspace_horizontal_min',
            default_value='0.08',
            description='Nominal minimum base-to-target XY reach without translating the base (metres).',
        ),
        DeclareLaunchArgument(
            'workspace_horizontal_max',
            default_value='0.70',
            description='Preparation radius: maximum base-to-target XY distance before stage-one base motion (metres).',
        ),
        DeclareLaunchArgument(
            'workspace_horizontal_tolerance',
            default_value='0.05',
            description='Acceptance margin added to both ends of the horizontal workspace (metres).',
        ),
        DeclareLaunchArgument(
            'workspace_reselection_position_tolerance',
            default_value='0.25',
            description='Reuse the aligned base yaw when a failed/done workspace target is reselected within this 3D distance (metres).',
        ),
        DeclareLaunchArgument(
            'd405_view_alignment_enabled',
            default_value='true',
            description='After close-range D435 yaw alignment, translate base forward/backward to place target in D405 view.',
        ),
        DeclareLaunchArgument(
            'd405_view_target_base_x',
            default_value='0.06',
            description='Desired target X in base_link before switching to D405 (metres).',
        ),
        DeclareLaunchArgument(
            'd405_view_base_x_tolerance',
            default_value='0.035',
            description='Allowed target base X error before switching to D405 (metres).',
        ),
        DeclareLaunchArgument(
            'd405_view_max_linear_adjustment',
            default_value='0.35',
            description='Maximum close-range forward/backward base adjustment allowed for D405 visibility (metres).',
        ),
        DeclareLaunchArgument(
            'd405_height_tolerance',
            default_value='0.01',
            description='Maximum D405 target-to-gripper contact height error allowed without lift adjustment (metres).',
        ),
        DeclareLaunchArgument(
            'restore_gamepad_on_shutdown',
            default_value='true',
            description='Switch Stretch back to gamepad mode when this launch is stopped.',
        ),
        DeclareLaunchArgument(
            'controller_url',
            default_value='http://192.168.0.201:11000',
            description='Controller URL for VLPoint server',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='PME033541/vla13',
            description='Model to load in the VLPoint GUI',
        ),
        DeclareLaunchArgument(
            'use_compressed_color',
            default_value='true',
            description='Use compressed color topic for GUI (/color/image_raw/compressed)',
        ),
        DeclareLaunchArgument(
            'enable_d405_view',
            default_value='true',
            description='Show D405 image in the GUI and use it for the second point selection.',
        ),
        DeclareLaunchArgument(
            'd405_camera_name',
            default_value='gripper_camera',
            description='RealSense camera_name for the D405.',
        ),
        DeclareLaunchArgument(
            'd405_serial_no',
            default_value='218622277570',
            description='D405 serial number.',
        ),
        DeclareLaunchArgument(
            'd405_topic_prefix',
            default_value='/gripper_camera',
            description='D405 ROS topic prefix.',
        ),
        DeclareLaunchArgument(
            'd405_camera_frame',
            default_value='gripper_camera_color_optical_frame',
            description='D405 color optical frame.',
        ),
        DeclareLaunchArgument(
            'd405_profile',
            default_value='640x480x30',
            description='D405 color/depth profile.',
        ),
        DeclareLaunchArgument(
            'd405_use_compressed_color',
            default_value='false',
            description='Use compressed D405 color topic in the GUI.',
        ),
        DeclareLaunchArgument(
            'd405_rotate_display',
            default_value='false',
            description='Rotate the D405 GUI view 90 degrees clockwise.',
        ),
        DeclareLaunchArgument(
            'require_point_confirmation',
            default_value='true',
            description='Show Yes/No confirmation dialog before publishing a selected point',
        ),
        DeclareLaunchArgument(
            'point_confirmation_timeout_sec',
            default_value='3.0',
            description='Confirmation dialog timeout in seconds; 0 disables timeout',
        ),
        DeclareLaunchArgument(
            'enable_respeaker',
            default_value='true',
            description='Launch ReSpeaker pipeline (/speech_to_text and /sound_direction)',
        ),
        DeclareLaunchArgument(
            'speech_language',
            default_value='en-US',
            description='Language for speech_to_text node',
        ),
        DeclareLaunchArgument(
            'speech_self_cancellation',
            default_value='true',
            description='Pause speech recognition while sound_play is speaking',
        ),
        OpaqueFunction(function=launch_setup),
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=[
                    ExecuteProcess(
                        cmd=[
                            'timeout',
                            '2',
                            'ros2',
                            'service',
                            'call',
                            '/switch_to_gamepad_mode',
                            'std_srvs/srv/Trigger',
                            '{}',
                        ],
                        condition=IfCondition(LaunchConfiguration('restore_gamepad_on_shutdown')),
                        output='screen',
                    ),
                ]
            )
        ),
    ])

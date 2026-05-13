#!/usr/bin/env python3
"""
White Point Pipeline Dual-Camera Launch
======================================
保留原本 white_point_pipeline.launch.py 給單相機使用。
本檔為雙相機版本：
- PRIMARY_CAMERA: GUI + 點選 + white_point_to_3d
- SECONDARY_CAMERA: white_point_full_motion 末段導引（建議 d405）
"""

# 主要相機（GUI/點選）
CAMERA = 'd435i'
# 次要相機（末段導引）
SECONDARY_CAMERA = 'd405'

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import yaml
import os


def load_all_cameras_with_fallback():
    """從 cameras.yaml 讀設定，若缺少 d405 則提供 fallback。"""
    pkg_dir = get_package_share_directory('white_point_pipeline')
    config_path = os.path.join(pkg_dir, 'config', 'cameras.yaml')

    with open(config_path, 'r') as f:
        all_cameras = yaml.safe_load(f) or {}

    # fallback: 讓 dual launch 在 d405 被註解時仍可啟動
    fallback = {
        'd405': {
            'serial_no': '218622277570',
            'topic_prefix': '/d405',
            'color_optical_frame': 'd405_color_optical_frame',
            'link_name': 'd405_link',
            'link_adjusted_name': 'd405_link_adjusted',
            'tf_publisher_name': 'd405_tf_adjuster',
            'link_connector_name': 'd405_link_connector',
        }
    }

    for key, value in fallback.items():
        if key not in all_cameras:
            all_cameras[key] = value

    return all_cameras


def load_camera_config(camera_name):
    all_cameras = load_all_cameras_with_fallback()

    if camera_name not in all_cameras:
        available = ', '.join(all_cameras.keys())
        raise ValueError(
            f"未知的相機 '{camera_name}'。可用選項: {available}"
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
            print(f'[INFO] 已載入 {camera_name} 校準配置: {calibration_file}')
        except Exception as e:
            print(f'[WARN] 無法載入 {camera_name} 校準配置: {e}')

    return defaults


def add_camera_nodes(nodes, cam_name, cam_cfg, cal_cfg):
    """加入單台相機所需的 TF + realsense 節點。"""
    nodes.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=cam_cfg['link_connector_name'],
        arguments=[
            '--x', cal_cfg['x'], '--y', cal_cfg['y'], '--z', cal_cfg['z'],
            '--roll', cal_cfg['roll'], '--pitch', cal_cfg['pitch'], '--yaw', cal_cfg['yaw'],
            '--frame-id', 'camera_bottom_screw_frame',
            '--child-frame-id', cam_cfg['link_name'],
        ],
    ))

    nodes.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=cam_cfg['tf_publisher_name'],
        arguments=[
            '--x', cal_cfg['x'], '--y', cal_cfg['y'], '--z', cal_cfg['z'],
            '--roll', cal_cfg['roll'], '--pitch', cal_cfg['pitch'], '--yaw', cal_cfg['yaw'],
            '--frame-id', 'link_head_tilt',
            '--child-frame-id', cam_cfg['link_adjusted_name'],
        ],
    ))

    nodes.append(Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name=cam_name,
        namespace='',
        parameters=[{
            'camera_name': cam_name,
            'serial_no': cam_cfg['serial_no'],
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


def launch_setup(context, *args, **kwargs):
    primary_camera_name = LaunchConfiguration('camera').perform(context)
    secondary_camera_name = LaunchConfiguration('secondary_camera').perform(context)

    controller_url = LaunchConfiguration('controller_url').perform(context)
    model_path = LaunchConfiguration('model_path').perform(context)
    use_compressed_color = LaunchConfiguration('use_compressed_color').perform(context)
    keyboard_autonomy_phase_topic = LaunchConfiguration('keyboard_autonomy_phase_topic').perform(context)
    keyboard_autonomy_idle_phase = LaunchConfiguration('keyboard_autonomy_idle_phase').perform(context)

    primary_cam = load_camera_config(primary_camera_name)
    secondary_cam = load_camera_config(secondary_camera_name)

    primary_cal = load_calibration_defaults(primary_camera_name)
    secondary_cal = load_calibration_defaults(secondary_camera_name)

    # 主要相機給 GUI / 點選 / 3D
    primary_prefix = primary_cam['topic_prefix']
    use_compressed = str(use_compressed_color).lower() in ('1', 'true', 'yes', 'on')
    primary_color_topic = (
        f'{primary_prefix}/color/image_raw/compressed'
        if use_compressed else f'{primary_prefix}/color/image_raw'
    )
    primary_depth_topic = f'{primary_prefix}/aligned_depth_to_color/image_raw'
    primary_camera_info_topic = f'{primary_prefix}/color/camera_info'
    primary_optical_frame = primary_cam['color_optical_frame']

    # 次要相機給 full_motion 末段導引
    secondary_depth_topic = f"{secondary_cam['topic_prefix']}/aligned_depth_to_color/image_raw"
    secondary_camera_info_topic = f"{secondary_cam['topic_prefix']}/color/camera_info"
    secondary_optical_frame = secondary_cam['color_optical_frame']

    print(f'[white_point_pipeline_dual] 主要相機: {primary_camera_name}')
    print(f'  color topic:  {primary_color_topic}')
    print(f'  depth topic:  {primary_depth_topic}')
    print(f'  camera frame: {primary_optical_frame}')

    print(f'[white_point_pipeline_dual] 次要相機: {secondary_camera_name}')
    print(f'  depth topic:  {secondary_depth_topic}')
    print(f'  camera frame: {secondary_optical_frame}')

    nodes = []

    nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('stretch_core'),
                'launch',
                'stretch_driver.launch.py'
            ])
        ]),
        launch_arguments={
            'mode': LaunchConfiguration('mode'),
            'broadcast_odom_tf': 'True',
        }.items()
    ))

    nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('respeaker_ros2'),
                'launch',
                'respeaker.launch.py'
            ])
        ]),
        condition=IfCondition(LaunchConfiguration('enable_respeaker')),
        launch_arguments={
            'language': LaunchConfiguration('speech_language'),
            'self_cancellation': LaunchConfiguration('speech_self_cancellation'),
            'launch_soundplay': 'True',
        }.items()
    ))

    nodes.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('stretch_launch'),
                'launch',
                'rplidar.launch.py'
            ])
        ])
    ))

    # 同時啟動兩台相機
    add_camera_nodes(nodes, primary_camera_name, primary_cam, primary_cal)
    if secondary_camera_name != primary_camera_name:
        add_camera_nodes(nodes, secondary_camera_name, secondary_cam, secondary_cal)
    else:
        print('[white_point_pipeline_dual] 注意：主次相機相同，僅啟動一台。')

    # nodes.append(Node(
    #     package='white_point_pipeline',
    #     executable='white_point_gui_unity',
    #     name='white_point_gui_unity',
    #     output='screen',
    #     parameters=[{
    #         'color_topic': primary_color_topic,
    #         'depth_topic': primary_depth_topic,
    #         'camera_info_topic': primary_camera_info_topic,
    #         'camera_frame': primary_optical_frame,
    #     }],
    #     arguments=[
    #         '--controller-url', controller_url,
    #         '--model-path', model_path,
    #     ],
    # ))

    # nodes.append(Node(
    #     package='white_point_pipeline',
    #     executable='white_point_to_3d',
    #     name='white_point_to_3d',
    #     output='screen',
    #     parameters=[{
    #         'depth_topic': primary_depth_topic,
    #         'camera_info_topic': primary_camera_info_topic,
    #         'camera_frame': primary_optical_frame,
    #     }],
    # ))

    # nodes.append(Node(
    #     package='white_point_pipeline',
    #     executable='white_point_full_motion',
    #     name='white_point_full_motion',
    #     output='screen',
    #     parameters=[{
    #         'depth_topic': secondary_depth_topic,
    #         'camera_info_topic': secondary_camera_info_topic,
    #         'camera_frame': secondary_optical_frame,
    #     }],
    # ))

    # nodes.append(Node(
    #     package='white_point_pipeline',
    #     executable='keyboard_nav_teleop',
    #     name='keyboard_nav_teleop',
    #     output='screen',
    #     emulate_tty=True,
    #     condition=IfCondition(LaunchConfiguration('enable_keyboard_teleop')),
    #     parameters=[{
    #         'block_teleop_when_autonomy_active': True,
    #         'autonomy_phase_topic': keyboard_autonomy_phase_topic,
    #         'autonomy_idle_phase': keyboard_autonomy_idle_phase,
    #     }],
    # ))

    nodes.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_publisher',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'map', '--child-frame-id', 'odom'
        ]
    ))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera',
            default_value=CAMERA,
            description='Primary camera key for GUI/selection (e.g. d435i / d415 / d405)',
        ),
        DeclareLaunchArgument(
            'secondary_camera',
            default_value=SECONDARY_CAMERA,
            description='Secondary camera key for full_motion final guidance (recommended d405)',
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='navigation',
            description='Stretch driver mode (position / navigation / manipulation)',
        ),
        DeclareLaunchArgument(
            'controller_url',
            default_value='http://10.0.0.1:11000',
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
            'enable_respeaker',
            default_value='true',
            description='Launch ReSpeaker pipeline (/speech_to_text and /sound_direction)',
        ),
        DeclareLaunchArgument(
            'enable_keyboard_teleop',
            default_value='true',
            description='Launch stretch_core keyboard teleop node',
        ),
        DeclareLaunchArgument(
            'keyboard_autonomy_phase_topic',
            default_value='/white_point_selection_phase',
            description='Autonomy phase topic used by keyboard lock',
        ),
        DeclareLaunchArgument(
            'keyboard_autonomy_idle_phase',
            default_value='select_first_point',
            description='Phase value considered idle for keyboard unlock',
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
    ])

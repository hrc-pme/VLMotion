#!/usr/bin/env python3
"""
White Point Pipeline Launch 檔
======================================
在下方修改 CAMERA 即可切換相機，所有 topic、frame、TF 自動跟著變。
相機配置存在 config/cameras.yaml。
"""

# ╔══════════════════════════════════════════╗
# ║  要換相機？只改這一行！  'd415' / 'd435i' ║
# ╚══════════════════════════════════════════╝
CAMERA = 'd435i'
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import yaml, os


# ─────────────────────────────────────────────
# 工具函數
# ─────────────────────────────────────────────

def load_camera_config(camera_name):
    """從 config/cameras.yaml 載入指定相機的配置"""
    pkg_dir = get_package_share_directory('white_point_pipeline')
    config_path = os.path.join(pkg_dir, 'config', 'cameras.yaml')

    with open(config_path, 'r') as f:
        all_cameras = yaml.safe_load(f)

    if camera_name not in all_cameras:
        available = ', '.join(all_cameras.keys())
        raise ValueError(
            f"未知的相機 '{camera_name}'。可用選項: {available}\n"
            f"配置檔: {config_path}"
        )
    return all_cameras[camera_name]


def load_calibration_defaults(camera_name):
    """從 ~/.config/camera_tf_calibration/ 載入校準默認值"""
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
            print(f'       位移: X={defaults["x"]}m Y={defaults["y"]}m Z={defaults["z"]}m')
            print(f'       旋轉: R={defaults["roll"]} P={defaults["pitch"]} Y={defaults["yaw"]} (rad)')
        except Exception as e:
            print(f'[WARN] 無法載入校準配置: {e}')

    return defaults


# ─────────────────────────────────────────────
# OpaqueFunction：根據 camera 參數動態產生節點
# ─────────────────────────────────────────────

def launch_setup(context, *args, **kwargs):
    camera_name = CAMERA
    controller_url = LaunchConfiguration('controller_url').perform(context)
    model_path = LaunchConfiguration('model_path').perform(context)
    use_compressed_color = LaunchConfiguration('use_compressed_color').perform(context)
    require_point_confirmation = LaunchConfiguration('require_point_confirmation').perform(context)
    point_confirmation_timeout_sec = LaunchConfiguration('point_confirmation_timeout_sec').perform(context)

    cam = load_camera_config(camera_name)
    cal = load_calibration_defaults(camera_name)

    prefix = cam['topic_prefix']          # e.g. '/d415'
    serial = cam['serial_no']
    optical_frame = cam['color_optical_frame']
    link_name = cam['link_name']
    link_adjusted = cam['link_adjusted_name']

    # 組合 topic 名稱
    use_compressed = str(use_compressed_color).lower() in ('1', 'true', 'yes', 'on')
    color_topic = f'{prefix}/color/image_raw/compressed' if use_compressed else f'{prefix}/color/image_raw'
    depth_topic = f'{prefix}/aligned_depth_to_color/image_raw'
    camera_info_topic = f'{prefix}/color/camera_info'

    print(f'[white_point_pipeline] 使用相機: {camera_name}')
    print(f'  color topic:  {color_topic}')
    print(f'  depth topic:  {depth_topic}')
    print(f'  camera frame: {optical_frame}')

    nodes = []
   # -------------------------
   #Stretch Driver ros2 launch stretch_core stretch_driver.launch.py mode:=navigation broadcast_odom_tf:=True
   # -------------------------
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

    # -------------------------
    # Optional ReSpeaker + speech_to_text
    # -------------------------
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

    # ── 1. 相機 TF 連接器 ──
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

    # ── 2. 相機 TF 調整器 ──
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

    # ── 3. RealSense 相機節點 ──
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

    # ── 4. White Point GUI ──
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
            'require_point_confirmation': str(require_point_confirmation).lower() in ('1', 'true', 'yes', 'on'),
            'point_confirmation_timeout_sec': float(point_confirmation_timeout_sec),
        }],
        arguments=[
            '--controller-url', controller_url,
            '--model-path', model_path,
        ],
    ))

    # ── 5. Pixel → TF → 3D ──
    nodes.append(Node(
        package='white_point_pipeline',
        executable='white_point_to_3d',
        name='white_point_to_3d',
        output='screen',
        parameters=[{
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'camera_frame': optical_frame,
        }],
    ))

    # ── 6. Full Motion Controller ──
    nodes.append(Node(
        package='white_point_pipeline',
        executable='white_point_full_motion',
        name='white_point_full_motion',
        output='screen',
        parameters=[{
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'camera_frame': optical_frame,
        }],
    ))

    # ── 7. map -> odom 靜態 TF（身份變換）──
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
            '--child-frame-id', 'odom'
        ]
    ))

    return nodes


# ─────────────────────────────────────────────
# Launch Description
# ─────────────────────────────────────────────

def generate_launch_description():
    return LaunchDescription([
        # === Launch 參數 ===
        DeclareLaunchArgument(
            'mode',
            default_value='navigation',
            description='Stretch driver mode (position / navigation / manipulation)',
        ),
        DeclareLaunchArgument(
            'controller_url',
            default_value='http://192.168.0.70:11000',
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

        # === 動態產生相機 + pipeline 節點 ===
        OpaqueFunction(function=launch_setup),
    ])

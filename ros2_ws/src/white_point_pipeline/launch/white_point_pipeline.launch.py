#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import yaml
import os


def load_calibration_defaults(camera_name='d435i'):
    """從 camera_tf_calibration package 的配置文件載入校準默認值"""
    defaults = {
        'x': '0.0', 'y': '0.0', 'z': '0.0',
        'roll': '0.0', 'pitch': '0.0', 'yaw': '0.0'
    }
    
    # 優先從新的 camera_tf_calibration 配置目錄載入
    calibration_file = os.path.expanduser(f'~/.config/camera_tf_calibration/{camera_name}_calibration.yaml')
    
    # 如果不存在，嘗試舊的配置位置
    if not os.path.exists(calibration_file):
        calibration_file = os.path.expanduser('~/.config/white_point_pipeline/camera_tf_calibration.yaml')
    
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


def generate_launch_description():

    # -------------------------
    # Get package directory for config files
    # -------------------------
    pkg_share = FindPackageShare('white_point_pipeline')
    
    # 載入校準默認值
    cal_defaults = load_calibration_defaults()

    # -------------------------
    # Launch Arguments (相機 TF 變換參數)
    # -------------------------
    # 位置參數 - 默認值從配置文件載入
    camera_tf_x_arg = DeclareLaunchArgument(
        'camera_tf_x', default_value=cal_defaults['x'],
        description='D435i X offset (meters)')
    camera_tf_y_arg = DeclareLaunchArgument(
        'camera_tf_y', default_value=cal_defaults['y'],
        description='D435i Y offset (meters)')
    camera_tf_z_arg = DeclareLaunchArgument(
        'camera_tf_z', default_value=cal_defaults['z'],
        description='D435i Z offset (meters)')
    
    # 旋轉參數 (弧度) - 默認值從配置文件載入
    camera_tf_roll_arg = DeclareLaunchArgument(
        'camera_tf_roll', default_value=cal_defaults['roll'],
        description='D435i roll rotation (radians)')
    camera_tf_pitch_arg = DeclareLaunchArgument(
        'camera_tf_pitch', default_value=cal_defaults['pitch'],
        description='D435i pitch rotation (radians)')
    camera_tf_yaw_arg = DeclareLaunchArgument(
        'camera_tf_yaw', default_value=cal_defaults['yaw'],
        description='D435i yaw rotation (radians)')
    
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
        default_value='wentao-yuan/robopoint-v1-vicuna-v1.5-13b',
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
    # 2. D435i 相機 TF 變換發布器
    # -------------------------
    # 校準方式：
    # 1. 運行 ros2 run white_point_pipeline camera_tf_calibrator 進行互動式校準
    # 2. 校準值會保存到 ~/.config/white_point_pipeline/camera_tf_calibration.yaml
    # 3. launch 時會自動載入，或用 launch 參數覆蓋
    # -------------------------
    
    # 相機 TF 連接器 - 使用載入的校準值或 launch 參數
    d435i_link_connector = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='d435i_link_connector',
        arguments=[
            '--x', LaunchConfiguration('camera_tf_x'),
            '--y', LaunchConfiguration('camera_tf_y'),
            '--z', LaunchConfiguration('camera_tf_z'),
            '--roll', LaunchConfiguration('camera_tf_roll'),
            '--pitch', LaunchConfiguration('camera_tf_pitch'),
            '--yaw', LaunchConfiguration('camera_tf_yaw'),
            '--frame-id', 'camera_bottom_screw_frame',
            '--child-frame-id', 'd435i_link'
        ]
    )
    
    # 這個節點會在 d435i_link 和 d435i_link_adjusted 之間發布靜態變換
    # 允許您調整相機的位置和角度,而不需要修改 URDF
    d435i_tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='d435i_tf_adjuster',
        arguments=[
            '--x', LaunchConfiguration('camera_tf_x'),
            '--y', LaunchConfiguration('camera_tf_y'),
            '--z', LaunchConfiguration('camera_tf_z'),
            '--roll', LaunchConfiguration('camera_tf_roll'),
            '--pitch', LaunchConfiguration('camera_tf_pitch'),
            '--yaw', LaunchConfiguration('camera_tf_yaw'),
            '--frame-id', 'link_head_tilt',  # 父 frame (頭部傾斜關節)
            '--child-frame-id', 'd435i_link_adjusted'  # 調整後的相機 frame
        ]
    )

    # -------------------------
    # 3. RealSense D435i Head Camera
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
            'publish_tf': True,  # 啟用 TF 發布以顯示點雲
            'tf_publish_rate': 0.0,  # 只發布靜態 TF
            'pointcloud.enable': True,
            'pointcloud.stream_filter': 2,  # 2 = color stream
            'pointcloud.allow_no_texture_points': False,
        }],
        output='screen'
    )

    # -------------------------
    # 3. RealSense D405 Wrist Camera
    # -------------------------
    # d405_camera = Node(
    #     package='realsense2_camera',
    #     executable='realsense2_camera_node',
    #     name='d405',
    #     namespace='',
    #     parameters=[{
    #         'camera_name': 'd405',
    #         'serial_no': '218622277570',
    #         'enable_color': True,
    #         'enable_depth': True,
    #         'align_depth.enable': True,
    #         'depth_module.profile': '640x480x30',
    #         'rgb_camera.profile': '640x480x30',
    #         'publish_tf': False,
    #     }],
    #     output='screen'
    # )

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
            'camera_frame': 'd435i_depth_optical_frame'
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
        # Launch 參數
        camera_tf_x_arg,
        camera_tf_y_arg,
        camera_tf_z_arg,
        camera_tf_roll_arg,
        camera_tf_pitch_arg,
        camera_tf_yaw_arg,
        mode_arg,
        controller_url_arg,
        model_path_arg,
        # 節點
        stretch_driver,
        d435i_link_connector,  # 相機 TF（從 base_link）
        d435i_tf_publisher,  # TF 變換發布器
        d435i_camera,
        # d405_camera,
        white_point_gui,
        white_point_to_3d,
        white_point_full_motion,
    ])

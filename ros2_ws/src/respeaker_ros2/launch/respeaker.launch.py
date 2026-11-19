from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    publish_tf_arg = DeclareLaunchArgument(
        'publish_tf',
        default_value='true',
        description='Publish tf of respeaker'
    )

    launch_soundplay_arg = DeclareLaunchArgument(
        'launch_soundplay',
        default_value='true',
        description='Launch sound_play node'
    )

    launch_speech_to_text_arg = DeclareLaunchArgument(
        'launch_speech_to_text',
        default_value='true',
        description='Launch the in-process speech-to-text node'
    )

    language_arg = DeclareLaunchArgument(
        'language',
        default_value='auto',
        description='Whisper language code, or auto for Mandarin/English detection'
    )

    model_size_arg = DeclareLaunchArgument(
        'model_size',
        default_value='small',
        description='faster-whisper multilingual model size'
    )

    model_dir_arg = DeclareLaunchArgument(
        'model_dir',
        default_value='/workspace/models/faster-whisper',
        description='Persistent faster-whisper download/cache directory'
    )

    local_files_only_arg = DeclareLaunchArgument(
        'local_files_only',
        default_value='true',
        description='Do not access the network while the speech node is running'
    )

    self_cancellation_arg = DeclareLaunchArgument(
        'self_cancellation',
        default_value='true',
        description='Self cancellation means halting speech_to_text while the robot is playing sound'
    )

    static_transformer_node = Node(
        condition=IfCondition(LaunchConfiguration('publish_tf')),
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'respeaker_base', '100']
    )

    # Stop PulseAudio user services (daemon + socket activation) to avoid ALSA device contention.
    # Some desktop/audio applications can cause PulseAudio to open and hold the ReSpeaker capture PCM device,
    # which may lead to respeaker_node failing during initialization (e.g., reporting 0 channels).
    stop_pulseaudio = ExecuteProcess(
        cmd=[
            'bash', '-lc',
            'systemctl --user stop pulseaudio.socket pulseaudio.service 2>/dev/null || true'
        ],
        output='screen'
    )

    start_pulseaudio = ExecuteProcess(
        cmd=[
            'bash', '-lc',
            'systemctl --user start pulseaudio.socket pulseaudio.service 2>/dev/null || true'
        ],
        output='screen'
    )

    respeaker_node = Node(
        package='respeaker_ros2',
        executable='respeaker_node',
        output='screen'
    )

    sound_play_node = Node(
        condition=IfCondition(LaunchConfiguration('launch_soundplay')),
        package='sound_play',
        executable='soundplay_node.py'
    )

    speech_to_text_node = Node(
        condition=IfCondition(LaunchConfiguration('launch_speech_to_text')),
        package='respeaker_ros2',
        executable='speech_to_text',
        parameters=[{
            'language': LaunchConfiguration('language'),
            'model_size': LaunchConfiguration('model_size'),
            'model_dir': LaunchConfiguration('model_dir'),
            'local_files_only': LaunchConfiguration('local_files_only'),
            'self_cancellation': LaunchConfiguration('self_cancellation'),
            'tts_tolerance': 0.5
        }]
    )

    return LaunchDescription([
        # publish_tf_arg,
        launch_soundplay_arg,
        launch_speech_to_text_arg,
        language_arg,
        model_size_arg,
        model_dir_arg,
        local_files_only_arg,
        self_cancellation_arg,
        # static_transformer_node,
        # Stop PulseAudio first, then start all microphone consumers only after
        # that command exits. Restart PulseAudio when this launch shuts down.
        stop_pulseaudio,
        RegisterEventHandler(
            OnProcessExit(
                target_action=stop_pulseaudio,
                on_exit=[respeaker_node, sound_play_node, speech_to_text_node],
            )
        ),
        RegisterEventHandler(
            OnShutdown(on_shutdown=[start_pulseaudio])
        ),
        # LogInfo(
        #     condition=IfCondition(LaunchConfiguration('publish_tf')),
        #     msg='Static transform publisher node will be launched.'
        # ),
        LogInfo(
            condition=IfCondition(LaunchConfiguration('launch_soundplay')),
            msg='Sound play node will be launched.'
        ),
    ])

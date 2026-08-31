from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Launch parameters
    controller_host = LaunchConfiguration('controller_host')
    controller_port = LaunchConfiguration('controller_port')
    dispatch_method = LaunchConfiguration('dispatch_method')

    worker_host = LaunchConfiguration('worker_host')
    worker_port = LaunchConfiguration('worker_port')
    controller_address = LaunchConfiguration('controller_address')
    worker_address = LaunchConfiguration('worker_address')
    model_path = LaunchConfiguration('model_path')
    device = LaunchConfiguration('device')
    limit_model_concurrency = LaunchConfiguration('limit_model_concurrency')
    stream_interval = LaunchConfiguration('stream_interval')

    # Common env to prefer workspace Python packages
    set_env = SetEnvironmentVariable('PYTHONUNBUFFERED', '1')

    # Controller process
    controller_cmd = ExecuteProcess(
        cmd=[
            'python3', '-m', 'point.serve.controller',
            '--host', controller_host,
            '--port', controller_port,
            '--dispatch-method', dispatch_method,
        ],
        name='vlpoint_controller',
        output='screen'
    )

    # Worker process
    worker_cmd = ExecuteProcess(
        cmd=[
            'python3', '-m', 'point.serve.model_worker',
            '--host', worker_host,
            '--port', worker_port,
            '--worker-address', worker_address,
            '--controller-address', controller_address,
            '--model-path', model_path,
            '--device', device,
            '--limit-model-concurrency', limit_model_concurrency,
            '--stream-interval', stream_interval,
            '--load-4bit',
        ],
        name='vlpoint_worker',
        output='screen'
    )

    return LaunchDescription([
        # Controller args
        DeclareLaunchArgument('controller_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('controller_port', default_value='11000'),
        DeclareLaunchArgument('dispatch_method', default_value='shortest_queue',
                              description='lottery | shortest_queue'),

        # Worker args
        DeclareLaunchArgument('worker_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('worker_port', default_value='22000'),
        DeclareLaunchArgument('controller_address', default_value='http://127.0.0.1:11000'),
        DeclareLaunchArgument('worker_address', default_value='http://127.0.0.1:22000'),
        DeclareLaunchArgument('model_path', default_value='wentao-yuan/robopoint-v1-vicuna-v1.5-13b'),
        DeclareLaunchArgument('device', default_value='cuda'),
        DeclareLaunchArgument('limit_model_concurrency', default_value='5'),
        DeclareLaunchArgument('stream_interval', default_value='1'),

        GroupAction([
            set_env,
            controller_cmd,
            worker_cmd,
        ]),
    ])

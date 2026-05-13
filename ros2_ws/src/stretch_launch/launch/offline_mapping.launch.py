import os

from ament_index_python.packages import (get_package_share_directory,
                                         get_package_share_path)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    stretch_core_path = get_package_share_directory("stretch_launch")
    stretch_navigation_path = get_package_share_directory("stretch_launch")
    navigation_package = str(get_package_share_path("stretch_launch"))

    rviz_param = DeclareLaunchArgument("use_rviz", default_value="true", choices=["true", "false"])

    teleop_type = DeclareLaunchArgument("teleop_type", default_value="joystick", description="how to teleop ('keyboard', 'joystick' or 'none')")

    declare_use_sim_time_argument = DeclareLaunchArgument("use_sim_time", default_value="false", description="Use simulation/Gazebo clock")

    stretch_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([stretch_core_path, "/launch/stretch_driver.launch.py"]),
        launch_arguments={"mode": "gamepad", "broadcast_odom_tf": "True"}.items(),
    )

    rplidar_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource([stretch_core_path, "/launch/rplidar.launch.py"]))

    base_teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([stretch_navigation_path, "/launch/teleop_twist.launch.py"]),
        launch_arguments={"teleop_type": LaunchConfiguration("teleop_type")}.items(),
    )

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([get_package_share_directory("nav2_bringup"), "/launch/rviz_launch.py"]),
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    ld = LaunchDescription(
        [
            rviz_param,
            teleop_type,
            declare_use_sim_time_argument,
            stretch_driver_launch,
            rplidar_launch,
            base_teleop_launch,
            rviz_launch,
        ]
    )

    ld.add_action(
        Node(
            package="slam_toolbox",
            executable="sync_slam_toolbox_node",
            name="slam_toolbox",
            output="screen",
            parameters=[
                PathJoinSubstitution([navigation_package, "config", "mapper_params_online_async.yaml"]),
            ],
        )
    )

    return ld

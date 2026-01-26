"""
Wrapper launch file for stretch_core's stretch_driver.launch.py
This provides backward compatibility for the stretch_launch package name.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """Generate launch description that includes stretch_core's driver."""
    
    # Include the stretch_driver launch file from stretch_core
    stretch_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('stretch_core'),
                'launch',
                'stretch_driver.launch.py'
            ])
        ])
    )
    
    return LaunchDescription([
        stretch_driver_launch
    ])

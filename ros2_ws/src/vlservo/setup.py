from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vlservo'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Install URDF so it can be found via ament_index_python
        (os.path.join('share', package_name), ['VLServo/stretch_uncalibrated.urdf']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='ROS2 package for Stretch robot visual servoing (VLServo)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # Console entry to launch the VLPoint GUI
            'vlservoing = VLServo.vlservoing:main',
            # ROS2 nodes
            'visual_servoing_ros2_node = VLServo.visual_servoing_ros2_node:main',
            'camera_subscriber = VLServo.camera_ros2_subscriber:main',
            'base_motion_ros2 = VLServo.base_motion_ros2:main',
        ],
    },
)

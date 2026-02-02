from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'camera_tf_calibration'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HRC Lab',
    maintainer_email='hrclab.nthu@gmail.com',
    description='Interactive camera TF calibration tool for ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'calibrator = camera_tf_calibration.calibrator:main',
            'multi_calibrator = camera_tf_calibration.multi_calibrator:main',
        ],
    },
)

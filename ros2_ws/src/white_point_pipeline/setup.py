from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'white_point_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.rviz') + glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hrc',
    maintainer_email='hrclab.nthu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'white_point_gui = white_point_pipeline.white_point_gui:main',
            'white_point_gui_standalone = white_point_pipeline.white_point_gui_standalone:main',
            'white_point_to_3d = white_point_pipeline.white_point_to_3d:main',
            'white_point_full_motion = white_point_pipeline.white_point_full_motion:main',
            'camera_tf_calibrator = white_point_pipeline.camera_tf_calibrator:main',
        ],
    },
)

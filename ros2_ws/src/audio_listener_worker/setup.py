from setuptools import find_packages, setup

package_name = 'audio_listener_worker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            ['launch/audio_listener.launch.py']),
        ('share/' + package_name + '/config',
            ['config/audio_listener.yaml']),
    ],
    install_requires=['setuptools', 'launch_ros', 'websocket-client'],
    zip_safe=True,
    maintainer='hrc',
    maintainer_email='hrclab.nthu@gmail.com',
    description='Audio listener worker for remote voice input over ROS2',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'audio_listener_node=audio_listener_worker.audio_listener_node:main',
        ],
    },
)

from setuptools import setup

package_name = 'vlmotion_host'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/host.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hrc',
    maintainer_email='hrc@example.com',
    description='VLMotion host session node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'host_session = vlmotion_host.host_session:main',
        ],
    },
)

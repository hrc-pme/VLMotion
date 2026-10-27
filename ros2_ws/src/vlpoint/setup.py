from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'vlpoint'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Launch wrappers for VLPoint controller and worker',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'counting_pass1 = point.serve.run_counting_pass1:main',
            'counting_three_stage = point.serve.run_three_stage_counting:main',
            'calibrate_counting_conformal = point.serve.calibrate_counting_conformal:main',
            'analyze_pass1 = point.serve.analyze_pass1_results:main',
            'build_intervention_regions = point.serve.build_intervention_regions:main',
        ],
    },
)

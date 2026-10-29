import os
from glob import glob
from setuptools import setup

package_name = 'sdt_uav_platform'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.sh')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='valeriia',
    maintainer_email='gritsenko5555@gmail.com',
    description='Drone platform (PX4 SITL, MAVROS, camera model, mission node) for semantic_twin',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'mission_node = sdt_uav_platform.mission_node:main',
        ],
    },
)

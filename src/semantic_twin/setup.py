import os
from glob import glob
from setuptools import setup

package_name = 'semantic_twin'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'models', 'person_actor'), glob('models/person_actor/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='valeriia-hrytsenko',
    maintainer_email='gritsenko5555@gmail.com',
    description='Semantic digital twin: person detection, visual SLAM integration and pixel-to-map transformation for search-and-rescue UAV missions.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cv_node = semantic_twin.cv_node:main',
            'integration_node = semantic_twin.integration_node:main',
            'ground_truth_node = semantic_twin.ground_truth_node:main',
            
        ],
    },
)

from setuptools import setup
import os
from glob import glob

package_name = 'aruco_transform'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS2 package for Aruco marker detection and TF broadcasting',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detector = aruco_transform.aruco_detector:main',
            'aruco_tf_publisher = aruco_transform.aruco_tf_publisher:main',
            'sim_camera_pose = aruco_transform.sim_camera_pose:main',
            'sim_camera_cal = aruco_transform.sim_camera_cal:main',
            'sim_camera_compressed_pose = aruco_transform.sim_camera_compressed_pose:main',
            'real_camera_pose = aruco_transform.real_camera_pose:main',
        ],
    },
)


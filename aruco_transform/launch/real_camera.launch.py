from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='aruco_transform', executable='real_camera_pose', output='screen'),
    ])
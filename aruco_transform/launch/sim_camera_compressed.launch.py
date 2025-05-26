from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='aruco_transform', executable='sim_camera_compressed_pose', output='screen'),
    ])
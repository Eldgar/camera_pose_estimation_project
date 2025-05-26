from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='aruco_transform', executable='aruco_detector', output='screen'),
        #Node(package='aruco_transform', executable='aruco_tf_publisher', output='screen'),
    ])

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                '-0.43135', '-0.32845', '0.36465',
                '1.571', '1.200', '0.000',
                'base_link', 'D415_link'
            ]
        ),
    ])

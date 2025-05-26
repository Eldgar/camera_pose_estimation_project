from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf_web_bridge',
            executable='tf_web_bridge',
            name='tf_web_bridge',
            output='screen'
        ),
        Node(
            package='tf_web_bridge',
            executable='web_action_bridge',
            name='web_action_bridge',
            output='screen'
        )
    ])

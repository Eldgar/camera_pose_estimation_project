from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("ur3e", package_name="starbot_moveit_config").to_moveit_configs()

    servo_yaml = PathJoinSubstitution([
        get_package_share_directory("starbot_moveit_config"),
        "config",
        "servo.yaml"
    ])

    return LaunchDescription([
        Node(
            package='moveit_servo',
            executable='servo_node',
            name='servo_node',
            output='screen',
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                {"use_sim_time": True},
                servo_yaml
            ],
        )
    ])

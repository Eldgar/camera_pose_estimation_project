from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("name", package_name="starbot_moveit_config").to_moveit_configs()

    # Move Group Node (delayed to wait for joint states)
    move_group_node = Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {"trajectory_execution.allowed_execution_duration_scaling": 2.0},
                    {"publish_robot_description_semantic": True},
                    {"use_sim_time": True},
                ],
    )

    return LaunchDescription([
        move_group_node
    ])

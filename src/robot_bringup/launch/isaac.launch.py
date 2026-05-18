from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_description = ParameterValue(Command([
        FindExecutable(name="xacro"), " ",
        PathJoinSubstitution([
            FindPackageShare("bluerov2_description"), "urdf", "bluerov2.xacro"
        ]),
    ]), value_type=str)

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{
                "robot_description": robot_description,
                "use_sim_time": True,
            }],
        ),
    ])

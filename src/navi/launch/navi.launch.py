from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    nav_cfg = os.path.join(
        get_package_share_directory("navi"), "config", "navigation.yaml"
    )

    return LaunchDescription([
        Node(
            package="navi",
            executable="position_controller",
            name="position_controller",
            parameters=[{"navigation_config": nav_cfg}],
            output="screen",
        ),
    ])

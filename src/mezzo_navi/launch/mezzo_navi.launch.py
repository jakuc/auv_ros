from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    ctrl_cfg = os.path.join(
        get_package_share_directory("mezzo_navi"), "config", "controller.yaml"
    )
    thr_cfg = os.path.join(
        get_package_share_directory("robot_bringup"), "config", "thrusters.yaml"
    )

    return LaunchDescription([
        Node(
            package="mezzo_navi",
            executable="velocity_controller",
            name="velocity_controller",
            parameters=[{"controller_config": ctrl_cfg}],
            output="screen",
        ),
        Node(
            package="mezzo_navi",
            executable="thruster_allocator",
            name="thruster_allocator",
            parameters=[{"thrusters_config": thr_cfg}],
            output="screen",
        ),
    ])

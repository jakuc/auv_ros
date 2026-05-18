from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory("mezzo_navi")
    ctrl_cfg = os.path.join(pkg, "config", "controller.yaml")
    ekf_cfg  = os.path.join(pkg, "config", "ekf.yaml")
    thr_cfg  = os.path.join(
        get_package_share_directory("robot_bringup"), "config", "thrusters.yaml")

    return LaunchDescription([
        Node(
            package="mezzo_navi",
            executable="depth_pose_converter",
            name="depth_pose_converter",
            output="screen",
        ),
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_node",
            parameters=[ekf_cfg],
            remappings=[("odometry/filtered", "/auv/odometry")],
            output="screen",
        ),
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

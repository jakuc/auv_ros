from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    nav_cfg = os.path.join(
        get_package_share_directory("navi"), "config", "navigation.yaml")

    position_controller = Node(
        package="navi",
        executable="position_controller",
        name="position_controller",
        parameters=[{"navigation_config": nav_cfg}],
        output="screen",
    )

    local_planner = Node(
        package="navi",
        executable="local_planner",
        name="local_planner",
        output="screen",
    )

    global_planner = Node(
        package="navi",
        executable="global_planner",
        name="global_planner",
        output="screen",
    )

    visualizer = Node(
        package="navi",
        executable="visualizer",
        name="visualizer",
        output="screen",
    )

    rviz_cfg = os.path.join(
        get_package_share_directory("navi"), "config", "nav.rviz")

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        output="screen",
    )

    return LaunchDescription([
        position_controller,
        local_planner,
        global_planner,
        visualizer,
        rviz,
    ])

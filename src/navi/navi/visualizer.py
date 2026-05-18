#!/usr/bin/env python3
"""
visualizer.py — Wizualizacja nawigacji dla RViz2.

  sub: /auv/pose          → actual_path (historia trasy)
  sub: /auv/setpoint      → marker aktualnego setpointa
  sub: /auv/planned_path  → relay do RViz2 + markery WP z sferami akceptacji

  pub: /auv/viz/actual_path   (nav_msgs/Path)
  pub: /auv/viz/markers       (visualization_msgs/MarkerArray)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration


class VisualizerNode(Node):
    def __init__(self):
        super().__init__("visualizer")

        self.declare_parameter("transit_radius",    1.0)
        self.declare_parameter("acceptance_radius", 0.3)
        self.declare_parameter("max_path_length",   500)

        self._transit_r = self.get_parameter("transit_radius").value
        self._acc_r     = self.get_parameter("acceptance_radius").value
        self._max_len   = self.get_parameter("max_path_length").value

        self._actual_path = Path()
        self._actual_path.header.frame_id = "world"
        self._planned_path: Path | None = None
        self._setpoint: PoseStamped | None = None

        self.create_subscription(PoseStamped, "/auv/pose",         self._cb_pose,     10)
        self.create_subscription(PoseStamped, "/auv/setpoint",     self._cb_setpoint, 10)
        self.create_subscription(Path,        "/auv/planned_path", self._cb_planned,  10)

        self._pub_actual  = self.create_publisher(Path,        "/auv/viz/actual_path", 10)
        self._pub_markers = self.create_publisher(MarkerArray, "/auv/viz/markers",     10)

        self.create_timer(0.1, self._publish)
        self.get_logger().info("Visualizer gotowy.")

    def _cb_pose(self, msg: PoseStamped) -> None:
        self._actual_path.poses.append(msg)
        if len(self._actual_path.poses) > self._max_len:
            self._actual_path.poses.pop(0)

    def _cb_setpoint(self, msg: PoseStamped) -> None:
        self._setpoint = msg

    def _cb_planned(self, msg: Path) -> None:
        self._planned_path = msg

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()
        self._actual_path.header.stamp = now
        self._pub_actual.publish(self._actual_path)

        markers = MarkerArray()
        mid = 0

        # Zaplanowana trasa
        if self._planned_path and self._planned_path.poses:
            line = Marker()
            line.header.frame_id = "world"
            line.header.stamp = now
            line.ns = "planned"
            line.id = mid; mid += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.05
            line.color = ColorRGBA(r=0.2, g=0.8, b=0.2, a=0.8)
            line.points = [p.pose.position for p in self._planned_path.poses]
            markers.markers.append(line)

            # Sfery akceptacji przy każdym WP
            for i, wp in enumerate(self._planned_path.poses):
                is_goal = (i == len(self._planned_path.poses) - 1)
                sphere = Marker()
                sphere.header.frame_id = "world"
                sphere.header.stamp = now
                sphere.ns = "wp_radius"
                sphere.id = mid; mid += 1
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose = wp.pose
                r = self._acc_r if is_goal else self._transit_r
                sphere.scale.x = sphere.scale.y = sphere.scale.z = r * 2.0
                if is_goal:
                    sphere.color = ColorRGBA(r=1.0, g=0.3, b=0.3, a=0.3)
                else:
                    sphere.color = ColorRGBA(r=0.3, g=0.8, b=0.3, a=0.2)
                markers.markers.append(sphere)

        # Aktualny setpoint
        if self._setpoint is not None:
            sp = Marker()
            sp.header.frame_id = "world"
            sp.header.stamp = now
            sp.ns = "setpoint"
            sp.id = mid; mid += 1
            sp.type = Marker.SPHERE
            sp.action = Marker.ADD
            sp.pose = self._setpoint.pose
            sp.scale.x = sp.scale.y = sp.scale.z = 0.2
            sp.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            markers.markers.append(sp)

        self._pub_markers.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

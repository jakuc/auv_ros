#!/usr/bin/env python3
"""
global_planner.py — Najprostsza wersja: goal → linia prosta → /auv/global_path.

  srv: /auv/set_goal     (auv_msgs/SetGoal)        — cel misji od operatora
  sub: /auv/pose         (geometry_msgs/PoseStamped)
  pub: /auv/global_path  (nav_msgs/Path)            — trasa do local_planner
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from auv_msgs.srv import SetGoal


class GlobalPlannerNode(Node):
    def __init__(self):
        super().__init__("global_planner")

        self.declare_parameter("waypoint_spacing", 1.0)  # [m]

        self._pose: PoseStamped | None = None

        self._sub = self.create_subscription(
            PoseStamped, "/auv/pose", self._cb_pose, 10)

        self._pub = self.create_publisher(Path, "/auv/global_path", 10)

        self.create_service(SetGoal, "/auv/set_goal", self._srv_set_goal)

        self.get_logger().info("GlobalPlanner gotowy.")

    def _cb_pose(self, msg: PoseStamped) -> None:
        self._pose = msg

    def _srv_set_goal(self, req: SetGoal.Request, res: SetGoal.Response):
        if self._pose is None:
            res.success = False
            res.message = "Brak pozycji robota — /auv/pose nie odebrane."
            return res

        spacing = self.get_parameter("waypoint_spacing").value
        start = self._pose.pose.position
        goal  = req.goal.pose.position

        dx = goal.x - start.x
        dy = goal.y - start.y
        dz = goal.z - start.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist < 1e-3:
            res.success = False
            res.message = "Cel zbyt blisko aktualnej pozycji."
            return res

        n_segments = max(1, int(dist / spacing))
        path = Path()
        path.header.frame_id = "world"
        path.header.stamp = self.get_clock().now().to_msg()

        for i in range(1, n_segments):
            t = i / n_segments
            wp = PoseStamped()
            wp.header.frame_id = "world"
            wp.pose.position.x = start.x + t * dx
            wp.pose.position.y = start.y + t * dy
            wp.pose.position.z = start.z + t * dz
            wp.pose.orientation.w = 1.0
            path.poses.append(wp)

        path.poses.append(req.goal)
        self._pub.publish(path)

        res.success = True
        res.message = f"Misja zaplanowana: {len(path.poses)} WP, dystans {dist:.1f} m."
        self.get_logger().info(res.message)
        return res


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
local_planner.py — Kolejka waypointów → /auv/setpoint.

Tryby:
  CRUISE  (0) — yaw face-forward podczas jazdy, obrót do yaw WP po dotarciu do XYZ
  WORKING (1) — niezależna interpolacja wszystkich DOF

  sub:  /auv/pose            (geometry_msgs/PoseStamped)
  pub:  /auv/setpoint        (geometry_msgs/PoseStamped)
  pub:  /auv/planned_path    (nav_msgs/Path)
  srv:  /auv/add_waypoint    (auv_msgs/AddWaypoint)
  srv:  /auv/clear_waypoints (auv_msgs/ClearWaypoints)
"""

import collections
import math

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from auv_msgs.srv import AddWaypoint, ClearWaypoints


def _quat_to_yaw(q) -> float:
    w, x, y, z = q.w, q.x, q.y, q.z
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_to_quat(yaw: float):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class LocalPlannerNode(Node):
    def __init__(self):
        super().__init__("local_planner")

        self.declare_parameter("acceptance_radius", 0.3)   # [m] — cel końcowy
        self.declare_parameter("acceptance_yaw",    0.1)   # [rad] ~6 deg — cel końcowy
        self.declare_parameter("transit_radius",    1.0)   # [m] — WP pośredni
        self.declare_parameter("rate_hz",           10.0)

        self._acc_r      = self.get_parameter("acceptance_radius").value
        self._acc_yaw    = self.get_parameter("acceptance_yaw").value
        self._transit_r  = self.get_parameter("transit_radius").value
        rate_hz       = self.get_parameter("rate_hz").value

        # kolejka: (PoseStamped, mode)
        self._queue: collections.deque = collections.deque()
        self._current_wp: PoseStamped | None = None
        self._current_mode: int = AddWaypoint.Request.CRUISE

        self._pose: PoseStamped | None = None

        self._pub      = self.create_publisher(PoseStamped, "/auv/setpoint",     10)
        self._pub_path = self.create_publisher(Path,        "/auv/planned_path", 10)
        self._sub = self.create_subscription(
            PoseStamped, "/auv/pose", self._cb_pose, 10)
        self.create_subscription(
            Path, "/auv/global_path", self._cb_global_path, 10)

        self.create_service(AddWaypoint,    "/auv/add_waypoint",    self._srv_add)
        self.create_service(ClearWaypoints, "/auv/clear_waypoints", self._srv_clear)

        self.create_timer(1.0 / rate_hz, self._loop)
        self.get_logger().info("LocalPlanner gotowy.")

    # ------------------------------------------------------------------

    def _srv_add(self, req: AddWaypoint.Request, res: AddWaypoint.Response):
        self._queue.append((req.pose, req.mode))
        if self._current_wp is None:
            self._advance()
        else:
            self._publish_planned_path()
        res.success = True
        res.message = f"Dodano WP #{len(self._queue) + (1 if self._current_wp else 0)}, tryb={'CRUISE' if req.mode == 0 else 'WORKING'}"
        self.get_logger().info(res.message)
        return res

    def _srv_clear(self, req: ClearWaypoints.Request, res: ClearWaypoints.Response):
        self._queue.clear()
        self._current_wp = None
        self._publish_planned_path()
        res.success = True
        res.message = "Kolejka wyczyszczona."
        self.get_logger().info(res.message)
        return res

    def _cb_global_path(self, msg: Path) -> None:
        self._queue.clear()
        self._current_wp = None
        for wp in msg.poses:
            self._queue.append((wp, AddWaypoint.Request.CRUISE))
        self._advance()
        self.get_logger().info(f"Nowa trasa z global_plannera: {len(msg.poses)} WP.")

    # ------------------------------------------------------------------

    def _publish_planned_path(self) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "world"
        if self._current_wp is not None:
            path.poses.append(self._current_wp)
        path.poses.extend(wp for wp, _ in self._queue)
        self._pub_path.publish(path)

    def _advance(self):
        if self._queue:
            self._current_wp, self._current_mode = self._queue.popleft()
            self.get_logger().info(
                f"Nowy aktywny WP: ({self._current_wp.pose.position.x:.2f}, "
                f"{self._current_wp.pose.position.y:.2f}, "
                f"{self._current_wp.pose.position.z:.2f}), "
                f"tryb={'CRUISE' if self._current_mode == 0 else 'WORKING'}"
            )
        else:
            self.get_logger().info("Kolejka pusta — trzymam ostatni setpoint.")
        self._publish_planned_path()

    def _cb_pose(self, msg: PoseStamped):
        self._pose = msg

    def _arrived(self) -> bool:
        if self._pose is None or self._current_wp is None:
            return False
        p = self._pose.pose.position
        t = self._current_wp.pose.position
        dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2 + (p.z - t.z)**2)

        is_goal = len(self._queue) == 0
        if is_goal:
            if dist > self._acc_r:
                return False
            yaw_robot  = _quat_to_yaw(self._pose.pose.orientation)
            yaw_target = _quat_to_yaw(self._current_wp.pose.orientation)
            return abs(_wrap(yaw_robot - yaw_target)) < self._acc_yaw
        else:
            return dist < self._transit_r

    # ------------------------------------------------------------------

    def _loop(self):
        if self._current_wp is None:
            return

        if self._arrived():
            self.get_logger().info("WP osiągnięty.")
            self._advance()
            if self._current_wp is None:
                return

        setpoint = PoseStamped()
        setpoint.header.stamp = self.get_clock().now().to_msg()
        setpoint.header.frame_id = "world"
        setpoint.pose.position = self._current_wp.pose.position

        if self._current_mode == AddWaypoint.Request.CRUISE and self._pose is not None:
            p = self._pose.pose.position
            t = self._current_wp.pose.position
            dx = t.x - p.x
            dy = t.y - p.y
            dist_xy = math.sqrt(dx * dx + dy * dy)

            if dist_xy > self._acc_r:
                # jadzie — face-forward
                yaw_ff = math.atan2(dy, dx)
                setpoint.pose.orientation = _yaw_to_quat(yaw_ff)
            else:
                # XYZ osiągnięte, obracamy do yaw WP
                setpoint.pose.orientation = self._current_wp.pose.orientation
        else:
            # WORKING — publikuj orientację WP bezpośrednio
            setpoint.pose.orientation = self._current_wp.pose.orientation

        self._pub.publish(setpoint)


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

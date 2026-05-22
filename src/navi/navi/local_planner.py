#!/usr/bin/env python3
"""
local_planner.py — Kolejka waypointów → /auv/setpoint.

Tryby:
  CRUISE  (0) — torpeda: obrót do kierunku jazdy → jazda → obrót do yaw WP
  WORKING (1) — wszystkie DOF jednocześnie, yaw niezależny od kierunku ruchu

  sub:  /auv/pose            (geometry_msgs/PoseStamped)
  pub:  /auv/setpoint        (geometry_msgs/PoseStamped)
  pub:  /auv/planned_path    (nav_msgs/Path)
  srv:  /auv/add_waypoint    (auv_msgs/AddWaypoint)
  srv:  /auv/clear_waypoints (auv_msgs/ClearWaypoints)
"""

import collections
import math
import pathlib

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from auv_msgs.srv import AddWaypoint, ClearWaypoints

# Fazy trybu CRUISE
_HEADING = 0  # obrót w miejscu do kierunku jazdy
_TRANSIT = 1  # jazda do WP
_ALIGN   = 2  # obrót w miejscu do yaw WP (tylko ostatni WP w kolejce)


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

        self.declare_parameter("navigation_config", "")
        cfg_path = self.get_parameter("navigation_config").value
        if not cfg_path:
            import ament_index_python.packages as ament
            share = ament.get_package_share_directory("navi")
            cfg_path = str(pathlib.Path(share) / "config" / "navigation.yaml")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)["navi"]["local_planner"]

        self._rate_hz       = float(cfg.get("rate_hz",            10.0))
        self._acc_r         = float(cfg.get("acceptance_radius",   0.3))
        self._acc_yaw       = float(cfg.get("acceptance_yaw",      0.1))
        self._transit_r     = float(cfg.get("transit_radius",      1.0))
        self._heading_tol   = float(cfg.get("heading_tol",         0.15))  # [rad] ~9 deg
        self._v_cruise      = float(cfg.get("v_cruise",            0.4))
        self._a_max         = float(cfg.get("a_max",               0.15))
        self._omega_max     = float(cfg.get("omega_max",           0.3))

        self._dt = 1.0 / self._rate_hz

        # kolejka: (PoseStamped, mode)
        self._queue: collections.deque = collections.deque()
        self._current_wp: PoseStamped | None = None
        self._current_mode: int = AddWaypoint.Request.CRUISE
        self._cruise_phase: int = _HEADING

        self._pose: PoseStamped | None = None

        # marchewka pozycji
        self._ref_pos = np.zeros(3)
        self._v_ref   = 0.0

        # marchewka yaw
        self._yaw_ref = 0.0

        self._pub      = self.create_publisher(PoseStamped, "/auv/setpoint",     10)
        self._pub_path = self.create_publisher(Path,        "/auv/planned_path", 10)
        self._sub = self.create_subscription(
            PoseStamped, "/auv/pose", self._cb_pose, 10)
        self.create_subscription(
            Path, "/auv/global_path", self._cb_global_path, 10)

        self.create_service(AddWaypoint,    "/auv/add_waypoint",    self._srv_add)
        self.create_service(ClearWaypoints, "/auv/clear_waypoints", self._srv_clear)

        self.create_timer(self._dt, self._loop)
        self.get_logger().info(
            f"LocalPlanner gotowy. v_cruise={self._v_cruise} m/s, "
            f"a_max={self._a_max} m/s², omega_max={self._omega_max} rad/s"
        )

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
            if self._pose is not None:
                p = self._pose.pose.position
                self._ref_pos = np.array([p.x, p.y, p.z])
                self._yaw_ref = _quat_to_yaw(self._pose.pose.orientation)
            else:
                t = self._current_wp.pose.position
                self._ref_pos = np.array([t.x, t.y, t.z])
                self._yaw_ref = _quat_to_yaw(self._current_wp.pose.orientation)
            self._v_ref = 0.0
            self._cruise_phase = _HEADING
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

    # ------------------------------------------------------------------

    def _step_yaw_carrot(self, yaw_desired: float) -> None:
        err = _wrap(yaw_desired - self._yaw_ref)
        step = float(np.clip(err, -self._omega_max * self._dt, self._omega_max * self._dt))
        self._yaw_ref = _wrap(self._yaw_ref + step)

    def _step_carrot(self) -> None:
        t = self._current_wp.pose.position
        wp = np.array([t.x, t.y, t.z])
        delta = wp - self._ref_pos
        dist = float(np.linalg.norm(delta))

        if dist < 1e-3:
            self._v_ref = 0.0
            return

        direction = delta / dist
        d_brake = (self._v_ref ** 2) / (2.0 * self._a_max) if self._a_max > 0 else 0.0

        if dist > d_brake:
            self._v_ref = min(self._v_ref + self._a_max * self._dt, self._v_cruise)
        else:
            self._v_ref = max(self._v_ref - self._a_max * self._dt, 0.0)

        step = self._v_ref * self._dt
        if step >= dist:
            self._ref_pos = wp.copy()
            self._v_ref = 0.0
        else:
            self._ref_pos += direction * step

    # ------------------------------------------------------------------

    def _loop(self):
        if self._current_wp is None:
            return

        if self._current_mode == AddWaypoint.Request.WORKING:
            self._loop_working()
        else:
            self._loop_cruise()

    def _loop_working(self):
        if self._pose is None:
            return

        p = self._pose.pose.position
        t = self._current_wp.pose.position
        dist = math.sqrt((p.x-t.x)**2 + (p.y-t.y)**2 + (p.z-t.z)**2)
        yaw_robot  = _quat_to_yaw(self._pose.pose.orientation)
        yaw_wp     = _quat_to_yaw(self._current_wp.pose.orientation)

        is_final = len(self._queue) == 0
        arrived = (dist < self._acc_r if not is_final
                   else dist < self._acc_r and abs(_wrap(yaw_robot - yaw_wp)) < self._acc_yaw)

        if arrived:
            self.get_logger().info("WP osiągnięty (WORKING).")
            self._advance()
            if self._current_wp is None:
                return

        self._step_carrot()
        self._step_yaw_carrot(_quat_to_yaw(self._current_wp.pose.orientation))
        self._publish_setpoint()

    def _loop_cruise(self):
        if self._pose is None:
            return

        p = self._pose.pose.position
        t = self._current_wp.pose.position
        yaw_robot  = _quat_to_yaw(self._pose.pose.orientation)
        yaw_wp     = _quat_to_yaw(self._current_wp.pose.orientation)

        dx = t.x - p.x
        dy = t.y - p.y
        dist = math.sqrt(dx*dx + dy*dy + (t.z - p.z)**2)
        yaw_to_wp = math.atan2(dy, dx)

        is_final = len(self._queue) == 0

        if self._cruise_phase == _HEADING:
            # stoimy w miejscu, obracamy się do kierunku jazdy
            self._step_yaw_carrot(yaw_to_wp)
            if abs(_wrap(yaw_robot - yaw_to_wp)) < self._heading_tol:
                self._cruise_phase = _TRANSIT
                self.get_logger().info("CRUISE: obrót gotowy → START TRANSIT")

        elif self._cruise_phase == _TRANSIT:
            self._step_carrot()
            self._step_yaw_carrot(yaw_to_wp)

            if is_final and dist < self._acc_r:
                self._cruise_phase = _ALIGN
                self.get_logger().info("CRUISE: pozycja osiągnięta → START ALIGN")
            elif not is_final and dist < self._transit_r:
                self.get_logger().info("WP osiągnięty (CRUISE tranzyt).")
                self._advance()
                if self._current_wp is None:
                    return

        elif self._cruise_phase == _ALIGN:
            # stoimy w miejscu przy WP, obracamy się do zadanego yaw
            self._step_yaw_carrot(yaw_wp)
            if abs(_wrap(yaw_robot - yaw_wp)) < self._acc_yaw:
                self.get_logger().info("CRUISE: obrót końcowy gotowy → WP osiągnięty.")
                self._advance()
                if self._current_wp is None:
                    return

        self._publish_setpoint()

    def _publish_setpoint(self):
        setpoint = PoseStamped()
        setpoint.header.stamp = self.get_clock().now().to_msg()
        setpoint.header.frame_id = "world"
        setpoint.pose.position.x = float(self._ref_pos[0])
        setpoint.pose.position.y = float(self._ref_pos[1])
        setpoint.pose.position.z = float(self._ref_pos[2])
        setpoint.pose.orientation = _yaw_to_quat(self._yaw_ref)
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

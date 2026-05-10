#!/usr/bin/env python3
"""
velocity_controller.py — Inner loop: błąd prędkości → wrench ciała.

Odpowiada za poziom companion computer (RT Linux). Odbiera setpoint
prędkości z navi (/auv/vel_setpoint) i mierzoną pozę (/auv/pose),
szacuje prędkość z różnicy pozycji, oblicza wymagany wrench (PID)
i publikuje go do alokatora silników.

  sub:  /auv/pose        (geometry_msgs/PoseStamped)  — 50 Hz
  sub:  /auv/vel_setpoint (geometry_msgs/Twist)        — 10 Hz z navi
  pub:  /auv/cmd_wrench  (geometry_msgs/Wrench)        — body frame, 50 Hz
"""

import math
import pathlib
import yaml

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Wrench

from mezzo_navi.pid import PID


# ---------------------------------------------------------------------------

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [    2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x)],
        [    2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)],
    ], dtype=float)


def _quat_diff_omega(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    w1, x1, y1, z1 = q_prev
    w2, x2, y2, z2 = q_curr
    dx = -w2*x1 + x2*w1 - y2*z1 + z2*y1
    dy = -w2*y1 + x2*z1 + y2*w1 - z2*x1
    dz = -w2*z1 - x2*y1 + y2*x1 + z2*w1
    return 2.0 * np.array([dx, dy, dz]) / max(dt, 1e-6)


def _make_pid(cfg: dict, preset: float = 0.0) -> PID:
    return PID(
        kp=float(cfg["kp"]),
        ki=float(cfg.get("ki", 0.0)),
        kd=float(cfg.get("kd", 0.0)),
        max_integral=float(cfg["max_integral"]) if "max_integral" in cfg else None,
        max_output=float(cfg["max_output"])   if "max_output"   in cfg else None,
        preset_integral=preset,
    )


# ---------------------------------------------------------------------------

class VelocityControllerNode(Node):
    def __init__(self):
        super().__init__("velocity_controller")

        self.declare_parameter("controller_config", "")
        cfg_path = self.get_parameter("controller_config").value
        if not cfg_path:
            import ament_index_python.packages as ament
            share = ament.get_package_share_directory("mezzo_navi")
            cfg_path = str(pathlib.Path(share) / "config" / "controller.yaml")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)["mezzo_navi"]["velocity_controller"]

        preset_z = float(cfg["z"].get("preset_integral", 0.0))
        self._pid = {
            "vx":    _make_pid(cfg["x"]),
            "vy":    _make_pid(cfg["y"]),
            "vz":    _make_pid(cfg["z"], preset=preset_z),
            "roll":  _make_pid(cfg["roll_damp"]),
            "pitch": _make_pid(cfg["pitch_damp"]),
            "yaw":   _make_pid(cfg["yaw"]),
        }

        # Setpoint prędkości w układzie ciała (aktualizowany przez navi)
        self._v_sp = np.zeros(6)

        self._prev_pos  = None
        self._prev_quat = None
        self._prev_time = None

        self._sub_pose = self.create_subscription(
            PoseStamped, "/auv/pose", self._cb_pose, 10)
        self._sub_vsp  = self.create_subscription(
            Twist, "/auv/vel_setpoint", self._cb_vel_setpoint, 10)
        self._pub = self.create_publisher(Wrench, "/auv/cmd_wrench", 10)

        self.get_logger().info(f"VelocityController gotowy. Config: {cfg_path}")

    def _cb_vel_setpoint(self, msg: Twist) -> None:
        self._v_sp = np.array([
            msg.linear.x,  msg.linear.y,  msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z,
        ])

    def _cb_pose(self, msg: PoseStamped) -> None:
        pos  = np.array([msg.pose.position.x,
                         msg.pose.position.y,
                         msg.pose.position.z])
        quat = np.array([msg.pose.orientation.w, msg.pose.orientation.x,
                         msg.pose.orientation.y, msg.pose.orientation.z])
        t_now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self._prev_pos is None:
            self._prev_pos  = pos
            self._prev_quat = quat
            self._prev_time = t_now
            return

        dt = max(t_now - self._prev_time, 1e-4)

        # Prędkość w układzie ciała z różnicy pozycji
        v_lin_world = (pos - self._prev_pos) / dt
        v_ang_world = _quat_diff_omega(self._prev_quat, quat, dt)
        R = _quat_to_rot(quat)
        v_body = np.concatenate([R.T @ v_lin_world, R.T @ v_ang_world])

        self._prev_pos  = pos
        self._prev_quat = quat
        self._prev_time = t_now

        # Inner PID
        e = self._v_sp - v_body
        msg_out = Wrench()
        msg_out.force.x  = self._pid["vx"].step(e[0], dt)
        msg_out.force.y  = self._pid["vy"].step(e[1], dt)
        msg_out.force.z  = self._pid["vz"].step(e[2], dt)
        # Roll/pitch: tłumienie — setpoint = 0, bez wkładu z navi
        msg_out.torque.x = self._pid["roll"].step(-v_body[3], dt)
        msg_out.torque.y = self._pid["pitch"].step(-v_body[4], dt)
        msg_out.torque.z = self._pid["yaw"].step(e[5], dt)
        self._pub.publish(msg_out)


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = VelocityControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

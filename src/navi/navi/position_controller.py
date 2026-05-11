#!/usr/bin/env python3
"""
position_controller.py — Outer loop: błąd pozycji → setpoint prędkości.

Poziom main computer (best-effort Linux). Odbiera pozę robota, porównuje
z zadanym setpointem pozycji i publikuje setpoint prędkości ciała do
companion computera (mezzo_navi/velocity_controller).

  sub: /auv/pose         (geometry_msgs/PoseStamped) — 50 Hz z symulatora/IMU+DVL
  pub: /auv/vel_setpoint (geometry_msgs/Twist)       — body frame, ~10 Hz

Prosta proporcjonalna regulacja P — wystarczająca dla outer loop.
Bardziej zaawansowane algorytmy (Pure Pursuit, MPC) zastąpią ten węzeł
gdy dojdzie lokalny planer trajektorii.
"""

import math
import pathlib
import yaml

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from rcl_interfaces.msg import SetParametersResult


# ---------------------------------------------------------------------------

def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [    2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x)],
        [    2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)],
    ], dtype=float)


def _quat_to_yaw(q: np.ndarray) -> float:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------

class PositionControllerNode(Node):
    def __init__(self):
        super().__init__("position_controller")

        self.declare_parameter("navigation_config", "")
        cfg_path = self.get_parameter("navigation_config").value
        if not cfg_path:
            import ament_index_python.packages as ament
            share = ament.get_package_share_directory("navi")
            cfg_path = str(pathlib.Path(share) / "config" / "navigation.yaml")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)["navi"]["position_controller"]

        sp = cfg["setpoint"]
        self._sp_pos = np.array([float(sp["x"]), float(sp["y"]), float(sp["z"])])
        self._sp_yaw = float(sp["yaw"])

        gains = cfg["gains"]
        self._kp     = {ax: float(gains[ax]["kp"])      for ax in ("x", "y", "z", "yaw")}
        self._max_vel = {ax: float(gains[ax]["max_vel"]) for ax in ("x", "y", "z", "yaw")}

        rate_hz = float(cfg.get("rate_hz", 10.0))

        cmd_tau = float(cfg.get("command_filter", 0.0))
        dt_outer = 1.0 / rate_hz
        self._cmd_alpha = cmd_tau / (cmd_tau + dt_outer) if cmd_tau > 0.0 else 0.0
        self._cmd_filt  = np.zeros(4)  # [vx, vy, vz, wyaw]

        self._latest_pos  = None
        self._latest_quat = None
        self._vel_override: np.ndarray | None = None
        self._override_time: float = 0.0

        for ax in ("x", "y", "z", "yaw"):
            self.declare_parameter(f"{ax}.kp",      self._kp[ax])
            self.declare_parameter(f"{ax}.max_vel", self._max_vel[ax])
        self.add_on_set_parameters_callback(self._on_param_change)

        self._sub = self.create_subscription(
            PoseStamped, "/auv/pose", self._cb_pose, 10)
        self._sub_sp = self.create_subscription(
            PoseStamped, "/auv/setpoint", self._cb_setpoint, 10)
        self._sub_override = self.create_subscription(
            Twist, "/auv/vel_override", self._cb_vel_override, 10)
        self._pub = self.create_publisher(Twist, "/auv/vel_setpoint", 10)
        self._timer = self.create_timer(1.0 / rate_hz, self._outer_loop)

        self.get_logger().info(
            f"PositionController gotowy. Setpoint: {self._sp_pos}, yaw={self._sp_yaw:.2f} rad"
        )

    def _on_param_change(self, params) -> SetParametersResult:
        for p in params:
            parts = p.name.split(".")
            if len(parts) != 2:
                continue
            ax, field = parts
            if ax not in self._kp:
                continue
            val = float(p.value)
            if field == "kp":
                self._kp[ax] = val
            elif field == "max_vel":
                self._max_vel[ax] = val
            self.get_logger().info(f"Position param {p.name} → {val}")
        return SetParametersResult(successful=True)

    def _cb_vel_override(self, msg: Twist) -> None:
        self._vel_override = np.array([
            msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z])
        self._override_time = self.get_clock().now().nanoseconds * 1e-9

    def _cb_setpoint(self, msg: PoseStamped) -> None:
        self._sp_pos = np.array([msg.pose.position.x,
                                  msg.pose.position.y,
                                  msg.pose.position.z])
        q = np.array([msg.pose.orientation.w, msg.pose.orientation.x,
                      msg.pose.orientation.y, msg.pose.orientation.z])
        if np.linalg.norm(q) > 1e-6:
            self._sp_yaw = _quat_to_yaw(q)
        self.get_logger().info(
            f"Nowy setpoint: pos={self._sp_pos}, yaw={self._sp_yaw:.2f} rad")

    def _cb_pose(self, msg: PoseStamped) -> None:
        self._latest_pos  = np.array([msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z])
        self._latest_quat = np.array([msg.pose.orientation.w, msg.pose.orientation.x,
                                      msg.pose.orientation.y, msg.pose.orientation.z])

    def _outer_loop(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        override_active = (
            self._vel_override is not None and (now - self._override_time) < 1.0
        )

        if override_active:
            raw = self._vel_override
        elif self._latest_pos is not None:
            pos  = self._latest_pos
            quat = self._latest_quat

            e_world = self._sp_pos - pos
            e_yaw   = _wrap(self._sp_yaw - _quat_to_yaw(quat))

            # XY: skaluj wektor jako całość żeby zachować kierunek do celu
            v_hor_w = np.array([self._kp["x"] * e_world[0], self._kp["y"] * e_world[1]])
            max_hor = min(self._max_vel["x"], self._max_vel["y"])
            speed_hor = np.linalg.norm(v_hor_w)
            if speed_hor > max_hor:
                v_hor_w = v_hor_w * (max_hor / speed_hor)

            vz   = float(np.clip(self._kp["z"] * e_world[2], -self._max_vel["z"], self._max_vel["z"]))
            wyaw = float(np.clip(self._kp["yaw"] * e_yaw, -self._max_vel["yaw"], self._max_vel["yaw"]))

            R = _quat_to_rot(quat)
            v_hor_body = R.T @ np.array([v_hor_w[0], v_hor_w[1], 0.0])

            raw = np.array([v_hor_body[0], v_hor_body[1], vz, wyaw])
        else:
            raw = np.zeros(4)

        if self._cmd_alpha > 0.0:
            self._cmd_filt = self._cmd_alpha * self._cmd_filt + (1.0 - self._cmd_alpha) * raw
        else:
            self._cmd_filt = raw

        msg = Twist()
        msg.linear.x  = float(self._cmd_filt[0])
        msg.linear.y  = float(self._cmd_filt[1])
        msg.linear.z  = float(self._cmd_filt[2])
        msg.angular.z = float(self._cmd_filt[3])
        self._pub.publish(msg)


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PositionControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

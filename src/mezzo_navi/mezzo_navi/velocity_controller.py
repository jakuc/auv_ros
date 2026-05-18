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
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped, Wrench
from rcl_interfaces.msg import SetParametersResult

class PID:
    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        max_integral: float | None = None,
        max_output:   float | None = None,
        preset_integral: float = 0.0,
        derivative_tau: float = 0.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral   = max_integral
        self.max_output     = max_output
        self.derivative_tau = derivative_tau
        self._integral      = preset_integral
        self._prev_error    = None
        self._deriv_filt    = 0.0

    def reset(self, preset_integral: float = 0.0) -> None:
        self._integral   = preset_integral
        self._prev_error = None
        self._deriv_filt = 0.0

    def step(self, error: float, dt: float) -> float:
        dt = max(dt, 1e-6)

        derivative = 0.0
        if self._prev_error is not None:
            raw = (error - self._prev_error) / dt
            if self.derivative_tau > 0.0:
                alpha = self.derivative_tau / (self.derivative_tau + dt)
                self._deriv_filt = alpha * self._deriv_filt + (1.0 - alpha) * raw
                derivative = self._deriv_filt
            else:
                derivative = raw
        self._prev_error = error

        # Anti-windup: całkuj tylko gdy output nie jest saturowany
        # lub gdy błąd redukuje saturację (przeciwny znak)
        output_no_i = self.kp * error + self.kd * derivative
        tentative = output_no_i + self.ki * (self._integral + error * dt)
        saturated = (
            self.max_output is not None and abs(tentative) > self.max_output
        )
        winding_up = saturated and (tentative * error > 0)
        if not winding_up:
            self._integral += error * dt
            if self.max_integral is not None:
                self._integral = float(np.clip(self._integral,
                                               -self.max_integral, self.max_integral))

        output = output_no_i + self.ki * self._integral
        if self.max_output is not None:
            output = float(np.clip(output, -self.max_output, self.max_output))
        return output


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
        max_output=float(cfg["max_output"])     if "max_output"   in cfg else None,
        preset_integral=preset,
        derivative_tau=float(cfg.get("derivative_filter", 0.0)),
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

        vel_tau = float(cfg.get("velocity_filter", 0.0))
        self._vel_alpha = vel_tau / (vel_tau + 0.02) if vel_tau > 0.0 else 0.0

        # Setpoint prędkości w układzie ciała (aktualizowany przez navi)
        self._v_sp = np.zeros(6)

        # Pomiar prędkości z symulatora/IMU/DVL (body frame)
        self._v_body_measured: np.ndarray | None = None

        self._prev_pos  = None
        self._prev_quat = None
        self._prev_time = None

        self._sub_pose = self.create_subscription(
            PoseStamped, "/auv/pose", self._cb_pose, 10)
        self._sub_vel  = self.create_subscription(
            TwistStamped, "/auv/velocity", self._cb_velocity, 10)
        self._sub_vsp  = self.create_subscription(
            Twist, "/auv/vel_setpoint", self._cb_vel_setpoint, 10)
        self._pub      = self.create_publisher(Wrench, "/auv/cmd_wrench", 10)
        self._pub_err  = self.create_publisher(Twist, "/auv/vel_error", 10)

        self._declare_pid_params(cfg)
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(f"VelocityController gotowy. Config: {cfg_path}")

    # (cfg_key w yaml) → (klucz w self._pid, pola do eksponowania)
    _AXES = {
        "x":          ("vx",    ["kp", "ki", "kd", "max_integral", "max_output"]),
        "y":          ("vy",    ["kp", "ki", "kd", "max_integral", "max_output"]),
        "z":          ("vz",    ["kp", "ki", "kd", "max_integral", "max_output", "preset_integral"]),
        "yaw":        ("yaw",   ["kp", "ki", "kd", "max_integral", "max_output"]),
        "roll_damp":  ("roll",  ["kp", "kd", "max_output"]),
        "pitch_damp": ("pitch", ["kp", "kd", "max_output"]),
    }

    def _declare_pid_params(self, cfg: dict) -> None:
        for cfg_key, (_, fields) in self._AXES.items():
            for field in fields:
                val = float(cfg[cfg_key].get(field, 0.0))
                self.declare_parameter(f"{cfg_key}.{field}", val)

    def _on_param_change(self, params) -> SetParametersResult:
        pid_map = {cfg_key: pid_key for cfg_key, (pid_key, _) in self._AXES.items()}
        for p in params:
            parts = p.name.split(".")
            if len(parts) != 2:
                continue
            cfg_key, field = parts
            if cfg_key not in pid_map:
                continue
            pid = self._pid[pid_map[cfg_key]]
            val = float(p.value)
            if   field == "kp":             pid.kp           = val
            elif field == "ki":             pid.ki           = val
            elif field == "kd":             pid.kd           = val
            elif field == "max_integral":   pid.max_integral = val
            elif field == "max_output":     pid.max_output   = val
            elif field == "preset_integral": pid.reset(preset_integral=val)
            self.get_logger().info(f"PID param {p.name} → {val}")
        return SetParametersResult(successful=True)

    def _cb_velocity(self, msg: TwistStamped) -> None:
        raw = np.array([
            msg.twist.linear.x,  msg.twist.linear.y,  msg.twist.linear.z,
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z,
        ])
        if self._vel_alpha > 0.0 and self._v_body_measured is not None:
            self._v_body_measured = self._vel_alpha * self._v_body_measured + (1.0 - self._vel_alpha) * raw
        else:
            self._v_body_measured = raw

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

        if self._v_body_measured is not None:
            v_body = self._v_body_measured
        else:
            # Fallback: różniczkowanie pozycji (głośne — używane gdy brak /auv/velocity)
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

        err_msg = Twist()
        err_msg.linear.x  = float(e[0])
        err_msg.linear.y  = float(e[1])
        err_msg.linear.z  = float(e[2])
        err_msg.angular.x = float(-v_body[3])  # roll: setpoint=0
        err_msg.angular.y = float(-v_body[4])  # pitch: setpoint=0
        err_msg.angular.z = float(e[5])
        self._pub_err.publish(err_msg)


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = VelocityControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
thruster_allocator.py — Alokacja silników: wrench ciała → komendy [N].

Poziom companion computer. Rozwiązuje odwrotny problem alokacji:
    thrusts = pinv(A) @ tau
gdzie A (6×6) to macierz alokacji zbudowana z geometrii silników.
Wynik klipowany do granic siły T200.

  sub: /auv/cmd_wrench   (geometry_msgs/Wrench)           — body frame
  pub: /auv/thruster_cmds (std_msgs/Float64MultiArray, [N]) — 6 silników
"""

import pathlib
import yaml

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64MultiArray


class ThrusterAllocatorNode(Node):
    def __init__(self):
        super().__init__("thruster_allocator")

        self.declare_parameter("thrusters_config", "")
        cfg_path = self.get_parameter("thrusters_config").value
        if not cfg_path:
            import ament_index_python.packages as ament
            share = ament.get_package_share_directory("robot_bringup")
            cfg_path = str(pathlib.Path(share) / "config" / "thrusters.yaml")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)["thrusters"]

        geom = cfg["geometry"]
        n = len(geom)
        A = np.zeros((6, n))
        for t in geom:
            i = t["id"]
            p = np.array(t["position"],  dtype=float)
            d = np.array(t["direction"], dtype=float)
            d /= np.linalg.norm(d)
            A[:3, i] = d
            A[3:, i] = np.cross(p, d)

        self._A_pinv  = np.linalg.pinv(A)
        self._min_thr = float(cfg["limits"]["min_thrust"])
        self._max_thr = float(cfg["limits"]["max_thrust"])

        self._pub = self.create_publisher(Float64MultiArray, "/auv/thruster_cmds", 10)
        self._sub = self.create_subscription(Wrench, "/auv/cmd_wrench", self._cb, 10)

        self.get_logger().info(f"ThrusterAllocator gotowy: {n} silnikow")

    def _cb(self, msg: Wrench) -> None:
        tau = np.array([
            msg.force.x,  msg.force.y,  msg.force.z,
            msg.torque.x, msg.torque.y, msg.torque.z,
        ])
        thrusts = np.clip(self._A_pinv @ tau, self._min_thr, self._max_thr)
        out = Float64MultiArray()
        out.data = thrusts.tolist()
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ThrusterAllocatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

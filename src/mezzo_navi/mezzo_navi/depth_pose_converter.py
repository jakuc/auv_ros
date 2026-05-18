#!/usr/bin/env python3
"""
depth_pose_converter.py — FluidPressure → PoseWithCovarianceStamped (tylko Z).

Przelicza ciśnienie hydrostatyczne na głębokość i publikuje jako pozycję Z
dla EKF. Tylko Z jest ustawione — reszta zerowa z nieskończoną kowariancją.

  sub: /auv/depth      (sensor_msgs/FluidPressure)
  pub: /auv/depth_pose (geometry_msgs/PoseWithCovarianceStamped)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure
from geometry_msgs.msg import PoseWithCovarianceStamped


class DepthPoseConverter(Node):
    def __init__(self):
        super().__init__("depth_pose_converter")

        self.declare_parameter("atm_pressure",   101325.0)
        self.declare_parameter("fluid_density",    1025.0)
        self.declare_parameter("gravity",             9.81)
        self.declare_parameter("z_variance",          0.01)  # [m²]

        self._atm     = self.get_parameter("atm_pressure").value
        self._density = self.get_parameter("fluid_density").value
        self._g       = self.get_parameter("gravity").value
        self._z_var   = self.get_parameter("z_variance").value

        self._sub = self.create_subscription(
            FluidPressure, "/auv/depth", self._cb, 10)
        self._pub = self.create_publisher(
            PoseWithCovarianceStamped, "/auv/depth_pose", 10)

    def _cb(self, msg: FluidPressure) -> None:
        depth = (msg.fluid_pressure - self._atm) / (self._density * self._g)
        z = -depth  # depth > 0 → z < 0

        out = PoseWithCovarianceStamped()
        out.header = msg.header
        out.header.frame_id = "world"
        out.pose.pose.position.z = z
        out.pose.pose.orientation.w = 1.0

        # Kowariancja 6×6 (x,y,z,roll,pitch,yaw): tylko Z znany, reszta ∞
        cov = [0.0] * 36
        cov[0]  = 1e9   # x nieznane
        cov[7]  = 1e9   # y nieznane
        cov[14] = self._z_var
        cov[21] = 1e9   # roll
        cov[28] = 1e9   # pitch
        cov[35] = 1e9   # yaw
        out.pose.covariance = cov

        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = DepthPoseConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

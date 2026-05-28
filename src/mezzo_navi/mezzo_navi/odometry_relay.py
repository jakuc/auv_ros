"""
odometry_relay.py — Konwertuje /auv/odometry (EKF) na /auv/pose i /auv/velocity.

Oddziela warstwę percepcji (EKF) od stosu nawigacji — navi i mezzo_navi
subskrybują standardowe topiki /auv/pose i /auv/velocity nie wiedząc czy
pochodzą z EKF czy z innego źródła.

Parametr use_sim_ground_truth (domyślnie false):
  false → źródło: /auv/odometry (EKF)
  true  → źródło: /auv/sim/pose + /auv/sim/velocity (ground truth z Isaaca)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from tf2_ros import TransformBroadcaster


class OdometryRelay(Node):

    def __init__(self):
        super().__init__("odometry_relay")
        self.declare_parameter("use_sim_ground_truth", False)
        use_sim = self.get_parameter("use_sim_ground_truth").value

        self._pub_pose = self.create_publisher(PoseStamped, "/auv/pose",     10)
        self._pub_vel  = self.create_publisher(TwistStamped, "/auv/velocity", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        if use_sim:
            self.create_subscription(PoseStamped,  "/auv/sim/pose",     self._cb_sim_pose, 10)
            self.create_subscription(TwistStamped, "/auv/sim/velocity", self._cb_sim_vel,  10)
            self.get_logger().info("OdometryRelay: ground truth (sim)")
        else:
            self.create_subscription(Odometry, "/auv/odometry", self._cb_ekf, 10)
            self.get_logger().info("OdometryRelay: EKF")

    def _broadcast_tf(self, pose_msg: PoseStamped) -> None:
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id  = "bluerov2/base_link"
        t.transform.translation.x = pose_msg.pose.position.x
        t.transform.translation.y = pose_msg.pose.position.y
        t.transform.translation.z = pose_msg.pose.position.z
        t.transform.rotation      = pose_msg.pose.orientation
        self._tf_broadcaster.sendTransform(t)

    def _cb_ekf(self, msg: Odometry) -> None:
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = msg.header.stamp
        pose_msg.header.frame_id = msg.header.frame_id
        pose_msg.pose            = msg.pose.pose
        self._pub_pose.publish(pose_msg)
        self._broadcast_tf(pose_msg)

        vel_msg = TwistStamped()
        vel_msg.header.stamp    = msg.header.stamp
        vel_msg.header.frame_id = msg.child_frame_id
        vel_msg.twist           = msg.twist.twist
        self._pub_vel.publish(vel_msg)

    def _cb_sim_pose(self, msg: PoseStamped) -> None:
        self._pub_pose.publish(msg)
        self._broadcast_tf(msg)

    def _cb_sim_vel(self, msg: TwistStamped) -> None:
        self._pub_vel.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

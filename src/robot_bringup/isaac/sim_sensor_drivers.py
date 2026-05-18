"""
sim_sensor_drivers.py — Symulacyjne drivery sensorów nawigacyjnych AUV.

Na prawdziwym robocie zastąpione przez drivery sprzętowe:
  SimAhrsDriver  → xsens_ros_mti_driver (MTi-600)
  SimDvlDriver   → driver WL-DVL lub podobny
  SimDepthDriver → driver Bar30 / Bar100

Wzorzec identyczny z SimThrusterDriver: klasy (nie węzły ROS), działają
wewnątrz procesu isaac_sim.py, tworzą publishery na przekazanym węźle ROS,
czytają stan fizyki bezpośrednio z obiektu Articulation.

Interfejs topikowy jest kontraktem między warstwą sterowników a EKF:
  /auv/ahrs   sensor_msgs/Imu                           (orientacja + ang_vel + accel)
  /auv/dvl    geometry_msgs/TwistWithCovarianceStamped  (prędkość body frame)
  /auv/depth  sensor_msgs/FluidPressure                 (ciśnienie hydrostatyczne)
"""

import math
import time
import numpy as np

from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import FluidPressure, Imu


def _quat_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    q = q_wxyz / np.linalg.norm(q_wxyz)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [    2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x)],
        [    2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)],
    ])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# ---------------------------------------------------------------------------

class SimAhrsDriver:
    """
    Symulacyjny driver AHRS (odpowiednik Xsens MTi-600).

    Publikuje orientację absolutną (kwaternion) + żyroskop + akcelerometr
    z szumem Gaussowskim. Szum yaw rośnie z aktywnością silników — model
    zakłóceń magnetycznych od prądów T200.

    Użycie (w isaac_sim.py):
        ahrs = SimAhrsDriver(ros_node, config["sensors"]["ahrs"])
        # w pętli, co pose_dt:
        ahrs.publish(robot, thruster_forces=sim_driver._thrusts)
    """

    def __init__(self, ros_node, config: dict):
        self._pub = ros_node.create_publisher(Imu, "/auv/ahrs", 10)
        self._node = ros_node

        self._rp_noise    = float(config.get("roll_pitch_noise_stddev", 0.003))
        self._yaw_base    = float(config.get("yaw_noise_base_stddev",   0.008))
        self._yaw_max     = float(config.get("yaw_noise_max_stddev",    0.17))
        self._gyro_noise  = float(config.get("gyro_noise_stddev",       0.003))
        self._accel_noise = float(config.get("accel_noise_stddev",      0.05))
        self._g           = float(config.get("gravity",                 9.81))
        self._thr_max     = float(config.get("thruster_max_force",      30.0))

        self._prev_vel: np.ndarray | None = None
        self._prev_t:   float | None = None
        self._rng = np.random.default_rng()

    def publish(self, robot, thruster_forces: np.ndarray | None = None) -> None:
        position, quat_wxyz = robot.get_world_pose()
        lin_vel_world = robot.get_linear_velocity()
        ang_vel_world = robot.get_angular_velocity()

        quat = np.asarray(quat_wxyz, dtype=float)
        R    = _quat_to_rot(quat)

        lin_vel_body = R.T @ np.asarray(lin_vel_world, dtype=float)
        ang_vel_body = R.T @ np.asarray(ang_vel_world, dtype=float)

        # aktywność silników → skala szumu yaw
        if thruster_forces is not None and len(thruster_forces) > 0:
            activity = float(np.clip(
                np.mean(np.abs(thruster_forces)) / self._thr_max, 0.0, 1.0))
        else:
            activity = 0.0

        yaw_std = self._yaw_base + (self._yaw_max - self._yaw_base) * activity

        # szum orientacji: małe obroty nałożone na GT kwaternion
        rp_err  = self._rng.normal(0.0, self._rp_noise, 2)
        yaw_err = self._rng.normal(0.0, yaw_std)
        q_noise = _quat_multiply(
            _quat_multiply(
                np.array([math.cos(rp_err[0]/2), math.sin(rp_err[0]/2), 0.0, 0.0]),
                np.array([math.cos(rp_err[1]/2), 0.0, math.sin(rp_err[1]/2), 0.0]),
            ),
            _yaw_quat(yaw_err),
        )
        q_meas = _quat_multiply(quat, q_noise)
        q_meas /= np.linalg.norm(q_meas)

        # żyroskop
        gyro = ang_vel_body + self._rng.normal(0.0, self._gyro_noise, 3)

        # siła właściwa: d/dt(v_body) - g_body  [to co mierzy akcelerometr]
        now = time.monotonic()
        if self._prev_vel is not None and self._prev_t is not None:
            dt = now - self._prev_t
            accel_body = (lin_vel_body - self._prev_vel) / dt if dt > 0.0 else np.zeros(3)
        else:
            accel_body = np.zeros(3)
        self._prev_vel = lin_vel_body.copy()
        self._prev_t   = now

        g_body         = R.T @ np.array([0.0, 0.0, -self._g])
        specific_force  = accel_body - g_body
        accel           = specific_force + self._rng.normal(0.0, self._accel_noise, 3)

        rp2  = self._rp_noise ** 2
        yaw2 = yaw_std ** 2
        gn2  = self._gyro_noise  ** 2
        an2  = self._accel_noise ** 2

        msg = Imu()
        msg.header.stamp    = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "bluerov2/ahrs_link"

        msg.orientation.w = float(q_meas[0])
        msg.orientation.x = float(q_meas[1])
        msg.orientation.y = float(q_meas[2])
        msg.orientation.z = float(q_meas[3])
        msg.orientation_covariance = [
            rp2, 0.0, 0.0,
            0.0, rp2, 0.0,
            0.0, 0.0, yaw2,
        ]

        msg.angular_velocity.x = float(gyro[0])
        msg.angular_velocity.y = float(gyro[1])
        msg.angular_velocity.z = float(gyro[2])
        msg.angular_velocity_covariance = [
            gn2, 0.0, 0.0,
            0.0, gn2, 0.0,
            0.0, 0.0, gn2,
        ]

        msg.linear_acceleration.x = float(accel[0])
        msg.linear_acceleration.y = float(accel[1])
        msg.linear_acceleration.z = float(accel[2])
        msg.linear_acceleration_covariance = [
            an2, 0.0, 0.0,
            0.0, an2, 0.0,
            0.0, 0.0, an2,
        ]

        self._pub.publish(msg)


# ---------------------------------------------------------------------------

class SimDvlDriver:
    """
    Symulacyjny driver DVL.

    Publikuje prędkość liniową w układzie ciała z szumem Gaussowskim.
    Na HW zastąpiony przez driver WL-DVL lub podobny (UART/Ethernet).

    Użycie (w isaac_sim.py):
        dvl = SimDvlDriver(ros_node, config["sensors"]["dvl"])
        dvl.publish(robot)
    """

    def __init__(self, ros_node, config: dict):
        self._pub  = ros_node.create_publisher(
            TwistWithCovarianceStamped, "/auv/dvl", 10)
        self._node = ros_node
        self._noise = float(config.get("vel_noise_stddev", 0.02))
        self._rng   = np.random.default_rng()

    def publish(self, robot) -> None:
        _, quat_wxyz  = robot.get_world_pose()
        lin_vel_world = robot.get_linear_velocity()

        R            = _quat_to_rot(np.asarray(quat_wxyz, dtype=float))
        lin_vel_body = R.T @ np.asarray(lin_vel_world, dtype=float)
        noisy        = lin_vel_body + self._rng.normal(0.0, self._noise, 3)
        var          = self._noise ** 2

        msg = TwistWithCovarianceStamped()
        msg.header.stamp    = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "bluerov2/dvl_link"
        msg.twist.twist.linear.x = float(noisy[0])
        msg.twist.twist.linear.y = float(noisy[1])
        msg.twist.twist.linear.z = float(noisy[2])
        cov = [0.0] * 36
        cov[0]  = var
        cov[7]  = var
        cov[14] = var
        msg.twist.covariance = cov
        self._pub.publish(msg)


# ---------------------------------------------------------------------------

class SimDepthDriver:
    """
    Symulacyjny driver czujnika głębokości (odpowiednik Bar30/Bar100).

    Publikuje ciśnienie hydrostatyczne obliczone z pozycji Z + szum.
    Na HW zastąpiony przez driver I2C Bar30 lub Bar100.

    Użycie (w isaac_sim.py):
        depth = SimDepthDriver(ros_node, config["sensors"]["depth"])
        depth.publish(robot)
    """

    def __init__(self, ros_node, config: dict):
        self._pub     = ros_node.create_publisher(FluidPressure, "/auv/depth", 10)
        self._node    = ros_node
        self._density = float(config.get("fluid_density",        1025.0))
        self._noise   = float(config.get("pressure_noise_stddev",  50.0))
        self._atm     = float(config.get("atm_pressure",        101325.0))
        self._g       = 9.81
        self._rng     = np.random.default_rng()

    def publish(self, robot) -> None:
        position, _ = robot.get_world_pose()
        z           = float(position[2])

        depth            = max(0.0, -z)
        pressure_true    = self._atm + self._density * self._g * depth
        pressure_meas    = pressure_true + self._rng.normal(0.0, self._noise)

        msg = FluidPressure()
        msg.header.stamp    = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "bluerov2/depth_link"
        msg.fluid_pressure  = pressure_meas
        msg.variance        = self._noise ** 2
        self._pub.publish(msg)

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

Modele błędów:
  Każdy sensor ma trzy składowe błędu:
    1. Szum biały (Gaussowski) — chwilowy, nieskorelowany
    2. Bias startowy (kalibracyjny) — losowany przy starcie, stały w sesji
    3. Random walk na biasie — powolny dryf przez cały czas działania
"""

import math
import time
import numpy as np

from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import FluidPressure, Imu, PointCloud2, PointField


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

class _BiasProcess:
    """
    Bias z losowym błądzeniem (random walk).

    Modeluje błąd kalibracyjny sensora: przy starcie losowany z rozkładu
    normalnego (niepewność kalibracji), potem powoli dryfuje w czasie.
    """

    def __init__(self, initial_std: float, walk_std_per_sqrt_s: float, rng):
        self._bias = rng.normal(0.0, initial_std)
        self._walk = walk_std_per_sqrt_s
        self._rng  = rng
        self._last_t = time.monotonic()

    @property
    def value(self) -> float:
        return self._bias

    def step(self) -> float:
        now = time.monotonic()
        dt  = max(now - self._last_t, 1e-6)
        self._last_t = now
        if self._walk > 0.0:
            self._bias += self._rng.normal(0.0, self._walk * math.sqrt(dt))
        return self._bias


# ---------------------------------------------------------------------------

class SimAhrsDriver:
    """
    Symulacyjny driver AHRS (odpowiednik Xsens MTi-600).

    Modele błędów:
      - Szum biały roll/pitch (spec MTi-600: ~0.2°)
      - Szum biały yaw adaptacyjny do aktywności silników (zakłócenia mag.)
      - Dryf yaw: random walk niezależny od silników (residual po magnetometrze)
      - Szum biały żyroskopu + bias startowy + random walk

    Użycie (w isaac_sim.py):
        ahrs = SimAhrsDriver(ros_node, config["sensors"]["ahrs"])
        ahrs.publish(robot, thruster_forces=sim_driver._thrusts)
    """

    def __init__(self, ros_node, config: dict):
        self._pub  = ros_node.create_publisher(Imu, "/auv/ahrs", 10)
        self._node = ros_node
        self._rng  = np.random.default_rng()

        self._rp_noise    = float(config.get("roll_pitch_noise_stddev", 0.003))
        self._yaw_base    = float(config.get("yaw_noise_base_stddev",   0.008))
        self._yaw_max     = float(config.get("yaw_noise_max_stddev",    0.17))
        self._gyro_noise  = float(config.get("gyro_noise_stddev",       0.003))
        self._accel_noise = float(config.get("accel_noise_stddev",      0.05))
        self._g           = float(config.get("gravity",                 9.81))
        self._thr_max     = float(config.get("thruster_max_force",      30.0))
        self._param_name  = "ahrs_enable_bias"

        self._yaw_drift = _BiasProcess(
            0.0, float(config.get("yaw_drift_walk_std", 0.0003)), self._rng)
        self._gyro_bias = [
            _BiasProcess(
                float(config.get("gyro_bias_initial_std", 0.001)),
                float(config.get("gyro_bias_walk_std",    0.0001)),
                self._rng)
            for _ in range(3)
        ]

        self._prev_vel: np.ndarray | None = None
        self._prev_t:   float | None = None

    def publish(self, robot, thruster_forces: np.ndarray | None = None) -> None:
        position, quat_wxyz = robot.get_world_pose()
        lin_vel_world = robot.get_linear_velocity()
        ang_vel_world = robot.get_angular_velocity()

        quat = np.asarray(quat_wxyz, dtype=float)
        R    = _quat_to_rot(quat)

        lin_vel_body = R.T @ np.asarray(lin_vel_world, dtype=float)
        ang_vel_body = R.T @ np.asarray(ang_vel_world, dtype=float)

        # Aktywność silników → skala szumu yaw od zakłóceń magnetycznych
        if thruster_forces is not None and len(thruster_forces) > 0:
            activity = float(np.clip(
                np.mean(np.abs(thruster_forces)) / self._thr_max, 0.0, 1.0))
        else:
            activity = 0.0

        yaw_noise_std = self._yaw_base + (self._yaw_max - self._yaw_base) * activity

        enable_bias = self._node.get_parameter(self._param_name).value

        # Orientacja: szum biały + dryf yaw
        rp_err    = self._rng.normal(0.0, self._rp_noise, 2)
        yaw_noise = self._rng.normal(0.0, yaw_noise_std)
        yaw_drift = self._yaw_drift.step() if enable_bias else 0.0
        yaw_err   = yaw_noise + yaw_drift

        q_noise = _quat_multiply(
            _quat_multiply(
                np.array([math.cos(rp_err[0]/2), math.sin(rp_err[0]/2), 0.0, 0.0]),
                np.array([math.cos(rp_err[1]/2), 0.0, math.sin(rp_err[1]/2), 0.0]),
            ),
            _yaw_quat(yaw_err),
        )
        q_meas = _quat_multiply(quat, q_noise)
        q_meas /= np.linalg.norm(q_meas)

        # Żyroskop: szum biały + bias z random walk
        gyro_biases = np.array([b.step() for b in self._gyro_bias]) if enable_bias else np.zeros(3)
        gyro = ang_vel_body + self._rng.normal(0.0, self._gyro_noise, 3) + gyro_biases

        # Akcelerometr
        now = time.monotonic()
        if self._prev_vel is not None and self._prev_t is not None:
            dt = now - self._prev_t
            accel_body = (lin_vel_body - self._prev_vel) / dt if dt > 0.0 else np.zeros(3)
        else:
            accel_body = np.zeros(3)
        self._prev_vel = lin_vel_body.copy()
        self._prev_t   = now

        g_body        = R.T @ np.array([0.0, 0.0, -self._g])
        specific_force = accel_body - g_body
        accel          = specific_force + self._rng.normal(0.0, self._accel_noise, 3)

        # Kowariancje — uwzględniają łączny szum (biały + dryf)
        yaw_total_std = math.sqrt(yaw_noise_std**2 + self._yaw_drift.value**2 + 1e-12)
        rp2  = self._rp_noise ** 2
        yaw2 = yaw_total_std  ** 2
        gn2  = (self._gyro_noise + abs(gyro_biases).mean()) ** 2
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
    Symulacyjny driver DVL (Water Linked DVL A50 Performance).

    Modele błędów wg datasheet DVL A50:
      - Błąd skali: ±0.1% prędkości (dominujący błąd DVL — zależy od prędkości dźwięku)
      - Szum biały: ±1 mm/s addytywny (niezależny od prędkości)
      - Bias kalibracyjny per-axis losowany przy starcie
      - Random walk na biasie (powolny dryf — temperatura zmienia prędkość dźwięku)

    Błąd skali modeluje niepewność prędkości dźwięku w wodzie: zmiana temperatury
    o 1°C → zmiana prędkości dźwięku ~3 m/s → błąd skali ~0.2%.

    Użycie (w isaac_sim.py):
        dvl = SimDvlDriver(ros_node, config["sensors"]["dvl"])
        dvl.publish(robot)
    """

    def __init__(self, ros_node, config: dict):
        self._pub  = ros_node.create_publisher(
            TwistWithCovarianceStamped, "/auv/dvl", 10)
        self._node = ros_node
        self._rng  = np.random.default_rng()

        self._noise       = float(config.get("vel_noise_stddev",    0.001))
        self._scale_std   = float(config.get("vel_scale_error_std", 0.001))
        self._scale_error = self._rng.normal(0.0, self._scale_std, 3)
        self._bias        = [
            _BiasProcess(
                float(config.get("vel_bias_initial_std", 0.001)),
                float(config.get("vel_bias_walk_std",    0.0001)),
                self._rng)
            for _ in range(3)
        ]
        self._param_name  = "dvl_enable_bias"

    def publish(self, robot) -> None:
        _, quat_wxyz  = robot.get_world_pose()
        lin_vel_world = robot.get_linear_velocity()

        R            = _quat_to_rot(np.asarray(quat_wxyz, dtype=float))
        lin_vel_body = R.T @ np.asarray(lin_vel_world, dtype=float)

        enable_bias = self._node.get_parameter(self._param_name).value
        biases      = np.array([b.step() for b in self._bias]) if enable_bias else np.zeros(3)
        scale_error = self._scale_error if enable_bias else np.zeros(3)

        # v_meas = v_true * (1 + scale_error) + noise + bias
        noisy  = lin_vel_body * (1.0 + scale_error) \
                 + self._rng.normal(0.0, self._noise, 3) \
                 + biases

        # Kowariancja: szum addytywny + składowa od błędu skali (zależy od prędkości)
        scale_var = (self._scale_std * lin_vel_body) ** 2 if enable_bias else np.zeros(3)
        var = self._noise**2 + scale_var + biases**2

        msg = TwistWithCovarianceStamped()
        msg.header.stamp    = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "bluerov2/base_link"
        msg.twist.twist.linear.x = float(noisy[0])
        msg.twist.twist.linear.y = float(noisy[1])
        msg.twist.twist.linear.z = float(noisy[2])
        cov = [0.0] * 36
        cov[0]  = float(var[0])
        cov[7]  = float(var[1])
        cov[14] = float(var[2])
        msg.twist.covariance = cov
        self._pub.publish(msg)


# ---------------------------------------------------------------------------

class SimDepthDriver:
    """
    Symulacyjny driver czujnika głębokości (odpowiednik Bar30/Bar100).

    Modele błędów:
      - Szum biały ciśnienia (Gaussowski)
      - Bias kalibracyjny losowany przy starcie
      - Random walk na biasie (dryf temperatury/kalibracji)

    Użycie (w isaac_sim.py):
        depth = SimDepthDriver(ros_node, config["sensors"]["depth"])
        depth.publish(robot)
    """

    def __init__(self, ros_node, config: dict):
        self._pub     = ros_node.create_publisher(FluidPressure, "/auv/depth", 10)
        self._node    = ros_node
        self._rng     = np.random.default_rng()

        self._density    = float(config.get("fluid_density",         1025.0))
        self._noise      = float(config.get("pressure_noise_stddev",   50.0))
        self._atm        = float(config.get("atm_pressure",        101325.0))
        self._g          = 9.81
        self._bias       = _BiasProcess(
            float(config.get("pressure_bias_initial_std", 20.0)),
            float(config.get("pressure_bias_walk_std",     0.5)),
            self._rng)
        self._param_name = "depth_enable_bias"

    def publish(self, robot) -> None:
        position, _ = robot.get_world_pose()
        z           = float(position[2])

        enable_bias   = self._node.get_parameter(self._param_name).value
        depth         = max(0.0, -z)
        pressure_true = self._atm + self._density * self._g * depth
        bias          = self._bias.step() if enable_bias else 0.0
        pressure_meas = pressure_true + self._rng.normal(0.0, self._noise) + bias

        total_var = self._noise**2 + (self._bias.value**2 if enable_bias else 0.0)

        msg = FluidPressure()
        msg.header.stamp    = self._node.get_clock().now().to_msg()
        msg.header.frame_id = "bluerov2/depth_link"
        msg.fluid_pressure  = pressure_meas
        msg.variance        = total_var
        self._pub.publish(msg)


# ---------------------------------------------------------------------------

class SimLidarDriver:
    """
    Idealny symulacyjny LiDAR 3D (pełna sfera) oparty na ray castingu PhysX.

    Substytut docelowego sensora galwanometrycznego — brak modelu błędów,
    dane idealne do testów SLAM.

    Użycie (w isaac_sim.py):
        lidar = SimLidarDriver(ros_node, config["sensors"]["lidar"])
        lidar.publish(robot)
    """

    def __init__(self, ros_node, config: dict):
        self._pub       = ros_node.create_publisher(PointCloud2, "/auv/lidar", 10)
        self._node      = ros_node
        self._max_range = float(config.get("max_range_m",          30.0))
        self._min_range = float(config.get("min_range_m",           0.5))
        self._frame_id  = str(config.get("frame_id", "bluerov2/base_link"))

        az_step = float(config.get("azimuth_step_deg",   2.0))
        el_step = float(config.get("elevation_step_deg", 2.0))
        self._ray_dirs = self._build_ray_dirs(az_step, el_step)

    @staticmethod
    def _build_ray_dirs(az_deg: float, el_deg: float) -> np.ndarray:
        azs = np.radians(np.arange(0.0, 360.0, az_deg))
        els = np.radians(np.arange(-90.0 + el_deg, 90.0, el_deg))
        az_grid, el_grid = np.meshgrid(azs, els)
        x = np.cos(el_grid) * np.cos(az_grid)
        y = np.cos(el_grid) * np.sin(az_grid)
        z = np.sin(el_grid)
        return np.stack([x, y, z], axis=-1).reshape(-1, 3)

    def publish(self, robot) -> None:
        from omni.physx import get_physx_scene_query_interface

        position, quat_wxyz = robot.get_world_pose()
        R      = _quat_to_rot(np.asarray(quat_wxyz, dtype=float))
        origin = np.asarray(position, dtype=float)

        dirs_world = (R @ self._ray_dirs.T).T

        physx    = get_physx_scene_query_interface()
        pts_body = []
        n_miss   = 0

        for dir_w in dirs_world:
            hit = physx.raycast_closest(tuple(origin), tuple(dir_w), self._max_range)
            if not (hit and hit["hit"]):
                n_miss += 1
                continue
            if float(hit["distance"]) < self._min_range:
                continue
            pt_world = np.asarray(hit["position"], dtype=float)
            pts_body.append(R.T @ (pt_world - origin))

        if not hasattr(self, "_dbg_printed"):
            self._dbg_printed = True
            pos = [round(float(x), 2) for x in origin]
            print(f"[lidar-dbg] robot pos={pos}  hits={len(pts_body)}  misses={n_miss}  total_rays={len(dirs_world)}")

        if not pts_body:
            return

        pts = np.array(pts_body, dtype=np.float32)

        msg = PointCloud2()
        msg.header.stamp    = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height    = 1
        msg.width     = len(pts)
        msg.fields    = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12
        msg.row_step     = 12 * len(pts)
        msg.is_dense     = True
        msg.data         = pts.tobytes()

        self._pub.publish(msg)

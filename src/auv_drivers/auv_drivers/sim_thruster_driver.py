"""
sim_thruster_driver.py — Symulacyjny odpowiednik sterownika ESC dla Isaac Sim.

W rzeczywistym robocie tę rolę pełni esc_driver.py uruchomiony na companion
computerze: odbiera komendy [N], przelicza je na PWM przez krzywą T200
i wysyła do ESC przez UART/serial.

Tutaj: ta sama granica ROS (/auv/thruster_cmds, Float64MultiArray, [N]),
ale zamiast ESC — przyłożenie sił do fizyki PhysX przez tensor API Isaac Sim.

Plugin NIE jest węzłem ROS — działa wewnątrz procesu isaac_sim.py.
Subskrypcja ROS aktualizuje bufor komend przez set_thrusts(); step(dt)
jest wywoływane jako physics callback przez world.add_physics_callback().
"""

import threading
import numpy as np


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] → macierz obrotu R (ciało → świat)."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [    2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x)],
        [    2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)],
    ], dtype=float)


class SimThrusterDriver:
    """
    Symulacyjny sterownik silników T200.

    Użycie (w isaac_sim.py):
        driver = SimThrusterDriver(thruster_config)
        driver.initialize(robot, robot_prim_path)
        world.add_physics_callback("sim_thruster_driver", driver.step)
        # W callbacku ROS:
        driver.set_thrusts(np.array(msg.data))
    """

    def __init__(self, config: dict):
        geom = config["thrusters"]["geometry"]
        n    = len(geom)

        # Macierz alokacji A (6×n): kolumna i = [d_i ; p_i × d_i]
        A = np.zeros((6, n))
        for t in geom:
            i = t["id"]
            p = np.array(t["position"],  dtype=float)
            d = np.array(t["direction"], dtype=float)
            d /= np.linalg.norm(d)
            A[:3, i] = d
            A[3:, i] = np.cross(p, d)

        self._A      = A
        self._n      = n
        self._lock   = threading.Lock()
        self._thrusts = np.zeros(n)

        self._robot    = None
        self._sim_view = None   # trzymamy referencję — inaczej GC zbierze view
        self._phx_art  = None
        self._n_links  = 0
        self._phx_idx  = None

    def initialize(self, robot, robot_path: str) -> None:
        """Inicjalizuj tensor API. Wywołaj po world.reset()."""
        self._robot = robot

        import omni.physics.tensors as phx_tensor
        self._sim_view = phx_tensor.create_simulation_view("numpy")
        self._sim_view.set_subspace_roots("/")
        self._phx_art = self._sim_view.create_articulation_view(
            f"{robot_path}/bluerov2_base_link"
        )
        self._n_links = self._phx_art.max_links
        self._phx_idx = np.array([0], dtype=np.uint32)
        print(f"[sim_thruster_driver] {self._n} silnikow, {self._n_links} linkow")

    def set_thrusts(self, thrusts: np.ndarray) -> None:
        """Aktualizuj komendy [N]. Wywoływane z ROS callback (inny wątek)."""
        with self._lock:
            n = min(len(thrusts), self._n)
            self._thrusts[:n] = thrusts[:n]

    def step(self, dt: float) -> None:
        """Physics callback — przyłącza siły silników do PhysX."""
        if self._phx_art is None:
            return

        with self._lock:
            thrusts = self._thrusts.copy()

        # Wrench w układzie ciała
        w        = self._A @ thrusts
        f_body   = w[:3]
        tau_body = w[3:]

        # Transformacja do układu świata (PhysX is_global=True)
        _, quat = self._robot.get_world_pose()
        R        = _quat_to_rot(np.asarray(quat, dtype=float))
        f_world   = R @ f_body
        tau_world = R @ tau_body

        forces  = np.zeros((1, self._n_links, 3), dtype=np.float32)
        torques = np.zeros((1, self._n_links, 3), dtype=np.float32)
        forces [0, 0, :] = f_world
        torques[0, 0, :] = tau_world
        self._phx_art.apply_forces_and_torques_at_position(
            forces, torques, None, self._phx_idx, True
        )

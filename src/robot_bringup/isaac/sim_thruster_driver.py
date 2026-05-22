"""
sim_thruster_driver.py — Symulacyjny odpowiednik sterownika ESC dla Isaac Sim.

W rzeczywistym robocie tę rolę pełni esc_driver.py uruchomiony na companion
computerze: odbiera komendy [N], przelicza je na PWM przez krzywą T200
i wysyła do ESC przez UART/serial.

Tutaj: ta sama granica ROS (/auv/thruster_cmds, Float64MultiArray, [N]),
ale zamiast ESC — przyłożenie sił do fizyki PhysX przez tensor API Isaac Sim.

Plugin NIE jest węzłem ROS — działa wewnątrz procesu isaac_sim.py.
Subskrypcja ROS aktualizuje bufor komend przez set_thrusts(); siły obliczane
przez compute_wrench_world() i aplikowane przez FossenPlugin.step() razem
z hydrodynamiką w jednym wywołaniu apply_forces_and_torques_at_position.
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
        fossen.initialize(robot, driver.get_phx_art(), driver)
        world.add_physics_callback("fossen_hydrodynamics", fossen.step)
        # Fossen łączy siły silników (driver.compute_wrench_world()) z hydrodynamiką
        # w jednym wywołaniu apply_forces — brak osobnego callbacka dla silnikow.
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
        """Inicjalizuj tensor API. Wywołaj po world.reset().

        Tworzy jedyny SimulationView w procesie — przekaż phx_art do FossenPlugin
        przez get_phx_art(), żeby uniknąć tworzenia drugiego view (unieważnia pierwsze).
        """
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

    def get_phx_art(self):
        """Zwróć współdzielony ArticulationView — używany przez FossenPlugin."""
        return self._phx_art

    def set_thrusts(self, thrusts: np.ndarray) -> None:
        """Aktualizuj komendy [N]. Wywoływane z ROS callback (inny wątek)."""
        with self._lock:
            n = min(len(thrusts), self._n)
            self._thrusts[:n] = thrusts[:n]

    def compute_wrench_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Oblicza wrench silników w układzie świata (bez aplikowania do PhysX).

        Wywoływane z FossenPlugin.step() — siły silników i hydrodynamiki łączone
        w jednym wywołaniu apply_forces_and_torques_at_position, żeby uniknąć
        problemu nadpisywania przy dwóch osobnych callbackach.
        """
        with self._lock:
            thrusts = self._thrusts.copy()

        w        = self._A @ thrusts
        f_body   = w[:3]
        tau_body = w[3:]

        _, quat = self._robot.get_world_pose()
        R        = _quat_to_rot(np.asarray(quat, dtype=float))
        return R @ f_body, R @ tau_body

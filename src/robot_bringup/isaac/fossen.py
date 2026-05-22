"""
fossen.py — Plugin hydrodynamiki 6-DOF dla Isaac Sim.

Implementuje równanie Fossena:
  (M_RB + M_A) * v_dot + C_A(v)*v + D(v)*v + g(eta) = tau

Isaac Sim obsługuje M_RB i C_RB wewnętrznie (PhysX). Plugin dodaje:
  - siłę wyporu + moment prostujący  [g(eta)]
  - efekt dodanej masy              [-M_A * a_body]
  - Coriolisa dodanej masy          [-C_A(v) * v]
  - tłumienie liniowe + kwadratowe  [-D(v) * v]

Konwencja układu ciała: X-przód, Y-lewo, Z-góra (zgodnie z Isaac Sim / ENU).
Grawitacja obsługiwana przez Isaac Sim — plugin nakłada jedynie pełną siłę wyporu B.
Wszystkie obliczenia w układzie ciała, wyniki transformowane do układu świata.
"""

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    """Macierz antysymetryczna: skew(a) @ b == a × b."""
    x, y, z = v
    return np.array([[ 0, -z,  y],
                     [ z,  0, -x],
                     [-y,  x,  0]], dtype=float)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Kwaternion [w, x, y, z] → macierz obrotu R (ciało → świat)."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def _quat_diff_omega(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    """Prędkość kątowa [rad/s] w układzie świata z różnicy kwaternionów.

    ω = 2 * Im(q_curr ⊗ q_prev⁻¹) / dt
    Dla małych dt przybliżenie numeryczne jest stabilne.
    """
    w1, x1, y1, z1 = q_prev
    w2, x2, y2, z2 = q_curr
    # Im(q_curr ⊗ q_prev⁻¹)  gdzie q_prev⁻¹ = [w1, -x1, -y1, -z1]
    dx = -w2*x1 + x2*w1 - y2*z1 + z2*y1
    dy = -w2*y1 + x2*z1 + y2*w1 - z2*x1
    dz = -w2*z1 - x2*y1 + y2*x1 + z2*w1
    return 2.0 * np.array([dx, dy, dz]) / max(dt, 1e-6)


class FossenPlugin:
    """
    Wtyczka hydrodynamiki Fossena dla swobodnie pływającego AUV-a.

    Użycie:
        plugin = FossenPlugin(robot_prim_path, config)
        plugin.initialize(robot)     # po world.reset(), robot = Articulation
        # w pętli:
        plugin.step(physics_dt)      # przed world.step()
    """

    def __init__(self, robot_prim_path: str, config: dict):
        p = config["fossen"]

        self._g        = float(p["gravity"])
        self._mass     = float(p["mass"])
        self._water_z  = float(p.get("water_surface_z", 0.0))
        self._enable_surface_check = bool(p.get("enable_surface_check", True))
        self._enable_buoyancy      = bool(p.get("enable_buoyancy", True))
        self._draft_z      = float(p.get("draft_z", 0.25))
        self._keel_offset  = float(p.get("keel_from_origin", self._draft_z))
        excess         = float(p["buoyancy_excess_kg"])
        # Pełna siła wyporu — Isaac Sim osobno nakłada grawitację -W.
        # Netto: B + (-W) = excess*g [N] w górę.
        self._B    = (self._mass + excess) * self._g

        self._cob  = np.array(p["cob_offset"], dtype=float)  # [m], układ ciała

        am = np.array(p["added_mass"],        dtype=float)
        dl = np.array(p["linear_damping"],    dtype=float)
        dq = np.array(p["quadratic_damping"], dtype=float)

        self._M_A = np.diag(am)  # 6×6 — dodana masa
        self._D_l = np.diag(dl)  # 6×6 — tłumienie liniowe
        self._D_q = np.diag(dq)  # 6×6 — tłumienie kwadratowe

        self._robot_path   = robot_prim_path
        self._robot        = None
        self._first_step   = True

    def initialize(self, robot, phx_art, thruster_driver=None) -> None:
        """Inicjalizuj plugin. Wywołaj po world.reset().

        robot            — Articulation (do odczytu pose/velocity)
        phx_art          — ArticulationView z jedynego SimulationView w procesie
        thruster_driver  — SimThrusterDriver; gdy podany, Fossen łączy siły silników
                           z hydrodynamiką w jednym wywołaniu apply_forces (unika nadpisywania)
        """
        self._robot      = robot
        self._phx_art    = phx_art
        self._thr_driver = thruster_driver
        self._phx_idx    = np.array([0], dtype=np.uint32)
        self._n_links    = phx_art.max_links
        print(f"[fossen] articulation: count={phx_art.count}, max_links={self._n_links}")

    # ------------------------------------------------------------------

    def _apply_wrench(self, force: np.ndarray, torque: np.ndarray) -> None:
        """Przykłada siłę i moment (układ świata) do bazy articulation.

        API wymaga kształtu (count, max_links, 3). Nakładamy siły tylko na
        link 0 (base_link), pozostałe linki (thrustery) = 0.
        """
        forces  = np.zeros((1, self._n_links, 3), dtype=np.float32)
        torques = np.zeros((1, self._n_links, 3), dtype=np.float32)
        forces [0, 0, :] = force
        torques[0, 0, :] = torque
        try:
            self._phx_art.apply_forces_and_torques_at_position(
                forces,
                torques,
                None,
                self._phx_idx,
                True,
            )
        except Exception as e:
            print(f"[fossen] _apply_wrench ERROR: {e}")

    # ------------------------------------------------------------------
    # Siły hydrodynamiczne w układzie ciała

    def _coriolis_added_mass_force(self, v: np.ndarray) -> np.ndarray:
        """
        -C_A(v) * v  (Fossen 2011, eq. 6.67) dla diagonalnej M_A.

        C_A = [  0₃,        -S(A * v₁) ]
              [ S(A * v₁),  -S(B * v₂) ]

        A = diag(X_du, Y_dv, Z_dw),  B = diag(K_dp, M_dq, N_dr)
        v₁ = [u,v,w],  v₂ = [p,q,r]
        """
        a = self._M_A.diagonal()[:3]
        b = self._M_A.diagonal()[3:]
        v1, v2 = v[:3], v[3:]

        S_av1 = _skew(a * v1)
        S_bv2 = _skew(b * v2)

        C_A = np.zeros((6, 6))
        C_A[:3, 3:] = -S_av1
        C_A[3:, :3] =  S_av1
        C_A[3:, 3:] = -S_bv2

        return -(C_A @ v)

    def _damping_force(self, v: np.ndarray) -> np.ndarray:
        """-(D_l + D_q * |v|) * v — tłumienie liniowe + kwadratowe."""
        return -(self._D_l @ v + self._D_q @ (np.abs(v) * v))

    def _added_mass_force(self, a: np.ndarray) -> np.ndarray:
        """-M_A * a_body — wirtualna inercja dodanej masy."""
        return -(self._M_A @ a)

    def _buoyancy_force_and_torque(self, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Siła wyporu B (w górę) i moment prostujący, w układzie ciała.

        Moment: r_CoB × F_wyporu — przy przechyle tworzy moment powrotny.
        Grawitacja (-W) obsługiwana przez Isaac Sim; nakładamy tylko +B.
        """
        Rt = R.T
        f_b_body = Rt @ np.array([0.0, 0.0, self._B])
        tau_body = np.cross(self._cob, f_b_body)
        return f_b_body, tau_body

    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """
        Jeden krok hydrodynamiki. Wywołaj przed world.step().

        Prędkości czytane bezpośrednio z PhysX (per-substep) — nie z USD,
        które aktualizuje się tylko co render_dt i dawało 5× spiki siły tłumienia.
        """
        if self._robot is None:
            raise RuntimeError("FossenPlugin.initialize(robot) nie zostało wywołane")

        # 1. Odczyt stanu
        position, quat  = self._robot.get_world_pose()
        position        = np.asarray(position, dtype=float)
        quat            = np.asarray(quat,     dtype=float)   # [w, x, y, z]
        v_lin_world     = np.asarray(self._robot.get_linear_velocity(),  dtype=float)
        v_ang_world     = np.asarray(self._robot.get_angular_velocity(), dtype=float)

        # 2. Pierwszy krok — nie przykładaj sił
        if self._first_step:
            self._first_step = False
            return

        # Ułamek zanurzonej objętości: 0 = całkowicie nad wodą, 1 = całkowicie pod wodą
        if self._enable_surface_check:
            submerged = float(np.clip(
                (self._water_z - position[2] + self._keel_offset) / self._draft_z,
                0.0, 1.0,
            ))
        else:
            submerged = 1.0

        R  = _quat_to_rot(quat)
        Rt = R.T

        # 4. Prędkości w układzie ciała
        v_body = np.concatenate([Rt @ v_lin_world, Rt @ v_ang_world])

        # 5. Składniki sił/momentów w układzie ciała (skalowane przez ułamek zanurzenia)
        f_ca        = self._coriolis_added_mass_force(v_body) * submerged
        f_d         = self._damping_force(v_body)             * submerged
        if self._enable_buoyancy:
            f_b, tau_b = self._buoyancy_force_and_torque(R)
            f_b   *= submerged
            tau_b *= submerged
        else:
            f_b   = np.zeros(3)
            tau_b = np.zeros(3)

        f_body   = f_ca[:3] + f_d[:3] + f_b
        tau_body = f_ca[3:] + f_d[3:] + tau_b

        # 7. Transformacja do układu świata
        f_world   = R @ f_body
        tau_world = R @ tau_body

        # 8. Dodaj siły silników (jeśli driver podany) — jeden zbiorczy apply
        # Skalowane przez submerged: silniki T200 potrzebują wody, nad powierzchnią = 0
        if self._thr_driver is not None:
            thr_f, thr_tau = self._thr_driver.compute_wrench_world()
            f_world   = f_world   + thr_f   * submerged
            tau_world = tau_world + thr_tau * submerged

        self._apply_wrench(f_world, tau_world)

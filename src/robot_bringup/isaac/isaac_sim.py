#!/usr/bin/env python3
"""
isaac_sim.py – Symulacja AUV BlueROV2 w Isaac Sim.

Uruchomienie (w kontenerze):
    OMNI_KIT_ALLOW_ROOT=1 python3 src/robot_bringup/isaac/isaac_sim.py

Równolegle uruchom węzły ROS2:
    ros2 launch robot_bringup isaac.launch.py
    ros2 launch navi navi.launch.py
    ros2 launch mezzo_navi mezzo_navi.launch.py

Publikuje:
    /auv/sim/pose      (geometry_msgs/PoseStamped)   – ground truth pozycja i orientacja, 50 Hz
    /auv/sim/velocity  (geometry_msgs/TwistStamped)  – ground truth prędkość w układzie ciała, 50 Hz

Subskrybuje:
    /auv/thruster_cmds (std_msgs/Float64MultiArray) – komendy silników [N] z mezzo_navi

Architektura sprzętu symulowanego:
    navi → mezzo_navi → /auv/thruster_cmds → [SimThrusterDriver] → PhysX
                                              ↑ w rzeczywistości:
                                              EscDriverNode (auv_drivers) → serial → ESC → T200
"""

import signal
import sys
import time
import pathlib
import yaml

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "renderer": "RayTracedLighting",
})

import omni.kit.commands
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation, ArticulationView
from omni.isaac.core.prims import XFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64MultiArray


def _quat_to_rot(q) -> np.ndarray:
    """Kwaternion [w, x, y, z] → macierz rotacji 3×3 (world←body)."""
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [    2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x)],
        [    2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)],
    ])

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fossen import FossenPlugin

from sim_thruster_driver import SimThrusterDriver
from sim_sensor_drivers import SimAhrsDriver, SimDvlDriver, SimDepthDriver, SimLidarDriver

# ---------------------------------------------------------------------------
URDF_PATH = "/tmp/bluerov2.urdf"  # generowany przez sim_robot.sh przed startem

_THIS_DIR        = pathlib.Path(__file__).parent
_FOSSEN_CONFIG   = _THIS_DIR.parent / "config" / "fossen.yaml"
_THRUSTER_CONFIG = _THIS_DIR.parent / "config" / "thrusters.yaml"
_SENSOR_CONFIG   = _THIS_DIR.parent / "config" / "sensors.yaml"
_WORLD_CONFIG    = _THIS_DIR.parent / "config" / "world.yaml"

ROBOT_START_Z  = 0.0   # [m]
POSE_RATE_HZ   = 50.0


# ---------------------------------------------------------------------------

def import_urdf(urdf_path: str) -> str:
    """Importuje URDF do aktualnego stage. Zwraca prim_path robota."""
    try:
        from omni.importer.urdf import _urdf as urdf_mod
    except ImportError:
        from isaacsim.asset.importer.urdf import _urdf as urdf_mod

    cfg = urdf_mod.ImportConfig()
    cfg.merge_fixed_joints    = False
    cfg.fix_base              = False   # AUV swobodnie pływa
    cfg.import_inertia_tensor = True
    cfg.distance_scale        = 1.0

    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=cfg,
    )

    return _find_robot_prim_path()


def _find_robot_prim_path() -> str:
    stage = omni.usd.get_context().get_stage()
    for candidate in ["/World/bluerov2", "/bluerov2"]:
        if stage.GetPrimAtPath(candidate).IsValid():
            return candidate
    world = stage.GetPrimAtPath("/World")
    if world.IsValid():
        for child in world.GetChildren():
            return child.GetPath().pathString
    return "/World/bluerov2"


# ---------------------------------------------------------------------------

def ensure_collision_api(stage, robot_prim_path: str) -> None:
    """Zapewnia geometrię kolizji na base_link.

    Isaac Sim 4.5 tworzy kolizje z URDF automatycznie (collisions/mesh_0/box).
    Jeśli już istnieją — pomijamy, żeby nie unieważnić tensor view przez
    usunięcie/podmianę primu zarządzanego przez PhysX.
    Fallback (stary Isaac Sim / brak kolizji): de-instancing + własny box.
    """
    collisions_prim = stage.GetPrimAtPath(
        f"{robot_prim_path}/bluerov2_base_link/collisions"
    )
    if collisions_prim.IsValid() and any(collisions_prim.GetChildren()):
        print("[isaac_sim] Collision geometry found from URDF importer — skipping custom collider")
        return

    # Fallback: importer nie stworzył kolizji — de-instancing + box.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)

    # NIE dodajemy RigidBodyAPI — link articulation jest już rigid body
    # przez articulation solver. Dodanie RigidBodyAPI tworzy drugi aktor
    # PhysX dla tego samego primu i blokuje apply_forces w tensor API.

    box_path = f"{robot_prim_path}/bluerov2_base_link/collisions/box"
    if not stage.GetPrimAtPath(box_path).IsValid():
        sx, sy, sz = 0.4576, 0.3442, 0.2552
        box = UsdGeom.Cube.Define(stage, box_path)
        box.GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(box.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(-0.01, 0.0, -sz / 2 + 0.07))
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        UsdGeom.Imageable(box.GetPrim()).MakeInvisible()
    print("[isaac_sim] Collision box created (URDF importer fallback)")


def add_rtx_lidar(robot_prim_path: str):
    """Tworzy RTX LiDAR prim. initialize() i add_point_cloud_data_to_frame()
    muszą być wywołane PÓŹNIEJ (po sim_driver.initialize()) z istniejącym sim view."""
    from omni.isaac.sensor import LidarRtx

    stage  = omni.usd.get_context().get_stage()
    parent = robot_prim_path
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
        if "base_link" in prim.GetName():
            parent = prim.GetPath().pathString
            break

    lidar_path = f"{parent}/Lidar"
    existing = stage.GetPrimAtPath(lidar_path)
    if existing.IsValid():
        stage.RemovePrim(lidar_path)
        print(f"[isaac_sim] Usunięto stary LiDAR prim: {lidar_path}")

    omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path="/Lidar",
        parent=parent,
        config="lidar_spherical",
    )
    lidar = LidarRtx(prim_path=lidar_path, name="auv_lidar")
    print(f"[isaac_sim] RTX LiDAR prim: {lidar_path}")
    return lidar


def probe_rtx_lidar_api() -> None:
    """Diagnostyka — wypisuje co jest dostępne w modułach sensorów Isaac Sim."""
    # Metody LidarRtx
    try:
        from omni.isaac.sensor import LidarRtx
        members = [x for x in dir(LidarRtx) if not x.startswith("_")]
        print(f"[lidar-api] LidarRtx methods: {members}")
    except Exception as e:
        print(f"[lidar-api] LidarRtx: {e}")

    # Dostępne annotatory replicatora
    try:
        import omni.replicator.core as rep
        annotators = rep.AnnotatorRegistry.get_registered_annotators()
        lidar_ann = [a for a in annotators if "lidar" in a.lower() or "point" in a.lower()]
        print(f"[lidar-api] Annotatory LiDAR/PointCloud: {lidar_ann}")
    except Exception as e:
        print(f"[lidar-api] AnnotatorRegistry: {e}")

    # Pliki konfigów JSON dla RTX LiDAR
    try:
        import importlib.util, os
        spec = importlib.util.find_spec("omni.isaac.sensor")
        if spec:
            pkg_dir = os.path.dirname(spec.origin)
            for root, _, files in os.walk(pkg_dir):
                for f in files:
                    if f.endswith(".json") and "lidar" in f.lower():
                        print(f"[lidar-api] config: {os.path.join(root, f)}")
    except Exception as e:
        print(f"[lidar-api] config search: {e}")


def add_tunnel_mesh(stage, config: dict) -> None:
    """Wczytuje mesh tunelu do sceny jako statyczny obiekt z kolizją.

    Przy pierwszym uruchomieniu Isaac Sim konwertuje OBJ → USD (cache obok .obj).
    Kolejne uruchomienia używają gotowego .usd — znacznie szybsze ładowanie.

    Po załadowaniu: ustaw pozycję/skalę ręcznie w GUI (Stage → Transform),
    odczytaj wartości z panelu właściwości i wpisz do world.yaml.
    """
    if not config.get("enabled", True):
        print("[isaac_sim] Tunel wyłączony w world.yaml — pomijam")
        return

    obj_path = _THIS_DIR.parent / config["obj_path"]
    if not obj_path.exists():
        print(f"[isaac_sim] Brak mesha tunelu: {obj_path} — pomijam")
        return

    usd_path = obj_path.with_suffix(".usd")
    if not usd_path.exists():
        print(f"[isaac_sim] Konwertuję {obj_path.name} → USD (jednorazowo, może chwilę potrwać)...")
        import asyncio
        import omni.kit.asset_converter as converter_module

        async def _convert():
            converter = converter_module.get_instance()
            task = converter.create_converter_task(str(obj_path), str(usd_path))
            return await task.wait_until_finished()

        loop = asyncio.get_event_loop()
        ok = loop.run_until_complete(_convert())
        if not ok:
            print(f"[isaac_sim] Konwersja OBJ → USD nieudana — pomijam tunel")
            return
        print(f"[isaac_sim] Konwersja zakończona: {usd_path.name}")

    from omni.isaac.core.utils.stage import add_reference_to_stage
    prim_path = "/World/tunnel"
    add_reference_to_stage(usd_path=str(usd_path), prim_path=prim_path)
    prim = stage.GetPrimAtPath(prim_path)

    pos      = config.get("position",     [0.0, 0.0, 0.0])
    rot_deg  = config.get("rotation_deg", [0.0, 0.0, 0.0])
    scale    = float(config.get("scale",  1.0))

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    if config.get("collision", True):
        # Kolizja musi być aplikowana na każdym UsdGeom.Mesh w hierarchii —
        # OBJ importer tworzy geometrię w child primach, nie na rootu referencji.
        n_meshes = 0
        for child in Usd.PrimRange(prim):
            print(f"[tunnel-dbg] prim: {child.GetPath()} type={child.GetTypeName()}")
            if child.GetTypeName() == "Mesh":
                UsdPhysics.CollisionAPI.Apply(child)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(child)
                mesh_api.GetApproximationAttr().Set("none")
                # physxCollision:doubleSided — promienie od wewnątrz trafiają w obie strony trójkąta
                child.CreateAttribute("physxCollision:doubleSided", Sdf.ValueTypeNames.Bool, False).Set(True)
                UsdGeom.Mesh(child).GetDoubleSidedAttr().Set(True)
                n_meshes += 1
        if n_meshes == 0:
            print(f"[isaac_sim] UWAGA: brak primów Mesh w tunelu — stosuję kolizję na rootu")
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("none")
            prim.CreateAttribute("physxCollision:doubleSided", Sdf.ValueTypeNames.Bool, False).Set(True)
        print(f"[isaac_sim] Kolizja tunelu: {n_meshes} mesh primów")

    print(f"[isaac_sim] Tunel załadowany: {usd_path.name}, pos={pos}, skala={scale}")


def setup_lighting(stage) -> None:
    # SphereLight wewnątrz tunelu — emituje we wszystkich kierunkach z punktu.
    # DomeLight (niebo) nie dociera do wnętrza zamkniętego mesha.
    light = UsdLux.SphereLight.Define(stage, "/World/tunnel_light")
    light.GetIntensityAttr().Set(500000.0)
    light.GetRadiusAttr().Set(0.5)
    UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(6.5, 0.0, -2.5))

    dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
    dome.GetIntensityAttr().Set(300.0)


def set_robot_start_pose(robot_prim_path: str) -> None:
    """Ustawia startową pozycję robota pod wodą."""
    xf = XFormPrim(robot_prim_path)
    xf.set_world_pose(
        position=[0.0, 0.0, ROBOT_START_Z],
        orientation=[1.0, 0.0, 0.0, 0.0],  # [w, x, y, z] — poziomo
    )
    print(f"[isaac_sim] Robot startuje na z={ROBOT_START_Z} m, orientacja pozioma")


# ---------------------------------------------------------------------------
class IsaacRosNode(Node):
    """Węzeł ROS 2 zintegrowany z pętlą Isaac Sim.

    Publikuje pozę robota. Subskrybuje komendy silników i przekazuje je
    do SimThrusterDriver — symulacyjnego odpowiednika sterownika ESC.
    """

    def __init__(self):
        super().__init__("isaac_sim")
        self.declare_parameter("physics_dt",       0.01)
        self.declare_parameter("render_dt",        0.05)
        self.declare_parameter("robot_start_z",    ROBOT_START_Z)
        self.declare_parameter("start_paused",     True)
        self.declare_parameter("ahrs_enable_bias",  False)
        self.declare_parameter("dvl_enable_bias",   False)
        self.declare_parameter("depth_enable_bias", False)

        self._pub_pose    = self.create_publisher(PoseStamped, "/auv/sim/pose", 10)
        self._pub_vel     = self.create_publisher(TwistStamped, "/auv/sim/velocity", 10)
        self._sim_driver  = None   # ustawiany przez set_thruster_driver()

        self._sub_thrusts = self.create_subscription(
            Float64MultiArray, "/auv/thruster_cmds", self._cb_thrusts, 10)

        self.get_logger().info("IsaacRosNode gotowy.")

    def set_thruster_driver(self, driver: SimThrusterDriver) -> None:
        """Podłącz SimThrusterDriver — wywoływane przed uruchomieniem pętli."""
        self._sim_driver = driver

    def _cb_thrusts(self, msg: Float64MultiArray) -> None:
        if self._sim_driver is not None:
            self._sim_driver.set_thrusts(np.array(msg.data, dtype=float))

    def publish_velocity(self, lin_vel_world, ang_vel_world, orientation, stamp) -> None:
        R = _quat_to_rot(orientation)
        lin_body = R.T @ np.asarray(lin_vel_world, dtype=float)
        ang_body = R.T @ np.asarray(ang_vel_world, dtype=float)
        msg = TwistStamped()
        msg.header.stamp    = stamp.to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x  = float(lin_body[0])
        msg.twist.linear.y  = float(lin_body[1])
        msg.twist.linear.z  = float(lin_body[2])
        msg.twist.angular.x = float(ang_body[0])
        msg.twist.angular.y = float(ang_body[1])
        msg.twist.angular.z = float(ang_body[2])
        self._pub_vel.publish(msg)

    def publish_pose(self, position, orientation, stamp) -> None:
        msg = PoseStamped()
        msg.header.stamp    = stamp.to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.w = float(orientation[0])
        msg.pose.orientation.x = float(orientation[1])
        msg.pose.orientation.y = float(orientation[2])
        msg.pose.orientation.z = float(orientation[3])
        self._pub_pose.publish(msg)


# ---------------------------------------------------------------------------
def main():
    rclpy.init()

    _shutdown = False

    def _signal_handler(sig, frame):
        nonlocal _shutdown
        print(f"[isaac_sim] Odebrano sygnał {sig} — zamykam...")
        _shutdown = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)

    if not pathlib.Path(URDF_PATH).exists():
        raise FileNotFoundError(f"Brak {URDF_PATH} — uruchom przez sim_robot.sh, nie bezpośrednio")

    with open(_THRUSTER_CONFIG, encoding="utf-8") as f:
        thruster_config = yaml.safe_load(f)

    # SimThrusterDriver tworzony przed węzłem, by callback ROS mógł od razu
    # przekazywać komendy (choć initialize() wywoływane dopiero po world.reset())
    sim_driver = SimThrusterDriver(thruster_config)

    ros_node = IsaacRosNode()
    ros_node.set_thruster_driver(sim_driver)

    with open(_SENSOR_CONFIG, encoding="utf-8") as f:
        sensor_config = yaml.safe_load(f)["sensors"]

    sensor_config["ahrs"]["enable_drift"]  = ros_node.get_parameter("ahrs_enable_bias").value
    sensor_config["dvl"]["enable_drift"]   = ros_node.get_parameter("dvl_enable_bias").value
    sensor_config["depth"]["enable_drift"] = ros_node.get_parameter("depth_enable_bias").value

    sim_ahrs  = SimAhrsDriver(ros_node,  sensor_config["ahrs"])
    sim_dvl   = SimDvlDriver(ros_node,   sensor_config["dvl"])
    sim_depth = SimDepthDriver(ros_node, sensor_config["depth"])

    physics_dt = ros_node.get_parameter("physics_dt").value
    render_dt  = ros_node.get_parameter("render_dt").value

    world = World(physics_dt=physics_dt, rendering_dt=render_dt, stage_units_in_meters=1.0)

    print(f"[isaac_sim] Importuję URDF: {URDF_PATH}")
    robot_prim_path = import_urdf(URDF_PATH)
    print(f"[isaac_sim] Robot: {robot_prim_path}")

    stage = omni.usd.get_context().get_stage()
    setup_lighting(stage)
    ensure_collision_api(stage, robot_prim_path)

    with open(_WORLD_CONFIG, encoding="utf-8") as f:
        world_config = yaml.safe_load(f)["world"]
    add_tunnel_mesh(stage, world_config["tunnel"])

    robot = world.scene.add(Articulation(prim_path=robot_prim_path))
    world.reset()

    with open(_FOSSEN_CONFIG, encoding="utf-8") as f:
        fossen_config = yaml.safe_load(f)

    sim_driver.initialize(robot, robot_prim_path)

    sim_lidar = SimLidarDriver(ros_node, sensor_config["lidar"])
    set_robot_start_pose(robot_prim_path)

    fossen = FossenPlugin(robot_prim_path, fossen_config)
    fossen.initialize(robot, sim_driver.get_phx_art(), sim_driver)
    world.add_physics_callback("fossen_hydrodynamics", fossen.step)

    print(f"[isaac_sim] Fossen: {_FOSSEN_CONFIG}")
    print(f"[isaac_sim] SimThrusterDriver: {_THRUSTER_CONFIG}")

    paused = ros_node.get_parameter("start_paused").value
    if paused:
        # world.reset() startuje timeline automatycznie — jawnie zatrzymujemy
        world.pause()
        print("[isaac_sim] Symulacja ZATRZYMANA (start_paused=true). Spacja w oknie Isaac → odpal.")
    else:
        print("[isaac_sim] Pętla symulacji uruchomiona.")

    pose_dt     = 1.0 / POSE_RATE_HZ
    ahrs_dt     = 1.0 / float(sensor_config["ahrs"]["rate_hz"])
    dvl_dt      = 1.0 / float(sensor_config["dvl"]["rate_hz"])
    depth_dt    = 1.0 / float(sensor_config["depth"]["rate_hz"])
    lidar_dt    = 1.0 / float(sensor_config["lidar"]["rate_hz"])

    last_pose_t  = time.monotonic()
    last_ahrs_t  = time.monotonic()
    last_dvl_t   = time.monotonic()
    last_depth_t = time.monotonic()
    step_dt      = 1.0 / 60.0

    # LiDAR w physics callback — odpala się przed renderem, niezależnie od stalli GPU
    _last_lidar = [time.monotonic()]
    def _lidar_step(dt):
        now = time.monotonic()
        if now - _last_lidar[0] >= lidar_dt:
            _last_lidar[0] = now
            sim_lidar.publish(robot)
    world.add_physics_callback("lidar_scan", _lidar_step)

    while simulation_app.is_running() and not _shutdown:
        t0 = time.monotonic()

        if paused:
            # Renderuj GUI bez kroku fizyki — robot stoi w miejscu.
            # Publikuj zerową prędkość żeby velocity_controller nie nakręcał
            # integrala na starym pomiarze z momentu inicjalizacji.
            world.render()
            rclpy.spin_once(ros_node, timeout_sec=0.0)
            position, orientation = robot.get_world_pose()
            stamp = ros_node.get_clock().now()
            ros_node.publish_pose(position, orientation, stamp)
            ros_node.publish_velocity(
                [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], orientation, stamp)
            if world.is_playing():
                paused = False
                print("[isaac_sim] Symulacja uruchomiona.")
                # Resetuj timery żeby uniknąć spikea z nagromadzonego dt
                last_pose_t = last_ahrs_t = last_dvl_t = last_depth_t = time.monotonic()
                _last_lidar[0] = time.monotonic()
            elapsed = time.monotonic() - t0
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)
            continue

        world.step(render=True)
        rclpy.spin_once(ros_node, timeout_sec=0.0)

        now = time.monotonic()
        if now - last_pose_t >= pose_dt:
            last_pose_t = now
            position, orientation = robot.get_world_pose()
            stamp = ros_node.get_clock().now()
            ros_node.publish_pose(position, orientation, stamp)
            ros_node.publish_velocity(
                robot.get_linear_velocity(),
                robot.get_angular_velocity(),
                orientation,
                stamp,
            )

        if now - last_ahrs_t >= ahrs_dt:
            last_ahrs_t = now
            sim_ahrs.publish(robot, thruster_forces=sim_driver._thrusts)

        if now - last_dvl_t >= dvl_dt:
            last_dvl_t = now
            sim_dvl.publish(robot)

        if now - last_depth_t >= depth_dt:
            last_depth_t = now
            sim_depth.publish(robot)

        elapsed = time.monotonic() - t0
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

    print("[isaac_sim] Zamykam symulację...")
    ros_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
    print("[isaac_sim] Zamknięto.")


if __name__ == "__main__":
    main()

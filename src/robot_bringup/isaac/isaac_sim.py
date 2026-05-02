#!/usr/bin/env python3
"""
isaac_sim.py – Symulacja AUV BlueROV2 w Isaac Sim.

Uruchomienie (w kontenerze):
    OMNI_KIT_ALLOW_ROOT=1 python3 src/robot_bringup/isaac/isaac_sim.py

Równolegle uruchom węzły ROS2:
    ros2 launch robot_bringup isaac.launch.py

Publikuje:
    /auv/pose  (geometry_msgs/PoseStamped) – pozycja i orientacja robota, 50 Hz
"""

import time

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
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import XFormPrim
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import pathlib

# ---------------------------------------------------------------------------
URDF_PATH = "/tmp/bluerov2.urdf"  # generowany przez sim_robot.sh przed startem

ROBOT_START_Z  = -5.0   # [m] — pod wodą
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
def add_water_plane(stage) -> None:
    """Dodaje wizualną płaszczyznę wody na z=0."""
    plane = UsdGeom.Mesh.Define(stage, "/World/water_surface")
    plane.GetPointsAttr().Set([
        Gf.Vec3f(-50, -50, 0), Gf.Vec3f( 50, -50, 0),
        Gf.Vec3f( 50,  50, 0), Gf.Vec3f(-50,  50, 0),
    ])
    plane.GetFaceVertexCountsAttr().Set([4])
    plane.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    plane.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.2, 0.6)])
    UsdGeom.Imageable(plane.GetPrim()).MakeVisible()
    print("[isaac_sim] Płaszczyzna wody dodana na z=0")


def ensure_collision_api(stage, robot_prim_path: str) -> None:
    """Fix collision: disable USD instancing, add RigidBodyAPI + box collision shape.

    Isaac Sim URDF importer in in-memory mode leaves collisions/ Xforms empty and
    creates the robot as instanced USD (child prims are read-only proxies).
    """
    # Must disable instancing before any authoring on child prims.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)

    # RigidBodyAPI on base_link — required for PhysX to bind collision shapes.
    base_link = stage.GetPrimAtPath(f"{robot_prim_path}/bluerov2_base_link")
    if base_link.IsValid() and not base_link.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(base_link)

    # Create collision box (importer leaves collisions/ empty in in-memory mode).
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


def setup_lighting(stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
    dome.GetIntensityAttr().Set(300.0)


def set_robot_start_pose(robot_prim_path: str) -> None:
    """Ustawia startową pozycję robota pod wodą."""
    xf = XFormPrim(robot_prim_path)
    xf.set_world_pose(
        position=[0.0, 0.0, ROBOT_START_Z],
    )
    print(f"[isaac_sim] Robot startuje na z={ROBOT_START_Z} m")


# ---------------------------------------------------------------------------
class IsaacRosNode(Node):
    def __init__(self):
        super().__init__("isaac_sim")
        self.declare_parameter("physics_dt",    0.01)
        self.declare_parameter("render_dt",     0.05)
        self.declare_parameter("robot_start_z", ROBOT_START_Z)

        self._pub_pose = self.create_publisher(PoseStamped, "/auv/pose", 10)
        self.get_logger().info("IsaacRosNode gotowy.")

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
    ros_node = IsaacRosNode()

    if not pathlib.Path(URDF_PATH).exists():
        raise FileNotFoundError(f"Brak {URDF_PATH} — uruchom przez sim_robot.sh, nie bezpośrednio")

    physics_dt = ros_node.get_parameter("physics_dt").value
    render_dt  = ros_node.get_parameter("render_dt").value

    world = World(physics_dt=physics_dt, rendering_dt=render_dt, stage_units_in_meters=1.0)

    print(f"[isaac_sim] Importuję URDF: {URDF_PATH}")
    robot_prim_path = import_urdf(URDF_PATH)
    print(f"[isaac_sim] Robot: {robot_prim_path}")

    stage = omni.usd.get_context().get_stage()
    setup_lighting(stage)
    add_water_plane(stage)
    ensure_collision_api(stage, robot_prim_path)

    world.scene.add_default_ground_plane(z_position=-10.0)
    robot = world.scene.add(Articulation(prim_path=robot_prim_path))
    world.reset()

    set_robot_start_pose(robot_prim_path)

    pose_dt      = 1.0 / POSE_RATE_HZ
    last_pose_t  = time.monotonic()
    step_dt      = 1.0 / 60.0

    print("[isaac_sim] Pętla symulacji uruchomiona.")

    while simulation_app.is_running():
        t0 = time.monotonic()
        world.step(render=True)
        rclpy.spin_once(ros_node, timeout_sec=0.0)

        now = time.monotonic()
        if now - last_pose_t >= pose_dt:
            last_pose_t = now
            position, orientation = robot.get_world_pose()
            ros_node.publish_pose(position, orientation, ros_node.get_clock().now())

        elapsed = time.monotonic() - t0
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

    ros_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()

"""Load two YAM arms and execute a validated Qwen policy in Isaac Sim 5.1."""

import asyncio
import json
import os
import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation, SingleGeometryPrim, SingleXFormPrim
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    get_current_stage,
    open_stage_async,
)
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.app import get_app
from omni.kit.commands import execute
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import PhysxSchema, Usd, UsdPhysics
from scipy.spatial.transform import Rotation, Slerp

from bricksim.assets.stages import DEFAULT_STAGE_PATH
from bricksim.cap.basket import Basket, BulkBasket
from bricksim.cap.kinematics import YamKinematics
from bricksim.cap.policy import Assembly
from bricksim.cap.yam import YamArm
from bricksim.core import (
    AssemblyThresholds,
    are_parts_connected,
    import_lego,
    set_assembly_thresholds,
)
from bricksim.utils.physics_step import wait_for_physics_step

ROOT = Path(__file__).resolve().parents[1]


class Demo:
    """Headless demo with scene inventory, screenshots and explicit result evidence."""

    def __init__(self):
        """Read target, policy and output paths from environment variables."""
        self.target = Assembly(
            Path(
                os.environ.get(
                    "YAM_TARGET", str(ROOT / "demos/assemblies/four_brick_tower.json")
                )
            )
        )
        self.output = Path(os.environ.get("YAM_OUTPUT", str(ROOT / "outputs/yam")))
        self.output.mkdir(parents=True, exist_ok=True)
        self.policy = Path(os.environ.get("YAM_POLICY", str(self.output / "policy.py")))
        self.scene_only = os.environ.get("YAM_SCENE_ONLY") == "1"
        self.basket = Basket() if os.environ.get("YAM_BASKET") == "1" else None
        self.seed = int(os.environ.get("YAM_SEED", "7"))
        supply_count = int(os.environ.get("YAM_SUPPLY_COUNT", "4"))
        if supply_count not in (4, 50):
            raise ValueError("YAM_SUPPLY_COUNT must be 4 or 50")
        if supply_count == 50:
            if self.basket is None:
                raise ValueError("Fifty-brick supply requires YAM_BASKET=1")
            self.basket = BulkBasket()
        self.supply_paths: dict[int, str] = {}
        if self.basket is not None:
            parts = self.target.topology["parts"]
            if {p["id"] for p in parts} != {0, 1, 2, 3, 4} or any(
                [p["payload"][key] for key in ("L", "W", "H")] != [2, 4, 3]
                for p in parts
                if p["id"] != 0
            ):
                raise ValueError("Basket mode requires four 2x4x3 bricks with IDs 1..4")
        self.paths: dict[int, str] = {}
        self.events: list[dict[str, object]] = []
        self.source_poses: dict[int, np.ndarray] = {}
        self.press_depths: tuple[float, ...] | None = None
        self.top_press = False
        self.grasp_width: float | None = None

    async def capture(self, name: str):
        """Save a rendered frame after the viewport settles.

        Args:
            name: PNG basename inside the output directory.
        """
        viewport = get_active_viewport()
        viewport.set_texture_resolution((1280, 720))
        for _ in range(60):
            await get_app().next_update_async()  # ty: ignore[unresolved-attribute]
        destination = self.output / name
        await capture_viewport_to_file(viewport, str(destination)).wait_for_result(0)
        for _ in range(200):
            if destination.exists() and destination.stat().st_size > 0:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError(f"Capture was not written: {destination}")

    async def run(self):
        """Import both arms, populate the table and execute physical actions."""
        actions = (
            [] if self.scene_only else self.target.parse_policy(self.policy.read_text())
        )
        enable_extension("isaacsim.asset.importer.urdf")
        await open_stage_async(str(DEFAULT_STAGE_PATH))
        world = World(backend="numpy", device="cpu", physics_prim_path="/physicsScene")
        assert isinstance(world, World)
        await world.initialize_simulation_context_async()
        stage = get_current_stage()
        urdf = ROOT / "resources/robots/yam/yam.urdf"
        cache = ROOT / ".cache/yam/yam.usd"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            success, config = execute("URDFCreateImportConfig")
            if not success:
                raise RuntimeError("URDF importer configuration failed")
            config.fix_base = True
            config.merge_fixed_joints = False
            config.make_default_prim = True
            config.create_physics_scene = False
            config.import_inertia_tensor = True
            config.convex_decomp = True
            config.set_collision_from_visuals(True)
            success, _ = execute(
                "URDFParseAndImportFile",
                urdf_path=str(urdf),
                import_config=config,
                dest_path=str(cache),
            )
            if not success:
                raise RuntimeError("YAM URDF conversion failed")
        pad = PhysicsMaterial(
            "/World/YamPad", static_friction=2.5, dynamic_friction=2.0, restitution=0
        )
        robots = {}
        kinematics = YamKinematics(urdf)
        for name, side in [("left", -1), ("right", 1)]:
            path = f"/World/Yam_{name}"
            add_reference_to_stage(str(cache), path)
            SingleXFormPrim(path).set_world_pose(position=[-0.02, side * 0.26, 0])
            for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
                # Generated USD stubs omit these runtime-verified schema classes.
                if prim.HasAPI(UsdPhysics.DriveAPI):  # ty: ignore[unresolved-attribute]
                    drive = UsdPhysics.DriveAPI.Get(  # ty: ignore[unresolved-attribute]
                        prim,
                        "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular",  # ty: ignore[unresolved-attribute]
                    )
                    finger = prim.IsA(UsdPhysics.PrismaticJoint)  # ty: ignore[unresolved-attribute]
                    drive.GetStiffnessAttr().Set(2000.0 if finger else 100000.0)
                    drive.GetDampingAttr().Set(50.0 if finger else 1000.0)
                    drive.GetMaxForceAttr().Set(15.0 if finger else 80.0)
                if prim.HasAPI(UsdPhysics.CollisionAPI):  # ty: ignore[unresolved-attribute]
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision.CreateContactOffsetAttr(0.0003)
                    collision.CreateRestOffsetAttr(0.0)
                    if "tip_" in str(prim.GetPath()):
                        SingleGeometryPrim(str(prim.GetPath())).apply_physics_material(
                            pad
                        )
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):  # ty: ignore[unresolved-attribute]
                    articulation = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                    articulation.CreateSolverPositionIterationCountAttr(64)
                    articulation.CreateEnabledSelfCollisionsAttr(False)
            robot = SingleArticulation(path, name=f"yam_{name}")
            # Scene.add accepts articulations; its SDK annotation is too narrow.
            world.scene.add(robot)  # ty: ignore[invalid-argument-type]
            robots[name] = robot
        thresholds = AssemblyThresholds()
        thresholds.distance_tolerance = 0.001
        thresholds.max_penetration = 0.005
        thresholds.z_angle_tolerance = np.deg2rad(5)
        thresholds.required_force = 1.0
        thresholds.yaw_tolerance = np.deg2rad(5)
        thresholds.position_tolerance = 0.002
        set_assembly_thresholds(thresholds)
        scattered = {} if self.basket is None else self.basket.scatter(self.seed)
        scattered.update(self.source_poses)
        if self.basket is not None:
            basket = self.basket
            center = (basket.minimum + basket.maximum) / 2
            size = basket.maximum - basket.minimum
            walls = [
                ("floor", [*center, basket.floor / 2], [*size, basket.floor]),
                (
                    "left",
                    [center[0], basket.minimum[1] - basket.wall / 2, basket.rim / 2],
                    [size[0] + 2 * basket.wall, basket.wall, basket.rim],
                ),
                (
                    "right",
                    [center[0], basket.maximum[1] + basket.wall / 2, basket.rim / 2],
                    [size[0] + 2 * basket.wall, basket.wall, basket.rim],
                ),
                (
                    "front",
                    [basket.minimum[0] - basket.wall / 2, center[1], basket.rim / 2],
                    [basket.wall, size[1], basket.rim],
                ),
                (
                    "back",
                    [basket.maximum[0] + basket.wall / 2, center[1], basket.rim / 2],
                    [basket.wall, size[1], basket.rim],
                ),
            ]
            for name, position, scale in walls:
                FixedCuboid(
                    f"/World/Basket/{name}",
                    name=f"basket_{name}",
                    position=np.array(position),
                    scale=np.array(scale),
                    size=1.0,
                    color=np.array([0.22, 0.32, 0.38]),
                )
        # Spawn disconnected bricks before physics; never teleport during actions.
        scene_parts = self.target.topology["parts"]
        if isinstance(self.basket, BulkBasket):
            template = next(part for part in scene_parts if part["id"] != 0)
            baseplate = next(part for part in scene_parts if part["id"] == 0)
            scene_parts = [baseplate] + [dict(template, id=part) for part in scattered]
        for index, part in enumerate(scene_parts):
            topology = dict(
                self.target.topology,
                parts=[part],
                connections=[],
                pose_hints=[
                    {"part": part["id"], "pos": [0, 0, 0], "rot": [1, 0, 0, 0]}
                ],
            )
            position = (
                [0.155 if self.basket is not None else 0.32, 0, 0]
                if part["id"] == 0
                else [
                    0.24 + 0.15 * ((index - 1) // 2),
                    -0.14 if index % 2 else 0.14,
                    0.001,
                ]
            )
            rotation = (1.0, 0.0, 0.0, 0.0)
            if part["id"] in scattered:
                pose = scattered[part["id"]]
                position = pose[:3, 3].tolist()
                quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat(
                    scalar_first=True
                )
                rotation = (
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                )
            paths, _ = import_lego(
                json=topology,
                env_id=-1,
                ref_pos=(position[0], position[1], position[2]),
                ref_rot=rotation,
            )
            if isinstance(self.basket, BulkBasket) and part["id"] != 0:
                self.supply_paths.update(paths)
            else:
                self.paths.update(paths)
        for path in [
            "/World/Cube",
            "/World/scene/roomScene/colliders/floor",
            "/World/scene/roomScene/colliders/walls",
            "/World/scene/roomScene/colliders/windows",
            "/World/scene/roomScene/renderables",
        ]:
            stage.GetPrimAtPath(path).SetActive(False)
        await world.reset_async()
        for name, side in [("left", -1), ("right", 1)]:
            robot = robots[name]
            home = np.eye(4)
            home[:3, :3] = np.diag([1, -1, -1] if name == "left" else [-1, 1, -1])
            home[:3, 3] = [0.28, -side * 0.10, 0.32]
            joints = kinematics.inverse(home, np.array([0, 1.5, 1.5, 0, 0, 0]))
            positions = np.zeros(8)
            for i, value in enumerate(np.r_[joints, -0.04, -0.04], 1):
                positions[robot.get_dof_index(f"joint{i}")] = value
            robot.set_joint_positions(positions)
            robot.set_joints_default_state(positions=positions)
            robot.get_articulation_controller().set_gains(
                kps=np.array(
                    [2000 if "7" in n or "8" in n else 100000 for n in robot.dof_names]
                ),
                kds=np.array(
                    [50 if "7" in n or "8" in n else 1000 for n in robot.dof_names]
                ),
            )
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=positions)
            )
        for _ in range(120):
            await wait_for_physics_step(world)
        arms = {
            name: YamArm(world, robot, urdf, f"/World/Yam_{name}")
            for name, robot in robots.items()
        }
        inventory = {
            "embodiment": "dual I2RT YAM v1",
            "control": "joint position drives + numerical IK",
            "parts": scene_parts,
            "supply_count": len(scene_parts) - 1,
            "supply_paths": self.supply_paths,
            "paths": self.paths,
            "arms": {name: robot.dof_names for name, robot in robots.items()},
            "source": "basket" if self.basket is not None else "table",
            "perception": "simulator world poses",
            "seed": self.seed if self.basket is not None else None,
            "initial_poses": {
                part: YamArm.pose(path).tolist()
                for part, path in (self.paths | self.supply_paths).items()
            },
        }
        (self.output / "inventory.json").write_text(json.dumps(inventory, indent=2))
        set_camera_view(np.array([1.1, -1.05, 0.85]), np.array([0.25, 0, 0.13]))
        await self.capture("yam-scene.png")
        loose_paths = (
            dict(self.supply_paths)
            if self.supply_paths
            else {part: path for part, path in self.paths.items() if part != 0}
        )
        for connection, name in actions:
            part = next(
                p for p in self.target.topology["parts"] if p["id"] == connection.hole
            )
            source_id = connection.hole
            rejected = {}
            if isinstance(self.basket, BulkBasket):
                observations = {
                    key: YamArm.pose(path) for key, path in loose_paths.items()
                }
                candidates, rejected = self.basket.candidates(
                    observations, side=-1 if name == "left" else 1
                )
                arm = arms[name]
                for candidate in candidates:
                    target = observations[candidate].copy()
                    target[:3, :3] = target[:3, :3] @ np.diag(
                        [-1, 1, -1] if name == "right" else [1, -1, -1]
                    )
                    target[:3, 3] += observations[candidate][:3, 2] * 0.0071
                    joints = arm.joint_target.copy()
                    try:
                        # Screen sampled approach poses before committing to this pick.
                        start = arm.pose(arm.flange_path) @ arm.tool
                        above = target.copy()
                        above[2, 3] += 0.10
                        for waypoint in (above, target):
                            rotations = Slerp(
                                [0, 1],
                                Rotation.from_matrix(
                                    np.stack([start[:3, :3], waypoint[:3, :3]])
                                ),
                            )
                            for fraction in np.linspace(0, 1, 16):
                                probe = waypoint.copy()
                                probe[:3, :3] = rotations(fraction).as_matrix()
                                probe[:3, 3] = (
                                    start[:3, 3] * (1 - fraction)
                                    + waypoint[:3, 3] * fraction
                                )
                                flange = (
                                    np.linalg.inv(arm.base)
                                    @ probe
                                    @ np.linalg.inv(arm.tool)
                                )
                                joints = arm.kinematics.inverse(flange, joints)
                            start = waypoint
                        # Physics poses can drift while reachability is evaluated.
                        latest = {
                            key: YamArm.pose(path) for key, path in loose_paths.items()
                        }
                        self.basket.check_grasp(candidate, latest, margin=0.001)
                    except (RuntimeError, ValueError) as error:
                        rejected[candidate] = str(error)
                        continue
                    source_id = candidate
                    self.paths[connection.hole] = loose_paths[candidate]
                    break
                else:
                    raise RuntimeError(f"No reachable clear bulk brick: {rejected}")
                print(
                    f"YAM_SELECT target={connection.hole} supply={source_id} "
                    f"remaining={len(loose_paths)} rejected={len(rejected)}",
                    flush=True,
                )
            pick = await arms[name].assemble(
                connection,
                self.paths,
                part["payload"]["H"] * 0.0032,
                reverse=name == "right",
                basket=self.basket,
                loose_paths=loose_paths,
                source_id=source_id,
                press_depths=self.press_depths,
                grasp_width=self.grasp_width,
                top_press=self.top_press,
            )
            del loose_paths[source_id]
            if isinstance(self.basket, BulkBasket):
                pick.update(
                    supply_id=source_id, remaining=len(loose_paths), rejected=rejected
                )
            self.events.append(
                {
                    "stud": connection.stud,
                    "hole": connection.hole,
                    "arm": name,
                    "connected": True,
                    "pick": pick,
                }
            )
            await self.capture(f"step-{connection.hole}.png")
        connected = (
            {}
            if self.scene_only
            else {
                str(c.hole): are_parts_connected(self.paths[c.stud], self.paths[c.hole])
                for c in self.target.connections.values()
            }
        )
        result = {
            "status": "scene_only" if self.scene_only else "success",
            "connections": connected,
            "actions": self.events,
            "remaining_supply": len(loose_paths),
            "source": "basket" if self.basket is not None else "table",
            "seed": self.seed if self.basket is not None else None,
        }
        if isinstance(self.basket, BulkBasket):
            remaining_poses = {
                part: YamArm.pose(path) for part, path in loose_paths.items()
            }
            outside = [
                part
                for part, pose in remaining_poses.items()
                if np.any(pose[:2, 3] < self.basket.minimum)
                or np.any(pose[:2, 3] > self.basket.maximum)
                or not self.basket.floor - 0.002 <= pose[2, 3] <= self.basket.rim
            ]
            if outside:
                raise RuntimeError(f"Unconsumed supply left the basket: {outside}")
            result["remaining_supply_poses"] = {
                part: pose.tolist() for part, pose in remaining_poses.items()
            }
        if not self.scene_only and not all(connected.values()):
            raise RuntimeError(f"Assembly validation failed: {connected}")
        (self.output / "result.json").write_text(json.dumps(result, indent=2))
        print("YAM_RESULT " + json.dumps(result), flush=True)
        await self.capture("yam-result.png")
        tower_x = float(YamArm.pose(self.paths[0])[0, 3])
        set_camera_view(
            np.array([tower_x + 0.23, -0.35, 0.28]), np.array([tower_x, 0, 0.025])
        )
        await self.capture("yam-tower.png")
        await world.stop_async()


async def main(demo: Demo | None = None):
    """Run a scene, save failure evidence and request application exit.

    Args:
        demo: Configured scene, or the default tower when omitted.
    """
    if demo is None:
        demo = Demo()
    try:
        (demo.output / "result.json").write_text('{"status": "running"}\n')
        await demo.run()
    except Exception as error:
        traceback.print_exc()
        (demo.output / "result.json").write_text(
            json.dumps(
                {"status": "failed", "error": str(error), "actions": demo.events},
                indent=2,
            )
        )
        await demo.capture("yam-failure.png")
        get_app().post_quit(1)
        raise
    get_app().post_quit(0)

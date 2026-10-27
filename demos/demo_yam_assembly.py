"""Load two YAM arms and execute a validated Qwen policy in Isaac Sim 5.1."""

import asyncio
import json
import os
import traceback
from pathlib import Path

import numpy as np
from isaacsim.core.api.materials import PhysicsMaterial
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

from bricksim.assets.stages import DEFAULT_STAGE_PATH
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
        self.paths: dict[int, str] = {}
        self.events: list[dict[str, object]] = []

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
        # Spawn disconnected bricks before physics; never teleport during actions.
        for index, part in enumerate(self.target.topology["parts"]):
            topology = dict(
                self.target.topology,
                parts=[part],
                connections=[],
                pose_hints=[
                    {"part": part["id"], "pos": [0, 0, 0], "rot": [1, 0, 0, 0]}
                ],
            )
            position = (
                [0.32, 0, 0]
                if part["id"] == 0
                else [
                    0.24 + 0.15 * ((index - 1) // 2),
                    -0.14 if index % 2 else 0.14,
                    0.001,
                ]
            )
            paths, _ = import_lego(
                json=topology,
                env_id=-1,
                ref_pos=(position[0], position[1], position[2]),
                ref_rot=(1, 0, 0, 0),
            )
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
            "parts": self.target.topology["parts"],
            "paths": self.paths,
            "arms": {name: robot.dof_names for name, robot in robots.items()},
        }
        (self.output / "inventory.json").write_text(json.dumps(inventory, indent=2))
        set_camera_view(np.array([1.1, -1.05, 0.85]), np.array([0.25, 0, 0.13]))
        await self.capture("yam-scene.png")
        for connection, name in actions:
            part = next(
                p for p in self.target.topology["parts"] if p["id"] == connection.hole
            )
            await arms[name].assemble(
                connection,
                self.paths,
                part["payload"]["H"] * 0.0032,
                reverse=name == "right",
            )
            self.events.append(
                {
                    "stud": connection.stud,
                    "hole": connection.hole,
                    "arm": name,
                    "connected": True,
                }
            )
            await self.capture(f"step-{connection.hole}.png")
        connected = {
            str(c.hole): are_parts_connected(self.paths[c.stud], self.paths[c.hole])
            for c in self.target.connections.values()
        }
        result = {
            "status": "scene_only" if self.scene_only else "success",
            "connections": connected,
            "actions": self.events,
        }
        if not self.scene_only and not all(connected.values()):
            raise RuntimeError(f"Assembly validation failed: {connected}")
        (self.output / "result.json").write_text(json.dumps(result, indent=2))
        print("YAM_RESULT " + json.dumps(result), flush=True)
        await self.capture("yam-result.png")
        set_camera_view(np.array([0.55, -0.35, 0.28]), np.array([0.32, 0, 0.025]))
        await self.capture("yam-tower.png")
        await world.stop_async()


async def main():
    """Run the demo, save failure evidence and request application exit."""
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

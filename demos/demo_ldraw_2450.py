"""Drop an LDraw 2450 mesh as an approximate convex rigid body in Isaac Sim.

This experiment does not create LEGO stud interfaces or snap connections.
"""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import get_current_stage, open_stage_async
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.app import get_app
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Gf, UsdGeom, UsdPhysics, Vt

from bricksim.assets.stages import DEFAULT_STAGE_PATH
from bricksim.utils.physics_step import wait_for_physics_step


async def main():
    """Drop the mesh, save its settled pose and capture a rendered image."""
    source = Path(os.environ["LDRAW_OBJ"])
    output = Path(os.environ.get("LDRAW_OUTPUT", "outputs/ldraw-2450-drop"))
    output.mkdir(parents=True, exist_ok=True)
    points = []
    indices = []
    for line in source.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            points.append(Gf.Vec3f(*(float(value) for value in fields[1:])))
        elif fields and fields[0] == "f":
            indices.extend(int(value) - 1 for value in fields[1:])
    await open_stage_async(str(DEFAULT_STAGE_PATH))
    world = World(backend="numpy", device="cpu", physics_prim_path="/physicsScene")
    assert isinstance(world, World)
    await world.initialize_simulation_context_async()
    stage = get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Ldraw2450")
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.58, 0.54, 0.45)])
    mesh.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.05))
    prim = mesh.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)  # ty: ignore[unresolved-attribute]
    UsdPhysics.CollisionAPI.Apply(prim)  # ty: ignore[unresolved-attribute]
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")  # ty: ignore[unresolved-attribute]
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.002)  # ty: ignore[unresolved-attribute]
    FixedCuboid(
        "/World/DropFloor",
        name="drop_floor",
        position=np.array([0, 0, -0.005]),
        scale=np.array([0.15, 0.15, 0.01]),
        size=1.0,
        color=np.array([0.25, 0.3, 0.34]),
    )
    for path in [
        "/World/Cube",
        "/World/scene/roomScene/colliders/floor",
        "/World/scene/roomScene/colliders/walls",
        "/World/scene/roomScene/colliders/windows",
        "/World/scene/roomScene/renderables",
    ]:
        stage.GetPrimAtPath(path).SetActive(False)
    await world.reset_async()
    body = SingleXFormPrim("/World/Ldraw2450")
    start, _ = body.get_world_pose()
    for _ in range(600):
        await wait_for_physics_step(world)
    end, rotation = body.get_world_pose()
    result = {
        "part": "2450",
        "start_position_m": start.tolist(),
        "end_position_m": end.tolist(),
        "end_quaternion_wxyz": rotation.tolist(),
        "collision": (
            "convex hull of visual mesh; underside hollows and stud contacts "
            "approximated"
        ),
        "mass_kg": 0.002,
        "functional_stud_connections": False,
    }
    if not 0.001 < end[2] < 0.008 or start[2] - end[2] < 0.03:
        raise RuntimeError(f"Drop did not settle on floor: {result}")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("LDRAW_DROP_RESULT " + json.dumps(result), flush=True)
    set_camera_view(np.array([0.09, -0.12, 0.10]), np.array([0, 0, 0]))
    viewport = get_active_viewport()
    viewport.set_texture_resolution((1280, 720))
    for _ in range(60):
        await get_app().next_update_async()  # ty: ignore[unresolved-attribute]
    capture = capture_viewport_to_file(viewport, str(output / "drop.png"))
    await capture.wait_for_result(0)
    for _ in range(100):
        if (output / "drop.png").exists():
            break
        await asyncio.sleep(0.05)
    await world.stop_async()
    get_app().post_quit(0)

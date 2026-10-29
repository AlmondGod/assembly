"""Contact-driven two-YAM pumpkin assembly with a measured-motion replay.

The round foundation is fixtured; supply pieces rest on visible tray supports.
Set PUMPKIN_LIMIT=1 for the first insertion coupon, or51for the complete model.
"""

import cProfile
import json
import os
import runpy
import time
import traceback
from pathlib import Path
from xml.etree.ElementTree import parse

import carb.settings
import numpy as np
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.stage import get_current_stage
from omni.kit.app import get_app
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

from bricksim.cap.interactive import InteractiveSession
from bricksim.cap.pumpkin import PumpkinParts
from bricksim.cap.yam import YamArm
from bricksim.core import deallocate_part
from bricksim.utils.physics_video import PhysicsRecording

ROOT = Path(__file__).resolve().parents[1]
BaseDemo = runpy.run_path(str(ROOT / "demos/demo_yam_assembly.py"))["Demo"]


class Demo(BaseDemo):
    """Reuse tested arm import and drives, with rendering independently disabled."""

    def __init__(self):
        """Use the standard arm scene with this demo's organized-tray session."""
        super().__init__()
        self.control = "interactive"

    async def capture(self, name: str):
        """Avoid RTX image capture; measured frames are recorded separately.

        Args:
            name: Unused legacy screenshot name.
        """
        return None


async def run_pumpkin(self: InteractiveSession):
    """Pick, lift, insert, release and validate each supported target part.

    Args:
        self: Initialized interactive scene with two physical YAM arms.
    """
    started = time.monotonic()
    output = self.output
    stage = get_current_stage()
    await self.world.stop_async()
    for path in self.paths.values():
        if not deallocate_part(path):
            raise RuntimeError(f"Unable to remove default scene part {path}")
    if os.environ.get("PUMPKIN_GPU_PHYSICS") == "1":
        physics = self.world.get_physics_context()
        physics.enable_gpu_dynamics(True)
        physics.set_broadphase_type("GPU")
        physics.set_gpu_total_aggregate_pairs_capacity(16384)
        physics.set_gpu_found_lost_aggregate_pairs_capacity(16384)
    physics_dt = 1 / int(os.environ.get("PUMPKIN_PHYSICS_HZ", "240"))
    self.world.set_simulation_dt(physics_dt=physics_dt, rendering_dt=physics_dt)
    settle_steps = int(np.ceil(1.5 / physics_dt))
    parts = PumpkinParts(
        stage, canonical_clutch=os.environ.get("PUMPKIN_CANONICAL_CLUTCH") == "1"
    )
    # Narrow extensions are compound colliders on the existing dynamic fingers.
    # URDF origins are column-vector transforms; USD matrices use row vectors.
    robot_xml = parse(ROOT / "resources/robots/yam/yam.urdf").getroot()
    dimensions = np.array(parts.grasp_plan["fingertip_dimensions_m"])
    tool_offset = float(parts.grasp_plan["tool_flange_offset_m"])
    inner_gap = float(parts.grasp_plan["fingertip_closed_inner_gap_m"])
    torsional_patch = float(os.environ.get("PUMPKIN_TORSIONAL_PATCH_M", "0"))
    for name in self.arms:
        for tip, joint_index, side in [("tip_left", 7, -1), ("tip_right", 8, 1)]:
            origin = robot_xml.find(f"joint[@name='joint{joint_index}']/origin")
            assert origin is not None
            joint_frame = np.eye(4)
            joint_frame[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.attrib["rpy"], sep=" ")
            ).as_matrix()
            joint_frame[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
            pad_frame = np.eye(4)
            pad_frame[:3, 3] = [
                side * (dimensions[0] + inner_gap) / 2,
                0,
                tool_offset + 0.002 - dimensions[2] / 2,
            ]
            local = np.linalg.inv(joint_frame) @ pad_frame
            cube = UsdGeom.Cube.Define(stage, f"/World/Yam_{name}/{tip}/PumpkinPad")
            cube.CreateSizeAttr(1.0)
            cube.AddTransformOp().Set(Gf.Matrix4d(*local.T.reshape(-1).tolist()))
            cube.AddScaleOp().Set(Gf.Vec3f(*map(float, dimensions)))
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.6, 0.95)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())  # ty: ignore[unresolved-attribute]
            contact = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            contact.CreateContactOffsetAttr(0.0003)
            contact.CreateRestOffsetAttr(0.0)
            contact.CreateMinTorsionalPatchRadiusAttr(torsional_patch)
            contact.CreateTorsionalPatchRadiusAttr(2 * torsional_patch)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                UsdShade.Material(stage.GetPrimAtPath("/World/YamPad")),
                materialPurpose="physics",
            )
    material = PhysicsMaterial(
        "/World/PumpkinMaterial",
        static_friction=1.5,
        dynamic_friction=1.2,
        restitution=0.0,
    )
    initial = parts.spawn(material)
    # The SDK accepts batched prim views; its Scene.add annotation is narrower.
    self.world.scene.add(parts.rigid)  # ty: ignore[invalid-argument-type]
    await self.world.reset_async()
    # Contact views disable sleeping globally; only the manipulated part needs
    # continuous force reports. Other dynamic parts still wake on real contact.
    parts.rigid.set_sleep_thresholds(np.full(51, 0.00005, dtype=np.float32))
    for arm in self.arms.values():
        positions = arm.robot.get_joint_positions()
        arm.joint_target = positions[arm.indices].copy()
        arm.grip_target = -0.04
        arm.inverse_residual_limit = 0.003
        arm.tool[2, 3] = tool_offset
        arm.park_pose = YamArm.pose(arm.flange_path) @ arm.tool
    for _ in range(settle_steps):
        await self.arms["left"].tick()
    supply_evidence = []
    for part, expected_pose in initial.items():
        if part == 44:
            continue
        actual_pose = YamArm.pose(parts.paths[part])
        supply_evidence.append(
            {
                "part": part,
                "translation_m": float(
                    np.linalg.norm(actual_pose[:3, 3] - expected_pose[:3, 3])
                ),
                "rotation_rad": float(
                    Rotation.from_matrix(
                        expected_pose[:3, :3].T @ actual_pose[:3, :3]
                    ).magnitude()
                ),
                "actual_pose": actual_pose.tolist(),
            }
        )
    (output / "supply-settling.json").write_text(json.dumps(supply_evidence, indent=2))
    print(
        "PUMPKIN_SUPPLY_SETTLED",
        max(x["translation_m"] for x in supply_evidence),
        max(x["rotation_rad"] for x in supply_evidence),
        flush=True,
    )
    if os.environ.get("PUMPKIN_SETUP_TEST") == "1":
        (output / "result.json").write_text(
            json.dumps(
                {
                    "status": "supply_setup_test",
                    "placements": [],
                    "placement_verified": False,
                    "supply": supply_evidence,
                },
                indent=2,
            )
        )
        return
    settings = carb.settings.get_settings()
    loop_paths = [
        "/app/runLoops/main/rateLimitEnabled",
        "/app/runLoops/main/rateLimitFrequency",
        "/app/renderer/sleepMsOnFocus",
        "/app/renderer/sleepMsOutOfFocus",
    ]
    print(
        "PUMPKIN_LOOP_SETTINGS",
        {path: settings.get(path) for path in loop_paths},
        flush=True,
    )
    settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
    settings.set_int("/app/renderer/sleepMsOnFocus", 0)
    settings.set_int("/app/renderer/sleepMsOutOfFocus", 0)
    recording = PhysicsRecording(
        stage,
        output,
        contact_sensor=parts.rigid,
        contact_filters=parts.contact_filter_paths,
        sample_period=float(os.environ.get("PUMPKIN_RECORD_INTERVAL", "0.2")),
    )
    self.world.add_physics_callback("pumpkin_recording", recording.record)
    events = []
    profiler = cProfile.Profile()
    if os.environ.get("PUMPKIN_PROFILE") == "1":
        profiler.enable()
    (output / "initial-layout.json").write_text(
        json.dumps({str(k): v.tolist() for k, v in initial.items()}, indent=2)
    )
    try:
        source_test = os.environ.get("PUMPKIN_GRASP_TEST")
        sequence = (
            [int(source_test)]
            if source_test is not None
            else parts.order()[: int(os.environ.get("PUMPKIN_LIMIT", "51"))]
        )
        for index, part in enumerate(sequence):
            recording.contact_source = parts.paths[part]
            recording.contact_watch = [
                parts.paths[placed] for placed in sorted(parts.placed) if placed != 44
            ]
            part_index = np.array([parts.rigid.prim_paths.index(parts.paths[part])])
            parts.rigid.set_sleep_thresholds(
                np.array([0.0], dtype=np.float32), indices=part_index
            )
            before = YamArm.pose(parts.paths[part])
            name = "left" if before[1, 3] < 0 else "right"
            arm = self.arms[name]
            plan = parts.grasps[part]
            grasp = before @ np.array(plan["grasp_in_part_m"])
            above = grasp.copy()
            above[2, 3] += 0.07
            width = float(plan["opening_width_m"])
            opening = float(plan["opening_m"])
            print(
                "PUMPKIN_PICK",
                index,
                part,
                parts.parts[part]["design"],
                name,
                flush=True,
            )
            await arm.grip(opening)
            if os.environ.get("PUMPKIN_PROFILE") == "1":
                profiler.dump_stats(str(output / "controller-profile.pstats"))
            await arm.move(above, speed=0.10, tolerance=0.003)
            await arm.move(grasp, speed=0.025, tolerance=0.003)
            if part in (0, 47):
                # Center before closure; instantaneous closure rolled Part47
                # by28degrees while it was still on its source supports.
                command = grasp.copy()
                for _ in range(2):
                    measured_tcp = YamArm.pose(arm.flange_path) @ arm.tool
                    command = grasp @ np.linalg.inv(measured_tcp) @ command
                    await arm.move(command, speed=0.01, tolerance=0.003)
            if part == 0:
                controller = arm.robot.get_articulation_controller()
                kps, kds = controller.get_gains()
                kps[arm.fingers] = 10000.0
                kds[arm.fingers] = 100.0
                controller.set_gains(kps=kps, kds=kds)
                controller.set_max_efforts(
                    np.array([5.0, 5.0]), joint_indices=arm.fingers
                )
                print(
                    "PUMPKIN_PLUME_GAINS",
                    controller.get_gains(),
                    "max_efforts",
                    controller.get_max_efforts(),
                    flush=True,
                )
                print(
                    "PUMPKIN_PLUME_CENTER_ERROR",
                    (YamArm.pose(arm.flange_path) @ arm.tool)[:3, 3] - grasp[:3, 3],
                    flush=True,
                )
            if part in (0, 47):
                for close_step in range(int(np.ceil(2.0 / physics_dt))):
                    closing = max(0.0, opening - (close_step + 1) * 0.015 * physics_dt)
                    arm.grip_target = -closing / 2
                    await arm.tick()
            else:
                await arm.grip(0.0)
            await arm.move(above, speed=0.07, tolerance=0.003)
            lifted = YamArm.pose(parts.paths[part])
            lift = float(lifted[2, 3] - before[2, 3])
            gap = float(-sum(arm.robot.get_joint_positions()[arm.fingers]))
            minimum_gap = min(0.001, float(plan["estimated_contact_width_m"]) * 0.2)
            if lift < 0.05 or not minimum_gap < gap < opening - 0.002:
                raise RuntimeError(f"Part{part} grasp failed: lift={lift}, gap={gap}")
            if source_test is not None:
                for _ in range(int(np.ceil(0.5 / physics_dt))):
                    await arm.tick()
                held = YamArm.pose(parts.paths[part])
                if held[2, 3] - before[2, 3] < 0.05:
                    raise RuntimeError(f"Part{part} dropped during grasp test")
                (output / "source-grasp.json").write_text(
                    json.dumps(
                        {
                            "status": "source_grasp_test",
                            "part": part,
                            "arm": name,
                            "lift_m": float(held[2, 3] - before[2, 3]),
                            "finger_gap_m": gap,
                            "before_pose": before.tolist(),
                            "held_pose": held.tolist(),
                            "placements": [],
                            "placement_verified": False,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        indent=2,
                    )
                )
                report = json.loads((output / "source-grasp.json").read_text())
                if os.environ.get("PUMPKIN_GRASP_TRANSPORT") == "1":
                    measured_tcp = YamArm.pose(arm.flange_path) @ arm.tool
                    relative = np.linalg.inv(held) @ measured_tcp
                    target_tcp = parts.targets[part] @ relative
                    transport_above = target_tcp.copy()
                    transport_above[2, 3] += float(plan["approach_height_m"])
                    await arm.move(transport_above, speed=0.1, tolerance=0.003)
                    await arm.move(target_tcp, speed=0.02, tolerance=0.003)
                    for _ in range(int(np.ceil(0.5 / physics_dt))):
                        await arm.tick()
                    transported = YamArm.pose(parts.paths[part])
                    position_error = float(
                        np.linalg.norm(transported[:3, 3] - parts.targets[part][:3, 3])
                    )
                    angle_error = float(
                        Rotation.from_matrix(
                            parts.targets[part][:3, :3].T @ transported[:3, :3]
                        ).magnitude()
                    )
                    if position_error > 0.003 or angle_error > 0.04:
                        raise RuntimeError(
                            f"Source grasp shifted during transport: "
                            f"{position_error}m/{angle_error}rad"
                        )
                    report.update(
                        {
                            "status": "source_grasp_and_transport_test",
                            "transport_verified": True,
                            "transport_position_error_m": position_error,
                            "transport_angle_error_rad": angle_error,
                            "transport_actual_body_pose": transported.tolist(),
                            "transport_target_body_pose": parts.targets[part].tolist(),
                        }
                    )
                (output / "result.json").write_text(json.dumps(report, indent=2))
                return
            if plan["via_park"]:
                await arm.move(arm.park_pose, speed=0.10, tolerance=0.003)
                lifted = YamArm.pose(parts.paths[part])
                if lifted[2, 3] - before[2, 3] < 0.05:
                    raise RuntimeError(f"Part{part} dropped during transport")
            relative_grasp = (
                np.linalg.inv(lifted) @ YamArm.pose(arm.flange_path) @ arm.tool
            )
            # Anchor goals to the measured support pose, not an assumed global CAD pose.
            parent = next(a for a, b in parts.edges if b == part and a in parts.placed)
            expected = np.linalg.inv(parts.targets[parent]) @ parts.targets[part]
            target = YamArm.pose(parts.paths[parent]) @ expected
            goal = target @ relative_grasp
            approach = goal.copy()
            approach[2, 3] += float(plan["approach_height_m"])
            await arm.move(approach, speed=0.10, tolerance=0.003)
            print("PUMPKIN_INSERT", part, "parent", parent, flush=True)
            seated = []
            hover_target = target.copy()
            hover_target[2, 3] += 0.004
            hover_command = goal.copy()
            hover_command[2, 3] += 0.004
            await arm.move(hover_command, speed=0.025, tolerance=0.003)
            required = {(a, b) for a, b in parts.edges if b == part}
            stopped_on_contact = False

            def stop_press() -> bool:
                """Cache measured seating or stop an excessive assembly-contact load.

                Returns:
                    True when the seat is loaded or total support load exceeds0.5N.
                """
                nonlocal stopped_on_contact
                seated.extend(parts.connect(part, physics_dt))
                stopped_on_contact = (
                    required <= parts.seated.keys()
                    or parts.latest_support_load[part] > 0.5
                )
                return stopped_on_contact

            press_command = hover_command.copy()
            # Reobserve the held part at each small descent instead of driving
            # one long press after the grip has already slipped on contact.
            for alignment, depth in enumerate(
                [0.002, 0.002, *np.linspace(0.002, -0.0008, 15), *[-0.0008] * 3]
            ):
                measured = YamArm.pose(parts.paths[part])
                desired = target.copy()
                desired[2, 3] += depth
                press_command = desired @ np.linalg.inv(measured) @ press_command
                await arm.move(
                    press_command,
                    speed=0.004,
                    tolerance=0.003,
                    minimum_duration=0.05,
                    settle_seconds=0.1,
                    stop_when=stop_press if index >= 4 else None,
                )
                if stopped_on_contact:
                    press_command = (
                        arm.base @ arm.kinematics.forward(arm.joint_target) @ arm.tool
                    )
                for _ in range(20):
                    await arm.tick()
                    if parts.broken:
                        raise RuntimeError(
                            f"Clutch joints broke: {sorted(parts.broken)}"
                        )
                    seated.extend(parts.connect(part, self.world.get_physics_dt()))
                required = {(a, b) for a, b in parts.edges if b == part}
                if required <= parts.seated.keys():
                    break
                measured = YamArm.pose(parts.paths[part])
                print(
                    "PUMPKIN_ALIGNMENT",
                    part,
                    alignment,
                    "error",
                    measured[:3, 3] - target[:3, 3],
                    flush=True,
                )
            required = {(a, b) for a, b in parts.edges if b == part}
            if not required <= parts.seated.keys():
                actual = YamArm.pose(parts.paths[part])
                force = np.asarray(
                    parts.rigid.get_net_contact_forces(
                        indices=np.array(
                            [parts.rigid.prim_paths.index(parts.paths[part])]
                        ),
                        dt=self.world.get_physics_dt(),
                    )
                ).tolist()
                raise RuntimeError(
                    f"Part{part} did not seat: target={target[:3, 3]}, "
                    f"actual={actual[:3, 3]}, force={force}"
                )
            # Remove compression before release; GPU contacts otherwise eject the
            # slightly penetrating part when the fingers stop supporting it.
            measured = YamArm.pose(parts.paths[part])
            unload_height = max(0.0, target[2, 3] + 0.0001 - measured[2, 3])
            if unload_height > 0.0001:
                relaxed = YamArm.pose(arm.flange_path) @ arm.tool
                relaxed[2, 3] += unload_height
                print("PUMPKIN_UNLOAD", part, unload_height, flush=True)
                await arm.move(relaxed, speed=0.01, tolerance=0.003)
            starting_gap = float(-sum(arm.robot.get_joint_positions()[arm.fingers]))
            released_gap = starting_gap
            seated = []
            retention_ready = False
            for release_step in range(int(np.ceil(2.0 / physics_dt))):
                # Ramp the opening rather than accelerating the jaws at full force.
                arm.grip_target = (
                    -min(opening, starting_gap + (release_step + 1) * 0.03 * physics_dt)
                    / 2
                )
                await arm.tick()
                if not retention_ready:
                    seated.extend(
                        parts.connect(part, self.world.get_physics_dt(), engage=True)
                    )
                    if required <= parts.connected.keys():
                        retention_ready = True
                        await get_app().next_update_async()  # ty: ignore[unresolved-attribute]
                released_gap = float(-sum(arm.robot.get_joint_positions()[arm.fingers]))
                if released_gap >= width + 0.004:
                    break
            if released_gap < width + 0.004:
                raise RuntimeError(f"Part{part} fingers did not clear: {released_gap}")
            if not retention_ready:
                raise RuntimeError(f"Part{part} did not retain its unloaded seat")
            arm.grip_target = -opening / 2
            await arm.move(approach, speed=0.07, tolerance=0.003)
            for _ in range(settle_steps):
                await arm.tick()
            # Drain simulation events, including joint breaks, before scoring.
            await get_app().next_update_async()  # ty: ignore[unresolved-attribute]
            actual = YamArm.pose(parts.paths[part])
            position_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            angle_error = float(
                Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude()
            )
            if position_error > 0.0015 or angle_error > 0.06:
                raise RuntimeError(
                    f"Part{part} moved after release: "
                    f"{position_error}m/{angle_error}rad"
                )
            if parts.broken:
                raise RuntimeError(f"Clutch joints broke: {sorted(parts.broken)}")
            parts.placed.add(part)
            parts.rigid.set_sleep_thresholds(
                np.array([0.00005], dtype=np.float32), indices=part_index
            )
            events.append(
                {
                    "part": part,
                    "design": parts.parts[part]["design"],
                    "arm": name,
                    "lift_m": lift,
                    "finger_gap_m": gap,
                    "connections": seated,
                    "simulation_seconds": recording.seconds,
                    "released_position_error_m": position_error,
                    "released_angle_error_rad": angle_error,
                }
            )
            (output / "progress.json").write_text(
                json.dumps(
                    {
                        "placements": events,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    indent=2,
                )
            )
            recording.save()
            print("PUMPKIN_PLACED", index + 1, "/", 51, part, flush=True)
            await arm.move(arm.park_pose, speed=0.10, tolerance=0.003)
        for _ in range(settle_steps):
            await self.arms["left"].tick()
        await get_app().next_update_async()  # ty: ignore[unresolved-attribute]
        if parts.broken:
            raise RuntimeError(f"Clutch joints broke: {sorted(parts.broken)}")
        final_connections = []
        for parent, child in parts.connected:
            expected = np.linalg.inv(parts.targets[parent]) @ parts.targets[child]
            actual = np.linalg.inv(YamArm.pose(parts.paths[parent])) @ YamArm.pose(
                parts.paths[child]
            )
            position_error = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
            angle_error = float(
                Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude()
            )
            if position_error > 0.0015 or angle_error > 0.06:
                raise RuntimeError(
                    f"Final connection {parent}->{child} moved: "
                    f"{position_error}m/{angle_error}rad"
                )
            final_connections.append(
                {
                    "parent": parent,
                    "child": child,
                    "position_error_m": position_error,
                    "angle_error_rad": angle_error,
                }
            )
        (output / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete" if len(events) == 51 else "partial_coupon",
                    "placements": events,
                    "foundation_fixture": 44,
                    "broken_joints": sorted(parts.broken),
                    "final_connections_after_parking": final_connections,
                    "retention_model": "contact-gated breakable fixed joints",
                    "clutch_frame": "canonical studs, measured bar"
                    if parts.canonical_clutch
                    else "measured",
                    "elapsed_seconds": time.monotonic() - started,
                },
                indent=2,
            )
        )
    finally:
        final_forces = np.asarray(parts.rigid.get_net_contact_forces(dt=physics_dt))
        (output / "final-contact-evidence.json").write_text(
            json.dumps(
                [
                    {
                        "path": path,
                        "normal_contact_force_N": force.tolist(),
                        "pose": YamArm.pose(path).tolist(),
                    }
                    for path, force in zip(parts.rigid.prim_paths, final_forces)
                ],
                indent=2,
            )
        )
        (output / "connection-evidence.json").write_text(
            json.dumps(
                {
                    "loaded_seats": list(parts.seated.values()),
                    "retained_connections": list(parts.connected.values()),
                    "broken_joints": sorted(parts.broken),
                },
                indent=2,
            )
        )
        profiler.disable()
        profiler.dump_stats(str(output / "controller-profile.pstats"))
        recording.save()
        self.world.remove_physics_callback("pumpkin_recording")


async def main():
    """Run one finite assembly trial and preserve failure evidence."""
    os.environ.update(YAM_CONTROL="assembly", YAM_BASKET="0")
    output = Path(os.environ["YAM_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    InteractiveSession.run = run_pumpkin
    try:
        await Demo().run()
    except Exception as error:
        traceback.print_exc()
        (output / "failure.json").write_text(
            json.dumps({"error": str(error)}, indent=2)
        )
        get_app().post_quit(1)
        return
    get_app().post_quit(0)

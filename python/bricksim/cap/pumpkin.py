"""Sourced pumpkin geometry, connectors and measured contact assembly.

All geometry uses metres, column-vector transforms and WXYZ quaternions.
Connections below model clutch retention with breakable fixed joints; their
strengths are approximations, not measured LEGO material parameters.
"""

import json
from pathlib import Path

import carb.events
import numpy as np
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.prims import RigidPrim
from omni.physx import get_physx_interface
from omni.physx.bindings._physx import SimulationEvent
from pxr import (
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,
    Vt,
)
from scipy.spatial.transform import Rotation

from bricksim.cap.yam import YamArm

RESOURCE = Path(__file__).resolve().parents[3] / "resources/assemblies/pumpkin40055"
LDD_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
COLORS = {
    "106": (0.96, 0.49, 0.10),
    "5": (0.78, 0.69, 0.51),
    "28": (0.11, 0.46, 0.23),
    "141": (0.12, 0.31, 0.18),
    "154": (0.45, 0.10, 0.14),
}


class PumpkinParts:
    """Create real rigid parts and gate retention on measured insertion.

    The round foundation is held by an assembly fixture. No other part is
    kinematic and no attachment is created between a gripper and a part.
    """

    def __init__(
        self,
        stage: Usd.Stage,
        resource: Path = RESOURCE,
        canonical_clutch: bool = False,
        canonical_foundation: bool = False,
        defer_correction: bool = False,
    ):
        """Load exact visual/collision assets and the target connection graph.

        Args:
            stage: Live simulator stage.
            resource: Directory containing sourced meshes and connector frames.
            canonical_clutch: Use nominal socket frames for constraint correction.
            canonical_foundation: Correct only the four foundation connections.
            defer_correction: Initially retain measured poses; correct after release.
        """
        self.stage = stage
        self.canonical_clutch = canonical_clutch
        self.canonical_foundation = canonical_foundation
        self.defer_correction = defer_correction
        self.resource = resource
        self.manifest = json.loads((resource / "assembly.json").read_text())
        self.connectors = json.loads((resource / "connectors.json").read_text())
        self.grasp_plan = json.loads((resource / "grasp-plan.json").read_text())
        self.grasps = {g["part"]: g for g in self.grasp_plan["grasps"]}
        self.parts = {p["id"]: p for p in self.manifest["parts"]}
        self.targets: dict[int, np.ndarray] = {}
        self.paths: dict[int, str] = {}
        self.rigid: RigidPrim
        self.bounds: dict[str, np.ndarray] = {}
        self.latest_support_load: dict[int, float] = {}
        self.connected: dict[tuple[int, int], dict[str, object]] = {}
        self.seated: dict[tuple[int, int], dict[str, object]] = {}
        self.placed = {44}
        self.broken: set[str] = set()
        self.joint_events = (
            get_physx_interface()
            .get_simulation_event_stream_v2()
            .create_subscription_to_pop(self.on_simulation_event)
        )
        self.edges: dict[tuple[int, int], int] = {}
        for connection in self.connectors["stud_connections"]:
            key = (connection["stud"], connection["socket"])
            self.edges[key] = self.edges.get(key, 0) + 1
        self.edges[(1, 0)] = 1  # Plume bar inserted in the cone's upper bore.
        for part in self.parts.values():
            transform = np.eye(4)
            transform[:3, :3] = (
                LDD_TO_Z_UP @ np.array(part["rotation_ldd"]) @ LDD_TO_Z_UP.T
            )
            transform[:3, 3] = LDD_TO_Z_UP @ (np.array(part["translation_ldd"]) * 0.01)
            transform[:3, 3] += self.grasp_plan["assembly_offset_world_m"]
            self.targets[part["id"]] = transform
            if part["design"] not in self.bounds:
                points = (
                    np.load(resource / "meshes" / (part["design"] + ".npz"))["vertices"]
                    @ LDD_TO_Z_UP.T
                )
                self.bounds[part["design"]] = np.array([points.min(0), points.max(0)])

    def spawn(self, material: PhysicsMaterial) -> dict[int, np.ndarray]:
        """Create fixture and51visible, separated supply parts before physics.

        Args:
            material: Existing USD physics material path.

        Returns:
            Initial root transforms for every part, including the foundation.
        """
        supply = json.loads((self.resource / "supply-posts.json").read_text())
        poses = {44: self.targets[44]}
        ordered = sorted(p for p in self.parts if p != 44)
        for part in ordered:
            bounds = self.bounds[self.parts[part]["design"]]
            extents = bounds[1] - bounds[0]
            pose = np.array(self.grasps[part]["source_pose_world_m"])
            center = (bounds[0] + bounds[1]) / 2
            poses[part] = pose
            center_world = pose[:3, :3] @ center + pose[:3, 3]
            world_size = np.abs(pose[:3, :3]) @ extents
            if part != 0:
                # Follow the underside, including cavities and inverted slopes.
                # Posts stay inside the footprint so the jaws have side access.
                for index, local_top in enumerate(
                    supply["designs"][self.parts[part]["design"]]
                ):
                    top = pose[:3, :3] @ np.array(local_top) + pose[:3, 3]
                    height = float(top[2])
                    support = FixedCuboid(
                        f"/World/PumpkinTray/Support{part}_{index}",
                        name=f"part_support_{part}_{index}",
                        position=np.array([*top[:2], height / 2]),
                        scale=np.array([0.0015, 0.0015, height]),
                        size=1.0,
                        color=np.array([0.22, 0.24, 0.28]),
                    )
                    contact = PhysxSchema.PhysxCollisionAPI.Apply(support.prim)
                    contact.CreateContactOffsetAttr(0.0001)
                    contact.CreateRestOffsetAttr(0.0)
            else:
                support = FixedCuboid(
                    "/World/PumpkinTray/Support0",
                    name="part_support_0",
                    position=np.array([*center_world[:2], 0.006]),
                    scale=np.array([world_size[0], world_size[1], 0.012]),
                    size=1.0,
                    color=np.array([0.22, 0.24, 0.28]),
                )
                contact = PhysxSchema.PhysxCollisionAPI.Apply(support.prim)
                contact.CreateContactOffsetAttr(0.0001)
                contact.CreateRestOffsetAttr(0.0)
            if part == 0:
                # Support the broad leaf away from the fingers' grasp band.
                # This is an open slot: the robot lifts the dynamic leaf out.
                for sign in [-1, 1]:
                    position = (
                        pose[:3, :3] @ np.array([-0.0005, sign * 0.0024, 0.005])
                        + pose[:3, 3]
                    )
                    scale = np.abs(pose[:3, :3]) @ np.array([0.003, 0.001, 0.006])
                    guide = FixedCuboid(
                        f"/World/PumpkinTray/LeafGuide{sign + 1}",
                        name=f"leaf_guide_{sign}",
                        position=position,
                        scale=scale,
                        size=1.0,
                        color=np.array([0.22, 0.24, 0.28]),
                    )
                    contact = PhysxSchema.PhysxCollisionAPI.Apply(guide.prim)
                    contact.CreateContactOffsetAttr(0.0001)
                    contact.CreateRestOffsetAttr(0.0)
                end_xy = (pose[:3, :3] @ np.array([0.0124, 0.0, 0.0]) + pose[:3, 3])[:2]
                height = float(pose[2, 3] + 0.012)
                end = FixedCuboid(
                    "/World/PumpkinTray/LeafEndGuide",
                    name="leaf_end_guide",
                    position=np.array([*end_xy, height / 2]),
                    scale=np.abs(pose[:3, :3]) @ np.array([0.001, 0.0048, height]),
                    size=1.0,
                    color=np.array([0.22, 0.24, 0.28]),
                )
                contact = PhysxSchema.PhysxCollisionAPI.Apply(end.prim)
                contact.CreateContactOffsetAttr(0.0001)
                contact.CreateRestOffsetAttr(0.0)
        for part, info in self.parts.items():
            path = (
                "/World/Pumpkin/Foundation44"
                if part == 44
                else f"/World/Pumpkin/Part{part}"
            )
            self.paths[part] = path
            root = UsdGeom.Xform.Define(self.stage, path)
            pose = poses[part]
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
            root.AddTranslateOp().Set(Gf.Vec3d(*map(float, pose[:3, 3])))
            root.AddOrientOp().Set(
                Gf.Quatf(float(quaternion[0]), Gf.Vec3f(*map(float, quaternion[1:])))
            )
            UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
                material.material, materialPurpose="physics"
            )
            body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())  # ty: ignore[unresolved-attribute]
            body.CreateKinematicEnabledAttr(part == 44)
            size = self.bounds[info["design"]][1] - self.bounds[info["design"]][0]
            mass = max(0.00012, float(np.prod(size)) * 1050 * 0.25)
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)  # ty: ignore[unresolved-attribute]
            rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
            rigid.CreateEnableCCDAttr(False)
            rigid.CreateSolverPositionIterationCountAttr(8)
            rigid.CreateSolverVelocityIterationCountAttr(1)
            visual = np.load(self.resource / "meshes" / (info["design"] + ".npz"))
            arrays = np.load(self.resource / "collisions" / (info["design"] + ".npz"))
            for name, points, faces, collision in [
                ("Visual", visual["vertices"], visual["faces"], False)
            ] + [
                (f"Collision{i}", arrays[f"v{i}"], arrays[f"f{i}"], True)
                for i in range(len(arrays.files) // 2)
            ]:
                mesh = UsdGeom.Mesh.Define(self.stage, path + "/" + name)
                points = points @ LDD_TO_Z_UP.T
                mesh.CreatePointsAttr(
                    Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points])
                )
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
                mesh.CreateFaceVertexIndicesAttr(
                    Vt.IntArray(faces.reshape(-1).tolist())
                )
                mesh.CreateSubdivisionSchemeAttr("none")
                if collision:
                    mesh.CreateVisibilityAttr("invisible")
                    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())  # ty: ignore[unresolved-attribute]
                    UsdPhysics.MeshCollisionAPI.Apply(  # ty: ignore[unresolved-attribute]
                        mesh.GetPrim()
                    ).CreateApproximationAttr("convexHull")
                    cooking = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(
                        mesh.GetPrim()
                    )
                    cooking.CreateMinThicknessAttr(0.00005)
                    contact = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
                    contact.CreateContactOffsetAttr(0.00008)
                    contact.CreateRestOffsetAttr(0.0)
                else:
                    mesh.CreateDoubleSidedAttr(True)
                    mesh.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[info["color_ldd"]])])
        self.contact_filter_paths = list(self.paths.values())
        for name in ["left", "right"]:
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(f"/World/Yam_{name}")):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):  # ty: ignore[unresolved-attribute]
                    self.contact_filter_paths.append(str(prim.GetPath()))
        self.rigid = RigidPrim(
            "/World/Pumpkin/Part.*",
            name="pumpkin_parts",
            reset_xform_properties=False,
            track_contact_forces=True,
            contact_filter_prim_paths_expr=self.contact_filter_paths,
            max_contact_count=8192,
        )
        return poses

    def order(self) -> list[int]:
        """Return a stable support-first order, with bar insertion last.

        Returns:
            IDs of the51parts to place on the fixtured foundation.
        """
        return list(self.grasp_plan["order"])

    def connect(
        self, part: int, dt: float, engage: bool = False
    ) -> list[dict[str, object]]:
        """Measure loaded insertion, then engage clutch on an unloaded seat.

        Args:
            part: Newly pressed part ID.
            dt: Physics step duration in seconds.
            engage: Create retention after removing the measured press load.

        Returns:
            Measured seating or newly created retention evidence; empty if invalid.
        """
        net_force = (
            np.asarray(
                self.rigid.get_net_contact_forces(
                    indices=np.array([self.rigid.prim_paths.index(self.paths[part])]),
                    dt=dt,
                )
            )
            .reshape(-1, 3)
            .sum(0)
        )
        pair_forces = np.asarray(
            self.rigid.get_contact_force_matrix(
                indices=np.array([self.rigid.prim_paths.index(self.paths[part])]), dt=dt
            )
        )[0]
        assert pair_forces.shape == (len(self.contact_filter_paths), 3)
        parent_indices = [
            self.contact_filter_paths.index(self.paths[parent])
            for parent, child in self.edges
            if child == part and parent in self.placed
        ]
        support_pairs = pair_forces[parent_indices]
        total_force = support_pairs.sum(0)
        support_load = float(np.linalg.norm(support_pairs, axis=1).sum())
        self.latest_support_load[part] = support_load
        evidence = []
        for (parent, child), studs in self.edges.items():
            if (
                child != part
                or parent not in self.placed
                or (parent, child) in self.connected
            ):
                continue
            force = pair_forces[self.contact_filter_paths.index(self.paths[parent])]
            a = YamArm.pose(self.paths[parent])
            b = YamArm.pose(self.paths[child])
            expected = np.linalg.inv(self.targets[parent]) @ self.targets[child]
            actual = np.linalg.inv(a) @ b
            position_error = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
            angle_error = float(
                Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude()
            )
            if engage:
                print(
                    "PUMPKIN_RETAIN",
                    parent,
                    child,
                    "position_mm",
                    position_error * 1000,
                    "angle_rad",
                    angle_error,
                    "contact_N",
                    force.tolist(),
                    flush=True,
                )
            if position_error > 0.0008 or angle_error > 0.05:
                continue
            if not engage:
                if total_force[2] < 0.15 or (parent, child) in self.seated:
                    continue
                item = {
                    "parent": parent,
                    "child": child,
                    "position_error_m": position_error,
                    "angle_error_rad": angle_error,
                    "net_contact_force_N": net_force.tolist(),
                    "support_contact_force_N": force.tolist(),
                    "total_support_contact_force_N": total_force.tolist(),
                    "type": "bar" if child == 0 else "stud",
                    "stud_count": studs,
                }
                self.seated[(parent, child)] = item
                evidence.append(item)
                continue
            if (parent, child) not in self.seated:
                continue
            break_force = 0.5 if child == 0 else 2.0 * studs
            if support_load > break_force / 4:
                continue
            # Measured mode retains the seat; canonical mode lets constraint
            # forces correct its residual error under the same break limits.
            joint = UsdPhysics.FixedJoint.Define(  # ty: ignore[unresolved-attribute]
                self.stage, f"/World/Pumpkin/Clutch_{parent}_{child}"
            )
            joint.CreateBody0Rel().SetTargets([self.paths[parent]])
            joint.CreateBody1Rel().SetTargets([self.paths[child]])
            canonical_rest = (
                (self.canonical_clutch or (self.canonical_foundation and parent == 44))
                and child != 0
                and not self.defer_correction
            )
            rest_frame = expected if canonical_rest else actual
            quaternion = Rotation.from_matrix(rest_frame[:3, :3]).as_quat(
                scalar_first=True
            )
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, rest_frame[:3, 3])))
            joint.CreateLocalRot0Attr().Set(
                Gf.Quatf(float(quaternion[0]), Gf.Vec3f(*map(float, quaternion[1:])))
            )
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
            joint.CreateCollisionEnabledAttr(False)
            joint.CreateBreakForceAttr(break_force)
            joint.CreateBreakTorqueAttr(0.002 if child == 0 else 0.01 * studs)
            item = {
                **self.seated[(parent, child)],
                "unloaded_seat_position_error_m": position_error,
                "unloaded_seat_angle_error_rad": angle_error,
                "engagement_net_contact_force_N": net_force.tolist(),
                "engagement_support_contact_force_N": force.tolist(),
                "engagement_total_support_contact_force_N": total_force.tolist(),
                "engagement_support_load_N": support_load,
                "retention_model": (
                    "contact-gated canonical clutch with constraint correction"
                    if canonical_rest
                    else "contact-gated measured clutch after unloading"
                ),
            }
            self.connected[(parent, child)] = item
            evidence.append(item)
        return evidence

    def on_simulation_event(self, event: carb.events.IEvent) -> None:
        """Record actual PhysX joint breaks, rather than trusting USD prim presence.

        Args:
            event: Physics simulation event, including an encoded joint path.
        """
        if event.type == int(SimulationEvent.JOINT_BREAK):
            encoded = event.payload["jointPath"]
            # NVIDIA's joint-break sample uses this runtime-only schema function.
            path = str(
                PhysicsSchemaTools.decodeSdfPath(encoded[0], encoded[1])  # ty: ignore[unresolved-attribute]
            )
            if path.startswith("/World/Pumpkin/Clutch_"):
                self.broken.add(path)
                print("PUMPKIN_JOINT_BREAK", path, flush=True)

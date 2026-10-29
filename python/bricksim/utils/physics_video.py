"""Capture live rigid-body poses and mesh assets for a CPU-rendered replay."""

import json
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from isaacsim.core.prims import RigidPrim
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


class PhysicsRecording:
    """Sample measured simulator poses, never planner targets."""

    def __init__(
        self,
        stage: Usd.Stage,
        output: Path,
        contact_sensor: RigidPrim | None = None,
        contact_filters: list[str] | None = None,
        sample_period: float = 0.2,
    ):
        """Export visible meshes and bind them to live rigid-body frames.

        Args:
            stage: Running simulator stage.
            output: Directory for mesh archive and measured frame archive.
            contact_sensor: Optional live body contact view for diagnostics.
            contact_filters: Paths in the sensor's filter column order.
            sample_period: Simulated seconds between measured samples.
        """
        self.output = output
        self.contact_sensor = contact_sensor
        self.contact_filters = contact_filters or []
        self.contact_source: str | None = None
        self.contact_watch: list[str] = []
        self.sample_period = sample_period
        self.bodies: list[str] = []
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []
        self.steps = 0
        self.seconds = 0.0
        self.started = perf_counter()
        self.read_seconds = 0.0
        self.next_sample_seconds = sample_period
        self.next_save_seconds = 2.0
        arrays = {}
        cache = UsdGeom.XformCache()
        count = 0
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not (
                path.startswith("/World/Yam_")
                or path.startswith("/World/Pumpkin/")
                or path.startswith("/World/PumpkinTray/")
            ):
                continue
            if (
                not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube))
                or "collision" in path.lower()
            ):
                continue
            mesh = UsdGeom.Gprim(prim)
            if mesh.ComputeVisibility() == "invisible":
                continue
            parent = prim
            while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):  # ty: ignore[unresolved-attribute]
                parent = parent.GetParent()
            body_index = -1
            if parent:
                body = str(parent.GetPath())
                if body not in self.bodies:
                    self.bodies.append(body)
                body_index = self.bodies.index(body)
                relative = np.array(cache.ComputeRelativeTransform(prim, parent)[0]).T
            else:
                relative = np.array(cache.GetLocalToWorldTransform(prim)).T
            if prim.IsA(UsdGeom.Cube):
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
                points = np.array(list(product([-size, size], repeat=3)))
                faces = [
                    (0, 1, 3),
                    (0, 3, 2),
                    (4, 6, 7),
                    (4, 7, 5),
                    (0, 4, 5),
                    (0, 5, 1),
                    (2, 3, 7),
                    (2, 7, 6),
                    (0, 2, 6),
                    (0, 6, 4),
                    (1, 5, 7),
                    (1, 7, 3),
                ]
            else:
                geometry = UsdGeom.Mesh(prim)
                points = np.array(geometry.GetPointsAttr().Get(), dtype=np.float32)
                counts = np.array(geometry.GetFaceVertexCountsAttr().Get())
                indices = np.array(geometry.GetFaceVertexIndicesAttr().Get())
                faces = []
                offset = 0
                for length in counts:
                    faces.extend(
                        (indices[offset], indices[offset + i], indices[offset + i + 1])
                        for i in range(1, length - 1)
                    )
                    offset += length
            points = (points @ relative[:3, :3].T + relative[:3, 3]).astype(np.float32)
            color = mesh.GetDisplayColorAttr().Get()
            arrays[f"v{count}"] = points
            arrays[f"f{count}"] = np.array(faces, dtype=np.int32)
            arrays[f"c{count}"] = np.array(
                color[0] if color else [0.55, 0.6, 0.68], dtype=np.float32
            )
            arrays[f"b{count}"] = np.array(body_index)
            count += 1
        arrays["body_paths"] = np.array(self.bodies)
        arrays["mesh_count"] = np.array(count)
        np.savez_compressed(output / "meshes.npz", **arrays)
        self.view = RigidPrim(
            self.bodies,
            name="pumpkin_recording_bodies",
            reset_xform_properties=False,
            prepare_contact_sensors=False,
        )
        self.view.initialize()
        self.body_order = np.array(
            [self.view.prim_paths.index(path) for path in self.bodies]
        )
        print("PUMPKIN_RECORDING_BODIES", len(self.bodies), flush=True)

    def record(self, dt: float) -> None:
        """Sample measured poses every0.2seconds of simulated time.

        Args:
            dt: Actual simulator step duration in seconds.
        """
        self.steps += 1
        self.seconds += dt
        if self.seconds + 1e-7 < self.next_sample_seconds:
            return
        self.next_sample_seconds += self.sample_period
        sample_started = perf_counter()
        positions, quaternions = self.view.get_world_poses()
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(self.bodies), axis=0)
        poses[:, :3, :3] = Rotation.from_quat(
            np.asarray(quaternions)[self.body_order], scalar_first=True
        ).as_matrix()
        poses[:, :3, 3] = np.asarray(positions)[self.body_order]
        self.frames.append(poses)
        self.read_seconds += perf_counter() - sample_started
        self.times.append(self.seconds)
        if self.contact_sensor is not None and self.contact_source is not None:
            sources = list(dict.fromkeys([self.contact_source, *self.contact_watch]))
            indices = np.array(
                [self.contact_sensor.prim_paths.index(path) for path in sources]
            )
            forces = np.asarray(
                self.contact_sensor.get_contact_force_matrix(indices=indices, dt=dt)
            )
            net = np.asarray(
                self.contact_sensor.get_net_contact_forces(indices=indices, dt=dt)
            )
            with (self.output / "contact-frames.jsonl").open("a") as log:
                for source, force_row, normal in zip(sources, forces, net, strict=True):
                    pairs = {
                        path: force.tolist()
                        for path, force in zip(self.contact_filters, force_row)
                        if np.linalg.norm(force) > 1e-5
                    }
                    log.write(
                        json.dumps(
                            {
                                "seconds": self.seconds,
                                "body": source,
                                "net_normal_force_N": normal.tolist(),
                                "pairs": pairs,
                            }
                        )
                        + "\n"
                    )
        if self.seconds + 1e-7 >= self.next_save_seconds:
            self.next_save_seconds += 2.0
            print(
                "PUMPKIN_PHYSICS",
                self.steps,
                self.seconds,
                "wall",
                perf_counter() - self.started,
                "recording",
                self.read_seconds,
                flush=True,
            )
            self.save()

    def save(self) -> None:
        """Write all captured poses and actual simulated timestamps."""
        np.savez_compressed(
            self.output / "physics-frames.tmp.npz",
            poses=np.array(self.frames),
            seconds=np.array(self.times),
            physics_steps=self.steps,
        )
        (self.output / "physics-frames.tmp.npz").replace(
            self.output / "physics-frames.npz"
        )

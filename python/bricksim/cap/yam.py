"""Physical dual-YAM assembly primitives for Isaac Sim 5.1.

Transforms use column vectors and metres; simulator quaternions are WXYZ.
This controller uses joint drives, not brick pose updates or attachment shortcuts.
"""

from pathlib import Path

import numpy as np
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.xforms import get_world_pose
from scipy.spatial.transform import Rotation, Slerp

from bricksim.cap.basket import Basket, BulkBasket
from bricksim.cap.kinematics import YamKinematics
from bricksim.cap.policy import Connection
from bricksim.core import are_parts_connected, compute_connection_transform
from bricksim.utils.physics_step import wait_for_physics_step


class YamArm:
    """Joint-driven YAM with Cartesian interpolation and a contact gripper."""

    def __init__(
        self, world: World, robot: SingleArticulation, urdf: Path, root_path: str
    ):
        """Bind an initialized articulation and validate its flange kinematics.

        Args:
            world: Running Isaac world.
            robot: Initialized YAM articulation with eight named joints.
            urdf: Source geometry and kinematic description.
            root_path: Referenced USD robot root (above its root_joint).
        """
        self.world = world
        self.robot = robot
        self.kinematics = YamKinematics(urdf)
        self.indices = np.array([robot.get_dof_index(f"joint{i}") for i in range(1, 7)])
        self.fingers = np.array([robot.get_dof_index(f"joint{i}") for i in (7, 8)])
        self.base = self.pose(root_path + "/base")
        self.flange_path = root_path + "/gripper"
        # Distal pad midpoint, measured from the official meshes in flange coordinates.
        self.tool = np.eye(4)
        self.tool[:3, 3] = [0, 0, 0.14465]
        self.park_pose = self.pose(self.flange_path) @ self.tool
        self.grip_target = -0.04
        self.joint_target = self.robot.get_joint_positions()[self.indices]
        error = np.linalg.norm(
            self.base @ self.kinematics.forward(self.joint_target)
            - self.pose(self.flange_path)
        )
        if error > 0.001:
            print(
                "YAM_FK_DIAGNOSTIC",
                self.joint_target,
                self.base,
                self.kinematics.forward(self.joint_target),
                self.pose(self.flange_path),
                flush=True,
            )
            raise RuntimeError(f"Imported YAM differs from URDF kinematics: {error}")
        print(f"YAM_KINEMATICS_OK {robot.name} error={error:.6g}", flush=True)

    @staticmethod
    def pose(path: str) -> np.ndarray:
        """Read an actual world pose as a column-vector transform.

        Args:
            path: USD prim path, tracked through physics transforms.

        Returns:
            World-to-prim 4x4 transform in metres.
        """
        if not get_current_stage().GetPrimAtPath(path).IsValid():
            raise ValueError(f"Missing pose prim: {path}")
        # Direct Fabric lookup avoids temporary view IDs persisting on other prims.
        position, quaternion = get_world_pose(path, fabric=True)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(
            quaternion, scalar_first=True
        ).as_matrix()
        transform[:3, 3] = position
        return transform

    async def tick(self):
        """Hold the latest arm and finger targets for one physics step."""
        self.robot.apply_action(
            ArticulationAction(
                joint_positions=np.r_[
                    self.joint_target, self.grip_target, self.grip_target
                ],
                joint_indices=np.r_[self.indices, self.fingers],
            )
        )
        await wait_for_physics_step(self.world)

    async def grip(self, width: float):
        """Command finger separation; closure can stall against a physical brick.

        Args:
            width: Desired gap in metres, from zero to 0.0939.
        """
        self.grip_target = -float(np.clip(width / 2, 0, 0.04695))
        for _ in range(120):
            await self.tick()

    async def move(
        self, target: np.ndarray, speed: float = 0.07, tolerance: float = 0.0015
    ):
        """Interpolate the actual TCP to a target using joint position drives.

        Args:
            target: World-to-TCP 4x4 transform in metres.
            speed: Maximum Cartesian translation speed in metres per second.
            tolerance: Position tolerance in metres; angular limit is 0.04 rad.
        """
        start = self.pose(self.flange_path) @ self.tool
        rotations = Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]]))
        interpolation = Slerp([0, 1], rotations)
        angle = (rotations[0].inv() * rotations[1]).magnitude()
        duration = max(
            np.linalg.norm(target[:3, 3] - start[:3, 3]) / speed, angle / 0.6, 0.4
        )
        steps = int(np.ceil(duration / self.world.get_physics_dt()))
        for index in range(1, steps + 1):
            fraction = index / steps
            waypoint = np.eye(4)
            waypoint[:3, :3] = interpolation(fraction).as_matrix()
            waypoint[:3, 3] = start[:3, 3] * (1 - fraction) + target[:3, 3] * fraction
            flange = np.linalg.inv(self.base) @ waypoint @ np.linalg.inv(self.tool)
            self.joint_target = self.kinematics.inverse(flange, self.joint_target)
            await self.tick()
        for _ in range(90):
            await self.tick()
        actual = self.pose(self.flange_path) @ self.tool
        position_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        rotation_error = Rotation.from_matrix(
            target[:3, :3].T @ actual[:3, :3]
        ).magnitude()
        if position_error > tolerance or rotation_error > 0.04:
            raise RuntimeError(
                f"{self.robot.name} motion failed: position={position_error:.6f} m, "
                f"rotation={rotation_error:.4f} rad"
            )

    async def assemble(
        self,
        connection: Connection,
        paths: dict[int, str],
        height: float,
        reverse: bool,
        basket: Basket | None = None,
        loose_paths: dict[int, str] | None = None,
        source_id: int | None = None,
        press_depths: tuple[float, ...] | None = None,
        grasp_width: float | None = None,
        top_press: bool = False,
    ) -> dict[str, object]:
        """Grasp and place one brick, then require an actual connection.

        Args:
            connection: Desired stud/hole IDs, stud offset and quarter-turn yaw.
            paths: Mapping from target part IDs to actual scene prims.
            height: Hole brick height in metres.
            reverse: Rotate the grasp 180 degrees about the brick's vertical axis.
            basket: Optional shallow basket for live approach clearance checks.
            loose_paths: All unplaced brick paths; required with a basket.
            source_id: Physical supply ID when distinct from the target hole ID.
            press_depths: Optional successive downward offsets in metres; stop
                once a physical connection forms. Defaults to the scene behavior.
            grasp_width: Optional open-finger gap in metres for known stock width.
            top_press: If side pressing fails, release and press with closed fingertips.

        Returns:
            Pick evidence including observed source pose, attempts and lift height.
        """
        stud, hole = paths[connection.stud], paths[connection.hole]
        opening = basket.grasp_width if isinstance(basket, BulkBasket) else 0.06
        if grasp_width is not None:
            opening = grasp_width
        attempts = 2 if basket is not None else 1
        pick: dict[str, object] = {}
        for attempt in range(1, attempts + 1):
            before = self.pose(hole)
            if basket is not None:
                if loose_paths is None:
                    raise ValueError("Basket picks require every unplaced brick path")
                observations = {
                    part: self.pose(path) for part, path in loose_paths.items()
                }
                physical_id = connection.hole if source_id is None else source_id
                before = observations[physical_id]
                basket.check_grasp(physical_id, observations)
            grasp = before.copy()
            grasp[:3, :3] = before[:3, :3] @ np.diag(
                [-1, 1, -1] if reverse else [1, -1, -1]
            )
            grasp[:3, 3] += before[:3, 2] * (height - 0.0025)
            above = grasp.copy()
            above[2, 3] += 0.10
            print(
                f"YAM_GRASP {hole} attempt={attempt} source={before[:3, 3]}", flush=True
            )
            await self.grip(opening)
            await self.move(above)
            await self.move(grasp, speed=0.025)
            await self.grip(0)
            await self.move(above, speed=0.035)
            lifted = self.pose(hole)
            lift = float(lifted[2, 3] - before[2, 3])
            if lift >= 0.065:
                pick = {
                    "attempts": attempt,
                    "source_pose": before.tolist(),
                    "lift_m": lift,
                }
                break
            await self.grip(opening)
            for _ in range(120):
                await self.tick()
            if attempt == attempts:
                raise RuntimeError(
                    f"Grasp failed: {hole} did not lift with the fingers"
                )
            print(f"YAM_RETRY {hole}: reobserving after failed lift", flush=True)
        relative_grasp = np.linalg.inv(lifted) @ self.pose(self.flange_path) @ self.tool
        quaternion, position = compute_connection_transform(
            stud_path=stud,
            stud_if=1,
            hole_path=hole,
            hole_if=0,
            offset=connection.offset,
            yaw=connection.yaw,
        )
        relative = np.eye(4)
        relative[:3, :3] = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
        relative[:3, 3] = position
        goal = self.pose(stud) @ relative @ relative_grasp
        approach = goal.copy()
        approach[2, 3] += 0.10
        print(f"YAM_PLACE {hole} onto {stud}", flush=True)
        await self.move(approach)
        if press_depths is None:
            press_depths = (0.0005, 0.0015, 0.0025) if basket is not None else (0.0005,)
        for depth in press_depths:
            press = goal.copy()
            press[2, 3] -= depth
            await self.move(press, speed=0.015, tolerance=0.003)
            for _ in range(120):
                await self.tick()
            error = np.linalg.inv(self.pose(stud) @ relative) @ self.pose(hole)
            connected = are_parts_connected(stud, hole)
            print(
                f"YAM_PRESS {connection.hole} depth={depth:.4f} connected={connected} "
                f"offset={error[:3, 3]} "
                f"angle={Rotation.from_matrix(error[:3, :3]).magnitude():.6g}",
                flush=True,
            )
            pick["press_depth_m"] = depth
            if connected:
                break
        await self.grip(opening)
        await self.move(approach, speed=0.035)
        if top_press and not are_parts_connected(stud, hole):
            # Apply downward contact directly when a side grip slides during seating.
            tamp = self.pose(stud) @ relative
            tamp[:3, :3] = tamp[:3, :3] @ np.diag(
                [-1, 1, -1] if reverse else [1, -1, -1]
            )
            # Stud height plus distal pad extent below the TCP, both in metres.
            tamp[2, 3] += height + 0.0017 + 0.0022
            hover = tamp.copy()
            hover[2, 3] += 0.025
            await self.grip(0)
            await self.move(hover)
            for depth in press_depths:
                press = tamp.copy()
                press[2, 3] -= depth
                await self.move(press, speed=0.01, tolerance=0.003)
                for _ in range(120):
                    await self.tick()
                connected = are_parts_connected(stud, hole)
                print(
                    f"YAM_TOP_PRESS {connection.hole} depth={depth:.4f} "
                    f"connected={connected}", flush=True,
                )
                pick["top_press_depth_m"] = depth
                if connected:
                    break
            await self.move(hover, speed=0.035)
            await self.grip(opening)
            await self.move(approach)
        if not are_parts_connected(stud, hole):
            raise RuntimeError(f"No physical connection formed: {stud} -> {hole}")
        actual_relative = np.linalg.inv(self.pose(stud)) @ self.pose(hole)
        position_error = np.linalg.norm(actual_relative[:3, 3] - relative[:3, 3])
        angle_error = Rotation.from_matrix(
            relative[:3, :3].T @ actual_relative[:3, :3]
        ).magnitude()
        if position_error > 0.002 or angle_error > np.deg2rad(5):
            raise RuntimeError("Physical connection differs from requested offset/yaw")
        print(
            f"YAM_CONNECTED {connection.stud} {connection.hole} "
            f"position_error={position_error:.6g} angle_error={angle_error:.6g}",
            flush=True,
        )
        await self.move(self.park_pose)
        return pick

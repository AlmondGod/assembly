"""Interactive direct-control session for a compact, physical brick pile."""

import asyncio
import json
import time
from pathlib import Path

import numpy as np
from isaacsim.core.api.world import World
from scipy.spatial.transform import Rotation

from bricksim.cap.cartesian import CartesianPolicy
from bricksim.cap.policy import Assembly
from bricksim.cap.yam import YamArm
from bricksim.core import are_parts_connected, compute_connection_transform


class InteractiveSession:
    """Exchange numbered files with the external planner without evaluating code."""

    def __init__(
        self,
        output: Path,
        world: World,
        arms: dict[str, YamArm],
        paths: dict[int, str],
        supply_paths: dict[int, str],
        assembly: Assembly,
    ):
        """Bind the physical scene and its task target.

        Args:
            output: Directory for immutable numbered command and state files.
            world: Live simulation world.
            arms: The two joint-driven YAM arms.
            paths: Baseplate and planner-bound target paths.
            supply_paths: Original physical source ID to simulator path.
            assembly: Four-brick desired connection graph.
        """
        self.output = output
        self.world = world
        self.arms = arms
        self.paths = paths
        self.supply_paths = supply_paths
        self.assembly = assembly
        self.bindings: dict[int, int] = {}
        self.events: list[dict[str, object]] = []
        self.valid_connections: dict[int, bool] = {}

    def snapshot(self, index: int, error: str | None = None) -> dict[str, object]:
        """Capture live world transforms and native connection evidence.

        Args:
            index: State sequence number, starting at zero.
            error: Most recent command error, if execution failed.

        Returns:
            JSON-compatible state; positions and transforms use world metres.
        """
        connections = {}
        for hole, connection in self.assembly.connections.items():
            if hole not in self.paths or connection.stud not in self.paths:
                self.valid_connections[hole] = False
                connections[str(hole)] = {
                    "bound": False,
                    "connected": False,
                    "valid": False,
                }
                continue
            stud, brick = self.paths[connection.stud], self.paths[hole]
            quaternion, position = compute_connection_transform(
                stud_path=stud,
                stud_if=1,
                hole_path=brick,
                hole_if=0,
                offset=connection.offset,
                yaw=connection.yaw,
            )
            relative = np.eye(4)
            relative[:3, :3] = Rotation.from_quat(
                quaternion, scalar_first=True
            ).as_matrix()
            relative[:3, 3] = position
            actual = np.linalg.inv(YamArm.pose(stud)) @ YamArm.pose(brick)
            position_error = float(np.linalg.norm(actual[:3, 3] - relative[:3, 3]))
            angle_error = float(
                Rotation.from_matrix(relative[:3, :3].T @ actual[:3, :3]).magnitude()
            )
            connected = bool(are_parts_connected(stud, brick))
            self.valid_connections[hole] = bool(
                connected and position_error <= 0.002 and angle_error <= np.deg2rad(5)
            )
            connections[str(hole)] = {
                "bound": True,
                "source_id": self.bindings[hole],
                "connected": connected,
                "position_error_m": position_error,
                "angle_error_rad": angle_error,
                "valid": self.valid_connections[hole],
            }
        state = {
            "index": index,
            "error": error,
            "bindings": self.bindings,
            "connections": connections,
            "baseplate": YamArm.pose(self.paths[0]).tolist(),
            "bricks": {
                str(part): YamArm.pose(path).tolist()
                for part, path in self.supply_paths.items()
            },
            "arms": {
                name: {
                    "tool_pose": (arm.pose(arm.flange_path) @ arm.tool).tolist(),
                    "finger_gap_m": float(
                        -sum(arm.robot.get_joint_positions()[arm.fingers])
                    ),
                    "grip_target_m": float(-2 * arm.grip_target),
                }
                for name, arm in self.arms.items()
            },
            "events": self.events,
        }
        path = self.output / f"state-{index:03}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)
        return state

    async def run(self) -> None:
        """Execute numbered commands until finish or an idle timeout."""
        index = 0
        state = self.snapshot(index)
        while index < 80:
            command_path = self.output / f"command-{index:03}.json"
            deadline = time.monotonic() + 300
            while not command_path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Interactive planner did not send a command")
                await asyncio.sleep(0.2)
            command = json.loads(command_path.read_text())
            kind = command.get("kind")
            error = None
            try:
                if kind == "finish":
                    state = self.snapshot(index + 1)
                    valid = list(self.valid_connections.values())
                    result = {
                        "status": "success"
                        if len(valid) == 4 and all(valid)
                        else "incomplete",
                        "planner": "interactive assistant",
                        "bindings": self.bindings,
                        "connections": state["connections"],
                        "event_count": len(self.events),
                    }
                    (self.output / "result.json").write_text(
                        json.dumps(result, indent=2)
                    )
                    print("INTERACTIVE_RESULT " + json.dumps(result), flush=True)
                    await self.world.stop_async()
                    return
                if kind == "bind":
                    target, source = command["target"], command["source"]
                    if (
                        type(target) is not int
                        or target not in self.assembly.connections
                        or type(source) is not int
                        or source not in self.supply_paths
                        or source in self.bindings.values()
                        or target in self.bindings
                    ):
                        raise ValueError("Invalid or already-bound target/source")
                    self.bindings[target] = source
                    self.paths[target] = self.supply_paths[source]
                elif kind == "motion":
                    actions = CartesianPolicy.parse(command["program"])
                    for action in actions:
                        arm = self.arms[action.arm]
                        if action.kind == "grip":
                            await arm.grip(action.values[0])
                        else:
                            pose = np.eye(4)
                            pose[:3, 3] = action.values[:3]
                            pose[:3, :3] = Rotation.from_quat(
                                action.values[3:7], scalar_first=True
                            ).as_matrix()
                            await arm.move(
                                pose, speed=action.values[7], tolerance=0.003
                            )
                else:
                    raise ValueError("Command kind must be bind, motion, or finish")
            except (RuntimeError, ValueError, SyntaxError) as exc:
                error = str(exc)
            self.events.append({"index": index, "command": command, "error": error})
            index += 1
            state = self.snapshot(index, error)
            print(f"INTERACTIVE_STATE {index} error={error}", flush=True)
        raise RuntimeError("Interactive command limit reached")

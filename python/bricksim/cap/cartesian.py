"""Bounded Cartesian programs: model-selected TCP poses and gripper commands.

This independent policy contract has no simulator dependencies. Positions are
world-frame metres, quaternion order is WXYZ, and speeds are metres/second.
"""

import ast
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CartesianAction:
    """A validated move or grip command with its unchanged numeric arguments."""

    kind: str
    arm: str
    values: tuple[float, ...]


class CartesianPolicy:
    """Parse literal commands without evaluating generated Python."""

    @staticmethod
    def parse(source: str) -> list[CartesianAction]:
        """Validate a finite motion program before any execution.

        Args:
            source: Python text containing only awaited robot.move/grip calls.

        Returns:
            At most 128 commands, preserving the model's order and targets.
        """
        if len(source) > 32768:
            raise ValueError("Cartesian policy exceeds 32 KiB")
        tree = ast.parse(source)
        if not 1 <= len(tree.body) <= 128:
            raise ValueError("Expected 1..128 Cartesian commands")
        actions = []
        for statement in tree.body:
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Await)
                and isinstance(statement.value.value, ast.Call)
            ):
                raise ValueError("Only awaited Cartesian calls are allowed")
            call = statement.value.value
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "robot"
                and call.func.attr in ("move", "grip")
                and not call.keywords
                and len(call.args) in ((5, 9) if call.func.attr == "move" else (2,))
            ):
                raise ValueError("Expected positional robot.move or robot.grip")
            arm_node = call.args[0]
            if not isinstance(arm_node, ast.Constant) or arm_node.value not in (
                "left",
                "right",
            ):
                raise ValueError("Unknown arm")
            numbers = []
            for node in call.args[1:]:
                sign = 1
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                    sign, node = -1, node.operand
                if (
                    not isinstance(node, ast.Constant)
                    or not isinstance(node.value, (int, float))
                    or isinstance(node.value, bool)
                ):
                    raise ValueError("Coordinates must be finite numeric literals")
                value = sign * float(node.value)
                if not math.isfinite(value):
                    raise ValueError("Coordinates must be finite")
                numbers.append(value)
            if call.func.attr == "move":
                if len(numbers) == 4:
                    orientation = (
                        [0.0, 1.0, 0.0, 0.0]
                        if arm_node.value == "left"
                        else [0.0, 0.0, 1.0, 0.0]
                    )
                    numbers = numbers[:3] + orientation + numbers[3:]
                x, y, z, qw, qx, qy, qz, speed = numbers
                if not (0.02 <= x <= 0.55 and -0.4 <= y <= 0.4 and 0 <= z <= 0.5):
                    raise ValueError("TCP outside experiment workspace")
                if abs(math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) - 1) > 0.001:
                    raise ValueError("Quaternion must have unit norm")
                if not 0.005 <= speed <= 0.10:
                    raise ValueError("Speed must be 0.005..0.10 m/s")
            elif not 0 <= numbers[0] <= 0.0939:
                raise ValueError("Gripper width must be 0..0.0939 m")
            actions.append(
                CartesianAction(call.func.attr, str(arm_node.value), tuple(numbers))
            )
        return actions

    @staticmethod
    def prompt(observation: dict[str, object], task: str = "assembly") -> str:
        """Describe the task and primitive API without supplying a motion program.

        Args:
            observation: Recorded scene poses, dimensions and desired brick poses.
            task: Either assembly or grasp; grasp includes a worked example.

        Returns:
            Text for the model, explicitly distinguishing goals from tool poses.
        """
        if task == "grasp":
            match observation:
                case {"initial_brick_pose": {"position": [x, y, z]}} if all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in (x, y, z)
                ):
                    pass
                case _:
                    raise ValueError(
                        "Grasp observation requires three numeric coordinates"
                    )
            return (
                "Output exactly five Python lines that grasp and lift a brick. "
                "Use only these two calls:\n"
                'await robot.grip("left", width)\n'
                'await robot.move("left", x, y, z, speed)\n'
                "All values must be numeric literals. The robot uses "
                "inverse kinematics. "
                "Units are metres. "
                "For a brick at x=0.20, y=-0.18 with bottom z=0.0000, "
                "here is a complete example:\n"
                'await robot.grip("left", 0.026)\n'
                'await robot.move("left", 0.20, -0.18, 0.1071, 0.04)\n'
                'await robot.move("left", 0.20, -0.18, 0.0071, 0.015)\n'
                'await robot.grip("left", 0.0)\n'
                'await robot.move("left", 0.20, -0.18, 0.1071, 0.04)\n\n'
                f"Now grasp a DIFFERENT brick at x={x:.3f}, y={y:.3f}, "
                f"bottom z={z:.3f}. "
                "It has the same height. Output only the five Python lines "
                "for the new brick."
            )
        if task != "assembly":
            raise ValueError("Task must be assembly or grasp")
        task_geometry = {
            key: observation[key]
            for key in (
                "initial_brick_pose",
                "desired_brick_pose",
                "brick_height_m",
                "brick_width_m",
            )
        }
        return (
            "Write a short robot program to pick up one LEGO brick and attach it "
            "to its desired position. Output only Python code, about 10 lines. "
            'API: await robot.move("left", x, y, z, speed) moves the fingertip '
            "midpoint to WORLD coordinates in metres with inverse kinematics. "
            "It keeps the gripper pointing downward. "
            'await robot.grip("left", width) sets the finger gap in metres. '
            'Use literal numbers and the literal string "left". No variables, '
            "loops or arithmetic expressions. No other functions. "
            "Opening width 0.026 clears the brick. Width 0 closes onto the brick. "
            "The brick position is its BOTTOM centre. Side grasp height is "
            "bottom z + brick_height_m - 0.0025. Keep that same grasp offset "
            "when placing at the desired brick position. Lift at least 0.065 m "
            "before moving horizontally. Approach from above, open, descend, "
            "close, lift, transport, lower, press 0.0005 m, release, retreat. "
            "Use speed 0.04 for travel and 0.015 near contact. "
            "You must calculate the coordinates; no pick/place function exists. "
            "This is a single observed scene, without automatic retries. "
            f"Task: {json.dumps(task_geometry)}"
        )

"""Validate a small Python action language without executing generated Python."""

import ast
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Connection:
    """Desired stud-to-hole connection; offsets are studs, yaw is quarter turns."""

    stud: int
    hole: int
    offset: tuple[int, int]
    yaw: int


class Assembly:
    """A rooted assembly target with one incoming connection per movable part."""

    def __init__(self, path: Path):
        """Load and validate a native topology JSON file.

        Args:
            path: Target file, with baseplate ID zero and standard interfaces.
        """
        self.topology = json.loads(path.read_text())
        parts = self.topology["parts"]
        self.ids = {part["id"] for part in parts}
        if len(parts) != len(self.ids) or 0 not in self.ids:
            raise ValueError("Part IDs must be unique and include baseplate 0")
        for part in parts:
            if type(part["id"]) is not int or part["type"] != "brick":
                raise ValueError("Only integer IDs and brick parts are supported")
            if any(
                type(part["payload"][axis]) is not int or part["payload"][axis] <= 0
                for axis in ("L", "W", "H")
            ):
                raise ValueError("Brick dimensions must be positive integers")
        self.connections: dict[int, Connection] = {}
        for item in self.topology["connections"]:
            stud, hole = item["stud_id"], item["hole_id"]
            offset, yaw = item["offset"], item["yaw"]
            if (
                type(stud) is not int
                or type(hole) is not int
                or stud not in self.ids
                or hole not in self.ids
                or hole == 0
                or hole == stud
                or hole in self.connections
                or item["stud_iface"] != 1
                or item["hole_iface"] != 0
                or len(offset) != 2
                or any(type(v) is not int for v in offset)
                or type(yaw) is not int
                or yaw not in range(4)
            ):
                raise ValueError("Invalid or unsupported connection")
            self.connections[hole] = Connection(stud, hole, tuple(offset), yaw)
        placed = {0}
        while True:
            available = {c.hole for c in self.connections.values() if c.stud in placed}
            if available <= placed:
                break
            placed |= available
        if placed != self.ids:
            raise ValueError("Target must be a connected acyclic assembly rooted at 0")

    def parse_policy(self, source: str) -> list[tuple[Connection, str]]:
        """Compile literal ``await robot.assemble(stud, hole, arm)`` statements.

        Args:
            source: At most 16 KiB of Python, containing only bounded action calls.

        Returns:
            Validated connections and arm names in execution order.
        """
        if len(source) > 16384:
            raise ValueError("Policy exceeds 16 KiB")
        tree = ast.parse(source)
        if len(tree.body) != len(self.connections) or len(tree.body) > 64:
            raise ValueError("Policy must assemble every target exactly once (max 64)")
        placed = {0}
        actions = []
        for statement in tree.body:
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Await)
                and isinstance(statement.value.value, ast.Call)
            ):
                raise ValueError(
                    "Only await robot.assemble(stud, hole, arm) is allowed"
                )
            call = statement.value.value
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "robot"
                and call.func.attr == "assemble"
            ):
                raise ValueError("Unsupported action or nonliteral argument")
            if len(call.args) == 3 and not call.keywords:
                arguments = call.args
            elif (
                not call.args
                and len(call.keywords) == 3
                and {kw.arg for kw in call.keywords} == {"stud", "hole", "arm"}
            ):
                arguments = [
                    next(kw.value for kw in call.keywords if kw.arg == key)
                    for key in ("stud", "hole", "arm")
                ]
            else:
                raise ValueError("Expected exactly stud, hole and arm")
            if not all(isinstance(arg, ast.Constant) for arg in arguments):
                raise ValueError("Action arguments must be literals")
            values = [arg.value for arg in arguments if isinstance(arg, ast.Constant)]
            stud, hole, arm = values
            if (
                type(stud) is not int
                or type(hole) is not int
                or arm not in ("left", "right")
                or hole not in self.connections
            ):
                raise ValueError("Unknown brick or arm")
            connection = self.connections[hole]
            if connection.stud != stud or stud not in placed or hole in placed:
                raise ValueError("Policy violates target or dependency order")
            placed.add(hole)
            actions.append((connection, str(arm)))
        return actions

    def prompt(self) -> str:
        """Return the grounded task and the complete supported policy contract."""
        edges = [
            {
                "stud": c.stud,
                "hole": c.hole,
                "arm": "left" if index % 2 == 0 else "right",
            }
            for index, c in enumerate(self.connections.values())
        ]
        return (
            "Write Python code for two YAM robot arms to build the given assembly. "
            "Return ONLY code, no markdown and no explanation. "
            "The ONLY allowed statement is await robot.assemble(stud, hole, arm). "
            'stud and hole must be integer literals; arm must be "left" or "right". '
            "Baseplate 0 is already placed. Every other brick must be placed once. "
            "Place a supporting brick before placing a brick on it. "
            "Among available bricks, place the smallest hole ID first. "
            "Alternate left and right arms, starting with left. "
            "Use exactly the arm specified for each desired connection. "
            "The simulator handles the target offsets and physical grasping. "
            f"Desired connections: {json.dumps(edges)}"
        )

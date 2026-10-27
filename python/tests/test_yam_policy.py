"""Tests for model policy rejection and assembly dependency validation."""

import json
from pathlib import Path

import pytest

from bricksim.cap.policy import Assembly

TARGET = Path(__file__).resolve().parents[2] / "demos/assemblies/four_brick_tower.json"
POLICY = "\n".join(f'await robot.assemble({i}, {i + 1}, "left")' for i in range(4))


def test_policy_preserves_model_order_and_arm():
    """Compile real model keyword syntax and retain arm assignments."""
    source = "\n".join(
        f'await robot.assemble(stud={i}, hole={i + 1}, arm="{arm}")'
        for i, arm in enumerate(["left", "right", "left", "right"])
    )
    actions = Assembly(TARGET).parse_policy(source)
    assert [c.hole for c, _ in actions] == [1, 2, 3, 4]
    assert [arm for _, arm in actions] == ["left", "right", "left", "right"]
    assert actions[0][0].offset == (9, 8)


@pytest.mark.parametrize(
    "source",
    [
        POLICY.replace('await robot.assemble(0, 1, "left")', "import os"),
        POLICY.replace("0, 1", '0, __import__("os").system("false")'),
        POLICY.replace('"left"', '"unknown"'),
        POLICY.replace("0, 1", "True, 1"),
        POLICY.replace("0, 1", "0, 4"),
        "\n".join(reversed(POLICY.splitlines())),
        POLICY + '\nawait robot.assemble(0, 1, "left")',
        POLICY.replace("robot.assemble", "robot.__class__"),
        "while True:\n    " + POLICY.replace("\n", "\n    "),
        "x" * 16385,
    ],
)
def test_invalid_programs_fail_before_execution(source: str):
    """Reject arbitrary Python, invalid references, duplication and wrong order."""
    with pytest.raises((ValueError, SyntaxError)):
        Assembly(TARGET).parse_policy(source)


def test_disconnected_cycle_is_rejected(tmp_path: Path):
    """A cycle with no support on the baseplate cannot be a target."""
    topology = json.loads(TARGET.read_text())
    topology["connections"][0]["stud_id"] = 4
    path = tmp_path / "cycle.json"
    path.write_text(json.dumps(topology))
    with pytest.raises(ValueError, match="acyclic"):
        Assembly(path)


def test_unmodified_four_brick_target_compiles():
    """All four requested connections must appear in the parsed program."""
    assert len(Assembly(TARGET).parse_policy(POLICY)) == 4

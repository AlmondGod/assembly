"""Direct-control programs preserve model commands and reject unsafe syntax."""

import pytest

from bricksim.cap.cartesian import CartesianPolicy


def test_cartesian_preserves_motion_and_gripper():
    """Negative coordinates, WXYZ orientation and opening remain unchanged."""
    actions = CartesianPolicy.parse(
        'await robot.grip("left", 0.026)\n'
        'await robot.move("left", 0.24, -0.14, 0.1071, 0, 1, 0, 0, 0.025)\n'
        'await robot.grip("left", 0)'
    )
    assert [a.kind for a in actions] == ["grip", "move", "grip"]
    assert actions[1].values == (0.24, -0.14, 0.1071, 0, 1, 0, 0, 0.025)


@pytest.mark.parametrize(
    "source",
    [
        "",
        "import os",
        'await robot.assemble(0,1,"left")',
        'await robot.grip("left", True)',
        'await robot.grip("left", 1e999)',
        'await robot.grip("left", 0.1)',
        'await robot.grip("left", -0.1)',
        'await robot.grip("third", 0)',
        'await robot.grip(arm="left", width=0)',
        'await robot.grip("left", 0.02+0.01)',
        'await robot.move("left",0.2,0,0.2,0,0,0,0,0.02)',
        'await robot.move("left",0.2,0,-0.1,0,1,0,0,0.02)',
        'await robot.move("left",0.2,0,0.2,0,1,0,0,0)',
        'await robot.move("left",0.2,0,0.2,0,1,0,0,1)',
        'await robot.grip("left", 0)\n' * 129,
        'while True:\n    await robot.grip("left", 0)',
    ],
)
def test_reject_invalid_cartesian_program(source: str):
    """No Python execution, nonfinite targets or unbounded commands are accepted."""
    with pytest.raises((ValueError, SyntaxError)):
        CartesianPolicy.parse(source)


def test_position_only_move_uses_documented_orientation():
    """Position-only IK calls use fixed downward poses for the chosen arm."""
    actions = CartesianPolicy.parse(
        'await robot.move("left", 0.24, -0.14, 0.1, 0.04)\n'
        'await robot.move("right", 0.24, 0.14, 0.1, 0.04)'
    )
    assert actions[0].values == (0.24, -0.14, 0.1, 0, 1, 0, 0, 0.04)
    assert actions[1].values == (0.24, 0.14, 0.1, 0, 0, 1, 0, 0.04)


def test_grasp_prompt_changes_position_and_keeps_example_distinct():
    """The example and the target must not share source coordinates."""
    prompt = CartesianPolicy.prompt(
        {"initial_brick_pose": {"position": [0.24, -0.14, 0.0]}}, task="grasp"
    )
    assert "x=0.20, y=-0.18" in prompt
    assert "x=0.240, y=-0.140, bottom z=0.000" in prompt
    assert 'await robot.move("left", 0.20, -0.18, 0.0071, 0.015)' in prompt

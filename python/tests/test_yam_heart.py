"""Verify the heart's brick geometry and each assigned arm's reachable poses."""

from collections import Counter
from pathlib import Path

import numpy as np

from bricksim.cap.heart import HeartLayout
from bricksim.cap.kinematics import YamKinematics
from bricksim.cap.policy import Assembly

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "demos/assemblies/eleven_brick_heart.json"


def test_heart_inventory_and_layer_footprints():
    """Reconstruct the published heart, with no overlaps and real stud support."""
    target = Assembly(TARGET)
    assert set(HeartLayout.poses()) == target.ids
    parts = {part["id"]: part["payload"] for part in target.topology["parts"]}
    assert Counter((p["L"], p["W"], p["H"]) for i, p in parts.items() if i) == {
        (2, 4, 3): 6,
        (2, 2, 3): 5,
    }
    # Place both small outer bricks before the middle bridge to clear the fingers.
    assert [parts[i]["W"] for i in (7, 8, 9)] == [2, 2, 4]
    origins = {0: np.array([0, 0, 0])}
    occupied: dict[int, set[tuple[int, int]]] = {}
    for connection in target.connections.values():
        support = origins[connection.stud]
        origin = support + [*connection.offset, 1]
        origins[connection.hole] = origin
        part, parent = parts[connection.hole], parts[connection.stud]
        footprint = {
            (x, y)
            for x in range(origin[0], origin[0] + part["L"])
            for y in range(origin[1], origin[1] + part["W"])
        }
        supporting_studs = {
            (x, y)
            for x in range(support[0], support[0] + parent["L"])
            for y in range(support[1], support[1] + parent["W"])
        }
        assert footprint & supporting_studs
        assert not footprint & occupied.setdefault(int(origin[2]), set())
        occupied[int(origin[2])] |= footprint
    expected_y = [
        {3, 4},
        set(range(2, 6)),
        set(range(8)),
        set(range(8)),
        set(range(8)),
        {1, 2, 5, 6},
    ]
    for level, columns in enumerate(expected_y, 1):
        assert occupied[level] == {(x, y + 6) for x in (9, 10) for y in columns}


def test_heart_policy_uses_two_arms_and_every_target_once():
    """All eleven target anchors can be executed in bottom-up order."""
    target = Assembly(TARGET)
    source = "\n".join(
        f'await robot.assemble({c.stud}, {c.hole}, "{arm}")'
        for c, arm in zip(
            target.connections.values(), ["left", "right"] * 6, strict=False
        )
    )
    actions = target.parse_policy(source)
    assert [c.hole for c, _ in actions] == list(range(1, 12))
    assert {arm for _, arm in actions} == {"left", "right"}


def test_heart_pick_and_place_poses_are_reachable():
    """Each arm reaches source, transfer and direct top-press poses."""
    target = Assembly(TARGET)
    model = YamKinematics(ROOT / "resources/robots/yam/yam.urdf")
    parts = {part["id"]: part["payload"] for part in target.topology["parts"]}
    origins = {0: np.array([0, 0, 0])}
    sources = HeartLayout.poses()
    assert len(sources) == 12
    for connection in target.connections.values():
        part = connection.hole
        origins[part] = origins[connection.stud] + [*connection.offset, 1]
        x, y, level = origins[part]
        placement = np.eye(4)
        placement[:3, 3] = [
            sources[0][0, 3] + (x + parts[part]["L"] / 2 - 10) * 0.008,
            (y + parts[part]["W"] / 2 - 10) * 0.008,
            0.0032 + (level - 1) * 0.0096,
        ]
        side = -1 if part % 2 else 1
        joints = np.array([0, 1.5, 1.5, 0, 0, 0.0])
        for pose in (sources[part], placement):
            flange = pose.copy()
            flange[:3, :3] = np.diag([1, -1, -1] if side == -1 else [-1, 1, -1])
            flange[2, 3] += 0.0096 - 0.0025 + 0.14465
            flange[:3, 3] -= [-0.02, side * 0.26, 0]
            joints = model.inverse(flange, joints)
            np.testing.assert_allclose(model.forward(joints), flange, atol=1e-4)
            flange[2, 3] += 0.10
            joints = model.inverse(flange, joints)
            np.testing.assert_allclose(model.forward(joints), flange, atol=1e-4)
            if pose is placement:
                for extra_height in (0.0064, 0.0314):
                    press_flange = flange.copy()
                    press_flange[2, 3] += extra_height - 0.10
                    joints = model.inverse(press_flange, joints)
                    np.testing.assert_allclose(
                        model.forward(joints), press_flange, atol=1e-4
                    )


def test_heart_outward_park_poses_are_reachable():
    """Both parked flanges can stay directly over their own side of the table."""
    model = YamKinematics(ROOT / "resources/robots/yam/yam.urdf")
    for side in (-1, 1):
        flange = np.eye(4)
        flange[:3, :3] = np.diag([1, -1, -1] if side == -1 else [-1, 1, -1])
        flange[:3, 3] = [0.28, 0.0, 0.32]
        joints = model.inverse(flange, np.array([0, 1.5, 1.5, 0, 0, 0]))
        np.testing.assert_allclose(model.forward(joints), flange, atol=1e-4)
        assert np.all(joints >= model.limits[0])
        assert np.all(joints <= model.limits[1])

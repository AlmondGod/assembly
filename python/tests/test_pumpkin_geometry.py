"""Check sourced inventory, support graph and real insertion voids."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

RESOURCE = Path(__file__).resolve().parents[2] / "resources/assemblies/pumpkin40055"


def test_inventory_and_collision_assets():
    """Every sourced instance has both visual and collision geometry."""
    manifest = json.loads((RESOURCE / "assembly.json").read_text())
    assert len(manifest["parts"]) == 52
    assert len({part["id"] for part in manifest["parts"]}) == 52
    for part in manifest["parts"]:
        visual = np.load(RESOURCE / "meshes" / (part["design"] + ".npz"))
        collision = np.load(RESOURCE / "collisions" / (part["design"] + ".npz"))
        assert visual["vertices"].shape[1] == 3
        assert visual["faces"].shape[1] == 3
        assert len(collision.files) >= 2
        for name in collision.files:
            if name.startswith("v"):
                assert ConvexHull(collision[name]).volume > 0


def test_support_graph_reaches_every_stud_connected_part():
    """The round foundation supports all50other stud-connected instances."""
    data = json.loads((RESOURCE / "connectors.json").read_text())
    edges = {(x["stud"], x["socket"]) for x in data["stud_connections"]}
    placed = {44}
    while True:
        ready = {
            b
            for _, b in edges
            if b not in placed and all(a in placed for a, child in edges if child == b)
        }
        if not ready:
            break
        placed.update(ready)
    assert placed == set(range(52)) - {0}


def test_collision_solids_leave_stud_socket_cores_open():
    """Insertion samples must not lie inside a convex collider.

    The 2.2mm test radius is below the2.70mm collision socket clearance and
    close to the2.35mm stud proxy radius. Convex hulls filling the cavities
    fail this test, even if they pass a floor-drop test.
    """
    data = json.loads((RESOURCE / "connectors.json").read_text())
    for design, ports in data["designs"].items():
        arrays = np.load(RESOURCE / "collisions" / (design + ".npz"))
        hulls = [
            ConvexHull(arrays[name]).equations
            for name in arrays.files
            if name.startswith("v")
        ]
        for port in ports:
            if port["gender"] != "F" or abs(port["radius"] - 0.0024) > 0.0001:
                continue
            axis = np.array(port["axis"])
            x = np.cross(axis, [1, 0, 0])
            x /= np.linalg.norm(x)
            z = np.cross(axis, x)
            points = np.array(
                [
                    np.array(port["position"])
                    + axis * depth
                    + radius * (x * np.cos(angle) + z * np.sin(angle))
                    for depth in [0.0004, 0.0008, 0.0012]
                    for radius in [0, 0.0018, 0.0022, 0.00255]
                    for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False)
                ]
            )
            for hull in hulls:
                assert not np.any(
                    (points @ hull[:, :3].T + hull[:, 3] < -0.00003).all(1)
                ), (design, port)


def test_grasp_plan_preserves_dependencies_and_separates_supply_parts():
    """The generated setup covers the inventory and avoids tray overlaps."""
    manifest = json.loads((RESOURCE / "assembly.json").read_text())
    parts = {p["id"]: p for p in manifest["parts"]}
    plan = json.loads((RESOURCE / "grasp-plan.json").read_text())
    connectors = json.loads((RESOURCE / "connectors.json").read_text())
    edges = {(c["stud"], c["socket"]) for c in connectors["stud_connections"]}
    edges.add((1, 0))
    assert len(plan["order"]) == 51
    assert set(plan["order"]) == set(parts) - {44}
    placed = {44}
    for part in plan["order"]:
        assert {a for a, b in edges if b == part} <= placed
        placed.add(part)
    assert {g["part"] for g in plan["grasps"]} == set(parts) - {44}
    conversion = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    bounds = []
    for grasp in plan["grasps"]:
        source = np.array(grasp["source_pose_world_m"])
        hand = np.array(grasp["grasp_in_part_m"])
        for transform in [source, hand]:
            assert np.isfinite(transform).all()
            np.testing.assert_allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8
            )
            np.testing.assert_allclose(transform[3], [0, 0, 0, 1])
        design = parts[grasp["part"]]["design"]
        points = np.load(RESOURCE / "meshes" / f"{design}.npz")["vertices"]
        world = points @ conversion.T @ source[:3, :3].T + source[:3, 3]
        assert world[:, 2].min() >= 0.0129
        bounds.append((world[:, :2].min(0), world[:, :2].max(0)))
    for index, (low, high) in enumerate(bounds):
        for other_low, other_high in bounds[index + 1 :]:
            overlap = np.minimum(high, other_high) - np.maximum(low, other_low)
            assert np.any(overlap < -0.0005)


def test_supply_posts_clear_the_complete_collision_solids():
    """No complete supply post may penetrate a hollow wall or sloped underside."""
    supply = json.loads((RESOURCE / "supply-posts.json").read_text())
    half = supply["post_width_m"] / 2
    conversion = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    for design, posts in supply["designs"].items():
        arrays = np.load(RESOURCE / "collisions" / f"{design}.npz")
        assert len(posts) >= 3
        for name in arrays.files:
            if not name.startswith("v"):
                continue
            vertices = arrays[name] @ conversion.T
            equations = ConvexHull(vertices).equations
            for x, y, top in posts:
                if np.any(vertices.max(0)[:2] < [x - half, y - half]) or np.any(
                    vertices.min(0)[:2] > [x + half, y + half]
                ):
                    continue
                intersection = linprog(
                    [0.0, 0.0, 0.0],
                    A_ub=equations[:, :3],
                    b_ub=-equations[:, 3],
                    bounds=[
                        (x - half, x + half),
                        (y - half, y + half),
                        (-0.013, top + 0.0003),
                    ],
                    method="highs",
                )
                assert intersection.status == 2, (design, x, y, top)


def test_cone_upper_bore_accepts_the_plume_stem():
    """The upper open stud must preserve its continuous bar-insertion passage."""
    arrays = np.load(RESOURCE / "collisions/59900.npz")
    points = np.array(
        [
            [radius * np.cos(angle), height, radius * np.sin(angle)]
            for height in [0.0068, 0.0080, 0.0096, 0.0110]
            for radius in [0.0, 0.0016, 0.0019]
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ]
    )
    for name in arrays.files:
        if not name.startswith("v"):
            continue
        planes = ConvexHull(arrays[name]).equations
        assert not np.any((points @ planes[:, :3].T + planes[:, 3] < -1e-6).all(1))

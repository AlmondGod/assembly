# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Compare stock finger and narrow-tip clearance against the CAD support sequence.

This is a static geometric audit, not a robot trajectory or successful grasp.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from bricksim.cap.kinematics import YamKinematics

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "resources/assemblies/pumpkin40055/grasp-plan.json",
)
args = parser.parse_args()
r = ROOT / "resources/assemblies/pumpkin40055"
p = {p["id"]: p for p in json.loads((r / "assembly.json").read_text())["parts"]}
c = json.loads((r / "connectors.json").read_text())
C = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
transforms = {}
bounds = {}
hulls = {}
for part, info in p.items():
    T = np.eye(4)
    T[:3, :3] = C @ info["rotation_ldd"] @ C.T
    T[:3, 3] = C @ np.array(info["translation_ldd"]) * 0.01 + [0.212, 0.024, 0.004]
    transforms[part] = T
    d = info["design"]
    if d not in bounds:
        v = np.load(r / "meshes" / f"{d}.npz")["vertices"] @ C.T
        bounds[d] = np.array([v.min(0), v.max(0)])
    a = np.load(r / "collisions" / f"{d}.npz")
    hs = []
    for i in range(len(a.files) // 2):
        v = a[f"v{i}"] @ C.T @ T[:3, :3].T + T[:3, 3]
        hs.append((v.min(0), v.max(0), ConvexHull(v).equations))
    hulls[part] = hs
edges = {(a["stud"], a["socket"]) for a in c["stud_connections"]} | {(1, 0)}
placed = {44}
order = []
while len(placed) < 52:
    ready = [
        i for i in p if i not in placed and all(a in placed for a, b in edges if b == i)
    ]
    part = min(
        ready, key=lambda i: (transforms[i][2, 3] + bounds[p[i]["design"]][1, 2], i)
    )
    placed.add(part)
    order.append(part)
faces = np.array(
    [
        [0, 1, 3],
        [0, 3, 2],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [2, 3, 7],
        [2, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
        [1, 5, 7],
        [1, 7, 3],
    ]
)


def intersections(triangles: np.ndarray, previous: set[int]) -> list[int]:
    """Find intersections with previously placed convex collision proxies.

    Args:
        triangles: World-space finger triangle vertices, shape(N,3,3), in metres.
        previous: IDs already placed in the support sequence.

    Returns:
        IDs whose collision solids intersect a finger surface.
    """
    hits = []
    tri_low = triangles.min(1)
    tri_high = triangles.max(1)
    whole_low = tri_low.min(0)
    whole_high = tri_high.max(0)
    for prior in previous:
        prior_low = np.min([h[0] for h in hulls[prior]], axis=0)
        prior_high = np.max([h[1] for h in hulls[prior]], axis=0)
        if np.any(whole_high < prior_low - 1e-6) or np.any(
            whole_low > prior_high + 1e-6
        ):
            continue
        found = False
        for low, high, planes in hulls[prior]:
            candidates = triangles[
                np.all(tri_high >= low - 1e-6, axis=1)
                & np.all(tri_low <= high + 1e-6, axis=1)
            ]
            for tri in candidates:
                polygon = list(tri)
                for plane in planes:
                    clipped = []
                    if not polygon:
                        break
                    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
                        d0 = start @ plane[:3] + plane[3]
                        d1 = end @ plane[:3] + plane[3]
                        if d0 <= -1e-6:
                            clipped.append(start)
                        if (d0 <= -1e-6) != (d1 <= -1e-6):
                            clipped.append(
                                start + (end - start) * ((-1e-6 - d0) / (d1 - d0))
                            )
                    polygon = clipped
                if polygon:
                    found = True
                    break
            if found:
                break
        if found:
            hits.append(prior)
    return hits


# Prototype 2x6x12mm pad extensions, TCP12mm below the stock tool frame.
stock = []
for side_sign in [-1, 1]:
    v = np.array(
        [
            [x, y, z]
            for x in [-0.001, 0.001]
            for y in [-0.003, 0.003]
            for z in [-0.010, 0.002]
        ]
    )
    v[:, 0] += side_sign * (0.008 + 0.001 + 0.0003)
    stock.append((v[faces], side_sign))
kin = YamKinematics(ROOT / "resources/robots/yam/yam.urdf")
tool = np.eye(4)
tool[2, 3] = 0.15665
source_poses = {44: transforms[44]}
source_indices = {}
for index, part in enumerate(sorted(set(p) - {44})):
    side = -1 if index % 2 == 0 else 1
    slot = index // 2
    b = bounds[p[part]["design"]]
    size = b[1] - b[0]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(
        "z", np.pi / 2 if size[0] > size[1] + 0.0001 else 0
    ).as_matrix()
    T[:3, 3] = [
        0.15 + 0.044 * (slot % 7),
        side * (0.10 + 0.064 * (slot // 7)),
        0.013 - b[0, 2],
    ]
    if part == 0:
        T[:2, 3] = [0.172, -0.18]
    T[:2, 3] -= (T[:3, :3] @ b.mean(0))[:2]
    source_poses[part] = T
    source_indices[part] = (index, side)
seeds = {}
for side in [-1, 1]:
    base = np.eye(4)
    base[:3, 3] = [-0.02, side * 0.26, 0]
    home = np.eye(4)
    home[:3, :3] = np.diag([1, -1, -1] if side == -1 else [-1, 1, -1])
    home[:3, 3] = [0.28, -side * 0.10, 0.32]
    seeds[side] = kin.inverse(
        home, np.array([0.0, 1.5, 1.5, 0, 0, 0]), residual_limit=0.003
    )


def active_width(part: int, grasp: np.ndarray) -> tuple[float, float] | None:
    """Estimate material span within the narrow finger contact window.

    Args:
        part: Source inventory ID.
        grasp: Column-vector part-to-TCP transform in metres.

    Returns:
        Minimum and maximum material X in TCP coordinates, or None if empty.
    """
    a = np.load(r / "meshes" / f"{p[part]['design']}.npz")
    v = a["vertices"] @ C.T
    v = (v - grasp[:3, 3]) @ grasp[:3, :3]
    triangles = v[a["faces"]]
    planes = np.array(
        [
            [0, 1, 0, -0.0028],
            [0, -1, 0, -0.0028],
            [0, 0, 1, -0.0018],
            [0, 0, -1, -0.010],
        ]
    )
    good = triangles[
        (triangles[:, :, 1].min(1) < 0.0028)
        & (triangles[:, :, 1].max(1) > -0.0028)
        & (triangles[:, :, 2].min(1) < 0.0018)
        & (triangles[:, :, 2].max(1) > -0.010)
    ]
    xs = []
    for tri in good:
        polygon = list(tri)
        for plane in planes:
            clipped = []
            if not polygon:
                break
            for start, end in zip(polygon, polygon[1:] + polygon[:1]):
                d0 = start @ plane[:3] + plane[3]
                d1 = end @ plane[:3] + plane[3]
                if d0 <= 0:
                    clipped.append(start)
                if (d0 <= 0) != (d1 <= 0):
                    clipped.append(start + (end - start) * (-d0 / (d1 - d0)))
            polygon = clipped
        xs.extend(x[0] for x in polygon)
    if not xs:
        return None
    return float(min(xs)), float(max(xs))


placed = {44}
reports = []
started = time.monotonic()
for part in order:
    d = p[part]["design"]
    b = bounds[d]
    size = b[1] - b[0]
    default_axis = 1 if size[0] > size[1] + 0.0001 else 0
    height = b[1, 2] - (
        0.0004 if size[2] < 0.0055 else 0.0015 if d == "54200" else 0.0037
    )
    if d == "64647":
        height = 0.0055
    heights = list(
        dict.fromkeys([height, b[1, 2] - 0.0004, b[1, 2] + 0.0004, b[1, 2] + 0.001])
    )
    chosen = None
    attempts = 0
    for z in heights:
        for axis in [default_axis, 1 - default_axis]:
            tangent = 1 - axis
            outward = (transforms[part] @ np.r_[b.mean(0), 1])[:2] - [0.22, 0]
            sign = np.sign(transforms[part][:2, tangent] @ outward)
            default_shift = (
                sign * min(0.002, size[tangent] / 4) if len(placed) > 1 else 0
            )
            if d == "64647":
                default_shift = 0.005 - b.mean(0)[tangent] if axis == 1 else 0.0
            shifts = list(
                dict.fromkeys(
                    [
                        default_shift,
                        0,
                        -default_shift,
                        min(0.003, size[tangent] / 3),
                        -min(0.003, size[tangent] / 3),
                    ]
                )
            )
            for shift in shifts:
                grasp = np.eye(4)
                grasp[:3, :3] = Rotation.from_euler(
                    "z", axis * np.pi / 2
                ).as_matrix() @ np.diag([1.0, -1, -1])
                grasp[:3, 3] = b.mean(0)
                grasp[2, 3] = z
                grasp[tangent, 3] += shift
                active = active_width(part, grasp)
                if active is None:
                    continue
                width = active[1] - active[0]
                if width < 0.0005:
                    continue
                grasp[:3, 3] += grasp[:3, 0] * np.mean(active)
                attempts += 1
                _, side = source_indices[part]
                base = np.eye(4)
                base[:3, 3] = [-0.02, side * 0.26, 0]
                park = np.diag([1, -1, -1] if side == -1 else [-1, 1, -1])
                target_choices = []
                for flip in [False, True]:
                    g = grasp.copy()
                    if flip:
                        g[:3, :3] = g[:3, :3] @ np.diag([-1, -1, 1])
                    goal = transforms[part] @ g
                    ts = []
                    for tri, side_sign in stock:
                        tri = tri.copy()
                        tri[:, :, 0] += side_sign * (width - 0.016) / 2
                        ts.extend(tri @ goal[:3, :3].T + goal[:3, 3])
                    hits = intersections(np.array(ts), placed)
                    if not hits:
                        target_choices.append(g)
                if not target_choices:
                    continue
                for source_delta in [0, np.pi]:
                    source = source_poses[part].copy()
                    source_center = (source @ np.r_[b.mean(0), 1])[:3]
                    source[:3, :3] = (
                        Rotation.from_euler("z", source_delta).as_matrix()
                        @ source[:3, :3]
                    )
                    source[:3, 3] = source_center - source[:3, :3] @ b.mean(0)
                    choices = [
                        (
                            Rotation.from_matrix(
                                park.T @ source[:3, :3] @ g[:3, :3]
                            ).magnitude(),
                            g,
                        )
                        for g in target_choices
                    ]
                    for _, g in sorted(choices, key=lambda x: x[0]):
                        try:
                            q = kin.inverse(
                                np.linalg.inv(base) @ source @ g @ np.linalg.inv(tool),
                                seeds[side],
                                residual_limit=0.003,
                            )
                            try:
                                kin.inverse(
                                    np.linalg.inv(base)
                                    @ transforms[part]
                                    @ g
                                    @ np.linalg.inv(tool),
                                    q,
                                    residual_limit=0.0001,
                                )
                                via_park = False
                            except RuntimeError:
                                kin.inverse(
                                    np.linalg.inv(base)
                                    @ transforms[part]
                                    @ g
                                    @ np.linalg.inv(tool),
                                    seeds[side],
                                    residual_limit=0.0001,
                                )
                                via_park = True
                        except RuntimeError:
                            continue
                        chosen = {
                            "part": part,
                            "design": d,
                            "grasp_in_part_m": g.tolist(),
                            "source_pose_world_m": source.tolist(),
                            "jaw_axis": axis,
                            "via_park": via_park,
                            # Recorded Part47 grip tilt makes the 70mm transit
                            # exceed reach; 30mm clears its low foundation.
                            "approach_height_m": 0.03 if part == 47 else 0.07,
                            "estimated_contact_width_m": width,
                            "opening_width_m": float(size[axis]),
                            "opening_m": float(
                                min(
                                    0.09,
                                    size[axis]
                                    + (0.008 if size[axis] > 0.032 else 0.014),
                                )
                            ),
                            "height_above_default_m": float(z - height),
                            "static_collisions": [],
                            "candidates_tested": attempts,
                        }
                        break
                    if chosen:
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen:
            break
    if not chosen:
        chosen = {
            "part": part,
            "design": d,
            "failed": True,
            "candidates_tested": attempts,
        }
    reports.append(chosen)
    placed.add(part)
    print(
        part,
        "FAILED"
        if chosen.get("failed")
        else (
            chosen["jaw_axis"],
            chosen["height_above_default_m"],
            chosen["candidates_tested"],
        ),
        flush=True,
    )
    args.output.with_suffix(".partial.json").write_text(
        json.dumps(
            {
                "static_only": True,
                "assembly_offset_world_m": [0.212, 0.024, 0.004],
                "order": order,
                "source_row_pitch_m": 0.064,
                "tool_flange_offset_m": 0.15665,
                "fingertip_dimensions_m": [0.002, 0.006, 0.012],
                "fingertip_closed_inner_gap_m": 0.0002,
                "source_urdf_sha256": hashlib.sha256(
                    (ROOT / "resources/robots/yam/yam.urdf").read_bytes()
                ).hexdigest(),
                "source_ik_residual_limit_m": 0.003,
                "target_ik_residual_limit_m": 0.0001,
                "planner": (
                    "Convex-proxy clearance and endpoint IK; "
                    "not physical trajectory validation"
                ),
                "elapsed_seconds": time.monotonic() - started,
                "grasps": reports,
            },
            indent=2,
        )
    )

failed = [g["part"] for g in reports if g.get("failed")]
if failed:
    raise RuntimeError(f"No feasible geometric grasp for parts {failed}")
args.output.with_suffix(".partial.json").replace(args.output)

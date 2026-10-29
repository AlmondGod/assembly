# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Generate hollow collision proxies with explicit rigid insertion clearances.

These are collision proxies, not exact material meshes. Visual geometry stays
unchanged. Stud sockets use2.70mmradius and bars2.00mm; retention is modeled separately.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1] / "resources/assemblies/pumpkin40055"
OUT = ROOT / "collisions"
MESH = ROOT / "meshes"


def clip(points: np.ndarray, normal: np.ndarray, bound: float) -> np.ndarray | None:
    """Intersect a convex solid with normal.dot(x)<=bound.

    Args:
        points: Convex vertices in metres.
        normal: Unit plane normal.
        bound: Signed plane offset in metres.

    Returns:
        Clipped vertices, or None if the intersection has zero volume.
    """
    distance = points @ normal - bound
    if np.all(distance <= 1e-10):
        return points
    if np.all(distance >= -1e-10):
        return None
    hull = ConvexHull(points)
    edges = set(
        tuple(sorted(e))
        for face in hull.simplices
        for e in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
    )
    crossings = [
        points[a] + (points[b] - points[a]) * distance[a] / (distance[a] - distance[b])
        for a, b in edges
        if distance[a] * distance[b] < -1e-22
    ]
    combined = np.unique(
        np.round(np.vstack([points[distance <= 1e-10], crossings]), 10), axis=0
    )
    if (
        len(combined) < 4
        or np.linalg.matrix_rank(combined - combined.mean(0), tol=1e-10) < 3
    ):
        return None
    hull = ConvexHull(combined)
    if hull.volume < 1e-15:
        return None
    return combined[hull.vertices]


ports = json.loads((ROOT / "connectors.json").read_text())["designs"]
for design, connectors in ports.items():
    points = np.load(MESH / f"{design}.npz")["vertices"]
    if design == "64647":
        # A flat blade gives the fingertips a real contact face. The visual
        # sculpt remains unchanged; the stepped bar preserves connector size.
        projection = points[points[:, 1] >= 0.0024 - 1e-7][:, :2]
        profile = projection[ConvexHull(projection).vertices]
        blade = np.array([[x, y, z] for x, y in profile for z in [-0.0004, 0.0004]])
        solids = [blade]
        for radius, bottom, top in [(0.0016, 0.0, 0.0024), (0.0008, -0.0024, 0.0)]:
            solids.append(
                np.array(
                    [
                        [radius * np.cos(angle), height, radius * np.sin(angle)]
                        for height in [bottom, top]
                        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False)
                    ]
                )
            )
    else:
        height = (
            0.0096
            if design in ["3010", "3009", "3039", "3660", "3665", "59900"]
            else 0.0032
        )
        if design == "54200":
            height = 0.00624
        body = points[points[:, 1] <= height + 1e-6]
        body = body[ConvexHull(body).vertices]
        solids = [body]
        if design in ["3710", "3010", "3795", "3020", "3666", "3009", "87580"]:
            low = points.min(0)
            high = points.max(0)
            high[1] = height
            boxes = [(np.array([low[0], height - 0.0012, low[2]]), high.copy())]
            for axis in [0, 2]:
                for sign in [0, 1]:
                    a = low.copy()
                    b = high.copy()
                    b[1] = height - 0.0012
                    if sign == 0:
                        b[axis] = a[axis] + 0.0012
                    else:
                        a[axis] = b[axis] - 0.0012
                    boxes.append((a, b))
            solids = [np.array(list(itertools.product(*zip(a, b)))) for a, b in boxes]
        center = (points.min(0) + points.max(0)) / 2
        extent = points.max(0) - points.min(0)
        for solid in solids:
            for axis in [0, 2]:
                solid[:, axis] = center[axis] + (solid[:, axis] - center[axis]) * (
                    (extent[axis] - 0.0002) / extent[axis]
                )
        for port in connectors:
            if port["gender"] != "M" or abs(port["radius"] - 0.0024) > 0.0001:
                continue
            axis = np.array(port["axis"])
            if axis[1] < 0.99:
                continue
            x = np.cross(axis, [1, 0, 0])
            x /= np.linalg.norm(x)
            z = np.cross(axis, x)
            origin = np.array(port["position"])
            cylinder = np.array(
                [
                    origin + axis * depth + radius * (x * np.cos(a) + z * np.sin(a))
                    for depth, radius in [
                        (0, 0.00235),
                        (0.0013, 0.00235),
                        (0.0016, 0.00205),
                    ]
                    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)
                ]
            )
            solids.append(cylinder)

    sockets = [
        (np.array(p["position"]), np.array(p["axis"]), 0.00270, 0.00185)
        for p in connectors
        if p["gender"] == "F" and abs(p["radius"] - 0.0024) < 0.0001
    ]
    sockets += [
        (np.array(p["position"]), np.array(p["axis"]), 0.00200, 0.00185)
        for p in connectors
        if p["gender"] == "F" and abs(p["radius"] - 0.0016) < 0.0001
    ]
    if design == "59900":
        bottom = next(p for p in connectors if p["gender"] == "F")
        sockets.append(
            (
                np.array(bottom["position"]) + np.array(bottom["axis"]) * 0.0064,
                np.array(bottom["axis"]),
                0.00200,
                0.005,
            )
        )
    for pos, axis, radius, depth in sockets:
        x = np.cross(axis, [1, 0, 0])
        x /= np.linalg.norm(x)
        z = np.cross(axis, x)
        # Circumscribed polygon: its minimum radius equals the desired clearance.
        planes = [
            (axis, float(axis @ pos) + depth),
            (-axis, float(-axis @ pos) + 0.00015),
        ]
        planes += [
            (n, float(n @ pos) + radius)
            for n in [
                x * np.cos(a) + z * np.sin(a)
                for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ]
        ]
        result = []
        for original in solids:
            if any(np.all(original @ n - b > 1e-9) for n, b in planes):
                result.append(original)
                continue
            remaining = original
            chunks = []
            for normal, bound in planes:
                if remaining is None:
                    break
                outside = clip(remaining, -normal, -bound)
                if outside is not None:
                    chunks.append(outside)
                remaining = clip(remaining, normal, bound)
            result.extend(chunks)
        solids = result
    arrays = {}
    for i, solid in enumerate(solids):
        arrays[f"v{i}"] = solid
        arrays[f"f{i}"] = ConvexHull(solid).simplices
    # Keys are exclusively vN/fN, never NumPy2's reserved allow_pickle keyword.
    np.savez_compressed(OUT / f"{design}.npz", **arrays)  # ty: ignore[invalid-argument-type]
    print(design, len(solids), flush=True)

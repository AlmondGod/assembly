# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Compute exact collision-free supply posts by convex linear programs."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

r = Path(__file__).resolve().parents[1] / "resources/assemblies/pumpkin40055"
C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
out = {}
for file in sorted((r / "collisions").glob("*.npz")):
    if file.stem in ["60474", "64647"]:
        continue
    a = np.load(file)
    v = np.load(r / "meshes" / file.name)["vertices"] @ C.T
    lo = v.min(0)
    hi = v.max(0)
    size = hi - lo
    hulls = []
    for i in range(len(a.files) // 2):
        w = a[f"v{i}"] @ C.T
        hulls.append((w.min(0), w.max(0), ConvexHull(w).equations))
    xs = np.linspace(lo[0] + 0.0018, hi[0] - 0.0018, max(2, int(size[0] / 0.003)))
    ys = np.linspace(lo[1] + 0.0018, hi[1] - 0.0018, max(2, int(size[1] / 0.003)))
    posts = []
    for x in xs:
        for y in ys:
            z = []
            for low, high, eq in hulls:
                if np.any(high[:2] < np.array([x, y]) - 0.00075) or np.any(
                    low[:2] > np.array([x, y]) + 0.00075
                ):
                    continue
                sol = linprog(
                    [0.0, 0.0, 1.0],
                    A_ub=eq[:, :3],
                    b_ub=-eq[:, 3],
                    bounds=[
                        (x - 0.00075, x + 0.00075),
                        (y - 0.00075, y + 0.00075),
                        (None, None),
                    ],
                    method="highs",
                )
                if sol.success:
                    z.append(float(sol.fun))
                elif sol.status != 2:
                    raise RuntimeError(sol.message)
            if z:
                posts.append([float(x), float(y), min(z) - 0.0004])
    out[file.stem] = posts
    print(file.stem, len(posts), flush=True)
(r / "supply-posts.json").write_text(
    json.dumps(
        {
            "post_width_m": 0.0015,
            "clearance_m": 0.0004,
            "description": (
                "Minimum convex-proxy Z over each complete square post footprint, "
                "solved by linear programming. Initial supply only; not attachments."
            ),
            "designs": out,
        },
        indent=2,
    )
    + "\n"
)

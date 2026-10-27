"""Flatten the pinned LDraw part 2450 geometry into a metre-scale OBJ mesh.

This exports visual triangles only. It does not create collision or stud joints.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

REVISION = "c43ed06a128a10ed65675166bf42eb6fd608b9b0"
BASE = f"https://raw.githubusercontent.com/pybricks/ldraw/{REVISION}/"


def flatten(
    name: str,
    matrix: np.ndarray,
    translation: np.ndarray,
    vertices: list[list[float]],
    triangles: list[tuple[int, int, int]],
    sources: dict[str, dict[str, str]],
    cache: Path,
    inverted: bool = False,
) -> None:
    """Append one LDraw file and its nested references to the output mesh.

    Args:
        name: Library-relative part or primitive name.
        matrix: Parent-to-world 3x3 linear transform in LDraw coordinates.
        translation: Parent-to-world translation in LDraw units.
        vertices: Destination vertex coordinates in metres.
        triangles: Destination zero-based triangle indices.
        sources: Fetched source metadata keyed by library path.
        cache: Local directory for pinned source files.
        inverted: Whether inherited BFC orientation is reversed.
    """
    candidates = ["parts/" + name, "p/" + name]
    raw = None
    for key in candidates:
        path = cache / key
        if path.exists():
            raw = path.read_bytes()
            break
    if raw is None:
        for key in candidates:
            try:
                with urlopen(BASE + key, timeout=20) as response:
                    raw = response.read()
            except HTTPError as error:
                if error.code == 404:
                    continue
                raise
            path = cache / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            break
    if raw is None:
        raise FileNotFoundError(f"LDraw source not found: {name}")
    text = raw.decode("utf-8")
    header = "\n".join(line for line in text.splitlines() if line.startswith("0 "))
    sources[key] = {
        "url": BASE + key,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "header": header,
    }
    invert_next = False
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[:3] == ["0", "BFC", "INVERTNEXT"]:
            invert_next = True
        elif fields[0] == "1":
            local_translation = np.array([float(value) for value in fields[2:5]])
            local_matrix = np.array([float(value) for value in fields[5:14]])
            local_matrix = local_matrix.reshape(3, 3)
            flatten(
                " ".join(fields[14:]).replace("\\", "/").lower(),
                matrix @ local_matrix,
                matrix @ local_translation + translation,
                vertices,
                triangles,
                sources,
                cache,
                inverted ^ invert_next ^ (np.linalg.det(local_matrix) < 0),
            )
            invert_next = False
        elif fields[0] in ("3", "4"):
            count = int(fields[0])
            points = np.array([float(value) for value in fields[2:]]).reshape(count, 3)
            transformed = points @ matrix.T + translation
            mapped = transformed[:, [0, 2, 1]] * [0.0004, 0.0004, -0.0004]
            start = len(vertices)
            vertices.extend(mapped.tolist())
            faces = [(0, 1, 2)] if count == 3 else [(0, 1, 2), (0, 2, 3)]
            for a, b, c in faces:
                face = (start + a, start + b, start + c)
                if inverted:
                    face = (start + a, start + c, start + b)
                triangles.append(face)


def main() -> None:
    """Fetch the pinned sources and write the mesh and attribution manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for OBJ and attribution")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    sources: dict[str, dict[str, str]] = {}
    flatten(
        "2450.dat", np.eye(3), np.zeros(3), vertices, triangles, sources,
        args.output / "sources",
    )
    with (args.output / "2450.obj").open("w") as obj:
        obj.write("# LDraw 2450, metres; attribution in sources.json\n")
        for vertex in vertices:
            obj.write("v " + " ".join(f"{value:.9g}" for value in vertex) + "\n")
        for triangle in triangles:
            obj.write("f " + " ".join(str(index + 1) for index in triangle) + "\n")
    bounds = np.array(vertices)
    manifest = {
        "part": "2450",
        "source_revision": REVISION,
        "units": "metres",
        "vertices": len(vertices),
        "triangles": len(triangles),
        "bounds_m": [bounds.min(axis=0).tolist(), bounds.max(axis=0).tolist()],
        "sources": sources,
        "limitations": "Visual mesh only; no collision, mass or stud connections.",
    }
    (args.output / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(
        {key: manifest[key] for key in ("vertices", "triangles", "bounds_m")},
        indent=2,
    ))


if __name__ == "__main__":
    main()

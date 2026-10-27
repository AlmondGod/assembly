"""Check nested LDraw transform conversion without network access."""

from pathlib import Path

import numpy as np
from scripts.ldraw_2450 import flatten


def test_nested_mesh_transform_and_invert(tmp_path: Path):
    """A translated primitive is flattened in metres and inversion is retained."""
    parts = tmp_path / "parts"
    primitives = tmp_path / "p"
    parts.mkdir()
    primitives.mkdir()
    (parts / "example.dat").write_text(
        "0 Example\n0 BFC INVERTNEXT\n"
        "1 16 20 0 0 1 0 0 0 1 0 0 0 1 child.dat\n"
    )
    (primitives / "child.dat").write_text(
        "0 Example primitive\n3 16 0 0 0 20 0 0 0 0 20\n"
    )
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    sources: dict[str, dict[str, str]] = {}
    flatten(
        "example.dat", np.eye(3), np.zeros(3), vertices, triangles, sources, tmp_path
    )
    np.testing.assert_allclose(
        vertices, [[0.008, 0, 0], [0.016, 0, 0], [0.008, 0.008, 0]]
    )
    assert triangles == [(0, 2, 1)]
    assert set(sources) == {"parts/example.dat", "p/child.dat"}

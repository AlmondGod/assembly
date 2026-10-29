# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "numba", "Pillow", "imageio-ffmpeg", "fast-simplification"]
# ///
"""Render measured simulator body poses without the RTX driver.

Usage: uv run --no-project scripts/render_physics_video.py RUN_DIRECTORY
This never creates or interpolates robot or part trajectories.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from fast_simplification import simplify
from imageio_ffmpeg import get_ffmpeg_exe
from numba import njit
from PIL import Image, ImageDraw, ImageFont


@njit(cache=True)
def raster(
    screen: np.ndarray, colors: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Rasterize orthographic triangles with a per-pixel depth buffer.

    Args:
        screen: Projected triangle vertices in pixel coordinates and depth.
        colors: RGB triangle colors.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        RGB image array, with occlusion resolved by measured geometry depth.
    """
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = 23
    pixels[:, :, 1] = 28
    pixels[:, :, 2] = 34
    zbuffer = np.full((height, width), -1e20)
    for i in range(len(screen)):
        t = screen[i]
        xmin = max(0, int(np.floor(t[:, 0].min())))
        xmax = min(width - 1, int(np.ceil(t[:, 0].max())))
        ymin = max(0, int(np.floor(t[:, 1].min())))
        ymax = min(height - 1, int(np.ceil(t[:, 1].max())))
        denominator = (t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0]) + (
            t[2, 0] - t[1, 0]
        ) * (t[0, 1] - t[2, 1])
        if abs(denominator) < 1e-8:
            continue
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                a = (
                    (t[1, 1] - t[2, 1]) * (x + 0.5 - t[2, 0])
                    + (t[2, 0] - t[1, 0]) * (y + 0.5 - t[2, 1])
                ) / denominator
                b = (
                    (t[2, 1] - t[0, 1]) * (x + 0.5 - t[2, 0])
                    + (t[0, 0] - t[2, 0]) * (y + 0.5 - t[2, 1])
                ) / denominator
                c = 1 - a - b
                if a >= -1e-7 and b >= -1e-7 and c >= -1e-7:
                    z = a * t[0, 2] + b * t[1, 2] + c * t[2, 2]
                    if z > zbuffer[y, x]:
                        zbuffer[y, x] = z
                        pixels[y, x, :] = colors[i]
    return pixels


def render_frame(
    triangles: np.ndarray,
    colors: np.ndarray,
    camera: np.ndarray,
    center: np.ndarray,
    scale: float,
    width: int,
    height: int,
) -> Image.Image:
    """Rasterize a view of measured world geometry.

    Args:
        triangles: World-space triangles in metres.
        colors: Per-triangle RGB colors, uint8.
        camera: Orthonormal camera matrix.
        center: View center in metres.
        scale: Pixels per metre.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        RGB PIL image with a depth buffer used for occlusion.
    """
    projected = (triangles - center) @ camera.T
    projected[:, :, :2] = projected[:, :, :2] * [scale, -scale] + [
        width / 2,
        height / 2,
    ]
    return Image.fromarray(raster(projected, colors, width, height))



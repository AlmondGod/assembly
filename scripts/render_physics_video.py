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


def main() -> None:
    """Load measured frames, render both arms and a close-up, and encode MP4."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--focus-part", type=int)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, default=float("inf"))
    args = parser.parse_args()
    root = args.directory
    assets = np.load(root / "meshes.npz")
    frames = np.load(root / "physics-frames.npz")
    if len(frames["poses"]) == 0:
        raise ValueError("No measured physics frames")
    meshes = []
    light = np.array([-0.3, -0.5, 1.0])
    light /= np.linalg.norm(light)
    for i in range(int(assets["mesh_count"])):
        vertices = assets[f"v{i}"].astype(np.float64)
        faces = assets[f"f{i}"].astype(np.int32)
        if (
            int(assets[f"b{i}"]) >= 0
            and len(faces) > 1200
            and "/Yam_" in str(assets["body_paths"][int(assets[f"b{i}"])])
        ):
            vertices, inverse = np.unique(
                np.round(vertices, 8), axis=0, return_inverse=True
            )
            faces = inverse[faces].astype(np.int32)
            vertices, faces = simplify(vertices, faces, target_count=1200)
        meshes.append((vertices, faces, int(assets[f"b{i}"]), assets[f"c{i}"]))
    focus_body = (
        list(assets["body_paths"]).index(f"/World/Pumpkin/Part{args.focus_part}")
        if args.focus_part is not None
        else None
    )
    focus_local_center = np.zeros(3)
    if focus_body is not None:
        vertices = next(v for v, _, body, _ in meshes if body == focus_body)
        focus_local_center = (vertices.min(0) + vertices.max(0)) / 2
    result = (
        json.loads((root / "result.json").read_text())
        if (root / "result.json").exists()
        else {}
    )
    grasp_probe = str(result.get("status", "")).startswith("source_grasp")
    assembly_center = np.array([0.28, 0.0, 0.05])
    for vertices, _, body, _ in meshes:
        if body >= 0 and str(assets["body_paths"][body]).endswith("/Foundation44"):
            local_center = (vertices.min(0) + vertices.max(0)) / 2
            measured = frames["poses"][0, body]
            assembly_center[:2] = (measured[:3, :3] @ local_center + measured[:3, 3])[
                :2
            ]
    eyes = [np.array([0.7, -1.0, 0.85]), np.array([0.7, -1.0, 0.6])]
    cameras = []
    for eye in eyes:
        eye /= np.linalg.norm(eye)
        right = np.cross([0, 0, 1], eye)
        right /= np.linalg.norm(right)
        up = np.cross(eye, right)
        cameras.append(np.array([right, up, eye]))
    font = ImageFont.load_default(size=23)
    small = ImageFont.load_default(size=17)
    progress = (
        json.loads((root / "progress.json").read_text())["placements"]
        if (root / "progress.json").exists()
        else []
    )
    success = (root / "result.json").exists() and json.loads(
        (root / "result.json").read_text()
    )["status"] == "complete"
    available = [
        index
        for index in range(len(frames["poses"]))
        if args.start_seconds <= frames["seconds"][index] <= args.end_seconds
    ]
    if not available:
        raise ValueError("No recorded frames within the requested time range")
    selected = available[:: args.stride]
    if selected[-1] != available[-1]:
        selected.append(available[-1])
    sample_dt = (
        float(np.median(np.diff(frames["seconds"])))
        if len(frames["seconds"]) > 1
        else 0.2
    )
    speed = sample_dt * args.stride * args.fps
    output = root / "robot-physics.mp4"
    with (root / "ffmpeg.log").open("w") as log:
        process = subprocess.Popen(
            [
                get_ffmpeg_exe(),
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1260x760",
                "-r",
                str(args.fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            stdin=subprocess.PIPE,
            stderr=log,
            stdout=subprocess.DEVNULL,
        )
        pipe = process.stdin
        assert pipe is not None
        for number, index in enumerate(selected):
            triangles = []
            shades = []
            poses = frames["poses"][index]
            close_center = assembly_center.copy()
            if focus_body is not None:
                measured = poses[focus_body]
                close_center = measured[:3, :3] @ focus_local_center + measured[:3, 3]
                close_center[2] += 0.005
            for vertices, faces, body, color in meshes:
                transform = poses[body] if body >= 0 else np.eye(4)
                world = vertices @ transform[:3, :3].T + transform[:3, 3]
                tri = world[faces]
                normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
                shade = 0.4 + 0.6 * np.abs(normal @ light)
                triangles.append(tri)
                shades.append(
                    np.minimum(255, 255 * color[None, :] * shade[:, None]).astype(
                        np.uint8
                    )
                )
            triangles = np.concatenate(triangles)
            colors = np.concatenate(shades)
            image = Image.new("RGB", (1260, 760), (23, 28, 34))
            image.paste(
                render_frame(
                    triangles,
                    colors,
                    cameras[0],
                    np.array([0.20, 0, 0.22]),
                    800,
                    840,
                    650,
                ),
                (0, 70),
            )
            image.paste(
                render_frame(
                    triangles,
                    colors,
                    cameras[1],
                    close_center,
                    10000 if focus_body is not None else 3500,
                    400,
                    400,
                ),
                (850, 100),
            )
            draw = ImageDraw.Draw(image)
            seconds = float(frames["seconds"][index])
            placed = sum(
                event.get("simulation_seconds", float("inf")) <= seconds
                for event in progress
            )
            draw.text(
                (24, 20),
                "YAM grasp and transport probe"
                if grasp_probe
                else "Two YAM arms · pumpkin assembly physics",
                font=font,
                fill=(245, 245, 240),
            )
            draw.text(
                (870, 68),
                f"Part {args.focus_part} close-up"
                if focus_body is not None
                else "Assembly close-up",
                font=small,
                fill=(245, 153, 60),
            )
            draw.text(
                (870, 600),
                "Contact grasp; no attachment"
                if grasp_probe
                else "Approximate clutch joints",
                font=small,
                fill=(190, 200, 210),
            )
            draw.text(
                (870, 530),
                "Assembly placements: 0 (probe only)"
                if grasp_probe
                else f"Placed: {placed} / 51 + fixtured base",
                font=small,
                fill=(235, 235, 232),
            )
            draw.text(
                (870, 565),
                f"Simulator time: {seconds:.1f} seconds",
                font=small,
                fill=(190, 200, 210),
            )
            draw.text(
                (24, 723),
                f"Recorded physics poses · {speed:.1f}x replay · CPU rendering",
                font=small,
                fill=(190, 200, 210),
            )
            draw.text(
                (870, 640),
                (
                    "Grasp + transport verified"
                    if result.get("transport_verified")
                    else "Source lift verified"
                )
                if grasp_probe
                else (
                    "Full assembly verified"
                    if success
                    else "Assembly trial · see result log"
                ),
                font=small,
                fill=(245, 153, 60),
            )
            if number == len(selected) - 1:
                image.save(root / "robot-final.png")
            pipe.write(image.tobytes())
            if number % 25 == 0:
                print("RENDER_PHYSICS", number, "/", len(selected), flush=True)
        pipe.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoding failed")
    (root / "video.json").write_text(
        json.dumps(
            {
                "video": str(output),
                "frames": len(selected),
                "measured_frames": len(frames["poses"]),
                "fps": args.fps,
                "replay_speed": speed,
                "source": "recorded physical body transforms; no interpolation",
                "full_assembly_complete": success,
            },
            indent=2,
        )
    )
    print(output.resolve())


if __name__ == "__main__":
    main()

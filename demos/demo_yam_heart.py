"""Build an eleven-brick heart from organized trays using exactly two YAM arms."""

import os
from pathlib import Path

import numpy as np
from demos import demo_yam_assembly
from isaacsim.core.utils.viewports import set_camera_view

from bricksim.cap.heart import HeartLayout

ROOT = Path(__file__).resolve().parents[1]


class HeartDemo(demo_yam_assembly.Demo):
    """Select the published six-layer heart and reachable tabletop source poses."""

    def __init__(self):
        """Configure the heart target and output directory before scene creation."""
        if os.environ.get("YAM_BASKET") == "1":
            raise ValueError("The heart demo uses organized tabletop source bricks")
        os.environ.setdefault(
            "YAM_TARGET", str(ROOT / "demos/assemblies/eleven_brick_heart.json")
        )
        os.environ.setdefault("YAM_OUTPUT", str(ROOT / "outputs/yam-heart"))
        super().__init__()
        self.source_poses = HeartLayout.poses()
        self.press_depths = (0.0005, 0.0015, 0.0025)
        self.grasp_width = 0.026
        self.top_press = True
        self.park_inset = 0.0
        if not self.scene_only:
            actions = self.target.parse_policy(self.policy.read_text())
            expected = [
                (part, "left" if part % 2 else "right") for part in range(1, 12)
            ]
            if [(c.hole, arm) for c, arm in actions] != expected:
                raise ValueError(
                    "Heart layout requires IDs 1–11 in order, alternating arms"
                )

    async def capture(self, name: str):
        """Save evidence, facing the heart's broad side for its final close-up.

        Args:
            name: PNG basename inside the output directory.
        """
        if name == "yam-tower.png":
            center = self.source_poses[0][:3, 3] + [0, 0, 0.03]
            set_camera_view(center + [0.24, -0.015, 0.09], np.array(center))
        await super().capture(name)


async def main():
    """Run the configured two-arm heart through the standard evidence runner."""
    await demo_yam_assembly.main(HeartDemo())

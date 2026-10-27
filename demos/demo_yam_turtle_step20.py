"""Prepare the two-plate subassembly from turtle set 31377, step 20."""

import os
from pathlib import Path

from demos import demo_yam_assembly

from bricksim.cap.heart import HeartLayout

ROOT = Path(__file__).resolve().parents[1]


class TurtleStep20Demo(demo_yam_assembly.Demo):
    """Use two organized plates and a neutral fixture for one turtle step."""

    def __init__(self):
        """Configure the isolated subassembly and reachable tabletop locations."""
        if os.environ.get("YAM_BASKET") == "1":
            raise ValueError("This step uses organized tabletop supply")
        os.environ.setdefault(
            "YAM_TARGET", str(ROOT / "demos/assemblies/turtle_step20.json")
        )
        os.environ.setdefault("YAM_OUTPUT", str(ROOT / "outputs/yam-turtle-step20"))
        os.environ.setdefault(
            "YAM_POLICY", str(ROOT / "demos/policies/turtle_step20.txt")
        )
        super().__init__()
        self.source_poses = {
            key: value for key, value in HeartLayout.poses().items() if key <= 2
        }
        self.source_supports = {}
        for part in (1, 2):
            self.source_poses[part][2, 3] = 0.0121
            self.source_supports[part] = self.source_poses[part][:3, 3].copy()
            self.source_supports[part][2] = 0.006
        self.grasp_width = 0.026
        self.top_press = True
        self.press_depths = (0.0005, 0.0015, 0.0025)
        self.press_tolerance = 0.007
        self.park_inset = 0.0


async def main():
    """Run the organized two-plate scene through the standard runner."""
    await demo_yam_assembly.main(TurtleStep20Demo())

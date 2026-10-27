"""Run the existing physical controller with reproducibly shuffled basket IDs."""

import json
import re

from demos.demo_yam_assembly import Demo
from demos.demo_yam_assembly import main as run_demo
from experiments.caprl.rollout import Scenario


class TrainingDemo(Demo):
    """Preserve physical validation while avoiding expensive screenshot capture."""

    def __init__(self):
        """Use the same source poses exposed in the training observation."""
        super().__init__()
        self.source_poses = Scenario(self.seed).poses()

    async def run(self):
        """Classify controller failures separately from unexpected runtime errors."""
        try:
            await super().run()
        except Exception as error:
            message = str(error)
            expected = isinstance(error, (RuntimeError, ValueError)) and (
                message.startswith(
                    (
                        "YAM target unreachable:",
                        "yam_left motion failed:",
                        "yam_right motion failed:",
                        "Grasp failed:",
                        "No physical connection formed:",
                        "Physical connection differs from requested offset/yaw",
                        "Assembly validation failed:",
                    )
                )
                or re.fullmatch(
                    r"Basket brick \d+ (is tilted; upright picks only|"
                    r"is not resting on the floor)|"
                    r"Basket wall blocks the grasp of brick \d+|"
                    r"Brick \d+ blocks the grasp of brick \d+",
                    message,
                )
                is not None
            )
            (self.output / "failure.json").write_text(
                json.dumps(
                    {
                        "exception_type": type(error).__name__,
                        "message": message,
                        "expected_controller_failure": expected,
                    },
                    indent=2,
                )
            )
            raise

    async def capture(self, name: str):
        """Persist verified progress at the existing scene and action boundaries.

        Args:
            name: Existing screenshot hook name, used as a progress label.
        """
        temporary = self.output / "progress.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "status": "running",
                    "stage": name,
                    "actions": self.events,
                },
                indent=2,
            )
        )
        temporary.replace(self.output / "progress.json")


async def main():
    """Run one complete physical rollout and retain standard result evidence."""
    await run_demo(TrainingDemo())

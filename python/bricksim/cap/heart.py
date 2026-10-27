"""Organized source poses for an eleven-brick, two-arm heart demonstration."""

import numpy as np


class HeartLayout:
    """Reachable tabletop positions in metres; brick long axes follow world Y."""

    @staticmethod
    def poses() -> dict[int, np.ndarray]:
        """Return initial world transforms for the baseplate and eleven bricks.

        Returns:
            IDs 0–11 mapped to 4×4 column-vector transforms in metres. Odd IDs
            are on the left arm's side, even IDs on the right arm's side.
        """
        baseplate = np.eye(4)
        baseplate[:3, 3] = [0.26, 0, 0]
        poses = {0: baseplate}
        for part in range(1, 12):
            pair = (part - 1) // 2
            pose = np.eye(4)
            pose[:3, 3] = [
                0.20 + 0.09 * (pair % 3),
                (-1 if part % 2 else 1) * (0.14 + 0.10 * (pair // 3)),
                0.001,
            ]
            poses[part] = pose
        return poses

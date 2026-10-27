"""Shallow basket geometry and conservative top-down grasp checks, in metres."""

import numpy as np
from scipy.spatial.transform import Rotation


class Basket:
    """Single-layer basket for four upright 2x4 bricks, using observed poses."""

    minimum = np.array([0.245, -0.17])
    maximum = np.array([0.425, 0.17])
    floor = 0.003
    rim = 0.028
    wall = 0.006
    # Open 60 mm finger gap plus pads and a conservative clearance margin.
    grasp_half_size = np.array([0.044, 0.023])
    brick_half_size = np.array([0.010, 0.018])

    def scatter(self, seed: int) -> dict[int, np.ndarray]:
        """Return reproducible, separated initial poses for part IDs 1 through 4.

        Args:
            seed: Nonnegative NumPy random seed.

        Returns:
            ID-to-world transforms using column vectors and metres. Bricks start
            1 mm above the basket floor, with varied yaw, then settle in physics.
        """
        generator = np.random.default_rng(seed)
        slots = [(0.305, -0.10), (0.305, 0.10), (0.37, -0.045), (0.37, 0.045)]
        for _ in range(100):
            poses = {}
            for index, slot in enumerate(slots, 1):
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_euler(
                    "z", generator.uniform(-20, 20), degrees=True
                ).as_matrix()
                pose[:2, 3] = np.array(slot) + generator.uniform(-0.004, 0.004, 2)
                pose[2, 3] = self.floor + 0.001
                poses[index] = pose
            try:
                for index in poses:
                    self.check_grasp(index, poses)
            except ValueError:
                continue
            return poses
        raise RuntimeError("Could not scatter four accessible bricks in the basket")

    @staticmethod
    def corners(pose: np.ndarray, half_size: np.ndarray) -> np.ndarray:
        """Project an oriented local XY rectangle into world XY.

        Args:
            pose: Column-vector world transform in metres.
            half_size: Rectangle half-extents along local X and Y, in metres.

        Returns:
            Four XY corners, counterclockwise, as a (4, 2) array.
        """
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half_size
        return corners @ pose[:2, :2].T + pose[:2, 3]

    def check_grasp(self, part: int, poses: dict[int, np.ndarray]):
        """Reject unsupported or obstructed vertical finger approaches.

        Args:
            part: Selected 2x4 brick ID.
            poses: Live world transforms of every unplaced brick in the basket.

        Raises:
            ValueError: Brick is tilted, stacked, outside the basket, or its
                open-finger footprint intersects a wall or neighboring brick.
                This checks the fingertip approach, not the whole robot path.
        """
        pose = poses[part]
        if pose[2, 2] < np.cos(np.deg2rad(5)):
            raise ValueError(f"Basket brick {part} is tilted; upright picks only")
        if not self.floor - 0.002 <= pose[2, 3] <= self.floor + 0.003:
            raise ValueError(f"Basket brick {part} is not resting on the floor")
        footprint = self.corners(pose, self.grasp_half_size)
        if np.any(footprint < self.minimum) or np.any(footprint > self.maximum):
            raise ValueError(f"Basket wall blocks the grasp of brick {part}")
        for other, neighbor in poses.items():
            if other == part:
                continue
            rectangle = self.corners(neighbor, self.brick_half_size)
            # Separating-axis test; touching also counts as blocked.
            axes = np.concatenate([pose[:2, :2].T, neighbor[:2, :2].T])
            separated = False
            for axis in axes:
                grasp_projection = footprint @ axis
                neighbor_projection = rectangle @ axis
                if (
                    grasp_projection.max() < neighbor_projection.min()
                    or neighbor_projection.max() < grasp_projection.min()
                ):
                    separated = True
                    break
            if not separated:
                raise ValueError(f"Brick {other} blocks the grasp of brick {part}")

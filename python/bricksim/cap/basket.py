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


class BulkBasket(Basket):
    """Fifty interchangeable 2x4 bricks scattered in a wide, shallow bin."""

    minimum = np.array([0.245, -0.30])
    maximum = np.array([0.56, 0.30])
    grasp_width = 0.026
    grasp_half_size = np.array([0.027, 0.023])

    def scatter(self, seed: int) -> dict[int, np.ndarray]:
        """Return fifty separated, randomized initial poses above the bin floor.

        Args:
            seed: Nonnegative NumPy random seed.

        Returns:
            Physical supply IDs 101..150 mapped to metre world transforms.
            XY and yaw vary independently; no brick is reserved for a target.
        """
        generator = np.random.default_rng(seed)
        poses = {}
        rectangles = []
        for index in range(50):
            for _ in range(1000):
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_euler(
                    "z", generator.uniform(-30, 30), degrees=True
                ).as_matrix()
                pose[:2, 3] = generator.uniform(
                    self.minimum + [0.03, 0.03],
                    self.maximum - [0.03, 0.03],
                )
                pose[2, 3] = self.floor + 0.001
                rectangle = self.corners(pose, self.brick_half_size + 0.003)
                if all(
                    any(
                        (rectangle @ axis).max() < (other @ axis).min()
                        or (other @ axis).max() < (rectangle @ axis).min()
                        for axis in np.concatenate([pose[:2, :2].T, rotation.T])
                    )
                    for other, rotation in rectangles
                ):
                    break
            else:
                raise RuntimeError("Could not populate the bulk basket")
            rectangles.append((rectangle, pose[:2, :2]))
            poses[101 + index] = pose
        return poses

    def check_grasp(self, part: int, poses: dict[int, np.ndarray]):
        """Reject tilted, rim-obstructed or covered candidates from live poses.

        Args:
            part: Physical supply ID of the selected 2x4 brick.
            poses: Actual world transforms of all remaining supply bricks.

        Raises:
            ValueError: The top-down open fingers cannot approach the candidate.
        """
        pose = poses[part]
        if pose[2, 2] < np.cos(np.deg2rad(5)):
            raise ValueError(f"Basket brick {part} is tilted; upright picks only")
        if pose[2, 3] < self.floor - 0.002:
            raise ValueError(f"Basket brick {part} is below the floor")
        footprint = self.corners(pose, self.grasp_half_size)
        if np.any(footprint < self.minimum) or np.any(footprint > self.maximum):
            raise ValueError(f"Basket wall blocks the grasp of brick {part}")
        for other, neighbor in poses.items():
            if other == part:
                continue
            # Gripper pads contact the upper sidewall; objects below that swept
            # volume cannot block the descending fingers. Bound tilted neighbors.
            top = (
                neighbor[2, 3]
                + abs(neighbor[2, 2]) * 0.0114
                + np.abs(neighbor[2, :2]) @ self.brick_half_size
            )
            if top < pose[2, 3] + 0.003:
                continue
            rectangle = self.corners(neighbor, self.brick_half_size)
            axes = np.concatenate([pose[:2, :2].T, neighbor[:2, :2].T])
            if not any(
                (footprint @ axis).max() < (rectangle @ axis).min()
                or (rectangle @ axis).max() < (footprint @ axis).min()
                for axis in axes
            ):
                raise ValueError(f"Brick {other} blocks the grasp of brick {part}")

    def candidates(
        self, poses: dict[int, np.ndarray], side: int | None = None
    ) -> tuple[list[int], dict[int, str]]:
        """Rank clear topmost candidates and retain explanations for rejections.

        Args:
            poses: Live world transforms of all remaining physical supply bricks.
            side: -1 for left/negative Y, +1 for right/positive Y, or all if None.

        Returns:
            Highest-first candidate IDs and rejected-ID reason mapping.
        """
        if side not in (None, -1, 1):
            raise ValueError("Arm side must be -1 or +1")
        accepted = []
        rejected = {}
        for part in sorted(poses, key=lambda key: (-poses[key][2, 3], key)):
            if side is not None and side * poses[part][1, 3] < 0:
                rejected[part] = "Brick is in the opposite arm's workspace"
                continue
            try:
                self.check_grasp(part, poses)
            except ValueError as error:
                rejected[part] = str(error)
            else:
                accepted.append(part)
        return accepted, rejected

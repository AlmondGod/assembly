"""Exercise basket clearance against walls, neighbors and unsupported piles."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bricksim.cap.basket import Basket
from bricksim.cap.kinematics import YamKinematics


def test_scatter_is_repeatable_and_accessible():
    """Every sampled layout permits all four open-finger approaches."""
    basket = Basket()
    for seed in range(100):
        poses = basket.scatter(seed)
        repeated = basket.scatter(seed)
        for part, pose in poses.items():
            np.testing.assert_array_equal(pose, repeated[part])
            basket.check_grasp(part, poses)
    assert not np.array_equal(basket.scatter(7)[1], basket.scatter(8)[1])


@pytest.mark.parametrize("rotation", [0, 30, 90])
def test_wall_rejects_open_fingers_even_when_brick_fits(rotation: int):
    """A brick near the rim cannot be picked through a physical wall."""
    basket = Basket()
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", rotation, degrees=True).as_matrix()
    pose[:3, 3] = [basket.minimum[0] + 0.02, 0, basket.floor]
    with pytest.raises(ValueError, match="wall"):
        basket.check_grasp(1, {1: pose})


def test_neighbor_blocks_rotated_grasp():
    """Reject a nearby brick inside the fingers' swept footprint."""
    basket = Basket()
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    pose[:3, 3] = [0.335, 0, basket.floor]
    neighbor = pose.copy()
    neighbor[:3, 3] += pose[:3, 0] * 0.035
    with pytest.raises(ValueError, match="Brick 2 blocks"):
        basket.check_grasp(1, {1: pose, 2: neighbor})
    neighbor[:3, 3] += pose[:3, 0] * 0.04
    basket.check_grasp(1, {1: pose, 2: neighbor})


def test_tilted_and_stacked_bricks_fail_explicitly():
    """Single-layer support must not silently extend to arbitrary piles."""
    basket = Basket()
    pose = basket.scatter(7)[1]
    pose[:3, :3] = Rotation.from_euler("x", 12, degrees=True).as_matrix()
    with pytest.raises(ValueError, match="tilted"):
        basket.check_grasp(1, {1: pose})
    pose[:3, :3] = np.eye(3)
    pose[2, 3] += 0.0096
    with pytest.raises(ValueError, match="floor"):
        basket.check_grasp(1, {1: pose})


@pytest.mark.parametrize("seed", [7, 11])
def test_scattered_grasps_and_lifts_are_reachable(seed: int):
    """Both selected arms can reach the basket pick and raised poses."""
    urdf = Path(__file__).resolve().parents[2] / "resources/robots/yam/yam.urdf"
    model = YamKinematics(urdf)
    for part, pose in Basket().scatter(seed).items():
        side = -1 if part % 2 else 1
        target = pose.copy()
        target[:3, :3] = pose[:3, :3] @ np.diag(
            [1, -1, -1] if side == -1 else [-1, 1, -1]
        )
        target[:3, 3] += pose[:3, 2] * (0.0096 - 0.0025)
        target[:3, 3] -= target[:3, 2] * 0.14465
        target[:3, 3] -= [-0.02, side * 0.26, 0]
        joints = model.inverse(target, np.array([0, 1.5, 1.5, 0, 0, 0.0]))
        target[2, 3] += 0.10
        solved = model.inverse(target, joints)
        np.testing.assert_allclose(model.forward(solved), target, atol=1e-4)

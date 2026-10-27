"""Check the URDF kinematic chain and reachable/unreachable IK behavior."""

from pathlib import Path

import numpy as np
import pytest

from bricksim.cap.kinematics import YamKinematics

URDF = Path(__file__).resolve().parents[2] / "resources/robots/yam/yam.urdf"


def test_known_flange_pose_and_inverse():
    """Pin a source geometry reference and recover a nearby reachable pose."""
    model = YamKinematics(URDF)
    joints = np.array([0, 1.5, 1.5, 0, 0, 0.0])
    transform = model.forward(joints)
    np.testing.assert_allclose(
        transform[:3, 3], [0.355921832, 0.00000492262, 0.436841067], atol=1e-8
    )
    solved = model.inverse(transform, joints + 0.05)
    np.testing.assert_allclose(model.forward(solved), transform, atol=1e-5)
    assert np.all(solved >= model.limits[0]) and np.all(solved <= model.limits[1])


def test_unreachable_pose_fails():
    """Do not emit joint targets for a point far outside the arm workspace."""
    model = YamKinematics(URDF)
    transform = np.eye(4)
    transform[:3, 3] = [10, 0, 0]
    with pytest.raises(RuntimeError, match="unreachable"):
        model.inverse(transform, np.array([0, 1.5, 1.5, 0, 0, 0.0]))

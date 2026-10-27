"""YAM URDF kinematics: column-vector transforms in metres and radians."""

from pathlib import Path
from xml.etree.ElementTree import parse

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class YamKinematics:
    """Six-joint numerical IK for the official YAM flange, +Z forward."""

    def __init__(self, urdf: Path):
        """Read the arm's six joint origins, axes and limits.

        Args:
            urdf: Official YAM URDF, with joint1 through joint6 in chain order.
        """
        tree = parse(urdf)
        self.origins = []
        self.axes = []
        limits = []
        for index in range(1, 7):
            joint = tree.find(f"joint[@name='joint{index}']")
            if joint is None:
                raise ValueError(f"Missing joint{index}")
            origin, axis, limit = (
                joint.find(tag) for tag in ("origin", "axis", "limit")
            )
            if origin is None or axis is None or limit is None:
                raise ValueError("Joint lacks origin, axis or limit")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.attrib["rpy"], sep=" ")
            ).as_matrix()
            transform[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ")
            self.origins.append(transform)
            self.axes.append(np.fromstring(axis.attrib["xyz"], sep=" "))
            limits.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
        self.limits = np.array(limits).T

    def forward(self, joints: np.ndarray) -> np.ndarray:
        """Return base-to-flange 4x4 homogeneous transform.

        Args:
            joints: Six arm positions in radians, joint1 through joint6.

        Returns:
            Column-vector homogeneous transform, translation in metres.
        """
        transform = np.eye(4)
        for origin, axis, angle in zip(self.origins, self.axes, joints, strict=True):
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ origin @ motion
        return transform

    def residual(self, joints: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return position error and orientation error scaled by 0.15 metres/radian.

        Args:
            joints: Six arm positions in radians.
            target: Desired base-to-flange column-vector 4x4 transform.

        Returns:
            Six residual components in metre-equivalent units.
        """
        actual = self.forward(joints)
        return np.r_[
            actual[:3, 3] - target[:3, 3],
            0.15 * Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec(),
        ]

    def inverse(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Solve a reachable flange pose within the URDF limits or fail.

        Args:
            target: Desired base-to-flange 4x4 transform.
            seed: Current six joint angles in radians.

        Returns:
            Six joint positions, with residual below 0.1 mm metre-equivalent.
        """
        result = least_squares(
            self.residual,
            np.clip(seed, *self.limits),
            bounds=self.limits,
            args=(target,),
            max_nfev=160,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if np.linalg.norm(result.fun) > 1e-4:
            raise RuntimeError(
                f"YAM target unreachable: residual={result.fun.tolist()}"
            )
        return result.x

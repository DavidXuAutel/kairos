"""FR3 numerical IK in pure NumPy (Panda/FR3 MDH + measured TCP tool offset)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Franka Emika Panda / FR3 modified DH (metres, radians). Matches libfranka docs.
_MDH = [
    # a, d, alpha
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
]
_FLANGE_D = 0.107  # flange along EE z after joint7


def quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(4)]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def rot_to_axisangle(R: np.ndarray) -> np.ndarray:
    """SO(3) -> axis-angle (rotation vector)."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if np.pi - theta < 1e-6:
        # Near 180 deg: use diagonal
        xx = (R[0, 0] + 1) * 0.5
        yy = (R[1, 1] + 1) * 0.5
        zz = (R[2, 2] + 1) * 0.5
        axis = np.sqrt(np.maximum([xx, yy, zz], 0.0))
        if axis[0] >= axis[1] and axis[0] >= axis[2]:
            axis[1] = np.copysign(axis[1], R[0, 1])
            axis[2] = np.copysign(axis[2], R[0, 2])
        elif axis[1] >= axis[2]:
            axis[0] = np.copysign(axis[0], R[0, 1])
            axis[2] = np.copysign(axis[2], R[1, 2])
        else:
            axis[0] = np.copysign(axis[0], R[0, 2])
            axis[1] = np.copysign(axis[1], R[1, 2])
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * theta
    w = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    w = w / (2.0 * np.sin(theta))
    return w * theta


def axisangle_to_rot(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = np.array(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _mdh_T(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


@dataclass
class SE3:
    matrix: np.ndarray  # 4x4

    @staticmethod
    def from_Rt(R: np.ndarray, t: np.ndarray) -> "SE3":
        T = np.eye(4)
        T[:3, :3] = np.asarray(R, dtype=np.float64)
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        return SE3(T)

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3]

    def inverse(self) -> "SE3":
        R = self.rotation
        t = self.translation
        Ri = R.T
        return SE3.from_Rt(Ri, -Ri @ t)

    def __matmul__(self, other: "SE3") -> "SE3":
        return SE3(self.matrix @ other.matrix)


def pose_to_se3(pos: np.ndarray, quat_xyzw: np.ndarray) -> SE3:
    return SE3.from_Rt(quat_xyzw_to_rot(quat_xyzw), pos)


@dataclass
class FR3IK:
    """Pure-numpy FR3 IK with measured TCP calibration."""

    def __post_init__(self) -> None:
        self.T_tool = SE3(np.eye(4))

    def fk_flange(self, q: np.ndarray) -> SE3:
        q = np.asarray(q, dtype=np.float64).reshape(7)
        T = np.eye(4)
        for i, (a, d, alpha) in enumerate(_MDH):
            T = T @ _mdh_T(a, d, alpha, float(q[i]))
        T = T @ _mdh_T(0.0, _FLANGE_D, 0.0, 0.0)
        return SE3(T)

    def calibrate_tool_from_measured(self, q: np.ndarray, T_meas: SE3) -> None:
        T_fk = self.fk_flange(q)
        self.T_tool = T_fk.inverse() @ T_meas

    def measured_pose(self, q: np.ndarray) -> SE3:
        return self.fk_flange(q) @ self.T_tool

    def _pose_error(self, T_cur: SE3, T_des: SE3) -> np.ndarray:
        """Body-frame approximate twist error [omega(3), v(3)]."""
        T_err = T_cur.inverse() @ T_des
        w = rot_to_axisangle(T_err.rotation)
        v = T_err.translation
        return np.concatenate([w, v])

    def _jacobian_numeric(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(7)
        T0 = self.fk_flange(q)
        J = np.zeros((6, 7), dtype=np.float64)
        for i in range(7):
            dq = np.zeros(7, dtype=np.float64)
            dq[i] = eps
            Ti = self.fk_flange(q + dq)
            J[:, i] = self._pose_error(T0, Ti) / eps
        return J

    def ik(
        self,
        T_meas_des: SE3,
        q0: np.ndarray,
        *,
        max_iter: int = 80,
        tol: float = 1e-4,
        damp: float = 1e-2,
        step: float = 0.8,
    ) -> tuple[np.ndarray, bool]:
        T_des_fk = T_meas_des @ self.T_tool.inverse()
        q = np.asarray(q0, dtype=np.float64).reshape(7).copy()
        for _ in range(max_iter):
            T_cur = self.fk_flange(q)
            err = self._pose_error(T_cur, T_des_fk)
            if float(np.linalg.norm(err)) < tol:
                return q, True
            J = self._jacobian_numeric(q)
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + damp * np.eye(6), err)
            # limit joint step
            n = float(np.linalg.norm(dq))
            if n > 0.2:
                dq = dq * (0.2 / n)
            q = q + step * dq
        return q, False

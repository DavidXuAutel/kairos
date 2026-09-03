"""Build LIBERO-compatible 8-D proprio from FR3 EEF pose + gripper.

Alignments vs raw Franka readings:
  1. xyz: subtract fixed origin so FR3 table maps near LIBERO workspace
  2. orientation: R_model = R_offset @ R_franka, then LIBERO quat2axisangle
  3. gripper: map Franka finger widths → antisymmetric [w, -w]

R_offset was calibrated 2026-07-23 on 125 look-down pose so aa → train mid.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np

# Default: shift FR3 base so table (~x=0.5–0.6) lands in LIBERO x∈[-0.49,0.21]
DEFAULT_ORIGIN_XYZ = (0.45, 0.0, 0.0)

# Calib: look-down FR3 quat → LIBERO state aa mid [2.308, -0.021, -0.234]
# R_offset = R_target @ R_franka.T  (row-major 3x3)
DEFAULT_R_OFFSET = (
    (-0.9519939421517739, 0.23281013982159773, 0.19876341155275828),
    (0.008027519328568954, -0.6300962111683618, 0.7764755486524786),
    (0.3060114584939603, 0.740795655225511, 0.5979788857817103),
)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Robosuite / LIBERO eval: (x,y,z,w) → axis-angle exponential coords."""
    q = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    aa = (q[:3] * (2.0 * math.acos(q[3]))) / den
    return aa.astype(np.float32)


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(4)]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
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
    q = np.array([x, y, z, w], dtype=np.float64)
    # Prefer w>=0 so quat2axisangle stays continuous near π
    if q[3] < 0:
        q = -q
    return q


def parse_r_offset(text: str) -> np.ndarray:
    """Parse 9 comma-separated floats (row-major) → 3x3."""
    parts = [p.strip() for p in str(text).replace(" ", "").split(",") if p.strip()]
    if len(parts) != 9:
        raise ValueError(f"r-offset must be 9 floats, got {len(parts)} from {text!r}")
    return np.asarray([float(x) for x in parts], dtype=np.float64).reshape(3, 3)


def franka_quat_to_libero_aa(
    quat_xyzw: np.ndarray,
    r_offset: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
) -> np.ndarray:
    """Apply R_offset @ R_franka then LIBERO quat2axisangle."""
    R_f = _quat_xyzw_to_rot(quat_xyzw)
    R_o = np.asarray(r_offset if r_offset is not None else DEFAULT_R_OFFSET, dtype=np.float64)
    if R_o.shape != (3, 3):
        raise ValueError(f"r_offset must be 3x3, got {R_o.shape}")
    R_m = R_o @ R_f
    return quat2axisangle(_rot_to_quat_xyzw(R_m))


def map_franka_gripper_to_libero(finger_widths: np.ndarray) -> np.ndarray:
    """Franka Hand (both ≥0) → LIBERO-style antisymmetric qpos [w, -w]."""
    f = np.asarray(finger_widths, dtype=np.float64).reshape(-1)
    if f.shape[0] < 1:
        raise ValueError("gripper needs at least 1 finger width")
    if f.shape[0] >= 2:
        w = 0.5 * (float(f[0]) + float(f[1]))
    else:
        w = float(f[0])
    w = float(np.clip(w, 0.0, 0.04))
    return np.array([w, -w], dtype=np.float32)


def apply_origin_offset(
    eef_pos: np.ndarray,
    origin_xyz: Sequence[float] = DEFAULT_ORIGIN_XYZ,
) -> np.ndarray:
    """Model-frame position: franka_xyz - origin (actions still executed in FR3)."""
    pos = np.asarray(eef_pos, dtype=np.float32).reshape(3)
    origin = np.asarray(origin_xyz, dtype=np.float32).reshape(3)
    return (pos - origin).astype(np.float32)


def parse_origin_xyz(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in str(text).replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"origin must be x,y,z got {text!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def build_proprio(
    eef_pos: np.ndarray,
    eef_axisangle: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    """Return float32 vector [xyz(3), axis-angle(3), gripper(2)] (already LIBERO-framed)."""
    pos = np.asarray(eef_pos, dtype=np.float32).reshape(-1)
    aa = np.asarray(eef_axisangle, dtype=np.float32).reshape(-1)
    grip = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)
    if pos.shape[0] != 3:
        raise ValueError(f"eef_pos must have 3 elems, got {pos.shape}")
    if aa.shape[0] != 3:
        raise ValueError(f"eef_axisangle must have 3 elems, got {aa.shape}")
    if grip.shape[0] != 2:
        raise ValueError(f"gripper_qpos must have 2 elems, got {grip.shape}")
    out = np.concatenate([pos, aa, grip]).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("proprio contains non-finite values")
    return out


def build_proprio_from_franka(
    eef_pos_franka: np.ndarray,
    quat_xyzw: np.ndarray,
    finger_widths: np.ndarray,
    *,
    origin_xyz: Sequence[float] = DEFAULT_ORIGIN_XYZ,
    r_offset: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
) -> np.ndarray:
    """Full FR3 → LIBERO 8-D raw proprio (before minmax normalize)."""
    pos = apply_origin_offset(eef_pos_franka, origin_xyz)
    aa = franka_quat_to_libero_aa(quat_xyzw, r_offset=r_offset)
    grip = map_franka_gripper_to_libero(finger_widths)
    return build_proprio(pos, aa, grip)

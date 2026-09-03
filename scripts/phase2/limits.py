"""Safety clamps for Phase-2 open-loop planning (no hardware)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ClampMode = Literal["per_axis", "linf"]


@dataclass(frozen=True)
class StepLimits:
    max_abs_xyz: float = 0.02  # metres per sample
    max_abs_rot: float = 0.05  # rad per sample (axis-angle magnitude proxy)


@dataclass(frozen=True)
class WorkspaceLimits:
    xyz_min: tuple[float, float, float] = (0.2, -0.4, 0.0)
    xyz_max: tuple[float, float, float] = (0.8, 0.4, 0.6)


def _linf_scale(v: np.ndarray, max_abs: float) -> np.ndarray:
    """Scale vector so max(|components|) <= max_abs, preserving direction."""
    peak = float(np.max(np.abs(v)))
    if peak <= max_abs or peak < 1e-12:
        return v
    return v * (max_abs / peak)


def clamp_eef_delta(
    delta7: np.ndarray,
    limits: StepLimits | None = None,
    *,
    mode: ClampMode = "per_axis",
) -> np.ndarray:
    """Limit translation (0:3) and rotation (3:6); leave gripper (6) unchanged.

    mode:
      - per_axis: independent clip (can distort direction when saturated)
      - linf: uniform scale so max(|xyz|) / max(|rot|) stay within limits
        (preserves approach direction — preferred for WAM→Franka)
    """
    limits = limits or StepLimits()
    x = np.asarray(delta7, dtype=np.float32).copy()
    if x.shape[-1] < 6:
        raise ValueError(f"expected >=6 action dims, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("action contains non-finite values")
    if mode == "per_axis":
        x[..., 0:3] = np.clip(x[..., 0:3], -limits.max_abs_xyz, limits.max_abs_xyz)
        x[..., 3:6] = np.clip(x[..., 3:6], -limits.max_abs_rot, limits.max_abs_rot)
    elif mode == "linf":
        x[..., 0:3] = _linf_scale(x[..., 0:3], limits.max_abs_xyz)
        x[..., 3:6] = _linf_scale(x[..., 3:6], limits.max_abs_rot)
    else:
        raise ValueError(f"unknown clamp mode: {mode}")
    return x


def reject_if_outside_workspace(
    xyz: np.ndarray,
    limits: WorkspaceLimits | None = None,
) -> None:
    limits = limits or WorkspaceLimits()
    p = np.asarray(xyz, dtype=np.float32).reshape(3)
    lo = np.asarray(limits.xyz_min, dtype=np.float32)
    hi = np.asarray(limits.xyz_max, dtype=np.float32)
    if np.any(p < lo) or np.any(p > hi):
        raise ValueError(f"xyz {p.tolist()} outside workspace {lo.tolist()}..{hi.tolist()}")

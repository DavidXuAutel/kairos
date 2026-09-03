"""Descend tracking / floor guards (pure, no robot deps)."""
from __future__ import annotations

import math
from typing import Any


def ee_tracking_errors(ee_cmd: list[float] | tuple[float, ...], ee_meas: list[float] | tuple[float, ...]) -> dict[str, Any]:
    """Compare commanded vs measured EE xyz (metres)."""
    cx, cy, cz = (float(ee_cmd[0]), float(ee_cmd[1]), float(ee_cmd[2]))
    mx, my, mz = (float(ee_meas[0]), float(ee_meas[1]), float(ee_meas[2]))
    ex, ey, ez = mx - cx, my - cy, mz - cz
    track_err_xy = math.hypot(ex, ey)
    track_err_z = abs(ez)
    track_err_xyz = math.sqrt(ex * ex + ey * ey + ez * ez)
    return {
        "track_err_xy": track_err_xy,
        "track_err_z": track_err_z,
        "track_err_xyz": track_err_xyz,
        "err_xyz": [ex, ey, ez],
    }


def xy_drift_from_anchor(
    ee_meas: list[float] | tuple[float, ...],
    anchor_xy: list[float] | tuple[float, ...],
) -> float:
    return math.hypot(float(ee_meas[0]) - float(anchor_xy[0]), float(ee_meas[1]) - float(anchor_xy[1]))


def descend_floor_z(
    *,
    approach_z: float,
    grasp_z: float,
    max_drop: float,
    workspace_z_min: float,
    grasp_slack: float = 0.005,
) -> float:
    """Lowest z allowed during scripted descend.

    Caps absolute drop from approach height and refuses to go far below grasp_z.
    """
    floors = [
        float(workspace_z_min),
        float(approach_z) - float(max_drop),
        float(grasp_z) - float(grasp_slack),
    ]
    return max(floors)


def check_descend_tracking(
    *,
    ee_cmd: list[float] | tuple[float, ...],
    ee_meas: list[float] | tuple[float, ...],
    anchor_xy: list[float] | tuple[float, ...],
    max_track_xy: float,
    max_track_z: float,
    max_track_xyz: float,
    max_anchor_xy: float,
) -> dict[str, Any]:
    """Return tracking audit fields; raise RuntimeError if thresholds exceeded."""
    errs = ee_tracking_errors(ee_cmd, ee_meas)
    anchor_xy_err = xy_drift_from_anchor(ee_meas, anchor_xy)
    out = {
        **errs,
        "anchor_xy": [float(anchor_xy[0]), float(anchor_xy[1])],
        "anchor_xy_err": float(anchor_xy_err),
        "ee_cmd": [float(ee_cmd[0]), float(ee_cmd[1]), float(ee_cmd[2])],
        "ee_measured": [float(ee_meas[0]), float(ee_meas[1]), float(ee_meas[2])],
    }
    reasons: list[str] = []
    if errs["track_err_xy"] > float(max_track_xy):
        reasons.append(
            f"|xy|_track={errs['track_err_xy']:.4f}>{float(max_track_xy):.4f}"
        )
    if errs["track_err_z"] > float(max_track_z):
        reasons.append(
            f"|z|_track={errs['track_err_z']:.4f}>{float(max_track_z):.4f}"
        )
    if errs["track_err_xyz"] > float(max_track_xyz):
        reasons.append(
            f"|xyz|_track={errs['track_err_xyz']:.4f}>{float(max_track_xyz):.4f}"
        )
    if anchor_xy_err > float(max_anchor_xy):
        reasons.append(
            f"|xy|_anchor={anchor_xy_err:.4f}>{float(max_anchor_xy):.4f}"
        )
    if reasons:
        raise RuntimeError(
            "descend tracking abort ("
            + ", ".join(reasons)
            + f"); ee_cmd={[round(x, 4) for x in out['ee_cmd']]} "
            f"ee_meas={[round(x, 4) for x in out['ee_measured']]} "
            "— refusing further z push (table-slam guard)"
        )
    return out

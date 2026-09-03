"""Approach-alignment policy for reliable pick (no robot deps)."""
from __future__ import annotations

from gripper_map import libero_gripper_wants_close


def approach_is_aligned(
    *,
    limited_xyz_inf: float,
    limited_xy_inf: float,
    ee_z: float,
    grip_denorm: float,
    approach_converge_xyz: float,
    approach_converge_xy: float,
    grasp_z: float,
    align_z_margin: float,
) -> tuple[bool, dict]:
    """Align requires lateral converge; never pass on near_table alone."""
    close_intent = libero_gripper_wants_close(grip_denorm)
    xyz_converged = limited_xyz_inf <= float(approach_converge_xyz)
    xy_ok = limited_xy_inf <= float(approach_converge_xy)
    near_table = float(ee_z) <= float(grasp_z) + float(align_z_margin)
    aligned = bool(xy_ok and (xyz_converged or (close_intent and near_table)))
    info = {
        "close_intent": close_intent,
        "xyz_converged": xyz_converged,
        "xy_ok": xy_ok,
        "near_table": near_table,
        "aligned": aligned,
        "limited_xyz_inf": float(limited_xyz_inf),
        "limited_xy_inf": float(limited_xy_inf),
        "z": float(ee_z),
        "grip": float(grip_denorm),
    }
    return aligned, info

#!/usr/bin/env python3
"""No-motion: read FR3 pose/gripper → LIBERO proprio → print raw/norm vs train range."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from action_pipeline import load_stats
from state_builder import (
    DEFAULT_ORIGIN_XYZ,
    DEFAULT_R_OFFSET,
    build_proprio,
    build_proprio_from_franka,
    franka_quat_to_libero_aa,
    map_franka_gripper_to_libero,
    parse_origin_xyz,
    quat2axisangle,
)
from state_normalize import normalize_proprio
from ik_fr3 import quat_xyzw_to_rot, rot_to_axisangle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stats",
        default=str(Path.home() / "kairos/benchmarks/libero_plus/libero_plus_dataset_stats.json"),
    )
    ap.add_argument(
        "--proprio-origin",
        default=",".join(str(x) for x in DEFAULT_ORIGIN_XYZ),
    )
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()
    origin = parse_origin_xyz(args.proprio_origin)
    stats = load_stats(args.stats)
    smin = np.asarray(stats["state"]["default"]["global_min"], dtype=np.float64)
    smax = np.asarray(stats["state"]["default"]["global_max"], dtype=np.float64)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    class Buf(Node):
        def __init__(self) -> None:
            super().__init__("proprio_compare")
            self.pose = self.grip = None
            self.create_subscription(
                PoseStamped,
                "/franka_robot_state_broadcaster/current_pose",
                lambda m: setattr(self, "pose", m),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                JointState, "/franka_gripper/joint_states", lambda m: setattr(self, "grip", m), 10
            )

    rclpy.init()
    node = Buf()
    t0 = time.time()
    while time.time() - t0 < args.timeout and (node.pose is None or node.grip is None):
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.pose is None:
        raise SystemExit("no current_pose")
    pos = np.array(
        [node.pose.pose.position.x, node.pose.pose.position.y, node.pose.pose.position.z],
        dtype=np.float64,
    )
    quat = np.array(
        [
            node.pose.pose.orientation.x,
            node.pose.pose.orientation.y,
            node.pose.pose.orientation.z,
            node.pose.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    fingers = (
        np.array(node.grip.position[:2], dtype=np.float64)
        if node.grip is not None and len(node.grip.position) >= 2
        else np.array([0.04, 0.04])
    )

    # Legacy (pre-fix)
    R = quat_xyzw_to_rot(quat)
    legacy = build_proprio(pos.astype(np.float32), rot_to_axisangle(R).astype(np.float32), fingers.astype(np.float32))
    legacy_n = normalize_proprio(legacy, stats)

    # Aligned (scheme A)
    aligned = build_proprio_from_franka(pos, quat, fingers, origin_xyz=origin)
    aligned_n = normalize_proprio(aligned, stats)

    labels = ["x", "y", "z", "aa0", "aa1", "aa2", "g0", "g1"]

    def report(name: str, raw: np.ndarray, n: np.ndarray) -> None:
        print(f"\n=== {name} ===")
        print(f"raw  {np.round(raw, 4).tolist()}")
        print(f"norm {np.round(n, 3).tolist()}")
        ood = [labels[i] for i in range(8) if abs(float(n[i])) > 1.0]
        print(f"|norm|>1: {ood or '(none)'}")

    print(f"franka_xyz={np.round(pos,4).tolist()} quat_xyzw={np.round(quat,4).tolist()}")
    print(f"fingers={np.round(fingers,4).tolist()} → libero_grip={map_franka_gripper_to_libero(fingers).tolist()}")
    print(f"origin={list(origin)}")
    print(f"R_offset default calibrated → train aa mid")
    print(f"train_min={np.round(smin,4).tolist()}")
    print(f"train_max={np.round(smax,4).tolist()}")
    print(
        f"raw_libero_aa={np.round(quat2axisangle(quat),4).tolist()} "
        f"aligned_aa={np.round(franka_quat_to_libero_aa(quat),4).tolist()} "
        f"legacy_aa={np.round(rot_to_axisangle(R),4).tolist()}"
    )
    report("LEGACY (before fix)", legacy, legacy_n)
    report("ALIGNED (origin + R_offset + grip)", aligned, aligned_n)

    out = {
        "origin": list(origin),
        "r_offset": [list(row) for row in DEFAULT_R_OFFSET],
        "franka_xyz": pos.tolist(),
        "fingers": fingers.tolist(),
        "legacy_raw": legacy.tolist(),
        "legacy_norm": legacy_n.tolist(),
        "aligned_raw": aligned.tolist(),
        "aligned_norm": aligned_n.tolist(),
    }
    path = Path("/tmp/kairos_proprio_compare.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

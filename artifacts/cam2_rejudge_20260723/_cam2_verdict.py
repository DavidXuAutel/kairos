#!/usr/bin/env python3
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import sys

sys.path.insert(0, str(Path.home() / "kairos/scripts/phase2"))
from ik_fr3 import FR3IK, rot_to_quat_xyzw
from action_pipeline import load_stats
from state_builder import build_proprio_from_franka, DEFAULT_ORIGIN_XYZ
from state_normalize import normalize_proprio
from camera_layout import center_crop_resize

out = Path("/tmp/kairos_cam2_rejudge_20260723_163355")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class J(Node):
    def __init__(self):
        super().__init__("fkj2")
        self.msg = None
        self.create_subscription(
            JointState, "/joint_states", lambda m: setattr(self, "msg", m), 10
        )


rclpy.init()
n = J()
t0 = time.time()
while time.time() - t0 < 5 and n.msg is None:
    rclpy.spin_once(n, timeout_sec=0.1)
names = list(n.msg.name)
pos = np.array(n.msg.position, float)
qmap = {names[i]: pos[i] for i in range(len(names))}
joints = np.array([qmap[f"fr3_joint{i}"] for i in range(1, 8)])
fingers = np.array([qmap["fr3_finger_joint1"], qmap["fr3_finger_joint2"]])
n.destroy_node()
rclpy.shutdown()

ik = FR3IK()
T = ik.measured_pose(joints)
xyz = T.translation
quat = rot_to_quat_xyzw(T.rotation)
stats = load_stats(
    str(Path.home() / "kairos/benchmarks/libero_plus/libero_plus_dataset_stats.json")
)
smin = np.asarray(stats["state"]["default"]["global_min"], float)
smax = np.asarray(stats["state"]["default"]["global_max"], float)
aligned = build_proprio_from_franka(xyz, quat, fingers, origin_xyz=DEFAULT_ORIGIN_XYZ)
aligned_n = normalize_proprio(aligned, stats)
labels = ["x", "y", "z", "aa0", "aa1", "aa2", "g0", "g1"]
ood = [labels[i] for i in range(8) if abs(float(aligned_n[i])) > 1.0]

prev = json.loads((out / "kairos_proprio_compare.json").read_text())
proprio = {
    "note": "FK flange+default tool (current_pose topic down); approximate",
    "joints": joints.tolist(),
    "fingers": fingers.tolist(),
    "ee_xyz_fk": xyz.tolist(),
    "ee_quat_xyzw_fk": quat.tolist(),
    "origin": list(DEFAULT_ORIGIN_XYZ),
    "aligned_raw": aligned.tolist(),
    "aligned_norm": aligned_n.tolist(),
    "ood_|norm|>1": ood,
    "train_state_min": smin.tolist(),
    "train_state_max": smax.tolist(),
    "prev_saved_aligned_norm": prev.get("aligned_norm"),
    "prev_saved_franka_xyz": prev.get("franka_xyz"),
}


def scene224(path):
    arr = np.asarray(Image.open(path).convert("RGB"))
    if arr.shape[1] > arr.shape[0] * 1.5:
        arr = arr[:, : arr.shape[1] // 2]
    if arr.shape[:2] != (224, 224):
        arr = center_crop_resize(arr, 224, 224)
    return arr


frames = {
    "live": out / "live_wam_224x448.png",
    "train": out / "train_smoke_t2_wam224x448.png",
    "prev_morning": out / "prev_morning_cam2.png",
    "prev_pose_fix": out / "prev_pose_fix_cam2.png",
    "prev_pose_fix2": out / "prev_pose_fix2_cam2.png",
}
fram = {}
for k, p in frames.items():
    if not p.exists():
        continue
    a = scene224(p).astype(np.float32)
    bright = float(a.mean())
    lower = float(a[112:, :, :].mean())
    upper = float(a[:112, :, :].mean())
    gray = a.mean(2)
    ys, xs = np.mgrid[0:224, 0:224]
    wgt = np.clip(gray - 30, 0, None)
    cy = float((wgt * ys).sum() / wgt.sum()) if wgt.sum() > 0 else 112.0
    gy = float(np.mean(np.abs(np.diff(gray, axis=0))))
    gx = float(np.mean(np.abs(np.diff(gray, axis=1))))
    fram[k] = {
        "brightness": bright,
        "lower_bright": lower,
        "upper_bright": upper,
        "lower_minus_upper": lower - upper,
        "mass_cy": cy,
        "grad_y": gy,
        "grad_x": gx,
    }

verdict = {
    "timestamp": "2026-07-23T16:36+08",
    "train_ref": (
        "LIBERO-plus sim smoke (h100_libero_spatial_task0_success.mp4) "
        "agentview|wrist — NOT real lerobot demos"
    ),
    "layout": "[cam2 scene | cam1 wrist] 224x448",
    "cam_rates_hz": {"cam1_compressed": 30.0, "cam2_compressed": 15.0},
    "image_metrics_scene224": fram,
    "proprio_fk": proprio,
    "qualitative": {
        "improved_after_cam2_move": [
            "Closer workspace framing vs morning/pose_fix (pen larger, more centered)",
            "Less distant table-edge composition; better fill of WAM 224 crop with task region",
            "pose_fix2 already similar; latest live is incremental tighten/center on pen",
        ],
        "still_differs_critical": [
            "Elevation/pitch: LIVE near eye-level across table; TRAIN agentview high oblique (~45-60 deg down)",
            "Domain gap: wood office vs sim cobblestone + household props (expected sim2real)",
        ],
        "still_differs_minor": [
            "Brightness: live scene ~111 vs train ~71 mean (lab brighter)",
            "Wrist: live often pen-only close-up without gripper fingers; train shows fingers+object (pose-dependent)",
            "Background clutter (monitors/plants) vs blank sim walls",
        ],
        "wam_closer_to_train": (
            "Partially — task-region scale/centering improved, but camera pitch still "
            "far from agentview; domain gap unchanged"
        ),
        "recommendation": (
            "Ready for WAM retest (dryrun / reliable_pick) to measure policy effect; "
            "optional further cam2 raise+tilt down if agentview match is priority. "
            "Restart franka_robot_state_broadcaster for live proprio dump."
        ),
    },
}
(out / "VERDICT.json").write_text(json.dumps(verdict, indent=2))
L = fram["live"]
Tm = fram["train"]
lines = [
    "# cam2 rejudge 2026-07-23",
    "",
    "## Verdict",
    verdict["qualitative"]["wam_closer_to_train"],
    "",
    "## Recommendation",
    verdict["qualitative"]["recommendation"],
    "",
    "## Key numbers (scene 224)",
    (
        f"- live brightness={L['brightness']:.1f}, mass_cy={L['mass_cy']:.1f}, "
        f"lower-upper={L['lower_minus_upper']:.1f}"
    ),
    (
        f"- train brightness={Tm['brightness']:.1f}, mass_cy={Tm['mass_cy']:.1f}, "
        f"lower-upper={Tm['lower_minus_upper']:.1f}"
    ),
    f"- prev_morning mass_cy={fram['prev_morning']['mass_cy']:.1f}",
    f"- prev_pose_fix2 mass_cy={fram['prev_pose_fix2']['mass_cy']:.1f}",
    (
        f"- FK ee_xyz={np.round(xyz, 4).tolist()} "
        f"aligned_norm={np.round(aligned_n, 3).tolist()} OOD={ood}"
    ),
    "",
    "## Artifacts",
    str(out),
]
(out / "VERDICT.md").write_text("\n".join(lines))
print(json.dumps(fram, indent=2))
print("proprio OOD", ood)
print("ee", np.round(xyz, 4).tolist())
print("aligned_norm", np.round(aligned_n, 3).tolist())
print("wrote", out / "VERDICT.json")

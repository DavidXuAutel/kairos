#!/usr/bin/env python3
"""Reliable pick: recover control → WAM approach chunks → descend → grasp → lift.

Never auto-resumes GELLO (that caused the post-motion fight). Default disarmed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from action_pipeline import denormalize_actions, load_stats
from approach_align import approach_is_aligned
from arming import ArmingGate
from descend_safety import check_descend_tracking, descend_floor_z, ee_tracking_errors
from camera_layout import (
    CAM_H,
    CAM_W,
    CAM1_COMPRESSED,
    CAM2_COMPRESSED,
    CONCAT_H,
    CONCAT_W,
    decode_compressed_image,
    franka_dual_to_wam_image,
    realsense_image_qos,
)
from gello_takeover import GelloTakeover, JOINT_NAMES
from gripper_map import libero_gripper_to_percent, libero_gripper_wants_close
from hw_helpers import (
    cont_gello,
    ensure_impedance_active,
    kill_hold_nodes,
    run_error_recovery,
    stop_gello,
)
from ik_fr3 import FR3IK, SE3, axisangle_to_rot, pose_to_se3, quat_xyzw_to_rot, rot_to_axisangle
from limits import StepLimits, WorkspaceLimits, clamp_eef_delta, reject_if_outside_workspace
from state_builder import (
    DEFAULT_ORIGIN_XYZ,
    build_proprio_from_franka,
    parse_origin_xyz,
)
from state_normalize import normalize_proprio


def _joint_vector(msg) -> np.ndarray:
    mp = {n: float(p) for n, p in zip(msg.name, msg.position)}
    return np.array([mp[n] for n in JOINT_NAMES], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="pick up a pen")
    ap.add_argument("--wam-url", default=os.environ.get("WAM_URL", "http://127.0.0.1:8005"))
    ap.add_argument("--stats", default=str(Path.home() / "kairos/benchmarks/libero_plus/libero_plus_dataset_stats.json"))
    ap.add_argument("--num-inference-steps", type=int, default=5)
    ap.add_argument(
        "--approach-replans",
        type=int,
        default=5,
        help="Fresh WAM vision chunks before grasp (more = better XY align)",
    )
    ap.add_argument("--approach-steps", type=int, default=4)
    ap.add_argument(
        "--max-abs-xyz",
        type=float,
        default=0.04,
        help="Per-step xyz limit (m); with --clamp-mode linf preserves direction",
    )
    ap.add_argument("--max-abs-rot", type=float, default=0.05)
    ap.add_argument(
        "--clamp-mode",
        choices=("linf", "per_axis"),
        default="linf",
        help="linf=scale preserving direction (fix empty-grasp); per_axis=old clip",
    )
    ap.add_argument(
        "--action-xyz-scale",
        type=float,
        default=0.45,
        help="Multiply denorm xyz before limit (dampen oversized WAM deltas)",
    )
    ap.add_argument(
        "--approach-converge-xyz",
        type=float,
        default=0.012,
        help="Stop approach when last limited |xyz|_inf below this (m)",
    )
    ap.add_argument(
        "--approach-converge-xy",
        type=float,
        default=0.015,
        help="Require last limited |xy|_inf <= this (m) before scripted descend",
    )
    ap.add_argument(
        "--align-z-margin",
        type=float,
        default=0.03,
        help="Treat as near-table when EE z <= grasp_z + margin (m)",
    )
    ap.add_argument(
        "--require-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse scripted descend until XY converged + near-table (never z-only)",
    )
    ap.add_argument(
        "--force-descend",
        action="store_true",
        help="Override --require-align and descend even if not aligned",
    )
    ap.add_argument("--grasp-z", type=float, default=0.195, help="Absolute EE z for grasp (m)")
    ap.add_argument("--lift-z", type=float, default=0.32)
    ap.add_argument("--descend-step", type=float, default=0.012)
    ap.add_argument(
        "--descend-settle-s",
        type=float,
        default=1.6,
        help="Settle time after each descend/lift micro-step (s); lab needs ~1.6",
    )
    ap.add_argument(
        "--descend-max-drop",
        type=float,
        default=0.12,
        help="Max absolute z drop from descend-start height (m); refuse lower",
    )
    ap.add_argument(
        "--descend-track-xy",
        type=float,
        default=0.015,
        help="Abort descend if |cmd-meas| XY after a step exceeds this (m)",
    )
    ap.add_argument(
        "--descend-track-z",
        type=float,
        default=0.025,
        help="Abort descend if |cmd-meas| Z after a step exceeds this (m)",
    )
    ap.add_argument(
        "--descend-track-xyz",
        type=float,
        default=0.030,
        help="Abort descend if |cmd-meas| XYZ after a step exceeds this (m)",
    )
    ap.add_argument(
        "--descend-anchor-xy",
        type=float,
        default=0.018,
        help="Abort descend if measured XY drifts from descend-start XY beyond this (m)",
    )
    ap.add_argument(
        "--approach-track-xy",
        type=float,
        default=0.035,
        help="Abort approach if |cmd-meas| XY after a step exceeds this (m); 0=disable",
    )
    ap.add_argument("--step-dt", type=float, default=0.18)
    ap.add_argument("--interp-substeps", type=int, default=20)
    ap.add_argument("--cam-w", type=int, default=CAM_W, help="Per-camera width before concat")
    ap.add_argument("--cam-h", type=int, default=CAM_H, help="Per-camera height before concat")
    ap.add_argument(
        "--flip180",
        action="store_true",
        help="Rotate each cam 180° before concat (LIBERO sim preprocess A/B)",
    )
    ap.add_argument(
        "--single-cam",
        action="store_true",
        help="Legacy: feed only cam1 (wrist) resized; default is 2-cam [scene|wrist]",
    )
    ap.add_argument("--width", type=int, default=CONCAT_W, help="Legacy single-cam / final W")
    ap.add_argument("--height", type=int, default=CONCAT_H, help="Legacy single-cam / final H")
    ap.add_argument("--i-approve-motion", action="store_true")
    ap.add_argument("--arm-token", default="")
    ap.add_argument("--resume-gello", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument(
        "--proprio-origin",
        default=",".join(str(x) for x in DEFAULT_ORIGIN_XYZ),
        help="FR3→LIBERO xyz origin offset (franka_xyz - origin). Default 0.45,0,0",
    )
    ap.add_argument(
        "--gripper-mode",
        choices=("hybrid", "model", "scripted"),
        default="hybrid",
        help="hybrid=model during approach + scripted close fallback; model=only model; scripted=ignore model grip",
    )
    args = ap.parse_args()
    origin_xyz = parse_origin_xyz(args.proprio_origin)

    expect = os.environ.get("KAIROS_ARM_TOKEN", "").strip()
    gate = ArmingGate()
    armed = False
    if not args.plan_only:
        if not args.i_approve_motion:
            raise SystemExit("need --i-approve-motion")
        if not expect or args.arm_token != expect:
            raise SystemExit("KAIROS_ARM_TOKEN / --arm-token mismatch")
        gate.issue_token(expect)
        gate.arm(args.arm_token)
        armed = True

    limits = StepLimits(max_abs_xyz=args.max_abs_xyz, max_abs_rot=args.max_abs_rot)
    ws = WorkspaceLimits(xyz_min=(0.30, -0.40, 0.08), xyz_max=(0.80, 0.40, 0.60))
    stats = load_stats(args.stats)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path.home() / "kairos" / "phase2_logs" / f"reliable_pick_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    frames = log_dir / "frames"
    frames.mkdir(exist_ok=True)

    kairos_root = Path(os.environ.get("KAIROS_ROOT", Path.home() / "kairos"))
    sys.path.insert(0, str(kairos_root / "benchmarks" / "common"))
    from clients.wam_http_client import WAMServiceClient

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import JointState

    class Sensors(Node):
        def __init__(self) -> None:
            super().__init__("kairos_reliable_pick")
            # cam1=wrist, cam2=scene (third-person); prefer compressed (raw often silent)
            self.cam1 = self.cam2 = None
            self.joints = self.pose = self.grip = None
            for qos in realsense_image_qos():
                self.create_subscription(
                    CompressedImage, CAM1_COMPRESSED, self._on_cam1, qos
                )
                self.create_subscription(
                    CompressedImage, CAM2_COMPRESSED, self._on_cam2, qos
                )
            self.create_subscription(JointState, "/franka/joint_states", self._j, 10)
            self.create_subscription(
                PoseStamped,
                "/franka_robot_state_broadcaster/current_pose",
                self._p,
                qos_profile_sensor_data,
            )
            self.create_subscription(JointState, "/franka_gripper/joint_states", self._g, 10)

        def _on_cam1(self, m):
            try:
                self.cam1 = decode_compressed_image(m)
            except Exception:
                pass

        def _on_cam2(self, m):
            try:
                self.cam2 = decode_compressed_image(m)
            except Exception:
                pass

        def _j(self, m):
            self.joints = m

        def _p(self, m):
            self.pose = m

        def _g(self, m):
            self.grip = m

        def spin_until(self, pred, timeout=20.0):
            t0 = time.time()
            while time.time() - t0 < timeout and not pred():
                rclpy.spin_once(self, timeout_sec=0.05)

        def wam_pil(self) -> Image.Image:
            """Training layout: [scene/cam2 | wrist/cam1] → 224×448."""
            if args.single_cam:
                if self.cam1 is None:
                    raise RuntimeError("cam1 (wrist) not ready")
                return (
                    Image.fromarray(self.cam1)
                    .convert("RGB")
                    .resize((args.width, args.height))
                )
            if self.cam1 is None or self.cam2 is None:
                raise RuntimeError(
                    f"need both cams for 2-cam WAM input "
                    f"(cam1/wrist={self.cam1 is not None}, cam2/scene={self.cam2 is not None})"
                )
            _, pil = franka_dual_to_wam_image(
                self.cam1,
                self.cam2,
                cam_h=args.cam_h,
                cam_w=args.cam_w,
                flip180=bool(args.flip180),
            )
            return pil

    client = WAMServiceClient(args.wam_url, load_engine_on_init=False)
    try:
        client.load_engine()
    except Exception as e:
        print(f"[pick] load_engine: {e}", flush=True)

    rclpy.init()
    node = Sensors()
    takeover = None
    gello_pids: list[int] = []
    audit: list[dict] = []

    def read_pose_q():
        node.spin_until(lambda: node.joints is not None and node.pose is not None)
        q = _joint_vector(node.joints)
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
        return q, pos, quat

    def move_ee(ik: FR3IK, takeover: GelloTakeover, q: np.ndarray, pos, R, target_pos, target_R=None):
        target_R = R if target_R is None else target_R
        target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        # Soft-clip z so approach cannot dive through table / workspace floor
        z_lo = float(ws.xyz_min[2]) + 0.002
        if target_pos[2] < z_lo:
            target_pos[2] = z_lo
        reject_if_outside_workspace(target_pos, ws)
        T = SE3.from_Rt(target_R, target_pos)
        q_des, ok = ik.ik(T, q)
        if not ok:
            raise RuntimeError("IK failed")
        if float(np.linalg.norm(q_des - q)) > 0.40:
            raise RuntimeError("joint jump too large")
        for s in range(max(1, args.interp_substeps)):
            a = (s + 1) / float(args.interp_substeps)
            takeover.set_goal((1 - a) * q + a * q_des)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(args.step_dt / float(args.interp_substeps))
        takeover.set_goal(q_des)
        # Always re-measure: commanded target ≠ tracked EE on this impedance stack.
        q_m, pos_m, quat_m = read_pose_q()
        R_m = quat_xyzw_to_rot(quat_m)
        ik.calibrate_tool_from_measured(q_m, pose_to_se3(pos_m, quat_m))
        return q_m, pos_m, R_m

    def wam_chunk(img, proprio_n):
        action = client.infer_action(
            input_image=img,
            robot_state=torch.from_numpy(proprio_n).unsqueeze(0),
            action_horizon=8,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            save_path=str(log_dir / f"wam_{len(audit):02d}.mp4"),
        )
        raw = action.detach().cpu().numpy()
        return raw[0] if raw.ndim == 3 else raw

    try:
        # --- recover ---
        kill_hold_nodes()
        gello_pids = stop_gello()
        q, pos, quat = read_pose_q()
        print(f"[pick] start ee={np.round(pos,4).tolist()} mode recover...", flush=True)
        if armed and not args.plan_only:
            run_error_recovery(node)
            ensure_impedance_active()

        ik = FR3IK()
        T_meas = pose_to_se3(pos, quat)
        ik.calibrate_tool_from_measured(q, T_meas)
        R = quat_xyzw_to_rot(quat)

        if args.plan_only:
            print("[pick] plan-only exit after sensor read", flush=True)
            return 0

        takeover = GelloTakeover(node, rate_hz=50.0)
        takeover.start(q)  # also stops gello again
        # freeze at current
        time.sleep(0.3)

        # Gripper via existing client (pauses impedance safely)
        from std_msgs.msg import Float32

        grip_pub = node.create_publisher(
            Float32, "/gripper/gripper_client/target_gripper_width_percent", 10
        )

        def set_gripper_percent(pct: float, settle_s: float = 2.5):
            msg = Float32()
            msg.data = float(np.clip(pct, 0.0, 1.0))
            for _ in range(8):
                grip_pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.05)
                time.sleep(0.05)
            time.sleep(settle_s)
            # Re-activate impedance after gripper Move (pause_arm_for_move).
            ensure_impedance_active()
            q_now, pos_now, quat_now = read_pose_q()
            takeover.set_goal(q_now)
            ik.calibrate_tool_from_measured(q_now, pose_to_se3(pos_now, quat_now))
            time.sleep(0.3)
            q_now, pos_now, quat_now = read_pose_q()
            takeover.set_goal(q_now)
            return q_now, pos_now, quat_xyzw_to_rot(quat_now)

        print("[pick] open gripper", flush=True)
        q, pos, R = set_gripper_percent(1.0)
        ensure_impedance_active()
        q, pos, quat = read_pose_q()
        R = quat_xyzw_to_rot(quat)
        takeover.set_goal(q)
        last_grip_denorm = -1.0  # LIBERO open
        approach_aligned = False
        last_limited_xyz_inf = float("inf")

        # --- WAM approach replans (fresh vision each chunk; stop when aligned) ---
        for r in range(args.approach_replans):
            def _cams_ready():
                if args.single_cam:
                    return node.cam1 is not None
                return node.cam1 is not None and node.cam2 is not None

            node.spin_until(_cams_ready, 15)
            q, pos, quat = read_pose_q()
            R = quat_xyzw_to_rot(quat)
            grip_q = np.array([0.04, 0.04], np.float32)
            if node.grip is not None and len(node.grip.position) >= 2:
                grip_q = np.array(node.grip.position[:2], np.float32)
            proprio = build_proprio_from_franka(pos, quat, grip_q, origin_xyz=origin_xyz)
            proprio_n = normalize_proprio(proprio, stats)
            print(
                f"[pick] proprio_raw={np.round(proprio,4).tolist()} "
                f"norm={np.round(proprio_n,3).tolist()} origin={list(origin_xyz)}",
                flush=True,
            )
            img = node.wam_pil()
            print(
                f"[pick] wam_image={img.size[0]}x{img.size[1]} "
                f"layout={'single-cam1' if args.single_cam else '[cam2|cam1]=scene|wrist'} "
                f"flip180={bool(args.flip180)} clamp={args.clamp_mode} "
                f"xyz_scale={args.action_xyz_scale}",
                flush=True,
            )
            img.save(frames / f"approach_{r:02d}.png")
            chunk = wam_chunk(img, proprio_n)
            cur_pos, cur_R, cur_q = pos.copy(), R.copy(), q.copy()
            stop_approach = False
            for i in range(min(args.approach_steps, chunk.shape[0])):
                den = denormalize_actions(chunk[i], stats).astype(np.float32, copy=True)
                den[0:3] *= float(args.action_xyz_scale)
                clamped = clamp_eef_delta(den, limits, mode=args.clamp_mode)
                last_grip_denorm = float(den[-1])
                last_limited_xyz_inf = float(np.max(np.abs(clamped[0:3])))
                print(
                    f"[pick] act r{r} i{i} den_xyz={np.round(den[0:3],4).tolist()} "
                    f"limited_xyz={np.round(clamped[0:3],4).tolist()} "
                    f"|xyz|_inf={last_limited_xyz_inf:.4f} grip={last_grip_denorm:.3f}",
                    flush=True,
                )
                cur_pos = cur_pos + clamped[0:3].astype(np.float64)
                cur_R = axisangle_to_rot(clamped[3:6].astype(np.float64)) @ cur_R
                cmd_pos = cur_pos.copy()
                cur_q, cur_pos, cur_R = move_ee(
                    ik, takeover, cur_q, cur_pos, cur_R, cmd_pos, cur_R
                )
                # Do NOT publish gripper during approach: franka_gripper_client
                # pause_arm_for_move deactivates impedance and freezes EE tracking.
                # Grip intent is applied only after descend (hybrid/model/scripted).
                track = ee_tracking_errors(cmd_pos, cur_pos)
                xy_inf = float(np.max(np.abs(clamped[0:2])))
                aligned, ainfo = approach_is_aligned(
                    limited_xyz_inf=last_limited_xyz_inf,
                    limited_xy_inf=xy_inf,
                    ee_z=float(cur_pos[2]),
                    grip_denorm=last_grip_denorm,
                    approach_converge_xyz=args.approach_converge_xyz,
                    approach_converge_xy=args.approach_converge_xy,
                    grasp_z=args.grasp_z,
                    align_z_margin=args.align_z_margin,
                )
                audit.append(
                    {
                        "phase": "approach",
                        "replan": r,
                        "i": i,
                        "ee_measured": cur_pos.tolist(),
                        "ee_cmd": cmd_pos.tolist(),
                        "track_err_xy": track["track_err_xy"],
                        "track_err_z": track["track_err_z"],
                        "denorm": den.tolist(),
                        "clamped": clamped.tolist(),
                        "grip_denorm": last_grip_denorm,
                        "limited_xyz_inf": last_limited_xyz_inf,
                        "limited_xy_inf": xy_inf,
                        "align": ainfo,
                    }
                )
                print(
                    f"[pick] approach r{r} i{i} ee_meas={np.round(cur_pos,4).tolist()} "
                    f"ee_cmd={np.round(cmd_pos,4).tolist()} "
                    f"track_xy={track['track_err_xy']:.4f} track_z={track['track_err_z']:.4f}",
                    flush=True,
                )
                if float(args.approach_track_xy) > 0.0 and track["track_err_xy"] > float(
                    args.approach_track_xy
                ):
                    raise RuntimeError(
                        f"approach tracking abort (|xy|_track={track['track_err_xy']:.4f} "
                        f"> {float(args.approach_track_xy):.4f}); "
                        f"ee_cmd={np.round(cmd_pos,4).tolist()} "
                        f"ee_meas={np.round(cur_pos,4).tolist()}"
                    )

                if aligned:
                    approach_aligned = True
                    stop_approach = True
                    print(
                        f"[pick] approach aligned "
                        f"(xy_ok={ainfo['xy_ok']} near_table={ainfo['near_table']} "
                        f"close_intent={ainfo['close_intent']} "
                        f"xyz_converged={ainfo['xyz_converged']} "
                        f"|xyz|_inf={last_limited_xyz_inf:.4f} |xy|_inf={xy_inf:.4f} "
                        f"z={cur_pos[2]:.3f})",
                        flush=True,
                    )
                    break
                if ainfo["close_intent"] and not ainfo["near_table"]:
                    print(
                        f"[pick] ignore premature close_intent "
                        f"(z={cur_pos[2]:.3f} > grasp_z+margin); continue approach",
                        flush=True,
                    )
                elif ainfo["near_table"] and not ainfo["xy_ok"]:
                    print(
                        f"[pick] near_table but |xy|_inf={xy_inf:.4f} "
                        f"> {args.approach_converge_xy}; keep approaching",
                        flush=True,
                    )
            q, pos, R = cur_q, cur_pos, cur_R
            if stop_approach:
                break
        else:
            # exhausted replans without early break — still require XY, never z-only
            xy_inf = float("inf")
            if audit:
                last = next(
                    (e for e in reversed(audit) if e.get("phase") == "approach"),
                    {},
                )
                xy_inf = float(last.get("limited_xy_inf", last_limited_xyz_inf))
            aligned, ainfo = approach_is_aligned(
                limited_xyz_inf=last_limited_xyz_inf,
                limited_xy_inf=xy_inf if np.isfinite(xy_inf) else last_limited_xyz_inf,
                ee_z=float(pos[2]),
                grip_denorm=last_grip_denorm,
                approach_converge_xyz=args.approach_converge_xyz,
                approach_converge_xy=args.approach_converge_xy,
                grasp_z=args.grasp_z,
                align_z_margin=args.align_z_margin,
            )
            approach_aligned = aligned
            print(
                f"[pick] max replans align_check "
                f"aligned={aligned} xy_ok={ainfo['xy_ok']} near_table={ainfo['near_table']} "
                f"|xy|_inf={ainfo['limited_xy_inf']:.4f} z={pos[2]:.3f} "
                f"grip={last_grip_denorm:.3f}",
                flush=True,
            )
        if args.require_align and not approach_aligned and not args.force_descend:
            raise RuntimeError(
                f"approach not aligned after {args.approach_replans} replans "
                f"(|xyz|_inf={last_limited_xyz_inf:.4f}, "
                f"need |xy|_inf<={args.approach_converge_xy} + "
                f"(xyz converge or close@table); grip={last_grip_denorm:.3f}); "
                f"refusing scripted grasp "
                f"(pass --force-descend to override, or raise --approach-replans)"
            )
        if not approach_aligned:
            print(
                f"[pick] WARNING: descending without align "
                f"(|xyz|_inf={last_limited_xyz_inf:.4f} grip={last_grip_denorm:.3f})",
                flush=True,
            )

        # --- scripted descend to grasp_z (only after XY/approach align) ---
        # Soft recover: do not drop tracking. Re-sync measured pose then hold.
        ensure_impedance_active()
        time.sleep(0.3)
        q, pos, quat = read_pose_q()
        R = quat_xyzw_to_rot(quat)
        ik.calibrate_tool_from_measured(q, pose_to_se3(pos, quat))
        takeover.set_goal(q)
        time.sleep(0.5)
        q, pos, quat = read_pose_q()
        R = quat_xyzw_to_rot(quat)
        takeover.set_goal(q)
        # Freeze XY at descend start; rebasing to drifted meas each step hid cumulative offset.
        anchor_xy = [float(pos[0]), float(pos[1])]
        approach_z0 = float(pos[2])
        z_floor = descend_floor_z(
            approach_z=approach_z0,
            grasp_z=float(args.grasp_z),
            max_drop=float(args.descend_max_drop),
            workspace_z_min=float(ws.xyz_min[2]) + 0.002,
            grasp_slack=0.005,
        )
        target_z = max(float(args.grasp_z), z_floor)
        if target_z > float(args.grasp_z) + 1e-6:
            print(
                f"[pick] WARNING: raise grasp target {args.grasp_z:.3f} -> {target_z:.3f} "
                f"(floor from approach_z={approach_z0:.3f} max_drop={args.descend_max_drop})",
                flush=True,
            )
        print(
            f"[pick] descend {pos[2]:.3f} -> {target_z:.3f} "
            f"(anchor_xy={[round(x, 4) for x in anchor_xy]} z_floor={z_floor:.3f})",
            flush=True,
        )
        stuck = 0
        while pos[2] - target_z > 0.003:
            z_before = float(pos[2])
            step = min(args.descend_step, z_before - target_z)
            cmd_pos = np.array([anchor_xy[0], anchor_xy[1], z_before - step], dtype=np.float64)
            if float(cmd_pos[2]) < z_floor - 1e-9:
                raise RuntimeError(
                    f"descend refused below z_floor={z_floor:.4f} "
                    f"(cmd_z={float(cmd_pos[2]):.4f}, approach_z0={approach_z0:.4f}, "
                    f"grasp_z={args.grasp_z}, max_drop={args.descend_max_drop})"
                )
            q, pos, R = move_ee(ik, takeover, q, pos, R, cmd_pos, R)
            # Extra settle beyond move_ee interp (lab impedance is slow).
            time.sleep(max(0.0, float(args.descend_settle_s) - float(args.step_dt)))
            q, pos, quat = read_pose_q()
            R = quat_xyzw_to_rot(quat)
            ik.calibrate_tool_from_measured(q, pose_to_se3(pos, quat))
            takeover.set_goal(q)
            track_info = check_descend_tracking(
                ee_cmd=cmd_pos.tolist(),
                ee_meas=pos.tolist(),
                anchor_xy=anchor_xy,
                max_track_xy=float(args.descend_track_xy),
                max_track_z=float(args.descend_track_z),
                max_track_xyz=float(args.descend_track_xyz),
                max_anchor_xy=float(args.descend_anchor_xy),
            )
            audit.append(
                {
                    "phase": "descend",
                    "z_before": z_before,
                    "z_floor": z_floor,
                    **track_info,
                }
            )
            print(
                f"[pick] descend measured ee={np.round(pos,4).tolist()} "
                f"cmd={np.round(cmd_pos,4).tolist()} "
                f"track_xy={track_info['track_err_xy']:.4f} "
                f"track_z={track_info['track_err_z']:.4f} "
                f"anchor_xy={track_info['anchor_xy_err']:.4f}",
                flush=True,
            )
            if abs(z_before - float(pos[2])) < 0.002:
                stuck += 1
                if stuck >= 2:
                    print(
                        f"[pick] descend weak track; re-ensure impedance at z={pos[2]:.4f}",
                        flush=True,
                    )
                    ensure_impedance_active()
                    takeover.set_goal(q)
                    time.sleep(0.5)
                    q, pos, quat = read_pose_q()
                    R = quat_xyzw_to_rot(quat)
                    takeover.set_goal(q)
                    # Re-check after re-ensure: still lagging vs last cmd → abort
                    check_descend_tracking(
                        ee_cmd=cmd_pos.tolist(),
                        ee_meas=pos.tolist(),
                        anchor_xy=anchor_xy,
                        max_track_xy=float(args.descend_track_xy),
                        max_track_z=float(args.descend_track_z),
                        max_track_xyz=float(args.descend_track_xyz),
                        max_anchor_xy=float(args.descend_anchor_xy),
                    )
                if stuck >= 4:
                    raise RuntimeError(
                        f"descend not tracking (z stuck at {pos[2]:.4f}); "
                        "check single /gello publisher and impedance active"
                    )
            else:
                stuck = 0

        # --- close gripper ---
        takeover.set_goal(q)
        if args.gripper_mode == "model":
            pct = libero_gripper_to_percent(last_grip_denorm)
            print(
                f"[pick] model gripper percent={pct:.2f} (denorm={last_grip_denorm:.3f})",
                flush=True,
            )
            q, pos, R = set_gripper_percent(pct, settle_s=3.0)
        elif args.gripper_mode == "hybrid":
            if libero_gripper_wants_close(last_grip_denorm):
                pct = libero_gripper_to_percent(last_grip_denorm)
                print(
                    f"[pick] hybrid close from model percent={pct:.2f} "
                    f"(denorm={last_grip_denorm:.3f})",
                    flush=True,
                )
                q, pos, R = set_gripper_percent(min(pct, 0.15), settle_s=3.0)
            else:
                print(
                    f"[pick] hybrid fallback scripted close "
                    f"(model grip={last_grip_denorm:.3f} still open-ish)",
                    flush=True,
                )
                q, pos, R = set_gripper_percent(0.0, settle_s=3.0)
        else:
            print("[pick] scripted close gripper", flush=True)
            q, pos, R = set_gripper_percent(0.0, settle_s=3.0)
        audit.append(
            {
                "phase": "grasp",
                "ee": pos.tolist(),
                "grip_denorm": last_grip_denorm,
                "gripper_mode": args.gripper_mode,
            }
        )

        # --- lift (measured feedback) ---
        print(f"[pick] lift -> z={args.lift_z}", flush=True)
        stuck = 0
        while args.lift_z - pos[2] > 0.003:
            z_before = float(pos[2])
            step = min(args.descend_step, args.lift_z - z_before)
            new_pos = pos.copy()
            new_pos[2] = z_before + step
            q, pos, R = move_ee(ik, takeover, q, pos, R, new_pos, R)
            time.sleep(max(0.0, float(args.descend_settle_s) - float(args.step_dt)))
            q, pos, quat = read_pose_q()
            R = quat_xyzw_to_rot(quat)
            ik.calibrate_tool_from_measured(q, pose_to_se3(pos, quat))
            takeover.set_goal(q)
            audit.append({"phase": "lift", "ee": pos.tolist()})
            print(f"[pick] lift measured ee={np.round(pos,4).tolist()}", flush=True)
            if float(pos[2]) - z_before < 0.0008:
                stuck += 1
                if stuck >= 4:
                    print(
                        f"[pick] lift progress slow at z={pos[2]:.4f}; accepting partial lift",
                        flush=True,
                    )
                    break
            else:
                stuck = 0

        (log_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        meta = {
            "prompt": args.prompt,
            "armed": armed,
            "grasp_z": args.grasp_z,
            "lift_z": args.lift_z,
            "resume_gello": bool(args.resume_gello),
            "proprio_origin": list(origin_xyz),
            "max_abs_xyz": args.max_abs_xyz,
            "clamp_mode": args.clamp_mode,
            "action_xyz_scale": args.action_xyz_scale,
            "approach_replans": args.approach_replans,
            "approach_converge_xyz": args.approach_converge_xyz,
            "approach_converge_xy": args.approach_converge_xy,
            "descend_settle_s": args.descend_settle_s,
            "descend_max_drop": args.descend_max_drop,
            "descend_track_xy": args.descend_track_xy,
            "descend_track_z": args.descend_track_z,
            "descend_track_xyz": args.descend_track_xyz,
            "descend_anchor_xy": args.descend_anchor_xy,
            "require_align": bool(args.require_align),
            "approach_aligned": bool(approach_aligned),
            "gripper_mode": args.gripper_mode,
            "flip180": bool(args.flip180),
            "final_ee": pos.tolist(),
        }
        (log_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[pick] DONE log_dir={log_dir}", flush=True)
        return 0
    except Exception:
        # Persist partial audit for failure diagnosis
        try:
            (log_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        except Exception:
            pass
        raise
    finally:
        if takeover is not None:
            # Keep publishing final goal briefly then stop thread; do NOT resume gello unless asked
            takeover.stop(resume_gello=False)
        if args.resume_gello:
            cont_gello(gello_pids)
        else:
            print("[pick] GELLO left STOPPED — use --resume-gello only after aligning leader", flush=True)
            # leave a hold process
            stop_gello()
            ensure_impedance_active()
            # restart hold file
            os.system("nohup python3 /tmp/kairos_hold_q.py >/tmp/kairos_hold.log 2>&1 &")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Restore FR3 EE to the recorded start/hover pose (XYZ + quat), then hold.

Default: lab-recorded current_pose (2026-07-23). Does NOT resume GELLO.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gello_takeover import GelloTakeover, JOINT_NAMES
from hw_helpers import ensure_impedance_active, kill_hold_nodes, stop_gello
from ik_fr3 import (
    FR3IK,
    SE3,
    axisangle_to_rot,
    pose_to_se3,
    quat_xyzw_to_rot,
    rot_to_axisangle,
    rot_to_quat_xyzw,
)
from limits import WorkspaceLimits, reject_if_outside_workspace

# Recorded start/hover pose on lab FR3 (2026-07-23 evening): current_pose + joint_states
# tilt(tool-z vs -world-z) ≈ 5.6°
DEFAULT_XYZ = (0.6135568022727966, 0.04296739026904106, 0.49905693531036377)
DEFAULT_QUAT_XYZW = (
    0.004798637383703515,
    0.9988086055538619,
    -0.00725589096438865,
    0.04801748168248377,
)
# Measured fr3_joint1..7 at recording time (restore targets cartesian; joints for reference/seed)
DEFAULT_JOINTS = (
    -0.1172100082039833,
    0.12329842150211334,
    0.21156899631023407,
    -1.4993666410446167,
    -0.050002798438072205,
    1.7140735387802124,
    -2.2501940727233887,
)


def _joint_vector(msg) -> np.ndarray:
    mp = {n: float(p) for n, p in zip(msg.name, msg.position)}
    return np.array([mp[n] for n in JOINT_NAMES], dtype=np.float64)


def _slerp_R(R0: np.ndarray, R1: np.ndarray, a: float) -> np.ndarray:
    a = float(np.clip(a, 0.0, 1.0))
    dR = R1 @ R0.T
    aa = rot_to_axisangle(dR)
    return axisangle_to_rot(aa * a) @ R0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", default=",".join(str(x) for x in DEFAULT_XYZ))
    ap.add_argument(
        "--quat",
        default=",".join(str(x) for x in DEFAULT_QUAT_XYZW),
        help="Target orientation xyzw (default: look-down calib)",
    )
    ap.add_argument("--keep-orientation", action="store_true")
    ap.add_argument("--step-m", type=float, default=0.025)
    ap.add_argument("--step-rad", type=float, default=0.12)
    ap.add_argument("--step-dt", type=float, default=0.16)
    ap.add_argument("--interp-substeps", type=int, default=18)
    ap.add_argument("--i-approve-motion", action="store_true")
    args = ap.parse_args()
    if not args.i_approve_motion:
        raise SystemExit("need --i-approve-motion")

    target = np.array([float(x) for x in args.xyz.split(",")], dtype=np.float64)
    if target.shape != (3,):
        raise SystemExit("--xyz needs 3 floats")
    q_tgt = np.array([float(x) for x in args.quat.split(",")], dtype=np.float64)
    if q_tgt.shape != (4,):
        raise SystemExit("--quat needs 4 floats xyzw")
    q_tgt = q_tgt / max(1e-9, float(np.linalg.norm(q_tgt)))
    if q_tgt[3] < 0:
        q_tgt = -q_tgt
    R_tgt = quat_xyzw_to_rot(q_tgt)

    ws = WorkspaceLimits(xyz_min=(0.30, -0.40, 0.08), xyz_max=(0.80, 0.40, 0.60))
    reject_if_outside_workspace(target, ws)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    class Sensors(Node):
        def __init__(self) -> None:
            super().__init__("kairos_restore_start")
            self.joints = None
            self.pose = None
            self.create_subscription(JointState, "/franka/joint_states", self._j, 10)
            self.create_subscription(
                PoseStamped,
                "/franka_robot_state_broadcaster/current_pose",
                self._p,
                qos_profile_sensor_data,
            )

        def _j(self, m):
            self.joints = m

        def _p(self, m):
            self.pose = m

        def spin_until(self, pred, timeout=20.0):
            t0 = time.time()
            while time.time() - t0 < timeout and not pred():
                rclpy.spin_once(self, timeout_sec=0.05)

        def read(self):
            self.spin_until(lambda: self.joints is not None and self.pose is not None)
            q = _joint_vector(self.joints)
            pos = np.array(
                [
                    self.pose.pose.position.x,
                    self.pose.pose.position.y,
                    self.pose.pose.position.z,
                ],
                dtype=np.float64,
            )
            quat = np.array(
                [
                    self.pose.pose.orientation.x,
                    self.pose.pose.orientation.y,
                    self.pose.pose.orientation.z,
                    self.pose.pose.orientation.w,
                ],
                dtype=np.float64,
            )
            return q, pos, quat

    kill_hold_nodes()
    stop_gello()
    os.system("pkill -9 -f kairos_hold_q.py >/dev/null 2>&1 || true")
    time.sleep(0.4)
    ensure_impedance_active()

    rclpy.init()
    node = Sensors()
    takeover = None
    try:
        q, pos, quat = node.read()
        R = quat_xyzw_to_rot(quat)
        if args.keep_orientation:
            R_tgt = R.copy()
            q_tgt = rot_to_quat_xyzw(R_tgt)

        print(
            f"[restore] from ee={np.round(pos,4).tolist()} quat={np.round(quat,4).tolist()}",
            flush=True,
        )
        print(
            f"[restore] to   ee={target.tolist()} quat={np.round(q_tgt,4).tolist()} "
            f"(lookdown={not args.keep_orientation})",
            flush=True,
        )

        ik = FR3IK()
        ik.calibrate_tool_from_measured(q, pose_to_se3(pos, quat))
        takeover = GelloTakeover(node, rate_hz=50.0)
        takeover.start(q)
        time.sleep(0.3)

        def _move_to(new_pos: np.ndarray, new_R: np.ndarray, q_seed: np.ndarray) -> np.ndarray:
            reject_if_outside_workspace(new_pos, ws)
            T = SE3.from_Rt(new_R, new_pos)
            q_des, ok = ik.ik(T, q_seed)
            if not ok:
                raise RuntimeError(f"IK failed at pos={new_pos.tolist()}")
            jump = float(np.linalg.norm(q_des - q_seed))
            if jump > 0.55:
                # halve the step once
                new_pos = q_seed and (pos + 0.5 * (new_pos - pos))  # bug placeholder
            return q_des

        # --- Phase A: XYZ only (keep current orientation) ---
        print("[restore] phase A: XYZ", flush=True)
        tol_p = max(0.012, float(args.step_m) * 0.5)
        stuck = 0
        last_err = float("inf")
        for _ in range(60):
            err_p = float(np.linalg.norm(target - pos))
            print(f"[restore][A] err_p={err_p:.4f} ee={np.round(pos,4).tolist()}", flush=True)
            if err_p <= tol_p:
                break
            if abs(err_p - last_err) < 0.001:
                stuck += 1
                if stuck >= 8:
                    print("[restore][A] stalled", flush=True)
                    break
            else:
                stuck = 0
            last_err = err_p
            step_p = min(float(args.step_m), err_p)
            new_pos = pos + (target - pos) * (step_p / err_p)
            reject_if_outside_workspace(new_pos, ws)
            q_des, ok = ik.ik(SE3.from_Rt(R, new_pos), q)
            if not ok:
                raise RuntimeError("IK failed (phase A)")
            jump = float(np.linalg.norm(q_des - q))
            print(f"[restore][A] jump={jump:.3f}", flush=True)
            if jump > 0.55:
                new_pos = pos + 0.5 * (new_pos - pos)
                q_des, ok = ik.ik(SE3.from_Rt(R, new_pos), q)
                if not ok or float(np.linalg.norm(q_des - q)) > 0.55:
                    raise RuntimeError(f"joint jump too large ({jump:.3f})")
            for s in range(max(1, args.interp_substeps)):
                a = (s + 1) / float(args.interp_substeps)
                takeover.set_goal((1 - a) * q + a * q_des)
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(args.step_dt / float(args.interp_substeps))
            time.sleep(0.12)
            q, pos, quat = node.read()
            R = quat_xyzw_to_rot(quat)
            ik.calibrate_tool_from_measured(q, pose_to_se3(pos, quat))
            takeover.set_goal(q)

        if args.keep_orientation:
            print(
                f"[restore] DONE (keep-ori) ee={np.round(pos,4).tolist()} "
                f"quat={np.round(quat,4).tolist()}",
                flush=True,
            )
            return 0

        # --- Phase B: orientation only at (near) target XYZ ---
        print("[restore] phase B: look-down orientation", flush=True)
        stuck = 0
        last_ang = float("inf")
        tol_r = max(0.05, float(args.step_rad) * 0.4)
        for _ in range(60):
            dR = R_tgt @ R.T
            aa = rot_to_axisangle(dR)
            ang = float(np.linalg.norm(aa))
            print(
                f"[restore][B] err_R={ang:.4f} ee={np.round(pos,4).tolist()} "
                f"quat={np.round(quat,4).tolist()}",
                flush=True,
            )
            if ang <= tol_r:
                break
            if abs(ang - last_ang) < 0.002:
                stuck += 1
                if stuck >= 10:
                    print("[restore][B] stalled", flush=True)
                    break
            else:
                stuck = 0
            last_ang = ang
            a_r = min(1.0, float(args.step_rad) / max(ang, 1e-9))
            new_R = _slerp_R(R, R_tgt, a_r)
            # hold position near target while rotating
            hold_pos = pos + 0.35 * (target - pos)
            reject_if_outside_workspace(hold_pos, ws)
            q_des, ok = ik.ik(SE3.from_Rt(new_R, hold_pos), q)
            if not ok:
                print("[restore][B] IK fail; reduce step", flush=True)
                new_R = _slerp_R(R, R_tgt, a_r * 0.4)
                q_des, ok = ik.ik(SE3.from_Rt(new_R, pos), q)
                if not ok:
                    raise RuntimeError("IK failed (phase B)")
            jump = float(np.linalg.norm(q_des - q))
            print(f"[restore][B] jump={jump:.3f}", flush=True)
            if jump < 1e-4:
                print("[restore][B] IK returned no motion; abort orientation", flush=True)
                break
            if jump > 0.60:
                new_R = _slerp_R(R, new_R, 0.4)
                q_des, ok = ik.ik(SE3.from_Rt(new_R, pos), q)
                if not ok or float(np.linalg.norm(q_des - q)) > 0.60:
                    raise RuntimeError(f"joint jump too large ({jump:.3f})")
            for s in range(max(1, args.interp_substeps)):
                a = (s + 1) / float(args.interp_substeps)
                takeover.set_goal((1 - a) * q + a * q_des)
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(args.step_dt / float(args.interp_substeps))
            time.sleep(0.15)
            q, pos, quat = node.read()
            R = quat_xyzw_to_rot(quat)
            ik.calibrate_tool_from_measured(q, pose_to_se3(pos, quat))
            takeover.set_goal(q)

        print(
            f"[restore] DONE ee={np.round(pos,4).tolist()} "
            f"quat={np.round(quat,4).tolist()}",
            flush=True,
        )
        return 0
    finally:
        if takeover is not None:
            try:
                takeover.stop(resume_gello=False)
            except Exception:
                pass
        ensure_impedance_active()
        os.system("nohup python3 /tmp/kairos_hold_q.py >/tmp/kairos_hold.log 2>&1 &")
        print("[restore] hold restarted; GELLO left STOPPED", flush=True)
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

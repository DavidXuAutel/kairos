#!/usr/bin/env python3
"""Phase-2 open-loop: cam+EEF -> WAM -> denorm/clamp -> IK -> /gello/joint_states.

Default is DISARMED. Hardware motion requires BOTH:
  --i-approve-motion
  --arm-token <token> matching KAIROS_ARM_TOKEN env (or --issue-and-arm for one-shot local)

Does not change Desk / robot network. Stops gello_publisher via SIGSTOP during motion.
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

# phase2 package on path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from action_pipeline import denormalize_actions, load_stats
from arming import ArmingGate
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
from ik_fr3 import (  # noqa: E402
    FR3IK,
    SE3,
    axisangle_to_rot,
    pose_to_se3,
    quat_xyzw_to_rot,
    rot_to_axisangle,
)
from limits import StepLimits, WorkspaceLimits, clamp_eef_delta, reject_if_outside_workspace
from state_builder import (
    DEFAULT_ORIGIN_XYZ,
    build_proprio_from_franka,
    parse_origin_xyz,
)
from state_normalize import normalize_proprio


def _joint_vector(msg) -> np.ndarray:
    name_to_pos = {n: float(p) for n, p in zip(msg.name, msg.position)}
    return np.array([name_to_pos[n] for n in JOINT_NAMES], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-2 FR3 open-loop pick (armed gate)")
    parser.add_argument("--prompt", default="pick up a pen")
    parser.add_argument("--wam-url", default=os.environ.get("WAM_URL", "http://127.0.0.1:8005"))
    parser.add_argument(
        "--stats",
        default=str(Path.home() / "kairos/benchmarks/libero_plus/libero_plus_dataset_stats.json"),
    )
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=8, help="Execute at most this many chunk steps")
    parser.add_argument("--step-dt", type=float, default=0.20)
    parser.add_argument("--interp-substeps", type=int, default=25)
    parser.add_argument("--max-abs-xyz", type=float, default=0.012)
    parser.add_argument("--max-abs-rot", type=float, default=0.035)
    parser.add_argument("--cam-w", type=int, default=CAM_W)
    parser.add_argument("--cam-h", type=int, default=CAM_H)
    parser.add_argument(
        "--single-cam",
        action="store_true",
        help="Legacy: only wrist cam1 resized; default is [scene/cam2 | wrist/cam1]",
    )
    parser.add_argument("--width", type=int, default=CONCAT_W)
    parser.add_argument("--height", type=int, default=CONCAT_H)
    parser.add_argument(
        "--proprio-origin",
        default=",".join(str(x) for x in DEFAULT_ORIGIN_XYZ),
        help="FR3→LIBERO xyz origin (franka_xyz - origin). Default 0.45,0,0",
    )
    parser.add_argument("--enable-gripper", action="store_true", help="Also command gripper percent")
    parser.add_argument("--i-approve-motion", action="store_true")
    parser.add_argument("--arm-token", default="")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Infer + plan joints; do not pause gello or publish commands",
    )
    parser.add_argument("--log-dir", default="")
    args = parser.parse_args()
    origin_xyz = parse_origin_xyz(args.proprio_origin)

    limits = StepLimits(max_abs_xyz=args.max_abs_xyz, max_abs_rot=args.max_abs_rot)
    stats = load_stats(args.stats)

    gate = ArmingGate()
    expect = os.environ.get("KAIROS_ARM_TOKEN", "").strip()
    armed = False
    if args.plan_only:
        print("[phase2] PLAN ONLY — no hardware commands", flush=True)
    else:
        if not args.i_approve_motion:
            raise SystemExit("refusing: pass --i-approve-motion (or use --plan-only)")
        if not expect:
            raise SystemExit("refusing: set env KAIROS_ARM_TOKEN first")
        if not args.arm_token or args.arm_token != expect:
            raise SystemExit("refusing: --arm-token does not match KAIROS_ARM_TOKEN")
        gate.issue_token(expect)
        gate.arm(args.arm_token)
        gate.require_armed()
        armed = True
        print("[phase2] ARMED — will command /gello/joint_states", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir or Path.home() / "kairos" / "phase2_logs" / ts)
    log_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = log_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    kairos_root = Path(os.environ.get("KAIROS_ROOT", Path.home() / "kairos"))
    common = kairos_root / "benchmarks" / "common"
    if str(common) not in sys.path:
        sys.path.insert(0, str(common))
    from clients.wam_http_client import WAMServiceClient

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32

    class Sensors(Node):
        def __init__(self) -> None:
            super().__init__("kairos_phase2_open_loop")
            self.cam1 = None  # wrist
            self.cam2 = None  # scene / third-person
            self.joints = None
            self.pose = None
            self.grip = None
            for qos in realsense_image_qos():
                self.create_subscription(
                    CompressedImage, CAM1_COMPRESSED, self._on_cam1, qos
                )
                self.create_subscription(
                    CompressedImage, CAM2_COMPRESSED, self._on_cam2, qos
                )
            self.create_subscription(JointState, "/franka/joint_states", self._on_j, 10)
            self.create_subscription(
                PoseStamped,
                "/franka_robot_state_broadcaster/current_pose",
                self._on_p,
                qos_profile_sensor_data,
            )
            self.create_subscription(JointState, "/franka_gripper/joint_states", self._on_g, 10)

        def _on_cam1(self, msg):
            try:
                self.cam1 = decode_compressed_image(msg)
            except Exception as e:
                self.get_logger().warn(f"cam1: {e}")

        def _on_cam2(self, msg):
            try:
                self.cam2 = decode_compressed_image(msg)
            except Exception as e:
                self.get_logger().warn(f"cam2: {e}")

        def _on_j(self, msg):
            self.joints = msg

        def _on_p(self, msg):
            self.pose = msg

        def _on_g(self, msg):
            self.grip = msg

        def cams_ready(self) -> bool:
            if args.single_cam:
                return self.cam1 is not None
            return self.cam1 is not None and self.cam2 is not None

        def wam_pil(self) -> Image.Image:
            if args.single_cam:
                return (
                    Image.fromarray(self.cam1)
                    .convert("RGB")
                    .resize((args.width, args.height))
                )
            _, pil = franka_dual_to_wam_image(
                self.cam1, self.cam2, cam_h=args.cam_h, cam_w=args.cam_w
            )
            return pil

    client = WAMServiceClient(args.wam_url, load_engine_on_init=False)
    try:
        client.load_engine()
    except Exception as e:
        print(f"[phase2] load_engine note: {e}", flush=True)

    rclpy.init()
    node = Sensors()
    takeover = None
    grip_pub = None
    audit: list[dict] = []
    try:
        t0 = time.time()
        while time.time() - t0 < 30.0 and (
            not node.cams_ready() or node.joints is None or node.pose is None
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.cams_ready() or node.joints is None or node.pose is None:
            raise RuntimeError(
                f"timeout waiting for cams/joints/pose "
                f"(cam1={node.cam1 is not None}, cam2={node.cam2 is not None})"
            )

        q = _joint_vector(node.joints)
        pos = np.array(
            [
                node.pose.pose.position.x,
                node.pose.pose.position.y,
                node.pose.pose.position.z,
            ],
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
        if node.grip is not None and len(node.grip.position) >= 2:
            grip_q = np.array(node.grip.position[:2], dtype=np.float32)
        else:
            grip_q = np.array([0.04, 0.04], dtype=np.float32)

        ik = FR3IK()
        T_meas = pose_to_se3(pos, quat)
        ik.calibrate_tool_from_measured(q, T_meas)
        print(
            f"[phase2] q0={np.round(q,4).tolist()} ee={np.round(pos,4).tolist()} "
            f"tool_z={ik.T_tool.translation[2]:.4f}",
            flush=True,
        )

        R = quat_xyzw_to_rot(quat)
        proprio = build_proprio_from_franka(pos, quat, grip_q, origin_xyz=origin_xyz)
        proprio_n = normalize_proprio(proprio, stats)
        print(f"[phase2] proprio_raw={np.round(proprio,4).tolist()}", flush=True)
        print(
            f"[phase2] proprio_norm={np.round(proprio_n,3).tolist()} origin={list(origin_xyz)}",
            flush=True,
        )

        img = node.wam_pil()
        print(
            f"[phase2] wam_image={img.size[0]}x{img.size[1]} "
            f"layout={'single-cam1' if args.single_cam else '[cam2|cam1]=scene|wrist'}",
            flush=True,
        )
        img.save(frames_dir / "frame_000.png")

        robot_state = torch.from_numpy(proprio_n).unsqueeze(0)
        t_inf0 = time.time()
        action_t = client.infer_action(
            input_image=img,
            robot_state=robot_state,
            action_horizon=args.action_horizon,
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            save_path=str(log_dir / "wam_out.mp4"),
        )
        dt_inf = time.time() - t_inf0
        raw = action_t.detach().cpu().numpy()
        if raw.ndim == 3:
            chunk = raw[0]
        elif raw.ndim == 2:
            chunk = raw
        else:
            raise ValueError(f"bad action shape {raw.shape}")
        n_steps = min(int(args.max_steps), int(chunk.shape[0]))
        print(f"[phase2] infer dt={dt_inf:.2f}s chunk={chunk.shape} exec_steps={n_steps}", flush=True)

        # Plan trajectory in measured EE frame
        cur_pos = pos.copy()
        cur_R = R.copy()
        cur_q = q.copy()
        plan = []
        for i in range(n_steps):
            den = denormalize_actions(chunk[i], stats)
            clamped = clamp_eef_delta(den, limits)
            dlt = clamped.astype(np.float64)
            cur_pos = cur_pos + dlt[0:3]
            # world-frame rotation increment
            cur_R = axisangle_to_rot(dlt[3:6]) @ cur_R
            reject_if_outside_workspace(
                cur_pos,
                WorkspaceLimits(
                    xyz_min=(0.25, -0.45, 0.05),
                    xyz_max=(0.85, 0.45, 0.65),
                ),
            )
            T_des = SE3.from_Rt(cur_R, cur_pos)
            q_des, ok = ik.ik(T_des, cur_q)
            if not ok:
                raise RuntimeError(f"IK failed at step {i}")
            dq = float(np.linalg.norm(q_des - cur_q))
            if dq > 0.35:
                raise RuntimeError(f"joint jump too large at step {i}: dq={dq:.3f} rad")
            plan.append(
                {
                    "i": i,
                    "raw": chunk[i].tolist(),
                    "denorm": den.tolist(),
                    "clamped": clamped.tolist(),
                    "ee_xyz": cur_pos.tolist(),
                    "q": q_des.tolist(),
                    "dq": dq,
                    "gripper_denorm": float(clamped[-1]),
                }
            )
            cur_q = q_des
            print(
                f"[phase2] plan[{i}] ee={np.round(cur_pos,4).tolist()} dq={dq:.4f} "
                f"g={clamped[-1]:.3f}",
                flush=True,
            )

        (log_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        meta = {
            "prompt": args.prompt,
            "armed": armed,
            "plan_only": bool(args.plan_only),
            "enable_gripper": bool(args.enable_gripper),
            "limits": {"max_abs_xyz": args.max_abs_xyz, "max_abs_rot": args.max_abs_rot},
            "infer_s": round(dt_inf, 3),
            "n_steps": n_steps,
        }
        (log_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if args.plan_only or not armed:
            print(f"[phase2] DONE plan-only log_dir={log_dir}", flush=True)
            return 0

        # --- hardware path ---
        takeover = GelloTakeover(node, rate_hz=50.0)
        takeover.start(q)
        if args.enable_gripper:
            grip_pub = node.create_publisher(Float32, "/gripper/gripper_client/target_gripper_width_percent", 10)

        q_prev = q.copy()
        for step in plan:
            q_tgt = np.asarray(step["q"], dtype=np.float64)
            for s in range(max(1, args.interp_substeps)):
                a = (s + 1) / float(args.interp_substeps)
                q_cmd = (1.0 - a) * q_prev + a * q_tgt
                takeover.set_goal(q_cmd)
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(args.step_dt / float(args.interp_substeps))
            q_prev = q_tgt
            audit.append({"executed": step, "t": time.time()})
            print(f"[phase2] executed step {step['i']}", flush=True)

        if args.enable_gripper and grip_pub is not None:
            # LIBERO: -1=open, +1=close → Franka percent 1=open
            from gripper_map import libero_gripper_to_percent

            g = float(plan[-1]["gripper_denorm"])
            percent = libero_gripper_to_percent(g)
            msg = Float32()
            msg.data = percent
            grip_pub.publish(msg)
            print(f"[phase2] gripper percent={percent:.2f} (from denorm={g:.3f})", flush=True)
            time.sleep(1.0)

        (log_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"[phase2] DONE motion log_dir={log_dir}", flush=True)
        return 0
    finally:
        if takeover is not None:
            takeover.stop(resume_gello=False)
            print("[phase2] GELLO left STOPPED to avoid leader mismatch fight", flush=True)
            os.system(
                "pgrep -f 'franka_gello_state_publisher/gello_publisher' | "
                "xargs -r kill -STOP; "
                "nohup python3 /tmp/kairos_hold_q.py >/tmp/kairos_hold.log 2>&1 &"
            )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

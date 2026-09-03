#!/usr/bin/env python3
"""Phase-1B: dual-cam + aligned LIBERO proprio -> WAM infer -> log only (NO motion).

ENABLE_MOTION is hard-coded False. This script must never create control publishers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Safety: never enable motion in Phase 1
ENABLE_MOTION = False

import numpy as np
import torch
from PIL import Image


def _require_no_motion() -> None:
    if ENABLE_MOTION:
        raise RuntimeError("ENABLE_MOTION must stay False in Phase-1 dry-run")


def main() -> int:
    _require_no_motion()
    parser = argparse.ArgumentParser(description="Franka sensor -> WAM dry-run (no motion)")
    parser.add_argument("--wam-url", default=os.environ.get("WAM_URL", "http://127.0.0.1:8005"))
    parser.add_argument("--prompt", default="pick up a pen")
    parser.add_argument("--num-infer", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--cam-w", type=int, default=224)
    parser.add_argument("--cam-h", type=int, default=224)
    parser.add_argument(
        "--single-cam",
        action="store_true",
        help="Legacy: only cam1 resized; default is [cam2 scene | cam1 wrist] 224x448",
    )
    parser.add_argument("--flip180", action="store_true", help="180° rotate cams before concat")
    parser.add_argument("--width", type=int, default=448, help="Legacy single-cam width")
    parser.add_argument("--height", type=int, default=224, help="Legacy single-cam height")
    parser.add_argument(
        "--stats",
        default="",
        help="LIBERO stats JSON (default: ~/kairos/benchmarks/libero_plus/libero_plus_dataset_stats.json)",
    )
    parser.add_argument(
        "--proprio-origin",
        default="",
        help="xyz origin override; default from state_builder.DEFAULT_ORIGIN_XYZ",
    )
    parser.add_argument(
        "--log-dir",
        default="",
        help="Output directory (default: ~/kairos/dryrun_logs/<timestamp>)",
    )
    parser.add_argument("--timeout-image-s", type=float, default=30.0)
    args = parser.parse_args()

    if ENABLE_MOTION:
        raise RuntimeError("refusing to run with motion enabled")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir or Path.home() / "kairos" / "dryrun_logs" / ts)
    log_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = log_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    jsonl_path = log_dir / "actions.jsonl"
    meta_path = log_dir / "meta.json"

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import JointState

    kairos_root = Path(os.environ.get("KAIROS_ROOT", Path.home() / "kairos"))
    common = kairos_root / "benchmarks" / "common"
    phase2 = kairos_root / "scripts" / "phase2"
    for p in (common, phase2):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from clients.wam_http_client import WAMServiceClient
    from action_pipeline import denormalize_actions, load_stats
    from camera_layout import (
        CAM1_COMPRESSED,
        CAM2_COMPRESSED,
        decode_compressed_image,
        franka_dual_to_wam_image,
        realsense_image_qos,
    )
    from state_builder import (
        DEFAULT_ORIGIN_XYZ,
        build_proprio_from_franka,
        parse_origin_xyz,
    )
    from state_normalize import normalize_proprio

    stats_path = args.stats or str(
        kairos_root / "benchmarks" / "libero_plus" / "libero_plus_dataset_stats.json"
    )
    stats = load_stats(stats_path)
    origin = parse_origin_xyz(args.proprio_origin) if args.proprio_origin else DEFAULT_ORIGIN_XYZ

    class SensorBuffer(Node):
        def __init__(self) -> None:
            super().__init__("kairos_phase1_dryrun")
            self.cam1 = None
            self.cam2 = None
            self.pose = None
            self.grip = None
            for qos in realsense_image_qos():
                self.create_subscription(
                    CompressedImage, CAM1_COMPRESSED, self._on_cam1, qos
                )
                self.create_subscription(
                    CompressedImage, CAM2_COMPRESSED, self._on_cam2, qos
                )
            self.create_subscription(
                PoseStamped,
                "/franka_robot_state_broadcaster/current_pose",
                self._on_pose,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                JointState, "/franka_gripper/joint_states", self._on_grip, 10
            )

        def _on_cam1(self, msg) -> None:
            try:
                self.cam1 = decode_compressed_image(msg)
            except Exception as e:
                self.get_logger().warn(f"cam1 convert failed: {e}")

        def _on_cam2(self, msg) -> None:
            try:
                self.cam2 = decode_compressed_image(msg)
            except Exception as e:
                self.get_logger().warn(f"cam2 convert failed: {e}")

        def _on_pose(self, msg) -> None:
            self.pose = msg

        def _on_grip(self, msg) -> None:
            self.grip = msg

        def cams_ready(self) -> bool:
            if args.single_cam:
                return self.cam1 is not None
            return self.cam1 is not None and self.cam2 is not None

        def proprio_ready(self) -> bool:
            return self.pose is not None

        def wam_pil(self) -> Image.Image:
            if args.single_cam:
                rgb = self.cam1
                if args.flip180:
                    rgb = np.ascontiguousarray(rgb[::-1, ::-1])
                return Image.fromarray(rgb).convert("RGB").resize((args.width, args.height))
            _, pil = franka_dual_to_wam_image(
                self.cam1,
                self.cam2,
                cam_h=args.cam_h,
                cam_w=args.cam_w,
                flip180=bool(args.flip180),
            )
            return pil

        def aligned_proprio(self) -> tuple[np.ndarray, np.ndarray]:
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
            if self.grip is not None and len(self.grip.position) >= 2:
                fingers = np.array(self.grip.position[:2], dtype=np.float64)
            else:
                fingers = np.array([0.04, 0.04], dtype=np.float64)
            raw = build_proprio_from_franka(pos, quat, fingers, origin_xyz=origin)
            return raw, normalize_proprio(raw, stats)

    meta = {
        "enable_motion": ENABLE_MOTION,
        "wam_url": args.wam_url,
        "cam_layout": "single-cam1" if args.single_cam else "[cam2|cam1]=scene|wrist",
        "flip180": bool(args.flip180),
        "proprio_origin": list(origin),
        "stats": stats_path,
        "prompt": args.prompt,
        "note": "Phase-1B: dual-cam + aligned LIBERO proprio -> WAM; no control publishers",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[dryrun] log_dir={log_dir}", flush=True)
    print(f"[dryrun] ENABLE_MOTION={ENABLE_MOTION}", flush=True)
    print(
        f"[dryrun] layout={'single-cam1' if args.single_cam else '[cam2|cam1]'} "
        f"flip180={bool(args.flip180)} origin={list(origin)}",
        flush=True,
    )

    client = WAMServiceClient(args.wam_url, load_engine_on_init=False)
    try:
        client.load_engine()
    except Exception as e:
        print(f"[dryrun] load_engine note: {e}", flush=True)

    rclpy.init()
    node = SensorBuffer()
    try:
        t0 = time.time()
        while (not node.cams_ready() or not node.proprio_ready()) and (
            time.time() - t0
        ) < args.timeout_image_s:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not node.cams_ready():
            raise RuntimeError(
                f"no cams within {args.timeout_image_s}s "
                f"(cam1={node.cam1 is not None}, cam2={node.cam2 is not None})"
            )
        if not node.proprio_ready():
            raise RuntimeError("no /franka_robot_state_broadcaster/current_pose")

        with jsonl_path.open("w", encoding="utf-8") as fout:
            for i in range(args.num_infer):
                rclpy.spin_once(node, timeout_sec=0.1)
                img = node.wam_pil()
                proprio_raw, proprio_n = node.aligned_proprio()
                ood = [j for j in range(8) if abs(float(proprio_n[j])) > 1.0]
                frame_path = frames_dir / f"frame_{i:03d}.png"
                img.save(frame_path)
                print(
                    f"[dryrun] saved {frame_path.name} size={img.size[0]}x{img.size[1]} "
                    f"proprio_norm={np.round(proprio_n,3).tolist()} ood_dims={ood or 'none'}",
                    flush=True,
                )

                robot_state = torch.from_numpy(proprio_n).unsqueeze(0)
                save_path = str(log_dir / f"wam_out_{i:03d}.mp4")
                t_infer0 = time.time()
                action = client.infer_action(
                    input_image=img,
                    robot_state=robot_state,
                    action_horizon=args.action_horizon,
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    save_path=save_path,
                )
                dt = time.time() - t_infer0
                raw_act = action.detach().cpu().numpy()
                # Log first-step denorm for inspection
                step0 = raw_act[0, 0] if raw_act.ndim == 3 else raw_act[0]
                den0 = denormalize_actions(step0, stats)
                row = {
                    "i": i,
                    "frame": str(frame_path),
                    "image_size": [img.size[0], img.size[1]],
                    "infer_s": round(dt, 3),
                    "proprio_raw": proprio_raw.tolist(),
                    "proprio_norm": proprio_n.tolist(),
                    "ood_dims": ood,
                    "action_shape": list(action.shape),
                    "action": raw_act.tolist(),
                    "denorm_step0": den0.tolist(),
                    "enable_motion": ENABLE_MOTION,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                print(
                    f"[dryrun] infer {i+1}/{args.num_infer} shape={tuple(action.shape)} "
                    f"dt={dt:.2f}s den0_xyz={np.round(den0[:3],4).tolist()} "
                    f"grip={float(den0[-1]):.3f}",
                    flush=True,
                )
                for _ in range(5):
                    rclpy.spin_once(node, timeout_sec=0.05)

        print(f"[dryrun] DONE frames={frames_dir} jsonl={jsonl_path}", flush=True)
        print("[dryrun] no motion publishers created", flush=True)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

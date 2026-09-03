"""Hardware helpers: precise GELLO pause, FCI error recovery, gripper Move with arm pause."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Optional

import numpy as np

JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]


def find_gello_publisher_pids() -> list[int]:
    """Match real gello binary only (avoid self-matching shell scripts)."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "franka_gello_state_publisher/gello_publisher"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.isdigit()]


def stop_gello() -> list[int]:
    pids = find_gello_publisher_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"[gello] SIGKILL pid={pid}", flush=True)
        except ProcessLookupError:
            pass
    return pids


def cont_gello(pids: Optional[list[int]] = None) -> None:
    for pid in pids if pids is not None else find_gello_publisher_pids():
        try:
            os.kill(pid, signal.SIGCONT)
            print(f"[gello] SIGCONT pid={pid}", flush=True)
        except ProcessLookupError:
            pass


def kill_hold_nodes() -> None:
    subprocess.run(["pkill", "-9", "-f", "/tmp/kairos_hold_q.py"], check=False)
    subprocess.run(["pkill", "-9", "-f", "kairos_hold_q"], check=False)
    # legacy inline hold shells
    subprocess.run(["pkill", "-9", "-f", 'super().__init__("kairos_hold_q")'], check=False)
    time.sleep(0.5)


def run_error_recovery(node, timeout_s: float = 15.0) -> bool:
    from franka_msgs.action import ErrorRecovery
    from rclpy.action import ActionClient

    client = ActionClient(node, ErrorRecovery, "/action_server/error_recovery")
    if not client.wait_for_server(timeout_sec=5.0):
        print("[recover] error_recovery server unavailable", flush=True)
        return False
    fut = client.send_goal_async(ErrorRecovery.Goal())
    t0 = time.time()
    while not fut.done() and time.time() - t0 < 5.0:
        import rclpy

        rclpy.spin_once(node, timeout_sec=0.05)
    gh = fut.result()
    if gh is None or not gh.accepted:
        print("[recover] goal rejected", flush=True)
        return False
    rf = gh.get_result_async()
    t0 = time.time()
    while not rf.done() and time.time() - t0 < timeout_s:
        import rclpy

        rclpy.spin_once(node, timeout_sec=0.05)
    ok = rf.result() is not None
    print(f"[recover] error_recovery ok={ok}", flush=True)
    return ok


def ensure_impedance_active(controller: str = "joint_impedance_controller") -> None:
    listed = subprocess.run(
        ["ros2", "control", "list_controllers"],
        capture_output=True,
        text=True,
        check=False,
    )
    line = next((ln for ln in (listed.stdout or "").splitlines() if controller in ln), "")
    if line and "active" in line and "inactive" not in line:
        print(f"[recover] {controller} already active", flush=True)
        return
    sw = subprocess.run(
        ["ros2", "control", "switch_controllers", "--activate", controller],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"[recover] activate {controller} rc={sw.returncode}", flush=True)


def switch_controller(activate: list[str], deactivate: list[str], timeout_s: float = 5.0) -> bool:
    cmd = ["ros2", "control", "switch_controllers"]
    for c in activate:
        cmd.extend(["--activate", c])
    for c in deactivate:
        cmd.extend(["--deactivate", c])
    sw = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout_s + 5)
    print(
        f"[ctrl] switch activate={activate} deactivate={deactivate} rc={sw.returncode}",
        flush=True,
    )
    return sw.returncode == 0


def gripper_move_width(node, width_m: float, speed: float = 0.08, pause_arm: bool = True) -> bool:
    """Move Franka Hand to width (m). Optionally pause impedance around Move."""
    from franka_msgs.action import Move
    from rclpy.action import ActionClient

    controller = "joint_impedance_controller"
    if pause_arm:
        switch_controller([], [controller])
        time.sleep(0.3)
    try:
        client = ActionClient(node, Move, "franka_gripper/move")
        if not client.wait_for_server(timeout_sec=5.0):
            print("[grip] move action unavailable", flush=True)
            return False
        goal = Move.Goal()
        goal.width = float(np.clip(width_m, 0.0, 0.08))
        goal.speed = float(speed)
        fut = client.send_goal_async(goal)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0:
            import rclpy

            rclpy.spin_once(node, timeout_sec=0.05)
        gh = fut.result()
        if gh is None or not gh.accepted:
            print("[grip] move rejected", flush=True)
            return False
        rf = gh.get_result_async()
        t0 = time.time()
        while not rf.done() and time.time() - t0 < 20.0:
            import rclpy

            rclpy.spin_once(node, timeout_sec=0.05)
        result = rf.result()
        ok = bool(result and result.result and result.result.success)
        print(f"[grip] move width={goal.width:.3f} ok={ok}", flush=True)
        return ok
    finally:
        if pause_arm:
            switch_controller([controller], [])
            time.sleep(0.4)
            ensure_impedance_active(controller)

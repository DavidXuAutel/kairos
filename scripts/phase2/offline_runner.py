"""Offline replay: denorm + clamp into FakeController (no ROS / no motion)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from action_pipeline import denormalize_actions, load_stats
from arming import ArmingGate
from fake_controller import FakeController
from limits import StepLimits, clamp_eef_delta


def replay_actions_jsonl(
    jsonl_path: str | Path,
    stats_path: str | Path,
    *,
    arm: bool = False,
    limits: StepLimits | None = None,
) -> list[dict[str, Any]]:
    stats = load_stats(stats_path)
    gate = ArmingGate()
    ctrl = FakeController()
    if arm:
        gate.issue_token("offline-test")
        gate.arm("offline-test")

    audit: list[dict[str, Any]] = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        raw = np.asarray(row["action"], dtype=np.float32)
        # Expect [B, T, 7] or [T, 7]
        if raw.ndim == 3:
            sample = raw[0, 0]
        elif raw.ndim == 2:
            sample = raw[0]
        else:
            sample = raw
        denorm = denormalize_actions(sample, stats)
        clamped = clamp_eef_delta(denorm, limits)
        entry = {
            "i": row.get("i"),
            "raw": sample.tolist(),
            "denorm": denorm.tolist(),
            "clamped": clamped.tolist(),
            "armed": gate.armed,
            "sent": False,
        }
        if gate.armed:
            ctrl.send({"action": clamped.tolist()})
            entry["sent"] = True
        audit.append(entry)
    return audit

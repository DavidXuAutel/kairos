"""Normalize / denormalize actions using LIBERO-plus dataset stats."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_stats(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _action_minmax(stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    action = stats.get("action", stats)
    if isinstance(action, dict) and "default" in action:
        action = action["default"]
    if "global_min" in action and "global_max" in action:
        amin = np.asarray(action["global_min"], dtype=np.float32)
        amax = np.asarray(action["global_max"], dtype=np.float32)
    elif "min" in action and "max" in action:
        amin = np.asarray(action["min"], dtype=np.float32)
        amax = np.asarray(action["max"], dtype=np.float32)
    else:
        raise KeyError("dataset stats missing action global_min/global_max")
    return amin.reshape(-1), amax.reshape(-1)


def denormalize_actions(raw: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Map normalized actions in roughly [-1, 1] back using global min/max."""
    x = np.asarray(raw, dtype=np.float32)
    amin, amax = _action_minmax(stats)
    if x.shape[-1] != amin.shape[0]:
        raise ValueError(f"action dim {x.shape[-1]} != stats dim {amin.shape[0]}")
    if not np.isfinite(x).all():
        raise ValueError("raw action contains non-finite values")
    # Match common (x+1)/2 * (max-min) + min denorm
    scale = amax - amin
    return ((x + 1.0) * 0.5) * scale + amin

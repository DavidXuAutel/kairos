"""Normalize 8-D proprio with LIBERO-plus dataset stats (minmax -> [-1, 1])."""
from __future__ import annotations

from typing import Any

import numpy as np

from action_pipeline import load_stats


def _state_minmax(stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    state = stats.get("state", stats)
    if isinstance(state, dict) and "default" in state:
        state = state["default"]
    if "global_min" in state and "global_max" in state:
        amin = np.asarray(state["global_min"], dtype=np.float32)
        amax = np.asarray(state["global_max"], dtype=np.float32)
    elif "min" in state and "max" in state:
        amin = np.asarray(state["min"], dtype=np.float32)
        amax = np.asarray(state["max"], dtype=np.float32)
    else:
        raise KeyError("dataset stats missing state global_min/global_max")
    return amin.reshape(-1), amax.reshape(-1)


def normalize_proprio(raw: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32).reshape(-1)
    amin, amax = _state_minmax(stats)
    if x.shape[0] != amin.shape[0]:
        raise ValueError(f"state dim {x.shape[0]} != stats dim {amin.shape[0]}")
    if not np.isfinite(x).all():
        raise ValueError("state contains non-finite values")
    scale = np.maximum(amax - amin, 1e-6)
    out = (2.0 * (x - amin) / scale - 1.0).astype(np.float32)
    # Match training LinearNormalizer.forward clamp
    return np.clip(out, -5.0, 5.0).astype(np.float32)

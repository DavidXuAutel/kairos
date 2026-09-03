"""Map LIBERO gripper action ↔ Franka Hand percent API."""
from __future__ import annotations

import numpy as np


def libero_gripper_to_percent(g: float) -> float:
    """LIBERO denorm gripper: -1=open, +1=close → Franka percent [0,1] (1=open)."""
    return float(np.clip(0.5 * (1.0 - float(g)), 0.0, 1.0))


def libero_gripper_wants_close(g: float, threshold: float = 0.0) -> bool:
    """True if action indicates close (g > threshold)."""
    return float(g) > float(threshold)

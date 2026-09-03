#!/usr/bin/env python3
"""HTTP first-frame smoke against local WAM (:8005 by default)."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

KAIROS_ROOT = Path(os.environ.get("KAIROS_ROOT", Path.home() / "kairos"))
COMMON = KAIROS_ROOT / "benchmarks" / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from clients.wam_http_client import WAMServiceClient  # noqa: E402


def main() -> int:
    endpoint = os.environ.get("WAM_URL", "http://127.0.0.1:8005")
    timeout_s = float(os.environ.get("WAM_SMOKE_TIMEOUT_S", "600"))
    client = WAMServiceClient(endpoint=endpoint, load_engine_on_init=True, timeout_s=timeout_s)

    # Synthetic 2-cam horizontal concat: H=224, W=448
    arr = (np.random.rand(224, 448, 3) * 255).astype("uint8")
    img = Image.fromarray(arr)
    robot_state = torch.zeros(1, 8, dtype=torch.float32)
    print(f"[smoke] sending /infer to {endpoint} ...", flush=True)
    action = client.infer_action(
        input_image=img,
        robot_state=robot_state,
        action_horizon=4,
        prompt=(
            "A video recorded from a robot point of view executing the following "
            "instruction: pick up the object"
        ),
        negative_prompt="",
        num_frames=1,
        num_inference_steps=4,
        cfg_scale=8.0,
        seed=0,
    )
    if not torch.isfinite(action).all():
        raise RuntimeError("action contains non-finite values")
    print("[smoke] OK action shape:", tuple(action.shape), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print("[smoke][ERROR]", repr(exc), flush=True)
        traceback.print_exc()
        raise SystemExit(1)

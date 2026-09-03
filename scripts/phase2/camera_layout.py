"""Match Kairos LIBERO-plus 2-cam training layout for FR3 RealSense.

Training (`libero_2cam.yaml` / eval_libero_single):
  primary (agentview) | wrist (eye_in_hand)  — horizontal concat → H=224, W=448

Lab mapping (after 2026-07-23 serial swap):
  cam1 = wrist D435
  cam2 = scene / third-person D435I

So model input is:  [cam2 | cam1]  each center-cropped to 224×224.

Prefer CompressedImage topics: raw Image often shows a publisher but delivers
0 Hz to rclpy subscribers on this lab stack; compressed is reliable (~30/15 fps).
"""
from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image

# Training defaults (libero_2cam.yaml video_size / per-cam shape)
CAM_H = 224
CAM_W = 224
CONCAT_H = 224
CONCAT_W = 448  # horizontal: 224 + 224

CAM1_COMPRESSED = "/cam1/cam1/color/image_raw/compressed"  # wrist
CAM2_COMPRESSED = "/cam2/cam2/color/image_raw/compressed"  # scene
CAM1_RAW = "/cam1/cam1/color/image_raw"
CAM2_RAW = "/cam2/cam2/color/image_raw"


def decode_compressed_image(msg) -> np.ndarray:
    """Decode sensor_msgs/CompressedImage → RGB uint8."""
    return np.asarray(Image.open(BytesIO(bytes(msg.data))).convert("RGB"), dtype=np.uint8)


def decode_raw_image(msg) -> np.ndarray:
    """Decode sensor_msgs/Image → RGB uint8."""
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        return arr.copy() if enc == "rgb8" else arr[:, :, ::-1].copy()
    if enc in ("rgba8", "bgra8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return arr.copy() if enc == "rgba8" else arr[:, :, ::-1].copy()
    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


def realsense_image_qos():
    """QoS matching realsense2_camera color publishers on this lab PC."""
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
    )

    reliable_tl = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )
    reliable_vol = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
    )
    return (reliable_tl, reliable_vol, qos_profile_sensor_data)


def center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Match eval_libero_single._center_crop_resize: scale to cover, then center crop."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {getattr(image, 'shape', None)}")
    pil_image = Image.fromarray(image).convert("RGB")
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR
    )
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def concat_scene_wrist(
    scene_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    *,
    cam_h: int = CAM_H,
    cam_w: int = CAM_W,
    layout: str = "horizontal",
    flip180: bool = False,
) -> np.ndarray:
    """Build training-style RGB: primary/scene left (or top), wrist right (or bottom)."""
    scene = scene_rgb
    wrist = wrist_rgb
    if flip180:
        # Match LIBERO get_libero_image obs[::-1, ::-1] (180° rotate)
        scene = np.ascontiguousarray(scene[::-1, ::-1])
        wrist = np.ascontiguousarray(wrist[::-1, ::-1])
    primary = center_crop_resize(scene, cam_w, cam_h)
    wrist_r = center_crop_resize(wrist, cam_w, cam_h)
    if layout == "horizontal":
        out = np.concatenate([primary, wrist_r], axis=1)
    elif layout == "vertical":
        out = np.concatenate([primary, wrist_r], axis=0)
    else:
        raise ValueError(f"layout must be horizontal|vertical, got {layout!r}")
    return out


def franka_dual_to_wam_image(
    cam1_wrist: np.ndarray,
    cam2_scene: np.ndarray,
    *,
    cam_h: int = CAM_H,
    cam_w: int = CAM_W,
    layout: str = "horizontal",
    flip180: bool = False,
) -> Tuple[np.ndarray, Image.Image]:
    """Map lab topics → WAM PIL image.

    cam1 = wrist, cam2 = third-person/scene → concat [scene | wrist].
    """
    rgb = concat_scene_wrist(
        cam2_scene,
        cam1_wrist,
        cam_h=cam_h,
        cam_w=cam_w,
        layout=layout,
        flip180=flip180,
    )
    return rgb, Image.fromarray(rgb).convert("RGB")

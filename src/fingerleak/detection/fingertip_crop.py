"""Crop fingertip ROIs from a hand image given MediaPipe landmarks.

Yeh stage 1 ka core hai — hum hand se 5 fingertips ke around square patches
nikalte hain jo aage super-resolution aur ridge extraction ke liye input banenge.

Tip landmarks (MediaPipe Hands):
    Thumb  = 4
    Index  = 8
    Middle = 12
    Ring   = 16
    Pinky  = 20

Crop size is computed adaptively from finger length (TIP - DIP distance) so
that a fingertip far from camera gets a smaller crop and one closer gets larger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from .mediapipe_hands import HandLandmarks


FINGERTIPS: Dict[str, int] = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
# DIP (penultimate joint) landmark for size estimation
FINGER_DIPS: Dict[str, int] = {
    "thumb": 3,
    "index": 7,
    "middle": 11,
    "ring": 15,
    "pinky": 19,
}


@dataclass
class FingertipCrop:
    """Single fingertip ROI."""
    finger: str
    tip_px: np.ndarray       # (2,) tip pixel coords
    bbox: np.ndarray         # (4,) [x1, y1, x2, y2]
    crop: np.ndarray         # the cropped image (H, W, 3) BGR
    crop_size_px: int        # side length of square crop


def crop_fingertips(
    image_bgr: np.ndarray,
    hand: HandLandmarks,
    padding_factor: float = 1.6,
    min_crop_px: int = 48,
    max_crop_px: int = 384,
    output_size: int = 128,
) -> List[FingertipCrop]:
    """Extract square fingertip patches from an image.

    Args:
        image_bgr: Source image in BGR (OpenCV format).
        hand: Detected hand landmarks.
        padding_factor: How much padding around the tip relative to TIP-DIP length.
        min_crop_px: Minimum crop side in pixels.
        max_crop_px: Maximum crop side in pixels.
        output_size: Resize each crop to this side (square). Set to 0 to skip resize.

    Returns:
        List of FingertipCrop, one per finger (5 total if all visible).
    """
    h, w = image_bgr.shape[:2]
    crops: List[FingertipCrop] = []

    for finger, tip_idx in FINGERTIPS.items():
        dip_idx = FINGER_DIPS[finger]
        tip = hand.landmarks_px[tip_idx].astype(np.float32)
        dip = hand.landmarks_px[dip_idx].astype(np.float32)

        # Adaptive crop size: finger segment length * padding
        seg_len = float(np.linalg.norm(tip - dip))
        side = int(np.clip(seg_len * padding_factor, min_crop_px, max_crop_px))
        half = side // 2

        cx, cy = int(tip[0]), int(tip[1])
        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w)
        y2 = min(cy + half, h)

        if x2 <= x1 or y2 <= y1:
            # Tip outside frame — skip
            continue

        patch = image_bgr[y1:y2, x1:x2].copy()

        if output_size > 0 and patch.size > 0:
            patch = cv2.resize(
                patch, (output_size, output_size), interpolation=cv2.INTER_CUBIC
            )

        crops.append(
            FingertipCrop(
                finger=finger,
                tip_px=tip.astype(np.int32),
                bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
                crop=patch,
                crop_size_px=side,
            )
        )
    return crops
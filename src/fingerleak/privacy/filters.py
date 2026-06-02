"""Privacy filters for fingertip regions.

Applies blur, pixelation, or emoji overlay to fingertip crops
to prevent fingerprint extraction from images/videos.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal

import cv2
import numpy as np

FilterMode = Literal["blur", "pixelate", "emoji", "blackout"]


@dataclass
class FilterResult:
    """Output of applying a privacy filter."""
    image: np.ndarray
    regions_filtered: int
    mode: str


def apply_blur(image: np.ndarray, bbox: tuple, kernel: int = 51) -> np.ndarray:
    out = image.copy()
    x, y, w, h = bbox
    H, W = out.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return out
    if kernel % 2 == 0:
        kernel += 1
    roi = out[y1:y2, x1:x2]
    out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    return out


def apply_pixelate(image: np.ndarray, bbox: tuple, blocks: int = 8) -> np.ndarray:
    out = image.copy()
    x, y, w, h = bbox
    H, W = out.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return out
    roi = out[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 2 or rw < 2:
        return out
    small = cv2.resize(roi, (max(1, blocks), max(1, blocks)),
                       interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    out[y1:y2, x1:x2] = pix
    return out


def apply_blackout(image: np.ndarray, bbox: tuple) -> np.ndarray:
    out = image.copy()
    x, y, w, h = bbox
    H, W = out.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = 0
    return out


def apply_emoji(image: np.ndarray, bbox: tuple, emoji_color=(0, 215, 255)) -> np.ndarray:
    out = image.copy()
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    radius = max(w, h) // 2
    cv2.circle(out, (cx, cy), radius, emoji_color, thickness=-1)
    cv2.circle(out, (cx, cy), radius, (0, 0, 0), thickness=2)
    return out


def apply_filter(
    image: np.ndarray,
    bboxes: List[tuple],
    mode: FilterMode = "blur",
) -> FilterResult:
    if mode not in ("blur", "pixelate", "blackout", "emoji"):
        raise ValueError(f"Unknown filter mode: {mode}")

    out = image.copy()
    for bbox in bboxes:
        if mode == "blur":
            out = apply_blur(out, bbox)
        elif mode == "pixelate":
            out = apply_pixelate(out, bbox)
        elif mode == "blackout":
            out = apply_blackout(out, bbox)
        elif mode == "emoji":
            out = apply_emoji(out, bbox)

    return FilterResult(image=out, regions_filtered=len(bboxes), mode=mode)
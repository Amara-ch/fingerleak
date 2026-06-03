"""Fallback fingertip detector using skin segmentation + ridge texture.

Used when MediaPipe fails (e.g., extreme close-ups of single fingertip
where no full hand context is visible).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class FallbackCrop:
    """A candidate fingertip crop detected without hand landmarks."""
    crop: np.ndarray
    bbox: tuple
    ridge_score: float


def _skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    """HSV + YCrCb skin segmentation — robust across skin tones."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    m1 = cv2.inRange(hsv, (0, 30, 60), (25, 200, 255))
    m2 = cv2.inRange(hsv, (160, 30, 60), (180, 200, 255))
    m3 = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _ridge_score(crop_bgr: np.ndarray) -> float:
    """Estimate fingerprint-ridge presence via Gabor response variance."""
    if crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    responses = []
    for theta in np.linspace(0, np.pi, 6, endpoint=False):
        kern = cv2.getGaborKernel(
            (15, 15), sigma=3.0, theta=theta,
            lambd=8.0, gamma=0.5, psi=0, ktype=cv2.CV_32F
        )
        resp = cv2.filter2D(gray, cv2.CV_32F, kern)
        responses.append(resp.var())
    score = float(np.max(responses))
    return min(score / 5000.0, 1.0)


def detect_fallback(
    img_bgr: np.ndarray,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.95,
    ridge_threshold: float = 0.15,
) -> List[FallbackCrop]:
    """Detect probable fingertip regions when MediaPipe fails."""
    if img_bgr is None or img_bgr.size == 0:
        return []

    H, W = img_bgr.shape[:2]
    img_area = H * W

    full_score = _ridge_score(img_bgr)
    if full_score >= ridge_threshold:
        skin = _skin_mask(img_bgr)
        skin_ratio = skin.sum() / 255.0 / img_area
        if skin_ratio >= 0.3:
            return [FallbackCrop(
                crop=img_bgr.copy(),
                bbox=(0, 0, W, H),
                ridge_score=full_score,
            )]

    mask = _skin_mask(img_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops: List[FallbackCrop] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        ratio = area / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        crop = img_bgr[y:y + h, x:x + w]
        score = _ridge_score(crop)
        if score >= ridge_threshold:
            crops.append(FallbackCrop(crop=crop, bbox=(x, y, w, h), ridge_score=score))

    crops.sort(key=lambda c: c.ridge_score, reverse=True)
    return crops[:5]

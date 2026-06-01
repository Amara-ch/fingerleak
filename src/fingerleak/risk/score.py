"""FingerLeak-Score: a calibrated privacy risk metric.

Yeh score answer karta hai: "Is selfie se fingerprint leak hone ka kitna risk hai?"

V0 (rule-based, this file):
    Combines 4 normalized signals into a single [0, 1] risk score:
        1. SHARPNESS (Laplacian variance)         — blur indicates low risk
        2. CROP SIZE (pixels)                     — small crops = far away
        3. RIDGE STRENGTH (Gabor mean response)   — ridge visibility
        4. PROXIMITY (1 / distance_cm)            — closer = higher risk

V1 (Week 9+, calibration.py):
    Replace rule-based weights with isotonic regression calibrated against
    actual fingerprint matcher accept-rates on a labeled dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# Score thresholds for human-friendly categorization
THRESHOLDS = {"low": 0.25, "medium": 0.50, "high": 0.75}

# Reference values for normalization (empirical priors; calibrate later)
REF_SHARPNESS = 200.0      # Laplacian variance — sharp images > 200
REF_SIZE_PX = 200.0        # crop side >= 200px is "good size"
REF_DISTANCE_CM = 20.0     # 20cm is "very close" → max proximity


@dataclass
class FingerLeakScore:
    """Per-fingertip risk score breakdown."""
    finger: str
    score: float                 # final risk in [0, 1]
    category: str                # Low / Medium / High / Critical
    components: dict             # individual normalized signal values
    raw: dict                    # raw (un-normalized) values, for inspection


def _laplacian_variance(image_bgr: np.ndarray) -> float:
    """Image sharpness via variance of Laplacian (higher = sharper)."""
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _categorize(score: float) -> str:
    if score < THRESHOLDS["low"]:
        return "Low"
    if score < THRESHOLDS["medium"]:
        return "Medium"
    if score < THRESHOLDS["high"]:
        return "High"
    return "Critical"


def compute_finger_score(
    finger: str,
    crop_bgr: np.ndarray,
    crop_size_px: int,
    ridge_strength: float,
    distance_cm: Optional[float],
    weights: Optional[dict] = None,
) -> FingerLeakScore:
    """Compute FingerLeak-Score for one fingertip.

    Args:
        finger: Finger name ("thumb", "index", ...).
        crop_bgr: Cropped fingertip image (BGR).
        crop_size_px: Original (pre-resize) crop side in pixels.
        ridge_strength: Mean Gabor response in [0, 1].
        distance_cm: Estimated camera-to-hand distance, or None.
        weights: Optional dict overriding default weights.

    Returns:
        FingerLeakScore with score, category, and component breakdown.
    """
    w = weights or {"sharp": 0.35, "size": 0.20, "ridge": 0.30, "prox": 0.15}

    # --- Raw signals ---
    sharp_raw = _laplacian_variance(crop_bgr)
    size_raw = float(crop_size_px)
    prox_raw = (1.0 / distance_cm) if (distance_cm and distance_cm > 1.0) else 0.0

    # --- Normalize to [0, 1] (saturating) ---
    sharp_n = float(np.clip(sharp_raw / REF_SHARPNESS, 0.0, 1.0))
    size_n = float(np.clip(size_raw / REF_SIZE_PX, 0.0, 1.0))
    ridge_n = float(np.clip(ridge_strength / 0.30, 0.0, 1.0))  # 0.30 ~ very strong
    prox_n = float(np.clip(prox_raw * REF_DISTANCE_CM, 0.0, 1.0))

    score = (
        w["sharp"] * sharp_n
        + w["size"] * size_n
        + w["ridge"] * ridge_n
        + w["prox"] * prox_n
    )
    score = float(np.clip(score, 0.0, 1.0))

    return FingerLeakScore(
        finger=finger,
        score=score,
        category=_categorize(score),
        components={
            "sharpness_norm": sharp_n,
            "size_norm": size_n,
            "ridge_norm": ridge_n,
            "proximity_norm": prox_n,
            "weights": w,
        },
        raw={
            "sharpness_laplacian": sharp_raw,
            "crop_size_px": size_raw,
            "ridge_strength": ridge_strength,
            "distance_cm": distance_cm,
        },
    )


def aggregate_frame_score(per_finger_scores: list[FingerLeakScore]) -> dict:
    """Aggregate per-finger scores into a single frame-level risk.

    Strategy: take the MAX (worst-case) — if even one finger leaks, frame leaks.
    Also report mean for completeness.
    """
    if not per_finger_scores:
        return {"max_score": 0.0, "mean_score": 0.0, "category": "Low", "n_fingers": 0}

    scores = [s.score for s in per_finger_scores]
    max_s = float(max(scores))
    mean_s = float(np.mean(scores))
    return {
        "max_score": max_s,
        "mean_score": mean_s,
        "category": _categorize(max_s),
        "n_fingers": len(per_finger_scores),
        "worst_finger": max(per_finger_scores, key=lambda x: x.score).finger,
    }
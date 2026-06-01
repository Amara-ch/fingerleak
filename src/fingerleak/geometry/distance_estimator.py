"""Estimate camera-to-hand distance from a single image using a hand-width prior.

Concept (Pinhole Camera Model):
    distance_cm = (real_object_width_cm * focal_length_px) / object_width_px

We use the "palm width" — distance between INDEX_MCP (5) and PINKY_MCP (17)
landmarks — as our object. Average adult palm width ≈ 8.5 cm.

Focal length depends on the camera. For typical smartphone selfie cameras:
    - iPhone front cam:    ~ 950 px (for 1080p capture)
    - Generic phone front: ~ 900 px
    - Webcam (720p):       ~ 700 px

Without EXIF / calibration we use a default. This gives a *rough* estimate
(±20% error) which is fine for our distance-conditioning experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detection.mediapipe_hands import HandLandmarks


# MediaPipe palm landmarks
INDEX_MCP = 5
PINKY_MCP = 17

# Anthropometric prior (adult average; can be calibrated per-user later)
PALM_WIDTH_CM = 8.5

# Reasonable defaults for unknown cameras
DEFAULT_FOCAL_LENGTH_PX = 900.0  # typical phone front-cam at ~1080p


@dataclass
class DistanceEstimate:
    """Result of distance estimation."""
    distance_cm: float
    palm_width_px: float
    focal_length_px: float
    confidence: str          # "high" / "medium" / "low"
    note: str                # explanation/caveats


def estimate_distance(
    hand: HandLandmarks,
    image_width_px: int,
    image_height_px: int,
    focal_length_px: float | None = None,
    palm_width_cm: float = PALM_WIDTH_CM,
) -> DistanceEstimate:
    """Estimate camera-to-hand distance using palm width.

    Args:
        hand: Detected hand landmarks.
        image_width_px: Image width in pixels.
        image_height_px: Image height in pixels.
        focal_length_px: Camera focal length in pixels. If None, default is used.
        palm_width_cm: Real-world palm width in cm (default 8.5 for adult).

    Returns:
        DistanceEstimate with distance_cm and metadata.
    """
    f_px = focal_length_px if focal_length_px is not None else DEFAULT_FOCAL_LENGTH_PX

    # Get pixel-space landmarks (already scaled in HandLandmarks.landmarks_px)
    p_index = hand.landmarks_px[INDEX_MCP].astype(np.float32)
    p_pinky = hand.landmarks_px[PINKY_MCP].astype(np.float32)

    palm_width_px = float(np.linalg.norm(p_index - p_pinky))

    if palm_width_px < 1.0:
        return DistanceEstimate(
            distance_cm=float("nan"),
            palm_width_px=palm_width_px,
            focal_length_px=f_px,
            confidence="low",
            note="Palm width too small — hand likely not visible properly.",
        )

    # Pinhole formula
    distance_cm = (palm_width_cm * f_px) / palm_width_px

    # Heuristic confidence based on palm size relative to image
    rel_size = palm_width_px / max(image_width_px, image_height_px)
    if rel_size > 0.20:
        confidence = "high"
        note = "Palm large in frame — distance estimate is reliable."
    elif rel_size > 0.08:
        confidence = "medium"
        note = "Palm moderately sized — estimate has ±20% error."
    else:
        confidence = "low"
        note = "Palm very small — distance estimate may be off by 30%+."

    return DistanceEstimate(
        distance_cm=float(distance_cm),
        palm_width_px=palm_width_px,
        focal_length_px=f_px,
        confidence=confidence,
        note=note,
    )
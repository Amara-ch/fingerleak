"""Peace sign (✌️) detection from MediaPipe hand landmarks.

Heuristic logic:
- Index (8) and Middle (12) fingers should be EXTENDED (tip far from wrist).
- Ring (16) and Pinky (20) fingers should be CURLED (tip near palm/MCP).
- Thumb (4) state is flexible (folded across palm or out — both are common).

Landmark indices (MediaPipe Hands):
    0  = Wrist
    4  = Thumb tip
    8  = Index tip
    12 = Middle tip
    16 = Ring tip
    20 = Pinky tip
    Each finger has 4 points (MCP, PIP, DIP, TIP).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mediapipe_hands import HandLandmarks


# MediaPipe landmark IDs
WRIST = 0
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
FINGER_MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


@dataclass
class PeaceSignResult:
    """Output of peace sign check."""
    is_peace_sign: bool
    confidence: float          # 0..1
    finger_states: dict        # e.g. {"index": "extended", "ring": "curled", ...}
    reason: str                # human-readable explanation


def _is_finger_extended(
    landmarks_norm: np.ndarray, finger: str, threshold_ratio: float = 1.05
) -> bool:
    """Check if a finger is extended.

    Logic: distance from WRIST to TIP should be greater than WRIST to PIP
    (when finger is straight, tip is the farthest point).
    """
    wrist = landmarks_norm[WRIST]
    tip = landmarks_norm[FINGER_TIPS[finger]]
    pip = landmarks_norm[FINGER_PIPS[finger]]
    d_wrist_tip = np.linalg.norm(tip - wrist)
    d_wrist_pip = np.linalg.norm(pip - wrist)
    return d_wrist_tip > d_wrist_pip * threshold_ratio


def detect_peace_sign(hand: HandLandmarks) -> PeaceSignResult:
    """Detect if a hand is showing a peace sign (✌️).

    Args:
        hand: HandLandmarks from HandDetector.

    Returns:
        PeaceSignResult with boolean, confidence, per-finger state, and reason.
    """
    lms = hand.landmarks  # normalized (21, 3)

    states = {}
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        states[finger] = "extended" if _is_finger_extended(lms, finger) else "curled"

    # Peace sign rule:
    index_ext = states["index"] == "extended"
    middle_ext = states["middle"] == "extended"
    ring_curled = states["ring"] == "curled"
    pinky_curled = states["pinky"] == "curled"

    is_peace = index_ext and middle_ext and ring_curled and pinky_curled

    # Confidence = fraction of conditions satisfied, weighted by detection score
    conditions = [index_ext, middle_ext, ring_curled, pinky_curled]
    cond_score = sum(conditions) / len(conditions)
    confidence = float(hand.score * cond_score)

    if is_peace:
        reason = "✌️ Index + Middle extended, Ring + Pinky curled."
    else:
        missing = []
        if not index_ext:
            missing.append("index not extended")
        if not middle_ext:
            missing.append("middle not extended")
        if not ring_curled:
            missing.append("ring not curled")
        if not pinky_curled:
            missing.append("pinky not curled")
        reason = "Not a peace sign: " + ", ".join(missing)

    return PeaceSignResult(
        is_peace_sign=is_peace,
        confidence=confidence,
        finger_states=states,
        reason=reason,
    )
"""Unit tests for peace sign detection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fingerleak.detection.mediapipe_hands import HandLandmarks
from fingerleak.detection.peace_sign import detect_peace_sign


def _make_hand(extended: list[bool]) -> HandLandmarks:
    """Build a synthetic hand with each finger extended/curled.

    extended: [thumb, index, middle, ring, pinky] booleans.
    """
    lms = np.zeros((21, 3), dtype=np.float32)
    # Wrist at origin
    lms[0] = [0, 0, 0]
    # MCPs (5, 9, 13, 17 — and thumb 2) at distance 1
    # PIPs (6, 10, 14, 18 — thumb 3) at distance 2
    # DIPs (7, 11, 15, 19 — thumb DIP not separate) at distance 3
    # TIPs (4, 8, 12, 16, 20) at distance 4 (extended) or 1 (curled)
    finger_map = {
        0: (2, 3, None, 4),   # thumb: MCP, PIP, --, TIP
        1: (5, 6, 7, 8),
        2: (9, 10, 11, 12),
        3: (13, 14, 15, 16),
        4: (17, 18, 19, 20),
    }
    for fidx, ext in enumerate(extended):
        mcp, pip, dip, tip = finger_map[fidx]
        col = fidx  # spread across X
        lms[mcp] = [col, 1, 0]
        lms[pip] = [col, 2, 0]
        if dip is not None:
            lms[dip] = [col, 3, 0]
        # TIP far if extended, near wrist if curled
        lms[tip] = [col, 4 if ext else 1, 0]

    return HandLandmarks(
        landmarks=lms,
        landmarks_px=(lms[:, :2] * 100).astype(np.int32),
        handedness="Right",
        score=0.95,
    )


def test_peace_sign_positive():
    """Index + middle extended, ring + pinky curled → peace sign."""
    hand = _make_hand([False, True, True, False, False])
    res = detect_peace_sign(hand)
    assert res.is_peace_sign is True
    assert res.confidence > 0.8


def test_open_hand_is_not_peace():
    """All fingers extended → not peace."""
    hand = _make_hand([True, True, True, True, True])
    res = detect_peace_sign(hand)
    assert res.is_peace_sign is False


def test_fist_is_not_peace():
    """All fingers curled → not peace."""
    hand = _make_hand([False, False, False, False, False])
    res = detect_peace_sign(hand)
    assert res.is_peace_sign is False


def test_finger_states_reported():
    """All 5 fingers reported in finger_states dict."""
    hand = _make_hand([False, True, True, False, False])
    res = detect_peace_sign(hand)
    assert set(res.finger_states.keys()) == {"thumb", "index", "middle", "ring", "pinky"}
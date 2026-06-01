"""Unit tests for geometry module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fingerleak.detection.mediapipe_hands import HandLandmarks
from fingerleak.geometry.distance_estimator import (
    estimate_distance,
    DEFAULT_FOCAL_LENGTH_PX,
    PALM_WIDTH_CM,
)


def _fake_hand(palm_width_px: float, image_size: int = 800) -> HandLandmarks:
    """Build a synthetic HandLandmarks with known palm width in pixels."""
    lms_norm = np.zeros((21, 3), dtype=np.float32)
    lms_px = np.zeros((21, 2), dtype=np.int32)
    # INDEX_MCP=5 and PINKY_MCP=17 with horizontal separation = palm_width_px
    cx = image_size // 2
    cy = image_size // 2
    lms_px[5] = [cx - int(palm_width_px / 2), cy]
    lms_px[17] = [cx + int(palm_width_px / 2), cy]
    return HandLandmarks(
        landmarks=lms_norm,
        landmarks_px=lms_px,
        handedness="Right",
        score=0.95,
    )


def test_distance_inverse_proportional_to_palm_px():
    """Doubling palm pixels should halve the distance (pinhole law)."""
    hand_far = _fake_hand(palm_width_px=100)
    hand_near = _fake_hand(palm_width_px=200)
    d_far = estimate_distance(hand_far, 800, 800).distance_cm
    d_near = estimate_distance(hand_near, 800, 800).distance_cm
    assert d_near == pytest.approx(d_far / 2, rel=0.01)


def test_distance_matches_pinhole_formula():
    """Distance = (palm_width_cm * focal_px) / palm_width_px."""
    palm_px = 150
    hand = _fake_hand(palm_width_px=palm_px)
    expected = (PALM_WIDTH_CM * DEFAULT_FOCAL_LENGTH_PX) / palm_px
    actual = estimate_distance(hand, 800, 800).distance_cm
    assert actual == pytest.approx(expected, rel=0.01)


def test_distance_confidence_low_for_tiny_palm():
    """Tiny palm in big image → low confidence."""
    hand = _fake_hand(palm_width_px=30, image_size=2000)
    result = estimate_distance(hand, 2000, 2000)
    assert result.confidence == "low"


def test_distance_confidence_high_for_large_palm():
    """Big palm in small image → high confidence."""
    hand = _fake_hand(palm_width_px=400, image_size=800)
    result = estimate_distance(hand, 800, 800)
    assert result.confidence == "high"


def test_distance_handles_invalid_palm():
    """Zero/tiny palm width → NaN distance, low conf."""
    hand = _fake_hand(palm_width_px=0)
    result = estimate_distance(hand, 800, 800)
    assert np.isnan(result.distance_cm)
    assert result.confidence == "low"
"""Unit tests for FingerLeak-Score."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fingerleak.risk.score import (
    compute_finger_score,
    aggregate_frame_score,
    _categorize,
)


def _dummy_crop(sharp: bool = True) -> np.ndarray:
    """Make a synthetic crop. Sharp = high-frequency pattern; blurry = uniform."""
    if sharp:
        # Checkerboard for high Laplacian variance
        c = np.indices((128, 128)).sum(axis=0) % 2
        return (c * 255).astype(np.uint8)[..., None].repeat(3, axis=2)
    return np.full((128, 128, 3), 128, dtype=np.uint8)


def test_categorize_thresholds():
    assert _categorize(0.10) == "Low"
    assert _categorize(0.30) == "Medium"
    assert _categorize(0.60) == "High"
    assert _categorize(0.90) == "Critical"


def test_score_in_unit_interval():
    crop = _dummy_crop(sharp=True)
    fs = compute_finger_score("index", crop, 100, 0.15, 30.0)
    assert 0.0 <= fs.score <= 1.0


def test_far_distance_lowers_score():
    """Same fingertip but farther → lower score."""
    crop = _dummy_crop(sharp=True)
    near = compute_finger_score("index", crop, 100, 0.15, 20.0)
    far = compute_finger_score("index", crop, 100, 0.15, 80.0)
    assert far.score < near.score


def test_blurry_lowers_score():
    """Same finger but blurrier → lower score."""
    sharp = compute_finger_score("index", _dummy_crop(sharp=True), 100, 0.15, 30.0)
    blurry = compute_finger_score("index", _dummy_crop(sharp=False), 100, 0.15, 30.0)
    assert blurry.score < sharp.score


def test_aggregate_max_strategy():
    """Frame score = max of per-finger scores."""
    crop = _dummy_crop()
    s1 = compute_finger_score("thumb", crop, 200, 0.20, 25.0)
    s2 = compute_finger_score("index", crop, 80, 0.05, 80.0)
    agg = aggregate_frame_score([s1, s2])
    assert agg["max_score"] == max(s1.score, s2.score)
    assert agg["worst_finger"] in ["thumb", "index"]


def test_aggregate_empty():
    """Empty list → safe defaults."""
    agg = aggregate_frame_score([])
    assert agg["n_fingers"] == 0
    assert agg["category"] == "Low"
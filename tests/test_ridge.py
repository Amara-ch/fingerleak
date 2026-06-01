"""Unit tests for Gabor ridge enhancement."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fingerleak.ridge.gabor import enhance_ridges, _build_gabor_bank


def test_gabor_bank_size():
    bank = _build_gabor_bank(ksize=21, n_orientations=8)
    assert len(bank) == 8
    for k in bank:
        assert k.shape == (21, 21)


def test_gabor_kernels_zero_mean():
    """Each kernel must be DC-removed (zero mean) — no sensitivity to brightness."""
    bank = _build_gabor_bank()
    for k in bank:
        assert abs(k.mean()) < 1e-5


def test_enhance_returns_uint8():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    res = enhance_ridges(img)
    assert res.enhanced.dtype == np.uint8
    assert res.enhanced.shape == (64, 64)
    assert 0 <= res.ridge_response <= 1.0


def test_enhance_handles_grayscale():
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    res = enhance_ridges(img)
    assert res.enhanced.shape == (64, 64)


def test_binarized_is_binary():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    res = enhance_ridges(img)
    unique_vals = np.unique(res.binarized)
    assert set(unique_vals.tolist()).issubset({0, 255})
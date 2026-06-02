"""Unit tests for privacy filters."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from fingerleak.privacy.filters import (
    apply_blur, apply_pixelate, apply_blackout, apply_emoji,
    apply_filter, FilterResult,
)


@pytest.fixture
def sample_image():
    """200x200 white image with a red square in the middle."""
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    img[50:150, 50:150] = [0, 0, 255]  # red square (BGR)
    return img


def test_blur_changes_region(sample_image):
    # bbox crosses red/white boundary so blur creates intermediate values
    bbox = (30, 30, 100, 100)
    out = apply_blur(sample_image, bbox, kernel=21)
    assert not np.array_equal(out[30:130, 30:130], sample_image[30:130, 30:130])
    # Outside region unchanged
    assert np.array_equal(out[:30, :], sample_image[:30, :])


def test_pixelate_reduces_unique_colors(sample_image):
    bbox = (50, 50, 100, 100)
    out = apply_pixelate(sample_image, bbox, blocks=4)
    region = out[50:150, 50:150].reshape(-1, 3)
    unique = np.unique(region, axis=0)
    assert len(unique) <= 16


def test_blackout_zeros_region(sample_image):
    bbox = (50, 50, 100, 100)
    out = apply_blackout(sample_image, bbox)
    assert np.all(out[50:150, 50:150] == 0)


def test_emoji_draws_circle(sample_image):
    bbox = (50, 50, 100, 100)
    out = apply_emoji(sample_image, bbox)
    center = out[100, 100]
    assert center[1] > 100  # green channel high (yellow-ish)


def test_apply_filter_returns_result(sample_image):
    bboxes = [(50, 50, 100, 100)]
    result = apply_filter(sample_image, bboxes, mode="blur")
    assert isinstance(result, FilterResult)
    assert result.regions_filtered == 1
    assert result.mode == "blur"
    assert result.image.shape == sample_image.shape


def test_apply_filter_invalid_mode(sample_image):
    with pytest.raises(ValueError):
        apply_filter(sample_image, [(0, 0, 10, 10)], mode="invalid")


def test_apply_filter_multiple_bboxes(sample_image):
    bboxes = [(10, 10, 30, 30), (100, 100, 30, 30)]
    result = apply_filter(sample_image, bboxes, mode="pixelate")
    assert result.regions_filtered == 2


def test_filter_handles_out_of_bounds(sample_image):
    bbox = (180, 180, 50, 50)
    out = apply_blur(sample_image, bbox, kernel=21)
    assert out.shape == sample_image.shape  # no crash
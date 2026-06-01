"""Gabor filter bank for fingerprint ridge enhancement.

Gabor filter ek band-pass filter hai jo specific frequency aur orientation
ke ridges ko enhance karta hai. Hum 8 different orientations (0° to 157.5°)
pe filter lagate hain aur har pixel pe MAX response lete hain — isse ridges
clear ho jate hain regardless of finger orientation.

References:
    - Hong, Wan, Jain (1998) "Fingerprint Image Enhancement: Algorithm and
      Performance Evaluation" IEEE T-PAMI
    - NIST NFIQ2 quality assessment uses similar Gabor responses
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class RidgeEnhancementResult:
    """Output of Gabor-based ridge enhancement."""
    enhanced: np.ndarray          # (H, W) uint8 — enhanced grayscale image
    orientation_field: np.ndarray  # (H, W) float32 — dominant orientation per pixel (radians)
    ridge_response: float          # scalar [0, 1] — average ridge "strength"
    binarized: np.ndarray          # (H, W) uint8 — thresholded ridge map (0 or 255)


def _build_gabor_bank(
    ksize: int = 21,
    sigma: float = 4.0,
    lambd: float = 8.0,
    gamma: float = 0.5,
    n_orientations: int = 8,
) -> List[np.ndarray]:
    """Build a bank of Gabor kernels at evenly spaced orientations.

    Args:
        ksize: Kernel size (odd integer).
        sigma: Standard deviation of the Gaussian envelope (controls scale).
        lambd: Wavelength of the sinusoid (controls ridge frequency).
        gamma: Spatial aspect ratio (0.5 = elongated along orientation).
        n_orientations: Number of orientations in [0, π).

    Returns:
        List of (ksize, ksize) float32 kernels.
    """
    kernels = []
    for i in range(n_orientations):
        theta = i * np.pi / n_orientations
        k = cv2.getGaborKernel(
            (ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=0,
            ktype=cv2.CV_32F,
        )
        # Normalize to zero-mean (removes DC component)
        k -= k.mean()
        kernels.append(k)
    return kernels


def enhance_ridges(
    image: np.ndarray,
    ksize: int = 21,
    sigma: float = 4.0,
    lambd: float = 8.0,
    gamma: float = 0.5,
    n_orientations: int = 8,
    apply_clahe: bool = True,
) -> RidgeEnhancementResult:
    """Enhance fingerprint ridges using a Gabor filter bank.

    Pipeline:
        1. Convert to grayscale.
        2. (Optional) CLAHE — adaptive histogram equalization for local contrast.
        3. Apply N Gabor filters at different orientations.
        4. At each pixel, take MAX absolute response across orientations.
        5. Track which orientation gave the max → orientation field.
        6. Otsu threshold → binarized ridge map.

    Args:
        image: Input image (BGR, RGB, or grayscale). Auto-converts to gray.
        ksize: Gabor kernel size.
        sigma: Gaussian envelope std.
        lambd: Sinusoid wavelength (≈ ridge spacing in px).
        gamma: Aspect ratio.
        n_orientations: Number of orientations.
        apply_clahe: Whether to apply CLAHE before filtering.

    Returns:
        RidgeEnhancementResult with enhanced image, orientation field,
        ridge response strength, and binarized ridge map.
    """
    # 1. Grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = gray.astype(np.uint8)

    # 2. CLAHE for local contrast
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 3. Build Gabor bank
    kernels = _build_gabor_bank(ksize, sigma, lambd, gamma, n_orientations)

    # 4. Apply each kernel; keep max response & corresponding orientation
    h, w = gray.shape
    max_response = np.zeros((h, w), dtype=np.float32)
    best_orient_idx = np.zeros((h, w), dtype=np.int32)
    gray_f = gray.astype(np.float32)

    for i, k in enumerate(kernels):
        resp = cv2.filter2D(gray_f, cv2.CV_32F, k)
        abs_resp = np.abs(resp)
        update_mask = abs_resp > max_response
        max_response = np.where(update_mask, abs_resp, max_response)
        best_orient_idx = np.where(update_mask, i, best_orient_idx)

    # 5. Normalize enhanced response to [0, 255]
    if max_response.max() > 0:
        enhanced = (max_response / max_response.max() * 255.0).astype(np.uint8)
    else:
        enhanced = np.zeros_like(gray)

    # Orientation field in radians
    orientation_field = best_orient_idx.astype(np.float32) * (np.pi / n_orientations)

    # 6. Binarize via Otsu
    _, binarized = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ridge response strength [0, 1]: mean of normalized response
    ridge_response = float(enhanced.mean() / 255.0)

    return RidgeEnhancementResult(
        enhanced=enhanced,
        orientation_field=orientation_field,
        ridge_response=ridge_response,
        binarized=binarized,
    )
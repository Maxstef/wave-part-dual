"""Simple camera and sensor simulations used by the notebooks."""

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


def sample_rggb(rgb: ArrayLike) -> NDArray[np.uint8]:
    """Sample an RGB image through an RGGB Bayer color-filter array."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")

    raw = np.zeros(image.shape[:2], dtype=image.dtype)
    raw[0::2, 0::2] = image[0::2, 0::2, 0]
    raw[0::2, 1::2] = image[0::2, 1::2, 1]
    raw[1::2, 0::2] = image[1::2, 0::2, 1]
    raw[1::2, 1::2] = image[1::2, 1::2, 2]
    return raw


def simulate_camera(
    rgb: ArrayLike,
    exposure: float = 1.0,
    gain: float = 1.0,
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0),
    focus_sigma: float = 0.0,
    noise_stddev: float = 0.018,
    seed: int = 4,
) -> NDArray[np.uint8]:
    """Apply a compact exposure, gain, white-balance, blur, and noise model."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if focus_sigma < 0 or noise_stddev < 0:
        raise ValueError("focus_sigma and noise_stddev cannot be negative")

    rng = np.random.default_rng(seed)
    optical_image = image.astype(np.float32) / 255.0
    if focus_sigma > 0:
        optical_image = cv2.GaussianBlur(optical_image, (0, 0), focus_sigma)

    signal = optical_image * exposure * np.asarray(white_balance)[None, None, :]
    measured = (signal + rng.normal(0, noise_stddev, signal.shape)) * gain
    return np.clip(measured * 255, 0, 255).astype(np.uint8)


def rolling_shutter(rgb: ArrayLike, total_shift: float) -> NDArray:
    """Shift successive rows to imitate motion during a rolling-shutter scan."""
    image = np.asarray(rgb)
    if image.ndim not in (2, 3):
        raise ValueError("rgb must be a 2D or 3D image")

    distorted = np.empty_like(image)
    for row in range(image.shape[0]):
        shift = round(total_shift * row / max(1, image.shape[0] - 1))
        distorted[row] = np.roll(image[row], shift, axis=0)
    return distorted

"""Small reusable building blocks for foreground and background estimation."""

import cv2
import numpy as np
from numpy.typing import NDArray


def motion_mask(
    previous: NDArray[np.uint8],
    current: NDArray[np.uint8],
    threshold: int = 25,
    kernel_size: int = 5,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    """Return grayscale frame difference, threshold mask, and morphology cleanup."""
    if previous.shape != current.shape:
        raise ValueError("Previous and current frames must have the same shape.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    difference = cv2.absdiff(current, previous)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
    _, raw_mask = cv2.threshold(gray_difference, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    return gray_difference, raw_mask, clean_mask


def foreground_boxes(
    mask: NDArray[np.uint8], minimum_area: int = 80
) -> list[tuple[int, int, int, int]]:
    """Return ``(x, y, width, height)`` boxes for large foreground components."""
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional single-channel image.")
    if minimum_area < 1:
        raise ValueError("minimum_area must be positive.")

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area >= minimum_area:
            boxes.append((int(x), int(y), int(width), int(height)))
    return boxes


def running_background_masks(
    frames: list[NDArray[np.uint8]],
    alpha: float = 0.08,
    threshold: int = 28,
) -> tuple[list[NDArray[np.uint8]], list[NDArray[np.uint8]]]:
    """Estimate running-average backgrounds and corresponding foreground masks."""
    if not frames:
        raise ValueError("frames must contain at least one image.")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1].")
    if any(frame.shape != frames[0].shape for frame in frames[1:]):
        raise ValueError("All frames must have the same shape.")

    background = frames[0].astype(np.float32)
    backgrounds = [frames[0].copy()]
    masks = [np.zeros(frames[0].shape[:2], dtype=np.uint8)]
    for frame in frames[1:]:
        difference = cv2.absdiff(frame, cv2.convertScaleAbs(background))
        gray_difference = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(
            gray_difference, threshold, 255, cv2.THRESH_BINARY
        )
        masks.append(mask)
        cv2.accumulateWeighted(frame, background, alpha)
        backgrounds.append(cv2.convertScaleAbs(background))
    return backgrounds, masks

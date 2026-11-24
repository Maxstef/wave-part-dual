"""Reusable corner detection and sparse Lucas-Kanade point tracking."""

import cv2
import numpy as np
from numpy.typing import NDArray


def detect_corners(
    gray: NDArray[np.uint8],
    mask: NDArray[np.uint8] | None = None,
    max_corners: int = 80,
    quality_level: float = 0.12,
    min_distance: float = 7,
    block_size: int = 7,
) -> NDArray[np.float32]:
    """Detect Shi-Tomasi corners and return them as an ``(N, 2)`` array."""
    if gray.ndim != 2:
        raise ValueError("gray must be a two-dimensional image.")
    if mask is not None and mask.shape != gray.shape:
        raise ValueError("mask and gray image must have the same shape.")
    if max_corners < 1:
        raise ValueError("max_corners must be positive.")
    if not 0 < quality_level <= 1:
        raise ValueError("quality_level must be in the interval (0, 1].")

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
        blockSize=block_size,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float32)
    return corners.reshape(-1, 2).astype(np.float32)


def track_lk_points(
    previous_gray: NDArray[np.uint8],
    current_gray: NDArray[np.uint8],
    points: NDArray[np.float32],
    win_size: tuple[int, int] = (21, 21),
    max_level: int = 3,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]:
    """Track sparse points once with pyramidal Lucas-Kanade optical flow."""
    if previous_gray.shape != current_gray.shape or previous_gray.ndim != 2:
        raise ValueError("Both grayscale frames must have the same 2D shape.")
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return points.copy(), np.empty(0, dtype=bool), np.empty(0, dtype=np.float32)

    tracked, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points.reshape(-1, 1, 2),
        None,
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None or error is None:
        return points.copy(), np.zeros(len(points), dtype=bool), np.full(len(points), np.inf)
    return (
        tracked.reshape(-1, 2).astype(np.float32),
        status.reshape(-1).astype(bool),
        error.reshape(-1).astype(np.float32),
    )


def track_points_forward_backward(
    previous_gray: NDArray[np.uint8],
    current_gray: NDArray[np.uint8],
    points: NDArray[np.float32],
    win_size: tuple[int, int] = (21, 21),
    max_level: int = 3,
    maximum_roundtrip_error: float = 1.5,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]:
    """Track forward and reject points that cannot return near their origin."""
    forward, forward_ok, _ = track_lk_points(
        previous_gray, current_gray, points, win_size, max_level
    )
    backward, backward_ok, _ = track_lk_points(
        current_gray, previous_gray, forward, win_size, max_level
    )
    original = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    roundtrip_error = np.linalg.norm(backward - original, axis=1)
    valid = forward_ok & backward_ok & (roundtrip_error <= maximum_roundtrip_error)
    return forward, valid, roundtrip_error.astype(np.float32)

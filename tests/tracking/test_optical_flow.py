import cv2
import numpy as np
import pytest

from wave_part_dual.tracking import (
    detect_corners,
    track_lk_points,
    track_points_forward_backward,
)


def _translated_texture(dx=5, dy=3):
    rng = np.random.default_rng(4)
    first = np.zeros((100, 120), dtype=np.uint8)
    texture = rng.integers(30, 230, (45, 50), dtype=np.uint8)
    first[25:70, 30:80] = texture
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    second = cv2.warpAffine(first, transform, (first.shape[1], first.shape[0]))
    return first, second


def test_detect_and_track_known_translation():
    first, second = _translated_texture()
    points = detect_corners(first, max_corners=40, quality_level=0.1)
    tracked, valid, _ = track_points_forward_backward(first, second, points)

    displacement = tracked[valid] - points[valid]
    assert valid.sum() >= 10
    assert np.allclose(np.median(displacement, axis=0), [5, 3], atol=0.25)


def test_lk_empty_points_remain_empty():
    frame = np.zeros((20, 20), dtype=np.uint8)
    tracked, valid, error = track_lk_points(
        frame, frame, np.empty((0, 2), dtype=np.float32)
    )

    assert tracked.shape == (0, 2)
    assert valid.size == 0
    assert error.size == 0


def test_detect_corners_returns_empty_for_flat_image():
    flat = np.full((30, 40), 127, dtype=np.uint8)
    assert detect_corners(flat).shape == (0, 2)


def test_detect_corners_rejects_color_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        detect_corners(np.zeros((10, 10, 3), dtype=np.uint8))

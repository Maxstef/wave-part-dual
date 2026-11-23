import numpy as np
import pytest

from wave_part_dual.tracking import (
    foreground_boxes,
    motion_mask,
    running_background_masks,
)


def test_motion_mask_detects_large_change_and_removes_isolated_noise():
    previous = np.zeros((40, 50, 3), dtype=np.uint8)
    current = previous.copy()
    current[10:25, 18:35] = 180
    current[2, 2] = 255

    difference, raw_mask, clean_mask = motion_mask(
        previous, current, threshold=30, kernel_size=3
    )

    assert difference.shape == previous.shape[:2]
    assert raw_mask[2, 2] == 255
    assert clean_mask[2, 2] == 0
    assert clean_mask[17, 25] == 255


def test_motion_mask_rejects_invalid_kernel_size():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive odd"):
        motion_mask(frame, frame, kernel_size=4)


def test_foreground_boxes_filters_small_components():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[3:6, 4:7] = 255
    mask[15:25, 20:35] = 255

    assert foreground_boxes(mask, minimum_area=20) == [(20, 15, 15, 10)]


def test_running_background_starts_from_first_frame_and_adapts():
    dark = np.zeros((12, 16, 3), dtype=np.uint8)
    bright = np.full_like(dark, 100)

    backgrounds, masks = running_background_masks(
        [dark, bright, bright], alpha=0.5, threshold=20
    )

    assert np.count_nonzero(masks[0]) == 0
    assert np.all(masks[1] == 255)
    assert np.all(masks[2] == 255)
    assert np.all(backgrounds[1] == 50)
    assert np.all(backgrounds[2] == 75)


def test_running_background_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        running_background_masks([])

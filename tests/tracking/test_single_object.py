import cv2
import numpy as np
import pytest

from wave_part_dual.tracking import box_iou, expand_box, match_template


def _textured_patch():
    patch = np.zeros((18, 24), dtype=np.uint8)
    cv2.rectangle(patch, (2, 3), (20, 15), 150, -1)
    cv2.circle(patch, (8, 8), 3, 255, -1)
    cv2.line(patch, (13, 4), (19, 13), 40, 2)
    return patch


def test_match_template_finds_known_location():
    template = _textured_patch()
    frame = np.zeros((80, 100), dtype=np.uint8)
    frame[35:53, 42:66] = template

    match = match_template(frame, template)

    assert match.box == (42, 35, 24, 18)
    assert match.score > 0.99
    assert match.scale == 1.0


def test_multiscale_match_recovers_larger_template():
    template = _textured_patch()
    larger = cv2.resize(template, (36, 27), interpolation=cv2.INTER_LINEAR)
    frame = np.zeros((90, 120), dtype=np.uint8)
    frame[40:67, 55:91] = larger

    match = match_template(frame, template, scales=(1.0, 1.5))

    assert match.box == (55, 40, 36, 27)
    assert match.scale == 1.5
    assert match.score > 0.99


def test_box_iou_handles_overlap_and_separation():
    assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)
    assert box_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_expand_box_clips_to_image():
    assert expand_box((3, 4, 10, 8), 6, (20, 30)) == (0, 0, 19, 18)


def test_match_template_rejects_color_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        match_template(
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.zeros((5, 5), dtype=np.uint8),
        )

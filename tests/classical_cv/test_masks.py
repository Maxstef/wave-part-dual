import numpy as np
import pytest

from wave_part_dual.classical_cv import mask_iou


def test_mask_iou_handles_overlap_and_empty_masks() -> None:
    prediction = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [1, 0]])

    assert mask_iou(prediction, target) == pytest.approx(1 / 3)
    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0


def test_mask_iou_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mask_iou(np.zeros((2, 2)), np.zeros((3, 3)))

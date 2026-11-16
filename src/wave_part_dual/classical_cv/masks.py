"""Small utilities for binary-mask experiments."""

import numpy as np
from numpy.typing import ArrayLike


def mask_iou(prediction: ArrayLike, target: ArrayLike) -> float:
    """Return intersection-over-union for two equally shaped binary masks."""
    prediction_mask = np.asarray(prediction, dtype=bool)
    target_mask = np.asarray(target, dtype=bool)
    if prediction_mask.shape != target_mask.shape:
        raise ValueError("prediction and target must have the same shape")

    intersection = np.logical_and(prediction_mask, target_mask).sum()
    union = np.logical_or(prediction_mask, target_mask).sum()
    return float(intersection / union) if union else 1.0

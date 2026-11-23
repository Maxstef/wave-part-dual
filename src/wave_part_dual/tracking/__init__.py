"""Reusable algorithms for motion estimation and object tracking."""

from .background import foreground_boxes, motion_mask, running_background_masks

__all__ = [
    "foreground_boxes",
    "motion_mask",
    "running_background_masks",
]

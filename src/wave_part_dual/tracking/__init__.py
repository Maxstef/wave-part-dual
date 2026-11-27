"""Reusable algorithms for motion estimation and object tracking."""

from .background import foreground_boxes, motion_mask, running_background_masks
from .kalman import (
    KalmanState,
    correct_position,
    filter_positions,
    initialize_constant_velocity,
    predict_constant_velocity,
)
from .optical_flow import detect_corners, track_lk_points, track_points_forward_backward
from .single_object import Box, TemplateMatch, box_iou, expand_box, match_template

__all__ = [
    "foreground_boxes",
    "motion_mask",
    "running_background_masks",
    "KalmanState",
    "initialize_constant_velocity",
    "predict_constant_velocity",
    "correct_position",
    "filter_positions",
    "detect_corners",
    "track_lk_points",
    "track_points_forward_backward",
    "Box",
    "TemplateMatch",
    "box_iou",
    "expand_box",
    "match_template",
]

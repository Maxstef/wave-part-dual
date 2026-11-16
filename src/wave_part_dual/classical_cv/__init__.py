"""Reusable building blocks for classical computer-vision experiments."""

from .calibration import calibrate_camera, create_artificial_calibration_data
from .camera import rolling_shutter, sample_rggb, simulate_camera
from .geometry import (
    compute_ground_points_from_pixels,
    distort_points,
    estimate_distance_from_size,
    look_at_world_to_camera,
    make_ground_camera_geometry,
    project_points,
    radial_distort,
)
from .masks import mask_iou

__all__ = [
    "calibrate_camera",
    "compute_ground_points_from_pixels",
    "create_artificial_calibration_data",
    "distort_points",
    "estimate_distance_from_size",
    "look_at_world_to_camera",
    "make_ground_camera_geometry",
    "mask_iou",
    "project_points",
    "radial_distort",
    "rolling_shutter",
    "sample_rggb",
    "simulate_camera",
]

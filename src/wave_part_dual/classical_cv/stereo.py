"""Reusable stereo-matching, depth, and reconstruction helpers."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sad_cost_curve(
    left_image: ArrayLike,
    right_image: ArrayLike,
    point: tuple[int, int],
    disparities: ArrayLike,
    block_size: int = 11,
) -> NDArray[np.float64]:
    """Return SAD costs for matching one left-image patch along its scanline."""
    left = np.asarray(left_image, dtype=float)
    right = np.asarray(right_image, dtype=float)
    disparity_values = np.asarray(disparities)

    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("left_image and right_image must be same-shaped 2D arrays")
    if block_size <= 0 or block_size % 2 == 0:
        raise ValueError("block_size must be a positive odd integer")
    if disparity_values.ndim != 1 or disparity_values.size == 0:
        raise ValueError("disparities must be a non-empty 1D sequence")
    if not np.all(np.isfinite(disparity_values)) or not np.all(disparity_values == disparity_values.astype(int)):
        raise ValueError("disparities must contain finite integers")

    x, y = point
    radius = block_size // 2
    height, width = left.shape
    if x - radius < 0 or x + radius >= width or y - radius < 0 or y + radius >= height:
        raise ValueError("the left patch must fit completely inside the image")

    disparity_values = disparity_values.astype(int)
    candidate_x = x - disparity_values
    if np.any(candidate_x - radius < 0) or np.any(candidate_x + radius >= width):
        raise ValueError("every candidate patch must fit completely inside the right image")

    left_patch = left[y - radius : y + radius + 1, x - radius : x + radius + 1]
    costs = []
    for candidate in candidate_x:
        right_patch = right[
            y - radius : y + radius + 1,
            candidate - radius : candidate + radius + 1,
        ]
        costs.append(np.abs(left_patch - right_patch).sum())
    return np.asarray(costs, dtype=float)


def disparity_to_depth(
    disparity: ArrayLike,
    focal_length_px: float,
    baseline_m: float,
) -> NDArray[np.float64]:
    """Convert positive rectified-stereo disparity to metric depth."""
    if focal_length_px <= 0 or baseline_m <= 0:
        raise ValueError("focal_length_px and baseline_m must be positive")

    disparity_values = np.asarray(disparity, dtype=float)
    depth = np.full(disparity_values.shape, np.nan, dtype=float)
    valid = np.isfinite(disparity_values) & (disparity_values > 0)
    depth[valid] = focal_length_px * baseline_m / disparity_values[valid]
    return depth


def depth_map_to_points(
    depth: ArrayLike,
    focal_length_px: float,
    principal_point: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Back-project a depth map to camera-frame points with square pixels."""
    depth_values = np.asarray(depth, dtype=float)
    if depth_values.ndim != 2:
        raise ValueError("depth must be a 2D array")
    if focal_length_px <= 0:
        raise ValueError("focal_length_px must be positive")
    if np.any(np.isfinite(depth_values) & (depth_values <= 0)):
        raise ValueError("finite depth values must be positive")

    height, width = depth_values.shape
    cx, cy = principal_point if principal_point is not None else (width / 2, height / 2)
    if not np.isfinite((cx, cy)).all():
        raise ValueError("principal_point must contain finite coordinates")

    v, u = np.mgrid[0:height, 0:width]
    x = (u - cx) * depth_values / focal_length_px
    y = (v - cy) * depth_values / focal_length_px
    return np.stack((x, y, depth_values), axis=-1)

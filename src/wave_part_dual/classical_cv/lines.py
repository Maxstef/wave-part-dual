"""Reusable line-fitting and Hough-voting helpers."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fit_line_ransac(
    x: ArrayLike,
    y: ArrayLike,
    residual_threshold: float = 1.3,
    trials: int = 250,
    seed: int = 5,
) -> tuple[float, float, NDArray[np.bool_]]:
    """Fit ``y = slope * x + intercept`` using two-point RANSAC."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.shape != y_values.shape:
        raise ValueError("x and y must be same-shaped 1D arrays")
    if x_values.size < 2 or not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("x and y must contain at least two finite points")
    if np.unique(x_values).size < 2:
        raise ValueError("at least two distinct x coordinates are required")
    if residual_threshold <= 0 or not np.isfinite(residual_threshold):
        raise ValueError("residual_threshold must be positive and finite")
    if trials <= 0:
        raise ValueError("trials must be positive")

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(x_values.size, dtype=bool)
    for _ in range(trials):
        first, second = rng.choice(x_values.size, size=2, replace=False)
        if x_values[first] == x_values[second]:
            continue
        slope = (y_values[second] - y_values[first]) / (
            x_values[second] - x_values[first]
        )
        intercept = y_values[first] - slope * x_values[first]
        inliers = (
            np.abs(y_values - (slope * x_values + intercept))
            <= residual_threshold
        )
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() < 2:
        raise RuntimeError("RANSAC did not find two supporting points")
    slope, intercept = np.polyfit(x_values[best_inliers], y_values[best_inliers], 1)
    return float(slope), float(intercept), best_inliers


def hough_accumulator(
    edge_image: ArrayLike,
    theta_degrees: ArrayLike | None = None,
) -> tuple[NDArray[np.uint32], NDArray[np.int64], NDArray[np.float64]]:
    """Build a basic rho-theta vote accumulator for a binary edge image."""
    edges = np.asarray(edge_image)
    if edges.ndim != 2:
        raise ValueError("edge_image must be a 2D array")
    theta_values = np.asarray(
        np.arange(-90, 90) if theta_degrees is None else theta_degrees,
        dtype=float,
    )
    if theta_values.ndim != 1 or theta_values.size == 0 or not np.isfinite(theta_values).all():
        raise ValueError("theta_degrees must be a non-empty finite 1D sequence")

    y_pixels, x_pixels = np.nonzero(edges)
    diagonal = int(np.ceil(np.hypot(*edges.shape)))
    rhos = np.arange(-diagonal, diagonal + 1, dtype=np.int64)
    accumulator = np.zeros((rhos.size, theta_values.size), dtype=np.uint32)

    for theta_index, theta in enumerate(np.deg2rad(theta_values)):
        rho_indices = np.rint(
            x_pixels * np.cos(theta) + y_pixels * np.sin(theta)
        ).astype(int) + diagonal
        np.add.at(accumulator[:, theta_index], rho_indices, 1)
    return accumulator, rhos, theta_values

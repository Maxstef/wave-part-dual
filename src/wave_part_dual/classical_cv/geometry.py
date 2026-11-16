"""Projection, distortion, and pixel-ray geometry helpers."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def project_points(points_xyz: ArrayLike, camera_matrix: ArrayLike) -> NDArray[np.float64]:
    """Project camera-frame 3D points into pixels with a pinhole camera matrix."""
    points = np.asarray(points_xyz, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (n, 3)")
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")
    if np.any(points[:, 2] <= 0):
        raise ValueError("all points must be in front of the camera")

    normalized = points[:, :2] / points[:, 2:3]
    return (matrix[:2, :2] @ normalized.T).T + matrix[:2, 2]


def look_at_world_to_camera(
    camera_position: ArrayLike,
    target: ArrayLike,
    world_up: ArrayLike = (0.0, 0.0, 1.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return rotation and translation for a camera looking at a world point."""
    position = np.asarray(camera_position, dtype=float)
    target_point = np.asarray(target, dtype=float)
    up = np.asarray(world_up, dtype=float)

    forward = target_point - position
    if np.linalg.norm(forward) == 0:
        raise ValueError("camera_position and target must differ")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) == 0:
        raise ValueError("view direction cannot be parallel to world_up")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    return rotation, -rotation @ position


def distort_points(
    x: ArrayLike,
    y: ArrayLike,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray, NDArray]:
    """Apply radial and tangential distortion to normalized image coordinates."""
    x_values, y_values = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    cx, cy = center
    x_centered = x_values - cx
    y_centered = y_values - cy
    radius_squared = x_centered**2 + y_centered**2
    radial_scale = 1 + k1 * radius_squared + k2 * radius_squared**2 + k3 * radius_squared**3
    x_distorted = (
        x_centered * radial_scale
        + 2 * p1 * x_centered * y_centered
        + p2 * (radius_squared + 2 * x_centered**2)
    )
    y_distorted = (
        y_centered * radial_scale
        + p1 * (radius_squared + 2 * y_centered**2)
        + 2 * p2 * x_centered * y_centered
    )
    return cx + x_distorted, cy + y_distorted


def radial_distort(
    x: ArrayLike,
    y: ArrayLike,
    k1: float,
    k2: float = 0.0,
    k3: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> tuple[NDArray, NDArray]:
    """Apply radial-only distortion to normalized image coordinates."""
    return distort_points(x, y, k1=k1, k2=k2, k3=k3, center=center)


def estimate_distance_from_size(
    focal_length_px: float,
    known_size: float,
    measured_size_px: ArrayLike,
) -> NDArray[np.float64]:
    """Estimate distance using focal length and an object's known physical size."""
    measured = np.asarray(measured_size_px, dtype=float)
    if focal_length_px <= 0 or known_size <= 0 or np.any(measured <= 0):
        raise ValueError("focal length and object sizes must be positive")
    return focal_length_px * known_size / measured


def make_ground_camera_geometry(
    camera_height: float = 1.6,
    pitch_degrees: float = 15.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return camera origin and camera-to-world rotation above flat ground."""
    if camera_height <= 0:
        raise ValueError("camera_height must be positive")

    pitch = np.deg2rad(pitch_degrees)
    camera_origin_world = np.array([0.0, 0.0, camera_height])
    right_world = np.array([1.0, 0.0, 0.0])
    forward_world = np.array([0.0, np.cos(pitch), -np.sin(pitch)])
    down_world = np.cross(forward_world, right_world)
    camera_to_world = np.column_stack([right_world, down_world, forward_world])
    return camera_origin_world, camera_to_world


def compute_ground_points_from_pixels(
    camera_matrix: ArrayLike,
    pixels: ArrayLike,
    camera_height: float = 1.6,
    pitch_degrees: float = 15.0,
) -> NDArray[np.float64]:
    """Intersect pixel viewing rays with the ideal ground plane ``Z = 0``."""
    matrix = np.asarray(camera_matrix, dtype=float)
    pixel_points = np.asarray(pixels, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")
    if pixel_points.ndim != 2 or pixel_points.shape[1] != 2:
        raise ValueError("pixels must have shape (n, 2)")

    origin, camera_to_world = make_ground_camera_geometry(camera_height, pitch_degrees)
    pixels_h = np.column_stack([pixel_points, np.ones(len(pixel_points))])
    rays_camera = (np.linalg.inv(matrix) @ pixels_h.T).T
    rays_world = (camera_to_world @ rays_camera.T).T
    if np.any(rays_world[:, 2] >= 0):
        raise ValueError("all selected pixels must point below the horizon")

    scale = -origin[2] / rays_world[:, 2]
    return origin + scale[:, None] * rays_world

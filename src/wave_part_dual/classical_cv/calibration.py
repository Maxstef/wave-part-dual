"""Synthetic camera-calibration data and estimation helpers."""

from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray


class SyntheticCalibrationData(NamedTuple):
    """Inputs, intermediate projections, and truth for a synthetic calibration."""

    object_points: list[NDArray[np.float32]]
    image_points: list[NDArray[np.float32]]
    ideal_image_points: list[NDArray[np.float64]]
    distorted_image_points: list[NDArray[np.float64]]
    image_size: tuple[int, int]
    camera_matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]


def create_artificial_calibration_data(
    camera_matrix: ArrayLike | None = None,
    distortion: ArrayLike | None = None,
    num_images: int = 20,
    image_size: tuple[int, int] = (640, 480),
    seed: int = 12,
    noise_stddev: float = 0.20,
    board_shape: tuple[int, int] = (9, 6),
    square_size: float = 0.22,
    rotation_bounds: tuple[ArrayLike, ArrayLike] = (
        (-0.50, -0.50, -0.25),
        (0.50, 0.50, 0.25),
    ),
    translation_bounds: tuple[ArrayLike, ArrayLike] = (
        (-0.35, -0.35, 2.4),
        (0.35, 0.35, 4.2),
    ),
    max_attempts: int = 10_000,
) -> SyntheticCalibrationData:
    """Generate noisy checkerboard observations from a known synthetic camera."""
    if num_images <= 0 or square_size <= 0 or noise_stddev < 0:
        raise ValueError("num_images and square_size must be positive; noise cannot be negative")

    rng = np.random.default_rng(seed)
    if camera_matrix is None:
        camera_matrix = np.array(
            [[700.0, 0.0, 318.0], [0.0, 710.0, 242.0], [0.0, 0.0, 1.0]]
        )
    camera_matrix = np.asarray(camera_matrix, dtype=float)
    if distortion is None:
        distortion = np.array([-0.15, 0.04, 0.002, -0.001, 0.0])
    distortion = np.asarray(distortion, dtype=float)

    columns, rows = board_shape
    board = np.zeros((columns * rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    board[:, :2] -= board[:, :2].mean(axis=0)

    object_points: list[NDArray[np.float32]] = []
    image_points: list[NDArray[np.float32]] = []
    ideal_points: list[NDArray[np.float64]] = []
    distorted_points: list[NDArray[np.float64]] = []

    attempts = 0
    while len(object_points) < num_images and attempts < max_attempts:
        attempts += 1
        rotation = rng.uniform(*rotation_bounds).astype(np.float64)
        translation = rng.uniform(*translation_bounds).astype(np.float64)
        ideal, _ = cv2.projectPoints(board, rotation, translation, camera_matrix, None)
        projected, _ = cv2.projectPoints(board, rotation, translation, camera_matrix, distortion)
        projected = projected.reshape(-1, 2)
        inside = (
            projected[:, 0].min() > 10
            and projected[:, 0].max() < image_size[0] - 10
            and projected[:, 1].min() > 10
            and projected[:, 1].max() < image_size[1] - 10
        )
        if not inside:
            continue

        noisy = projected + rng.normal(0, noise_stddev, projected.shape)
        object_points.append(board.copy())
        ideal_points.append(ideal.reshape(-1, 2))
        distorted_points.append(projected.copy())
        image_points.append(noisy.astype(np.float32).reshape(-1, 1, 2))

    if len(object_points) < num_images:
        raise RuntimeError("could not place enough fully visible checkerboards within max_attempts")

    return SyntheticCalibrationData(
        object_points,
        image_points,
        ideal_points,
        distorted_points,
        image_size,
        camera_matrix,
        distortion,
    )


def calibrate_camera(
    object_points: list[NDArray[np.float32]],
    image_points: list[NDArray[np.float32]],
    image_size: tuple[int, int],
    flags: int = cv2.CALIB_FIX_K3,
) -> dict[str, object]:
    """Estimate camera parameters and per-view reprojection errors."""
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags
    )

    view_errors = []
    for object_view, observed, rotation, translation in zip(
        object_points, image_points, rotations, translations
    ):
        reprojected, _ = cv2.projectPoints(
            object_view, rotation, translation, matrix, distortion
        )
        differences = observed.reshape(-1, 2) - reprojected.reshape(-1, 2)
        view_errors.append(np.sqrt(np.mean(np.sum(differences**2, axis=1))))

    return {
        "rms": float(rms),
        "K": matrix,
        "distortion": distortion,
        "rotation_vectors": rotations,
        "translation_vectors": translations,
        "view_errors": np.asarray(view_errors),
    }

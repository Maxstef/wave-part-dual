import numpy as np
import pytest

from wave_part_dual.classical_cv import (
    compute_ground_points_from_pixels,
    distort_points,
    estimate_distance_from_size,
    look_at_world_to_camera,
    project_points,
)


def test_project_points_uses_depth_and_intrinsics() -> None:
    matrix = np.array([[100.0, 0.0, 50.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])
    points = np.array([[1.0, 2.0, 2.0], [0.0, 0.0, 4.0]])

    assert np.allclose(project_points(points, matrix), [[100.0, 260.0], [50.0, 60.0]])


def test_look_at_maps_camera_to_origin_and_target_in_front() -> None:
    camera = np.array([2.0, -3.0, 1.5])
    target = np.array([0.0, 0.0, 0.5])
    rotation, translation = look_at_world_to_camera(camera, target)

    camera_in_camera_frame = rotation @ camera + translation
    target_in_camera_frame = rotation @ target + translation

    assert np.allclose(camera_in_camera_frame, 0.0)
    assert target_in_camera_frame[2] > 0
    assert np.allclose(rotation @ rotation.T, np.eye(3))


def test_distortion_is_identity_when_coefficients_are_zero() -> None:
    x = np.array([-0.5, 0.2])
    y = np.array([0.4, -0.3])

    distorted_x, distorted_y = distort_points(x, y)

    assert np.allclose(distorted_x, x)
    assert np.allclose(distorted_y, y)


def test_distance_from_known_size_inverts_apparent_size() -> None:
    distances = estimate_distance_from_size(500.0, 0.6, [300.0, 150.0])

    assert np.allclose(distances, [1.0, 2.0])


def test_ground_intersections_land_on_z_zero() -> None:
    matrix = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    pixels = np.array([[320.0, 300.0], [400.0, 380.0]])

    points = compute_ground_points_from_pixels(matrix, pixels, camera_height=1.6, pitch_degrees=15.0)

    assert np.allclose(points[:, 2], 0.0)
    assert np.all(points[:, 1] > 0)


def test_ground_intersection_rejects_pixels_above_horizon() -> None:
    matrix = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="below the horizon"):
        compute_ground_points_from_pixels(matrix, [[320.0, 0.0]], pitch_degrees=15.0)

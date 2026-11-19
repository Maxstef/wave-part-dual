import numpy as np
import pytest

from wave_part_dual.classical_cv import (
    depth_map_to_points,
    disparity_to_depth,
    sad_cost_curve,
)


def test_sad_cost_curve_finds_known_horizontal_shift() -> None:
    rng = np.random.default_rng(7)
    left = rng.integers(0, 256, size=(9, 18), dtype=np.uint8)
    right = np.roll(left, -3, axis=1)
    disparities = np.arange(0, 6)

    costs = sad_cost_curve(left, right, point=(10, 4), disparities=disparities, block_size=3)

    assert disparities[np.argmin(costs)] == 3
    assert costs[3] == 0


def test_sad_cost_curve_rejects_invalid_patches() -> None:
    image = np.zeros((7, 7), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive odd"):
        sad_cost_curve(image, image, point=(3, 3), disparities=[0], block_size=4)
    with pytest.raises(ValueError, match="left patch"):
        sad_cost_curve(image, image, point=(0, 3), disparities=[0], block_size=3)
    with pytest.raises(ValueError, match="candidate patch"):
        sad_cost_curve(image, image, point=(3, 3), disparities=[3], block_size=3)


def test_disparity_to_depth_converts_positive_values_and_marks_invalid_ones() -> None:
    depth = disparity_to_depth([0.0, 2.0, 4.0, np.nan], focal_length_px=100.0, baseline_m=0.12)

    assert np.isnan(depth[[0, 3]]).all()
    assert np.allclose(depth[[1, 2]], [6.0, 3.0])


def test_depth_map_to_points_back_projects_pixels() -> None:
    depth = np.full((2, 2), 2.0)

    points = depth_map_to_points(depth, focal_length_px=2.0, principal_point=(0.0, 0.0))

    assert points.shape == (2, 2, 3)
    assert np.allclose(points[0, 0], [0.0, 0.0, 2.0])
    assert np.allclose(points[1, 1], [1.0, 1.0, 2.0])


def test_stereo_geometry_rejects_nonphysical_parameters() -> None:
    with pytest.raises(ValueError, match="positive"):
        disparity_to_depth([1.0], focal_length_px=0.0, baseline_m=0.12)
    with pytest.raises(ValueError, match="positive"):
        depth_map_to_points([[1.0]], focal_length_px=-1.0)

import numpy as np
import pytest

from wave_part_dual.classical_cv import fit_line_ransac, hough_accumulator


def test_fit_line_ransac_ignores_distant_outlier() -> None:
    x = np.arange(6, dtype=float)
    y = 2 * x + 3
    y[-1] += 30

    slope, intercept, inliers = fit_line_ransac(
        x, y, residual_threshold=0.1, trials=100, seed=4
    )

    assert np.isclose(slope, 2)
    assert np.isclose(intercept, 3)
    assert np.array_equal(inliers, [True, True, True, True, True, False])


def test_fit_line_ransac_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same-shaped"):
        fit_line_ransac([0, 1], [0])
    with pytest.raises(ValueError, match="positive"):
        fit_line_ransac([0, 1], [0, 1], residual_threshold=0)
    with pytest.raises(ValueError, match="distinct"):
        fit_line_ransac([1, 1], [0, 1])


def test_hough_accumulator_peak_matches_horizontal_line() -> None:
    edges = np.zeros((8, 10), dtype=np.uint8)
    edges[3, 2:8] = 255

    accumulator, rhos, thetas = hough_accumulator(edges, [-90, 0, 90])
    peak_rho_index, peak_theta_index = np.unravel_index(
        np.argmax(accumulator), accumulator.shape
    )

    assert accumulator[peak_rho_index, peak_theta_index] == 6
    assert abs(rhos[peak_rho_index]) == 3
    assert abs(thetas[peak_theta_index]) == 90


def test_hough_accumulator_rejects_non_image_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        hough_accumulator([1, 0, 1])

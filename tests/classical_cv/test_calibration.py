import numpy as np

from wave_part_dual.classical_cv import calibrate_camera, create_artificial_calibration_data


def test_artificial_calibration_data_is_repeatable_and_well_shaped() -> None:
    first = create_artificial_calibration_data(num_images=4, seed=3)
    second = create_artificial_calibration_data(num_images=4, seed=3)

    assert len(first.object_points) == 4
    assert first.object_points[0].shape == (54, 3)
    assert first.image_points[0].shape == (54, 1, 2)
    assert np.array_equal(first.image_points[0], second.image_points[0])


def test_calibration_recovers_a_reasonable_camera_from_clean_evidence() -> None:
    data = create_artificial_calibration_data(num_images=16, noise_stddev=0.05)
    result = calibrate_camera(data.object_points, data.image_points, data.image_size)

    assert result["rms"] < 0.2
    assert np.allclose(
        np.diag(result["K"])[:2],
        np.diag(data.camera_matrix)[:2],
        rtol=0.03,
    )
    assert len(result["view_errors"]) == 16

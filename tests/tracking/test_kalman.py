import numpy as np
import pytest

from wave_part_dual.tracking import (
    correct_position,
    filter_positions,
    initialize_constant_velocity,
    predict_constant_velocity,
)


def test_constant_velocity_prediction_moves_position():
    state = initialize_constant_velocity((10, 20), velocity=(3, -2))
    predicted = predict_constant_velocity(state)

    np.testing.assert_allclose(predicted.mean, (13, 18, 3, -2))
    assert predicted.covariance[0, 0] > state.covariance[0, 0]


def test_correction_moves_prediction_toward_measurement():
    predicted = predict_constant_velocity(
        initialize_constant_velocity((0, 0), velocity=(2, 0))
    )
    corrected = correct_position(predicted, (8, 1), measurement_variance=4)

    assert predicted.mean[0] < corrected.mean[0] < 8
    assert 0 < corrected.mean[1] < 1
    assert corrected.covariance[0, 0] < predicted.covariance[0, 0]


def test_filter_predicts_during_missing_measurements():
    measurements = [np.array((2.0, 0.0)), None, None]
    estimates, predictions, uncertainty = filter_positions(
        measurements,
        initial_position=(0, 0),
        initial_velocity=(2, 0),
        process_variance=0.1,
    )

    assert estimates.shape == predictions.shape == uncertainty.shape == (3, 2)
    assert estimates[2, 0] > estimates[1, 0] > estimates[0, 0]
    assert uncertainty[2, 0] > uncertainty[0, 0]


def test_invalid_measurement_variance_is_rejected():
    state = initialize_constant_velocity((0, 0))
    with pytest.raises(ValueError, match="positive"):
        correct_position(state, (1, 1), measurement_variance=0)

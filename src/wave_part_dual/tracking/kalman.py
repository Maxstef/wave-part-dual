"""A small constant-velocity Kalman filter for two-dimensional tracking."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KalmanState:
    """Estimated ``[x, y, vx, vy]`` state and its uncertainty covariance."""

    mean: FloatArray
    covariance: FloatArray


def initialize_constant_velocity(
    position: ArrayLike,
    velocity: ArrayLike = (0.0, 0.0),
    position_variance: float = 25.0,
    velocity_variance: float = 100.0,
) -> KalmanState:
    """Create a two-dimensional constant-velocity filter state."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if position.shape != (2,) or velocity.shape != (2,):
        raise ValueError("position and velocity must each contain two values.")
    if position_variance < 0 or velocity_variance < 0:
        raise ValueError("variances must be non-negative.")
    mean = np.concatenate((position, velocity))
    covariance = np.diag(
        [position_variance, position_variance, velocity_variance, velocity_variance]
    ).astype(float)
    return KalmanState(mean, covariance)


def predict_constant_velocity(
    state: KalmanState,
    dt: float = 1.0,
    process_variance: float = 1.0,
) -> KalmanState:
    """Predict the next state while allowing uncertain acceleration."""
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if process_variance < 0:
        raise ValueError("process_variance must be non-negative.")
    transition = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    acceleration = np.array(
        [[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]], dtype=float
    )
    process_noise = process_variance * acceleration @ acceleration.T
    mean = transition @ state.mean
    covariance = transition @ state.covariance @ transition.T + process_noise
    return KalmanState(mean, covariance)


def correct_position(
    predicted: KalmanState,
    measurement: ArrayLike,
    measurement_variance: float = 16.0,
) -> KalmanState:
    """Correct a predicted state using one noisy ``[x, y]`` measurement."""
    measurement = np.asarray(measurement, dtype=float)
    if measurement.shape != (2,):
        raise ValueError("measurement must contain two values.")
    if measurement_variance <= 0:
        raise ValueError("measurement_variance must be positive.")
    observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    measurement_noise = np.eye(2) * measurement_variance
    innovation = measurement - observation @ predicted.mean
    innovation_covariance = (
        observation @ predicted.covariance @ observation.T + measurement_noise
    )
    gain = np.linalg.solve(
        innovation_covariance, observation @ predicted.covariance
    ).T
    mean = predicted.mean + gain @ innovation
    identity = np.eye(4)
    covariance = (identity - gain @ observation) @ predicted.covariance
    return KalmanState(mean, covariance)


def filter_positions(
    measurements: list[ArrayLike | None],
    initial_position: ArrayLike,
    initial_velocity: ArrayLike = (0.0, 0.0),
    *,
    process_variance: float = 1.0,
    measurement_variance: float = 16.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Filter optional measurements and return estimates, predictions, and uncertainty."""
    state = initialize_constant_velocity(initial_position, initial_velocity)
    estimates, predictions, uncertainties = [], [], []
    for index, measurement in enumerate(measurements):
        predicted = (
            state
            if index == 0
            else predict_constant_velocity(state, process_variance=process_variance)
        )
        predictions.append(predicted.mean[:2].copy())
        state = (
            predicted
            if measurement is None
            else correct_position(predicted, measurement, measurement_variance)
        )
        estimates.append(state.mean[:2].copy())
        uncertainties.append(np.sqrt(np.diag(state.covariance)[:2]))
    return np.asarray(estimates), np.asarray(predictions), np.asarray(uncertainties)

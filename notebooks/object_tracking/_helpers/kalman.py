"""Synthetic scenes and plotting helpers for tracking notebook 04."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from wave_part_dual.tracking import filter_positions


TRUE_COLOR = "#34A853"
MEASUREMENT_COLOR = "#F04E45"
ESTIMATE_COLOR = "#3776AB"
PREDICTION_COLOR = "#F4B400"


def make_motion_measurements(
    frame_count=45,
    noise_sigma=6.0,
    missing_range=(20, 28),
    seed=21,
    sudden_turn=False,
):
    """Create true positions plus noisy and temporarily missing observations."""
    rng = np.random.default_rng(seed)
    time = np.arange(frame_count, dtype=float)
    x = 34 + 7.2 * time
    y = 150 - 0.65 * time + 8 * np.sin(time / 6)
    if sudden_turn:
        turn = 27
        x[turn:] = x[turn] + 2.0 * (time[turn:] - turn)
        y[turn:] = y[turn] - 7.0 * (time[turn:] - turn)
    truth = np.column_stack((x, y))
    observed = truth + rng.normal(0, noise_sigma, truth.shape)
    available = np.ones(frame_count, dtype=bool)
    available[slice(*missing_range)] = False
    measurements = [point if keep else None for point, keep in zip(observed, available)]
    return truth, observed, measurements, available


def _style_trajectory_axis(ax, title):
    ax.set_title(title)
    ax.set_xlabel("x position (px)")
    ax.set_ylabel("y position (px)")
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")


def plot_noisy_measurements(truth, observed, available):
    """Show reality and the imperfect observations received by a tracker."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(*truth.T, color=TRUE_COLOR, linewidth=3, label="true path")
    axes[0].scatter(*observed.T, color=MEASUREMENT_COLOR, s=25, alpha=0.75, label="measurement")
    _style_trajectory_axis(axes[0], "Reality versus noisy measurements")
    axes[0].legend()

    frames = np.arange(len(truth))
    axes[1].plot(frames, truth[:, 0], color=TRUE_COLOR, linewidth=3, label="true x")
    axes[1].scatter(frames[available], observed[available, 0], color=MEASUREMENT_COLOR, s=25, label="measured x")
    axes[1].axvspan(np.flatnonzero(~available)[0], np.flatnonzero(~available)[-1], color="0.85", label="camera has no useful measurement")
    axes[1].set(title="The sensor occasionally goes silent", xlabel="frame", ylabel="x position (px)")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_predict_correct_example(previous, prediction, measurement, correction):
    """Visualize one prediction-and-correction step."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    points = [previous, prediction, measurement, correction]
    colors = ["0.25", PREDICTION_COLOR, MEASUREMENT_COLOR, ESTIMATE_COLOR]
    labels = ["previous estimate", "prediction", "measurement", "corrected estimate"]
    for point, color, label in zip(points, colors, labels):
        ax.scatter(*point, s=100, color=color, label=label, zorder=3)
    ax.annotate("predict", xy=prediction, xytext=previous, arrowprops={"arrowstyle": "->", "color": PREDICTION_COLOR, "lw": 2})
    ax.annotate("correct", xy=correction, xytext=prediction, arrowprops={"arrowstyle": "->", "color": ESTIMATE_COLOR, "lw": 2})
    ax.plot([prediction[0], measurement[0]], [prediction[1], measurement[1]], linestyle=":", color="0.45")
    ax.set(title="Prediction makes a guess; correction negotiates with evidence", xlabel="x position", ylabel="y position")
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_filter_result(truth, observed, available, estimates, predictions):
    """Compare noisy evidence, raw prediction, and corrected estimates."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    ax = axes[0]
    ax.plot(*truth.T, color=TRUE_COLOR, linewidth=3, label="truth")
    ax.scatter(*observed[available].T, color=MEASUREMENT_COLOR, s=22, alpha=0.7, label="available measurements")
    ax.plot(*predictions.T, color=PREDICTION_COLOR, linestyle="--", linewidth=2, label="before correction")
    ax.plot(*estimates.T, color=ESTIMATE_COLOR, linewidth=2.5, label="after correction")
    _style_trajectory_axis(ax, "The filter turns noisy dots into one plausible journey")
    ax.legend(fontsize=8)

    frames = np.arange(len(truth))
    axes[1].plot(frames, truth[:, 0], color=TRUE_COLOR, linewidth=3, label="true x")
    axes[1].scatter(frames[available], observed[available, 0], color=MEASUREMENT_COLOR, s=22, label="measured x")
    axes[1].plot(frames, estimates[:, 0], color=ESTIMATE_COLOR, linewidth=2.5, label="estimated x")
    axes[1].axvspan(np.flatnonzero(~available)[0], np.flatnonzero(~available)[-1], color="0.85", label="missing")
    axes[1].set(title="It keeps moving while nobody reports in", xlabel="frame", ylabel="x position (px)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_measurement_trust(truth, observed, measurements, initial_velocity=(7.2, 0.0), variances=(4.0, 36.0, 225.0)):
    """Compare filters that assign different reliability to measurements."""
    available = np.array([measurement is not None for measurement in measurements])
    fig, axes = plt.subplots(
        len(variances),
        1,
        figsize=(9, 4 * len(variances)),
        sharex=True,
        sharey=True,
    )
    for ax, variance in zip(axes, variances):
        estimates, _, _ = filter_positions(
            measurements,
            truth[0],
            initial_velocity,
            process_variance=1.0,
            measurement_variance=variance,
        )
        ax.plot(*truth.T, color=TRUE_COLOR, linewidth=3, label="truth")
        ax.scatter(*observed[available].T, color=MEASUREMENT_COLOR, s=14, alpha=0.35, label="measurements")
        ax.plot(*estimates.T, color=ESTIMATE_COLOR, linewidth=2.5, label="estimate")
        _style_trajectory_axis(ax, f"Measurement σ = {np.sqrt(variance):.0f} px")
    axes[0].legend(fontsize=8)
    fig.suptitle("Declared measurement noise is a trust dial")
    plt.tight_layout()
    plt.show()


def plot_uncertainty(truth, estimates, uncertainty, available):
    """Show position uncertainty growing without measurements and shrinking afterward."""
    frames = np.arange(len(truth))
    radius = np.linalg.norm(uncertainty, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].plot(frames, radius, color="#7E57C2", linewidth=2.5)
    axes[0].fill_between(frames, 0, radius, color="#7E57C2", alpha=0.18)
    axes[0].axvspan(np.flatnonzero(~available)[0], np.flatnonzero(~available)[-1], color="0.85")
    axes[0].set(title="Silence makes confidence leak away", xlabel="frame", ylabel="position uncertainty (px)")
    axes[0].grid(alpha=0.2)

    chosen = [np.flatnonzero(~available)[0] - 1, np.flatnonzero(~available)[-1], np.flatnonzero(~available)[-1] + 2]
    axes[1].plot(*truth.T, color=TRUE_COLOR, linewidth=2, alpha=0.5)
    for index, color in zip(chosen, ("#4CAF50", "#7E57C2", "#3776AB")):
        axes[1].scatter(*estimates[index], color=color, s=45)
        axes[1].add_patch(Circle(estimates[index], radius[index], fill=False, edgecolor=color, linewidth=2, label=f"frame {index}"))
    _style_trajectory_axis(axes[1], "Uncertainty circles before, during, and after occlusion")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_sudden_turn_comparison(truth, observed, measurements, process_variances=(0.05, 8.0)):
    """Compare rigid and flexible motion assumptions after a sudden turn."""
    fig, axes = plt.subplots(1, len(process_variances), figsize=(12, 4.5), sharex=True, sharey=True)
    for ax, variance in zip(axes, process_variances):
        estimates, _, _ = filter_positions(
            measurements,
            truth[0],
            initial_velocity=(7.2, 0),
            process_variance=variance,
            measurement_variance=36.0,
        )
        ax.plot(*truth.T, color=TRUE_COLOR, linewidth=3, label="truth")
        ax.scatter(*observed.T, color=MEASUREMENT_COLOR, s=15, alpha=0.4, label="measurements")
        ax.plot(*estimates.T, color=ESTIMATE_COLOR, linewidth=2.5, label="estimate")
        _style_trajectory_axis(ax, f"Process variance = {variance:g}")
    axes[0].legend(fontsize=8)
    fig.suptitle("A constant-velocity believer meets a robot with free will")
    plt.tight_layout()
    plt.show()

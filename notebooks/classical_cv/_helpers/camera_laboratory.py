"""Synthetic presentation material for notebook 02."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wave_part_dual.classical_cv import estimate_distance_from_size, radial_distort


def make_camera_scene(height: int = 280, width: int = 420) -> NDArray[np.uint8]:
    """Create a colorful synthetic scene for camera-behavior experiments."""
    image = np.full((height, width, 3), 225, dtype=np.uint8)
    image[:] = np.linspace(55, 220, width, dtype=np.uint8)[None, :, None]

    for x in range(0, width, 30):
        cv2.line(image, (x, 0), (x, height - 1), (110, 110, 110), 1)
    for y in range(0, height, 30):
        cv2.line(image, (0, y), (width - 1, y), (110, 110, 110), 1)

    colors = [(235, 55, 45), (45, 205, 70), (45, 90, 235), (240, 190, 35)]
    for index, color in enumerate(colors):
        x0 = 35 + index * 92
        cv2.rectangle(image, (x0, 45), (x0 + 65, 115), color, -1)

    cv2.circle(image, (105, 205), 42, (245, 130, 25), -1)
    cv2.rectangle(image, (185, 166), (365, 240), (245, 245, 245), -1)
    cv2.putText(
        image,
        "CAMERA?",
        (198, 213),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    return image


def plot_saturation_brightness_grid(saturation_levels, brightness_levels, column_names) -> None:
    """Show how saturation and value alter patches with one fixed red hue."""
    fig, axes = plt.subplots(len(brightness_levels), len(saturation_levels), figsize=(7, 5))
    for row, (value, row_name) in enumerate(brightness_levels):
        for column, (saturation, column_name) in enumerate(zip(saturation_levels, column_names)):
            rgb = np.array([value, value * (1 - saturation), value * (1 - saturation)])
            axes[row, column].imshow(np.ones((80, 120, 3)) * rgb)
            if row == 0:
                axes[row, column].set_title(f"S = {saturation:.2f}\n{column_name}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(row_name, fontsize=11)
    fig.suptitle("Same red hue: saturation changes colorfulness, brightness changes lightness", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_video_bandwidth(bandwidth_mb_s) -> None:
    """Plot uncompressed RGB bandwidth for named video modes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(bandwidth_mb_s)))
    bars = ax.barh(list(bandwidth_mb_s), list(bandwidth_mb_s.values()), color=colors)
    ax.bar_label(bars, fmt="%.0f MB/s", padding=5)
    ax.set_xlabel("Approximate uncompressed RGB bandwidth")
    ax.set_title("More pixels and more frames both charge rent")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()


def plot_depth_projection(pixels, image_size=(640, 480), group_size=4) -> None:
    """Compare projections of the same rectangle at two depths."""
    near, far = pixels[:group_size], pixels[group_size:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(near[:, 0], near[:, 1], s=100, label="Z = 2 m")
    ax.scatter(far[:, 0], far[:, 1], s=100, label="Z = 4 m")
    for group in (near, far):
        closed = np.vstack([group, group[0]])
        ax.plot(closed[:, 0], closed[:, 1])
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)
    ax.set_aspect("equal")
    ax.set(xlabel="u [pixels]", ylabel="v [pixels]", title="Same physical rectangle, different distance")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.show()


def plot_fov_curve(image_width, focal_lengths, highlighted=(250, 520, 1000)) -> None:
    """Plot horizontal field of view against focal length in pixels."""
    horizontal_fov = np.degrees(2 * np.arctan(image_width / (2 * focal_lengths)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(focal_lengths, horizontal_fov, linewidth=3)
    for focal in highlighted:
        angle = np.degrees(2 * np.arctan(image_width / (2 * focal)))
        ax.scatter(focal, angle, s=70)
        ax.annotate(f"f={focal}px\nFoV={angle:.1f}°", (focal, angle), xytext=(8, 8), textcoords="offset points")
    ax.set(xlabel="Focal length fx [pixels]", ylabel="Horizontal field of view [degrees]", title="Wide view and magnification negotiate over focal length")
    ax.grid(alpha=0.25)
    plt.show()


def plot_radial_distortion_grid(
    distortions,
    center=(0.0, 0.0),
    grid_shape=None,
    title="The same ideal grid under radial distortion",
) -> None:
    """Draw an ideal line grid under several radial-distortion models."""
    if grid_shape is None:
        grid_shape = (1, len(distortions))
    line_positions = np.linspace(-0.9, 0.9, 9)
    line_coordinate = np.linspace(-1.0, 1.0, 300)
    fig, axes = plt.subplots(*grid_shape, figsize=(3 * grid_shape[1], 3 * grid_shape[0]))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax, (label, coefficients) in zip(axes, distortions.items()):
        coefficients = np.atleast_1d(coefficients)
        k1, k2, k3 = np.pad(coefficients, (0, 3 - len(coefficients)))[:3]
        for fixed in line_positions:
            x_d, y_d = radial_distort(np.full_like(line_coordinate, fixed), line_coordinate, k1, k2, k3, center)
            ax.plot(x_d, y_d, color="#276FBF", linewidth=1)
            x_d, y_d = radial_distort(line_coordinate, np.full_like(line_coordinate, fixed), k1, k2, k3, center)
            ax.plot(x_d, y_d, color="#D1495B", linewidth=1)
        if center != (0.0, 0.0):
            ax.scatter(*center, color="black", marker="x", s=45, zorder=3)
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.axis("off")
    for ax in axes[len(distortions):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()


def plot_distance_rounding(true_distances, ideal_pixel_heights, absolute_error) -> None:
    """Show shrinking apparent size and distance error from pixel rounding."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 7))
    axes[0].plot(true_distances, ideal_pixel_heights, linewidth=3)
    axes[0].set(xlabel="True distance [m]", ylabel="Apparent object height [pixels]", title="Objects shrink with distance")
    axes[1].plot(true_distances, absolute_error, color="#D1495B", linewidth=3)
    axes[1].set(xlabel="True distance [m]", ylabel="Absolute distance error [m]", title="One-pixel rounding matters more far away")
    for ax in axes:
        ax.grid(alpha=0.25)
    plt.tight_layout()


def plot_distance_sensitivity(
    true_distances,
    measured_heights_px,
    true_focal_length_px,
    true_object_height_m,
    assumed_focal_lengths=(350.0, 500.0, 700.0),
    assumed_object_heights=(0.40, 0.60, 0.80),
) -> None:
    """Compare size-distance sensitivity while changing one assumption at a time."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for focal_length in assumed_focal_lengths:
        estimates = estimate_distance_from_size(focal_length, true_object_height_m, measured_heights_px)
        axes[0, 0].plot(true_distances, estimates, label=f"assumed f = {focal_length:.0f} px")
    axes[0, 0].plot(true_distances, true_distances, "k--", label="perfect estimate")
    axes[0, 0].set_title("Frozen pixels; change assumed focal length")
    axes[0, 0].set_ylabel("Estimated distance [m]")
    axes[0, 0].legend()

    for object_height in assumed_object_heights:
        estimates = estimate_distance_from_size(true_focal_length_px, object_height, measured_heights_px)
        axes[0, 1].plot(true_distances, estimates, label=f"assumed H = {object_height:.2f} m")
    axes[0, 1].plot(true_distances, true_distances, "k--", label="perfect estimate")
    axes[0, 1].set_title("Frozen pixels; change assumed object height")
    axes[0, 1].legend()

    plus_one = estimate_distance_from_size(true_focal_length_px, true_object_height_m, measured_heights_px + 1)
    minus_one = estimate_distance_from_size(true_focal_length_px, true_object_height_m, np.maximum(1, measured_heights_px - 1))
    axes[1, 0].plot(true_distances, np.abs(plus_one - true_distances), label="measured height +1 px (looks closer)")
    axes[1, 0].plot(true_distances, np.abs(minus_one - true_distances), label="measured height -1 px (looks farther)")
    axes[1, 0].set_title("Absolute distance error from one wrong pixel")
    axes[1, 0].set_ylabel("Absolute distance error [m]")
    axes[1, 0].legend()

    axes[1, 1].plot(true_distances, 100 / measured_heights_px, color="#D1495B", linewidth=2.5)
    axes[1, 1].set_title("One pixel becomes a larger fraction of a small object")
    axes[1, 1].set_ylabel("1 px / measured height [%]")
    for ax in axes.flat:
        ax.set_xlabel("True distance [m]")
        ax.grid(alpha=0.25)
    fig.suptitle("Distance estimation sensitivity: keep everything else frozen", fontsize=14)
    plt.tight_layout()
    plt.show()

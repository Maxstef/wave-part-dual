"""Synthetic scenes and presentation helpers for notebook 05."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wave_part_dual.classical_cv import fit_line_ransac, hough_accumulator


def make_line_scene(
    height: int = 280,
    width: int = 420,
    noise_sigma: float = 28,
    seed: int = 12,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Create a clean road scene and a reproducibly noisy observation."""
    clean = np.full((height, width), 30, dtype=np.uint8)
    cv2.line(clean, (70, height - 15), (170, 35), 225, 7)
    cv2.line(clean, (350, height - 15), (250, 35), 225, 7)

    def draw_tree(image, x, ground_y, scale=1.0):
        trunk_half_width = round(7 * scale)
        trunk_height = round(48 * scale)
        cv2.rectangle(
            image,
            (x - trunk_half_width, ground_y - trunk_height),
            (x + trunk_half_width, ground_y),
            135,
            -1,
        )
        crown_y = ground_y - trunk_height
        for dx, dy, radius in ((0, -20, 27), (-20, -5, 22), (20, -5, 22)):
            cv2.circle(
                image,
                (x + round(dx * scale), crown_y + round(dy * scale)),
                round(radius * scale),
                185,
                -1,
            )

    draw_tree(clean, x=82, ground_y=145, scale=0.85)
    draw_tree(clean, x=330, ground_y=205, scale=1.15)
    noise = np.random.default_rng(seed).normal(0, noise_sigma, clean.shape)
    noisy = np.clip(clean.astype(float) + noise, 0, 255).astype(np.uint8)
    return clean, noisy


def make_line_points(
    points_count: int = 32,
    noise_sigma: float = 0.7,
    seed: int = 21,
    outlier_indices=(23, 26, 28, 30),
    outlier_offsets=(12, 15, 18, 21),
):
    """Create noisy line observations and move selected points by fixed offsets."""
    rng = np.random.default_rng(seed)
    x_points = np.linspace(0, 10, points_count)
    true_y = 2.0 * x_points + 3.0
    measured_y = true_y + rng.normal(0, noise_sigma, points_count)
    indices = np.asarray(outlier_indices, dtype=int)
    offsets = np.asarray(outlier_offsets, dtype=float)
    if indices.size != offsets.size:
        raise ValueError("Each outlier index needs exactly one offset.")
    if np.any((indices < 0) | (indices >= points_count)):
        raise ValueError("Outlier indices must fit inside the generated point set.")
    observed_y = measured_y.copy()
    observed_y[indices] += offsets
    return x_points, true_y, measured_y, observed_y, indices


def _show_image_grid(images, shape, figsize):
    fig, axes = plt.subplots(*shape, figsize=figsize)
    axes = np.asarray(axes).ravel()
    for ax, (title, image) in zip(axes, images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_line_scene(clean, noisy):
    """Show the clean synthetic scene beside its noisy observation."""
    _show_image_grid(
        [("What actually exists", clean), ("What the camera has to endure", noisy)],
        (1, 2),
        (7, 3),
    )


def plot_smoothing_comparison(clean, noisy, kernel=(9, 9), gaussian_sigma=2.0):
    """Compare box and Gaussian smoothing on one noisy image."""
    box = cv2.blur(noisy, kernel)
    gaussian = cv2.GaussianBlur(noisy, kernel, sigmaX=gaussian_sigma)
    _show_image_grid(
        [
            ("Clean truth", clean),
            ("Noisy", noisy),
            (f"{kernel[0]}×{kernel[1]} box average", box),
            (f"{kernel[0]}×{kernel[1]} Gaussian", gaussian),
        ],
        (2, 2),
        (7, 5),
    )
    return {"Box average": box, "Gaussian": gaussian}


def plot_noise_smoothing_practice(clean, noisy, noise_sigma, kernel, gaussian_sigma=2.0):
    """Show a frozen high-noise scene before and after two smoothers."""
    smoothed = {
        "No smoothing": noisy,
        "Box average": cv2.blur(noisy, kernel),
        "Gaussian": cv2.GaussianBlur(noisy, kernel, sigmaX=gaussian_sigma),
    }
    _show_image_grid(
        [
            ("Clean truth", clean),
            (f"No smoothing, noise σ = {noise_sigma}", noisy),
            (f"Box average, kernel {kernel}", smoothed["Box average"]),
            (f"Gaussian, kernel {kernel}, σ = {gaussian_sigma:g}", smoothed["Gaussian"]),
        ],
        (2, 2),
        (7, 5),
    )
    return smoothed


def plot_smoothing_kernel_sweep(noisy, kernel_sizes, gaussian_sigma=2.0):
    """Compare box and Gaussian smoothing across kernel sizes."""
    fig, axes = plt.subplots(2, len(kernel_sizes), figsize=(9, 6), squeeze=False)
    for column, kernel in enumerate(kernel_sizes):
        versions = (
            ("Box average", cv2.blur(noisy, kernel)),
            ("Gaussian", cv2.GaussianBlur(noisy, kernel, sigmaX=gaussian_sigma)),
        )
        for row, (name, image) in enumerate(versions):
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row, column].set_title(f"{name}, kernel {kernel}")
            axes[row, column].axis("off")
    plt.tight_layout()
    plt.show()


def plot_smoothing_combinations(noisy, kernel=(7, 7), gaussian_sigma=2.0):
    """Compare repeated and mixed box/Gaussian smoothing."""
    box_once = cv2.blur(noisy, kernel)
    gaussian_once = cv2.GaussianBlur(noisy, kernel, sigmaX=gaussian_sigma)
    box_then_gaussian = cv2.GaussianBlur(box_once, kernel, sigmaX=gaussian_sigma)
    gaussian_then_box = cv2.blur(gaussian_once, kernel)
    versions = [
        ("No smoothing", noisy),
        ("Box once", box_once),
        ("Gaussian once", gaussian_once),
        ("Box → Gaussian", box_then_gaussian),
        ("Box twice", cv2.blur(box_once, kernel)),
        ("Gaussian twice", cv2.GaussianBlur(gaussian_once, kernel, sigmaX=gaussian_sigma)),
        ("Gaussian → Box", gaussian_then_box),
        (
            "Gaussian three times",
            cv2.GaussianBlur(
                cv2.GaussianBlur(gaussian_once, kernel, sigmaX=gaussian_sigma),
                kernel,
                sigmaX=gaussian_sigma,
            ),
        ),
    ]
    difference = cv2.absdiff(box_then_gaussian, gaussian_then_box)
    fig, axes = plt.subplots(3, 3, figsize=(9, 7))
    for ax, (name, image) in zip(axes.ravel(), versions):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
    axes.ravel()[8].imshow(difference, cmap="magma", vmin=0)
    axes.ravel()[8].set_title(f"Mixed-order difference\nmax = {difference.max()} gray levels")
    axes.ravel()[8].axis("off")
    plt.tight_layout()
    plt.show()


def plot_edge_maps(smoothed, sobel_x, sobel_y, gradient_magnitude, edge_map):
    """Show directional gradients, gradient magnitude, and Canny decisions."""
    _show_image_grid(
        [
            ("Sobel X", np.abs(sobel_x)),
            ("Sobel Y", np.abs(sobel_y)),
            ("Gradient strength", gradient_magnitude),
            ("Canny: thin decisions", edge_map),
        ],
        (2, 2),
        (7, 5),
    )


def plot_canny_threshold_sweep(smoothed, threshold_pairs):
    """Compare Canny results while keeping its input image fixed."""
    results = [cv2.Canny(smoothed, lower, upper) for lower, upper in threshold_pairs]
    side = int(np.ceil(np.sqrt(len(results) + 1)))
    fig, axes = plt.subplots(side, side, figsize=(10, 8))
    axes = axes.ravel()
    axes[0].imshow(smoothed, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Frozen input: Gaussian-smoothed noise")
    for ax, edges, (lower, upper) in zip(axes[1:], results, threshold_pairs):
        count = np.count_nonzero(edges)
        ax.imshow(edges, cmap="gray")
        ax.set_title(
            f"Canny ({lower}, {upper})\n{count:,} pixels promoted "
            f"({100 * count / edges.size:.2f}%)"
        )
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_least_squares_outliers(x, true_y, measured_y, observed_y, outlier_indices):
    """Contrast least squares before and after coordinated outliers."""
    fits = (np.polyfit(x, measured_y, 1), np.polyfit(x, observed_y, 1))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, values, (slope, intercept), title in zip(
        axes,
        (measured_y, observed_y),
        fits,
        ("Mostly honest points", "Four loud outliers coordinate their attack"),
    ):
        ax.scatter(x, values, color="#276FBF", label="measurements")
        ax.plot(x, true_y, "k--", label="true line")
        ax.plot(x, slope * x + intercept, color="#D1495B", linewidth=2.5,
                label=f"least squares: y={slope:.2f}x+{intercept:.2f}")
        ax.set(xlabel="x", ylabel="y", title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[1].scatter(x[outlier_indices], observed_y[outlier_indices],
                    facecolors="none", edgecolors="#F39C12", s=130, linewidth=2)
    plt.tight_layout()
    plt.show()
    return tuple(float(value) for fit in fits for value in fit)


def plot_ransac_result(x, true_y, observed_y, residual_threshold=1.3, trials=250, seed=5):
    """Compare least squares with one RANSAC fit and print both estimates."""
    ls_slope, ls_intercept = np.polyfit(x, observed_y, 1)
    slope, intercept, inliers = fit_line_ransac(
        x, observed_y, residual_threshold=residual_threshold, trials=trials, seed=seed
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x[inliers], observed_y[inliers], color="#276FBF", label="RANSAC inliers")
    ax.scatter(x[~inliers], observed_y[~inliers], facecolors="none",
               edgecolors="#F39C12", s=100, label="rejected points")
    ax.plot(x, true_y, "k--", label="true line")
    ax.plot(x, ls_slope * x + ls_intercept, color="#D1495B", linewidth=2,
            label="least squares: trusts everyone")
    ax.plot(x, slope * x + intercept, color="#2E8B57", linewidth=3,
            label="RANSAC: largest consensus")
    ax.set(xlabel="x", ylabel="y", title="The outliers may speak, but they no longer chair the meeting")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()
    print(f"Least squares: slope={ls_slope:.2f}, intercept={ls_intercept:.2f}")
    print(f"RANSAC:       slope={slope:.2f}, intercept={intercept:.2f}")
    print(f"RANSAC kept {inliers.sum()} of {len(x)} points as inliers")


def plot_ransac_parameter_grid(
    x, true_y, observed_y, residual_thresholds, trial_counts, seed=5
):
    """Compare RANSAC tolerance and trial-count combinations."""
    ls_slope, ls_intercept = np.polyfit(x, observed_y, 1)
    results = []
    for threshold in residual_thresholds:
        for trials in trial_counts:
            slope, intercept, inliers = fit_line_ransac(
                x, observed_y, residual_threshold=threshold, trials=trials, seed=seed
            )
            results.append((threshold, trials, slope, intercept, inliers))

    fig, axes = plt.subplots(
        len(residual_thresholds), len(trial_counts), figsize=(12, 9),
        sharex=True, sharey=True, squeeze=False,
    )
    for ax, (threshold, trials, slope, intercept, inliers) in zip(axes.ravel(), results):
        ax.scatter(x[inliers], observed_y[inliers], color="#276FBF", label="RANSAC inliers")
        ax.scatter(x[~inliers], observed_y[~inliers], facecolors="none",
                   edgecolors="#F39C12", s=100, label="rejected points")
        ax.plot(x, true_y, "k--", label="true line")
        ax.plot(x, ls_slope * x + ls_intercept, color="#D1495B", linewidth=2,
                label="least squares: trusts everyone")
        ax.plot(x, slope * x + intercept, color="#2E8B57", linewidth=3,
                label="RANSAC: largest consensus")
        ax.set(xlabel="x", ylabel="y",
               title=f"Threshold = {threshold}, trials = {trials}\n"
                     f"{inliers.sum()}/{len(x)} inliers, slope = {slope:.2f}")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Same witnesses, different tolerance and patience", fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"Least squares: slope={ls_slope:.2f}, intercept={ls_intercept:.2f}")
    for threshold, trials, slope, intercept, inliers in results:
        print(f"RANSAC threshold={threshold}, trials={trials}: slope={slope:.2f}, "
              f"intercept={intercept:.2f}, inliers={inliers.sum()}/{len(x)}")


def plot_hough_voting_curves(line_x, line_y, theta_degrees):
    """Show collinear pixels and their intersecting rho-theta curves."""
    theta_radians = np.deg2rad(theta_degrees)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].scatter(line_x, line_y, s=70, color="#D1495B")
    axes[0].plot(line_x, line_y, color="#276FBF", alpha=0.5)
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")
    axes[0].set(xlabel="image x", ylabel="image y", title="Four edge pixels sharing one line")
    for x, y in zip(line_x, line_y):
        rho_curve = x * np.cos(theta_radians) + y * np.sin(theta_radians)
        axes[1].plot(theta_degrees, rho_curve, label=f"pixel ({x:.0f}, {y:.0f})")
    axes[1].set(xlabel="θ [degrees]", ylabel="ρ [pixels]",
                title="Their voting curves meet at one line")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def _detect_hough_segments(edge_image, background, threshold, min_length, max_gap):
    segments = cv2.HoughLinesP(
        edge_image, rho=1, theta=np.pi / 180, threshold=threshold,
        minLineLength=min_length, maxLineGap=max_gap,
    )
    overlay = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    if segments is not None:
        for x1, y1, x2, y2 in segments[:, 0]:
            cv2.line(overlay, (x1, y1), (x2, y2), (220, 55, 55), 3)
    return segments, overlay


def plot_hough_detection(edge_map, background, threshold=55, min_length=70, max_gap=18):
    """Show Canny voters, their accumulator, and OpenCV line segments."""
    accumulator, rhos, thetas = hough_accumulator(edge_map)
    segments, overlay = _detect_hough_segments(
        edge_map, background, threshold, min_length, max_gap
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].imshow(edge_map, cmap="gray")
    axes[0].set_title("Canny voters")
    axes[1].imshow(accumulator, cmap="magma", aspect="auto",
                   extent=(thetas[0], thetas[-1], rhos[-1], rhos[0]))
    axes[1].set(xlabel="θ [degrees]", ylabel="ρ [pixels]",
                title="Vote accumulator: bright means agreement")
    axes[2].imshow(overlay)
    axes[2].set_title(f"OpenCV found {0 if segments is None else len(segments)} line segments")
    axes[0].axis("off")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()
    return segments


def plot_numbered_hough_segments(background, segments):
    """Draw every returned segment with a separate color and number."""
    overlay = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    colors = (plt.get_cmap("tab10")(np.arange(10))[:, :3] * 255).astype(np.uint8)
    if segments is not None:
        for number, (x1, y1, x2, y2) in enumerate(segments[:, 0], start=1):
            color = tuple(int(channel) for channel in colors[(number - 1) % 10])
            midpoint = ((x1 + x2) // 2 + 4, (y1 + y2) // 2 - 4)
            cv2.line(overlay, (x1, y1), (x2, y2), color, 4)
            cv2.putText(overlay, str(number), midpoint, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.putText(overlay, str(number), midpoint, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(overlay)
    count = 0 if segments is None else len(segments)
    ax.set_title(f"{count} Hough segments, {count} separate opinions")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_hough_parameter_grid(edge_image, background, configurations):
    """Compare probabilistic-Hough parameter combinations."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, threshold, min_length, max_gap) in zip(axes.ravel(), configurations):
        segments, overlay = _detect_hough_segments(
            edge_image, background, threshold, min_length, max_gap
        )
        count = 0 if segments is None else len(segments)
        ax.imshow(overlay)
        ax.set_title(f"{name}: {count} segments\nthreshold={threshold}, "
                     f"min length={min_length}, max gap={max_gap}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

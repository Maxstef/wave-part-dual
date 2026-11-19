"""Synthetic scenes and presentation helpers for notebook 04."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wave_part_dual.classical_cv import sad_cost_curve


def make_stereo_scene(
    height: int = 240,
    width: int = 420,
    seed: int = 8,
    objects=None,
    background_shift: int = 4,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.float32]]:
    """Create a rectified stereo pair with known piecewise-constant disparity."""
    rng = np.random.default_rng(seed)
    texture = rng.normal(125, 34, (height, width)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.0)
    texture += np.linspace(-28, 28, width, dtype=np.float32)[None, :]
    background = np.clip(texture, 0, 255).astype(np.uint8)

    left = background.copy()
    right = np.roll(background, -background_shift, axis=1)
    true_disparity = np.full((height, width), background_shift, dtype=np.float32)

    if objects is None:
        objects = [
            # name, center, size/radius, intensity, disparity, shape
            ("far crate", (285, 88), (92, 70), 185, 9, "rectangle"),
            ("near moon", (145, 155), 50, 225, 24, "circle"),
        ]

    for _name, center, size, intensity, disparity, shape in objects:
        left_mask = np.zeros_like(left)
        right_mask = np.zeros_like(right)
        right_center = (center[0] - disparity, center[1])

        if shape == "rectangle":
            half_width, half_height = size[0] // 2, size[1] // 2
            cv2.rectangle(
                left_mask,
                (center[0] - half_width, center[1] - half_height),
                (center[0] + half_width, center[1] + half_height),
                255,
                -1,
            )
            cv2.rectangle(
                right_mask,
                (right_center[0] - half_width, right_center[1] - half_height),
                (right_center[0] + half_width, right_center[1] + half_height),
                255,
                -1,
            )
        else:
            cv2.circle(left_mask, center, size, 255, -1)
            cv2.circle(right_mask, right_center, size, 255, -1)

        object_texture = np.clip(
            background.astype(float) * 0.45 + intensity * 0.55,
            0,
            255,
        ).astype(np.uint8)
        right_texture = np.roll(object_texture, -disparity, axis=1)
        left[left_mask > 0] = object_texture[left_mask > 0]
        right[right_mask > 0] = right_texture[right_mask > 0]
        true_disparity[left_mask > 0] = disparity

    # Details give matching something distinctive to hold onto.
    cv2.line(left, (105, 155), (185, 155), 45, 3)
    cv2.line(right, (81, 155), (161, 155), 45, 3)
    cv2.rectangle(left, (262, 70), (307, 106), 75, 3)
    cv2.rectangle(right, (253, 70), (298, 106), 75, 3)
    return left, right, true_disparity


def match_and_plot_patch(
    left_image,
    right_image,
    match_point,
    candidate_disparities,
    block_size=11,
    point_name="selected point",
):
    """Find and visualize the lowest-SAD horizontal match for one point."""
    sad_costs = sad_cost_curve(
        left_image,
        right_image,
        match_point,
        candidate_disparities,
        block_size,
    )
    best_disparity = candidate_disparities[np.argmin(sad_costs)]
    right_match_x = match_point[0] - best_disparity

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(np.hstack([left_image, right_image]), cmap="gray")
    axes[0].scatter(
        *match_point,
        s=90,
        facecolors="none",
        edgecolors="#D1495B",
        linewidth=2,
    )
    axes[0].scatter(
        left_image.shape[1] + right_match_x,
        match_point[1],
        s=90,
        facecolors="none",
        edgecolors="#276FBF",
        linewidth=2,
    )
    axes[0].plot(
        [match_point[0], left_image.shape[1] + right_match_x],
        [match_point[1], match_point[1]],
        linestyle="--",
        color="#2E8B57",
    )
    axes[0].set_title(f"{point_name}: best disparity = {best_disparity} px")
    axes[0].axis("off")

    axes[1].plot(candidate_disparities, sad_costs, "o-", markersize=3)
    axes[1].axvline(
        best_disparity,
        color="#D1495B",
        linestyle="--",
        label=f"minimum = {best_disparity} px",
    )
    axes[1].set(
        xlabel="candidate disparity [pixels]",
        ylabel="patch SAD cost",
        title="The winning shift is the least bad explanation",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    return {"best_disparity": best_disparity, "sad_costs": sad_costs}


def plot_disparity_comparison(
    left_image,
    true_disparity,
    estimated_disparity,
    valid_disparity,
    max_disparity=32,
):
    """Compare the left image, known disparity, and a masked estimate."""
    disparity_cmap = plt.colormaps["magma"].with_extremes(bad="#777777")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    axes[0].imshow(left_image, cmap="gray")
    axes[0].set_title("Left image")
    axes[1].imshow(true_disparity, cmap="magma", vmin=0, vmax=max_disparity)
    axes[1].set_title("True disparity")
    estimate_plot = axes[2].imshow(
        np.where(valid_disparity, estimated_disparity, np.nan),
        cmap=disparity_cmap,
        vmin=0,
        vmax=max_disparity,
    )
    axes[2].set_title("StereoBM estimate; gray = invalid")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(
        estimate_plot,
        ax=axes[1:],
        fraction=0.025,
        label="disparity [pixels]",
    )
    plt.show()


def plot_depth_geometry(
    true_depth,
    estimated_depth,
    focal_length_px,
    baselines=(0.06, 0.12, 0.24),
):
    """Show depth maps and the inverse disparity-to-depth relationship."""
    disparity_axis = np.linspace(1, 40, 300)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(true_depth, cmap="viridis", vmin=0.5, vmax=6)
    axes[0].set_title("True depth map")
    axes[1].imshow(estimated_depth, cmap="viridis", vmin=0.5, vmax=6)
    axes[1].set_title("Depth from StereoBM")
    for baseline in baselines:
        axes[2].plot(
            disparity_axis,
            focal_length_px * baseline / disparity_axis,
            label=f"baseline = {baseline:.2f} m",
        )
    axes[2].set(
        xlabel="disparity [pixels]",
        ylabel="depth [m]",
        title="Depth and disparity are not polite enough to be linear",
    )
    axes[2].set_ylim(0, 12)
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    for ax in axes[:2]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_point_cloud(points_3d, intensity_image, sample_step=4):
    """Plot a subsampled camera-frame point cloud using image intensities."""
    height, width, _ = points_3d.shape
    v, u = np.mgrid[0:height, 0:width]
    x, y, z = np.moveaxis(points_3d, -1, 0)
    sample = (u % sample_step == 0) & (v % sample_step == 0)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x[sample],
        z[sample],
        -y[sample],
        c=intensity_image[sample],
        cmap="gray",
        s=2,
        alpha=0.65,
    )
    ax.set(
        xlabel="X: sideways [m]",
        ylabel="Z: forward [m]",
        zlabel="Y: up [m]",
        title="Point cloud: the image has escaped into 3D",
    )
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 6)
    ax.view_init(elev=22, azim=-70)
    plt.tight_layout()
    plt.show()


def plot_active_ir_experiment(
    projected_disparity=12,
    match_point=(145, 75),
    block_size=15,
    image_shape=(150, 280),
    num_disparities=32,
    candidate_disparities=range(25),
    dot_count=320,
    seed=4,
):
    """Compare stereo matching on a plain surface with projected IR texture."""
    rng = np.random.default_rng(seed)
    height, width = image_shape
    candidate_disparities = np.asarray(candidate_disparities)

    plain_left = np.full((height, width), 145, dtype=np.uint8)
    plain_right = np.roll(plain_left, -projected_disparity, axis=1)

    dotted_left = plain_left.copy()
    for x, y in rng.integers([5, 5], [width - 5, height - 5], size=(dot_count, 2)):
        cv2.circle(
            dotted_left,
            (int(x), int(y)),
            1,
            int(rng.integers(45, 235)),
            -1,
        )
    dotted_right = np.roll(dotted_left, -projected_disparity, axis=1)

    plain_costs = sad_cost_curve(
        plain_left,
        plain_right,
        match_point,
        candidate_disparities,
        block_size=block_size,
    )
    dotted_costs = sad_cost_curve(
        dotted_left,
        dotted_right,
        match_point,
        candidate_disparities,
        block_size=block_size,
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    rows = (
        (plain_left, plain_right, plain_costs, "Plain surface"),
        (dotted_left, dotted_right, dotted_costs, "Projected IR dots"),
    )
    for row, (left_ir, right_ir, costs, label) in enumerate(rows):
        axes[row, 0].imshow(left_ir, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"{label}: left IR camera")
        axes[row, 1].imshow(right_ir, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"{label}: right IR camera")

        patch_radius = block_size // 2
        right_match_point = (match_point[0] - projected_disparity, match_point[1])
        for ax, centre, color in (
            (axes[row, 0], match_point, "#D1495B"),
            (axes[row, 1], right_match_point, "#276FBF"),
        ):
            ax.add_patch(
                plt.Rectangle(
                    (centre[0] - patch_radius, centre[1] - patch_radius),
                    block_size,
                    block_size,
                    fill=False,
                    edgecolor=color,
                    linewidth=1,
                )
            )
            ax.scatter(*centre, s=10, color=color)

        axes[row, 2].plot(candidate_disparities, costs, "o-", markersize=3)
        axes[row, 2].axvline(
            projected_disparity,
            color="#D1495B",
            linestyle="--",
            label=f"true disparity = {projected_disparity} px",
        )
        axes[row, 2].set(
            xlabel="candidate disparity [pixels]",
            ylabel="SAD cost",
        )
        axes[row, 2].grid(alpha=0.25)
        axes[row, 2].legend()
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    axes[0, 2].set_title("No unique winner: every shift looks the same")
    axes[1, 2].set_title("Dots create a clear matching minimum")
    plt.show()

    matcher = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size,
    )
    matcher.setTextureThreshold(5)
    matcher.setUniquenessRatio(8)
    plain_disparity = matcher.compute(plain_left, plain_right).astype(np.float32) / 16.0
    dotted_disparity = matcher.compute(dotted_left, dotted_right).astype(np.float32) / 16.0

    disparity_cmap = plt.colormaps["magma"].with_extremes(bad="#777777")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    titles = (
        "Plain surface: mostly invalid",
        "IR dots: disparity becomes recoverable",
    )
    for ax, disparity, title in zip(axes, (plain_disparity, dotted_disparity), titles):
        shown = ax.imshow(
            np.where(disparity > 0, disparity, np.nan),
            cmap=disparity_cmap,
            vmin=0,
            vmax=20,
        )
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(shown, ax=axes, fraction=0.035, label="disparity [pixels]")
    plt.show()

    radius = block_size // 2
    safe_area = np.zeros_like(plain_left, dtype=bool)
    safe_area[radius:-radius, num_disparities + radius:-radius] = True
    plain_coverage = ((plain_disparity > 0) & safe_area).sum() / safe_area.sum()
    dotted_coverage = ((dotted_disparity > 0) & safe_area).sum() / safe_area.sum()
    print(f"Trusted disparity on plain surface: {plain_coverage:.1%}")
    print(f"Trusted disparity with projected dots: {dotted_coverage:.1%}")
    return {
        "plain_coverage": plain_coverage,
        "dotted_coverage": dotted_coverage,
    }


def plot_orb_parameter_sweep(
    image,
    nfeatures_values=(30, 100, 300),
    fast_threshold_values=(5, 20, 50),
    fixed_fast_threshold=20,
    fixed_nfeatures=300,
):
    """Visualize how ORB capacity and selectivity change detected keypoints."""
    columns = max(len(nfeatures_values), len(fast_threshold_values))
    fig, axes = plt.subplots(2, columns, figsize=(4.3 * columns, 7))
    axes = np.asarray(axes, dtype=object).reshape(2, columns)

    for ax, nfeatures in zip(axes[0], nfeatures_values):
        detector = cv2.ORB_create(
            nfeatures=nfeatures,
            fastThreshold=fixed_fast_threshold,
        )
        keypoints = detector.detect(image, None)
        view = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(255, 80, 80),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        )
        ax.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        ax.set_title(f"nfeatures={nfeatures} → {len(keypoints)} retained")
        ax.axis("off")

    for ax, fast_threshold in zip(axes[1], fast_threshold_values):
        detector = cv2.ORB_create(
            nfeatures=fixed_nfeatures,
            fastThreshold=fast_threshold,
        )
        keypoints = detector.detect(image, None)
        view = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(80, 180, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        )
        ax.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        ax.set_title(f"fastThreshold={fast_threshold} → {len(keypoints)} found")
        ax.axis("off")

    for row, used in enumerate((len(nfeatures_values), len(fast_threshold_values))):
        for ax in axes[row, used:]:
            ax.axis("off")
    fig.suptitle("ORB parameter experiment: capacity versus selectivity", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_orb_matches(left_image, right_image, nfeatures=100, max_matches=35):
    """Detect, match, and visualize mutually consistent ORB descriptors."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    left_keypoints, left_descriptors = orb.detectAndCompute(left_image, None)
    right_keypoints, right_descriptors = orb.detectAndCompute(right_image, None)
    if left_descriptors is None or right_descriptors is None:
        raise ValueError("ORB found no descriptors to match")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(
        matcher.match(left_descriptors, right_descriptors),
        key=lambda match: match.distance,
    )
    shown_matches = matches[:max_matches]
    match_image = cv2.drawMatches(
        left_image,
        left_keypoints,
        right_image,
        right_keypoints,
        shown_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(12, 5))
    plt.imshow(cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB))
    plt.title("ORB keypoints + binary descriptors: sparse agreements between two witnesses")
    plt.axis("off")
    plt.show()

    vertical_errors = [
        abs(left_keypoints[match.queryIdx].pt[1] - right_keypoints[match.trainIdx].pt[1])
        for match in shown_matches
    ]
    median_vertical_error = np.median(vertical_errors) if vertical_errors else np.nan
    print(f"Detected keypoints: left={len(left_keypoints)}, right={len(right_keypoints)}")
    print(f"Median vertical mismatch among shown matches: {median_vertical_error:.2f} px")
    return {
        "left_keypoints": len(left_keypoints),
        "right_keypoints": len(right_keypoints),
        "shown_matches": len(shown_matches),
        "median_vertical_error": median_vertical_error,
    }

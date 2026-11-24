"""Presentation helpers for optical-flow and point-tracking notebook 02."""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from wave_part_dual.tracking import (
    detect_corners,
    track_points_forward_backward,
)


def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _robot_roi(mask, padding=5):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1)
    )
    return cv2.dilate(mask, kernel)


def plot_motion_vector_story(frames, centers, previous_index=11, current_index=12):
    """Introduce one known image displacement using the robot center."""
    previous_center = np.asarray(centers[previous_index])
    current_center = np.asarray(centers[current_index])
    displacement = current_center - previous_center

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(frames[previous_index])
    axes[0].scatter(*previous_center, color="#F04E45", s=45)
    axes[0].set_title(f"Frame {previous_index}: start")
    axes[1].imshow(frames[current_index])
    axes[1].scatter(*current_center, color="#2E8B57", s=45)
    axes[1].set_title(f"Frame {current_index}: arrival")
    axes[2].imshow(frames[current_index])
    axes[2].annotate(
        "",
        xy=current_center,
        xytext=previous_center,
        arrowprops={"arrowstyle": "->", "color": "#F04E45", "lw": 3},
    )
    axes[2].scatter(*previous_center, color="#F04E45", s=35, label="previous")
    axes[2].scatter(*current_center, color="#2E8B57", s=35, label="current")
    axes[2].set_title(f"Flow vector = ({displacement[0]}, {displacement[1]}) px")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_why_corners(blockSize=7, ksize=3):
    """Compare local evidence in flat, edge, and corner image patches."""
    patches = []
    flat = np.full((90, 90), 60, dtype=np.uint8)
    edge = flat.copy()
    edge[:, 45:] = 220
    corner = flat.copy()
    corner[45:, 45:] = 220
    for name, patch in (("Flat area", flat), ("Straight edge", edge), ("Corner", corner)):
        response = cv2.cornerMinEigenVal(patch, blockSize=blockSize, ksize=ksize)
        patches.append((name, patch, response))

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2))
    explanations = (
        "Every direction looks alike",
        "Position along the edge is ambiguous",
        "Two directions pin down the address",
    )
    for column, ((name, patch, response), explanation) in enumerate(
        zip(patches, explanations)
    ):
        axes[0, column].imshow(patch, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(name)
        axes[1, column].imshow(response, cmap="magma")
        axes[1, column].set_title(explanation, fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        "A tracker prefers patches with a recognizable address\n"
        f"blockSize={blockSize}, ksize={ksize}"
    )
    plt.tight_layout()
    plt.show()


def plot_corner_xy_gradients(ksize=7):
    """Show separate x- and y-direction brightness changes in demo patches."""
    flat = np.full((90, 90), 60, dtype=np.uint8)
    edge = flat.copy()
    edge[:, 45:] = 220
    corner = flat.copy()
    corner[45:, 45:] = 220
    patches = (("Flat area", flat), ("Straight edge", edge), ("Corner", corner))

    gradients = []
    for _, patch in patches:
        change_x = np.abs(cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=ksize))
        change_y = np.abs(cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=ksize))
        gradients.append((change_x, change_y))
    maximum = max(gradient.max() for pair in gradients for gradient in pair)

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2))
    for column, ((name, _), (change_x, change_y)) in enumerate(
        zip(patches, gradients)
    ):
        axes[0, column].imshow(change_x, cmap="magma", vmin=0, vmax=maximum)
        axes[0, column].set_title(f"{name}\nchange along x")
        axes[1, column].imshow(change_y, cmap="magma", vmin=0, vmax=maximum)
        axes[1, column].set_title("change along y")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        "Before the corner score: brightness changes measured separately\n"
        f"Sobel ksize={ksize}"
    )
    plt.tight_layout()
    plt.show()


def plot_corner_parameter_sweep(frame, roi_mask):
    """Show how Shi-Tomasi controls alter the selected points."""
    gray = _gray(frame)
    configurations = (
        (20, 0.20, 8, "Few selective points"),
        (80, 0.20, 8, "More allowed; quality unchanged"),
        (80, 0.04, 8, "Lower quality threshold"),
        (80, 0.04, 3, "Points may stand closer"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    results = {}
    for ax, (maximum, quality, distance, title) in zip(
        axes.ravel(), configurations
    ):
        points = detect_corners(
            gray,
            mask=roi_mask,
            max_corners=maximum,
            quality_level=quality,
            min_distance=distance,
        )
        ax.imshow(frame)
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=28, facecolors="none", edgecolors="#F04E45")
        ax.set_title(
            f"{title}\nmax={maximum}, quality={quality:g}, distance={distance} · {len(points)} found",
            fontsize=9,
        )
        ax.axis("off")
        results[(maximum, quality, distance)] = points
    plt.tight_layout()
    plt.show()
    return results


def plot_known_robot_flow(
    frames,
    robot_masks,
    centers,
    previous_index=11,
    current_index=12,
    win_size=(21, 21),
    max_level=3,
):
    """Track robot corners once and compare with its known displacement."""
    previous_gray = _gray(frames[previous_index])
    current_gray = _gray(frames[current_index])
    points = detect_corners(
        previous_gray,
        mask=_robot_roi(robot_masks[previous_index]),
        max_corners=40,
        quality_level=0.05,
        min_distance=4,
    )
    tracked, valid, roundtrip = track_points_forward_backward(
        previous_gray,
        current_gray,
        points,
        win_size=win_size,
        max_level=max_level,
    )
    good_start, good_end = points[valid], tracked[valid]
    measured = good_end - good_start
    true_displacement = (
        np.asarray(centers[current_index]) - np.asarray(centers[previous_index])
    )

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(frames[previous_index])
    axes[0].scatter(good_start[:, 0], good_start[:, 1], s=22, color="#F04E45")
    axes[0].set_title(f"{len(good_start)} recognizable points")
    axes[1].imshow(frames[current_index])
    for start, end in zip(good_start, good_end):
        axes[1].annotate(
            "", xy=end, xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#FFE066", "lw": 1.4},
        )
    axes[1].set_title("Lucas–Kanade estimates")
    axes[2].scatter(measured[:, 0], measured[:, 1], color="#276FBF", alpha=0.8)
    axes[2].scatter(
        *true_displacement,
        marker="*",
        s=180,
        color="#F04E45",
        label=f"truth {tuple(map(int, true_displacement))}",
    )
    axes[2].axhline(0, color="#999999", linewidth=0.7)
    axes[2].set(
        xlabel="horizontal displacement [px]",
        ylabel="vertical displacement [px]",
        title="Each point votes for a displacement",
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    axes[2].set_aspect("equal", adjustable="datalim")
    for ax in axes[:2]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return {
        "points": points,
        "tracked": tracked,
        "valid": valid,
        "roundtrip_error": roundtrip,
        "true_displacement": true_displacement,
    }


def plot_pyramid_experiment(
    frames,
    robot_masks,
    centers,
    previous_index=11,
    current_index=14,
    configurations=(
        ((15, 15), 0),
        ((31, 31), 0),
        ((15, 15), 2),
        ((31, 31), 2),
    ),
):
    """Compare LK settings on a deliberately large multi-frame displacement."""
    previous_gray = _gray(frames[previous_index])
    current_gray = _gray(frames[current_index])
    points = detect_corners(
        previous_gray,
        mask=_robot_roi(robot_masks[previous_index]),
        max_corners=35,
        quality_level=0.05,
        min_distance=4,
    )
    truth = np.asarray(centers[current_index]) - np.asarray(centers[previous_index])
    if not configurations:
        raise ValueError("configurations must contain at least one experiment.")
    columns = 2
    rows = (len(configurations) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(8, 3 * rows),
        squeeze=False,
    )
    results = {}
    for ax, (window, levels) in zip(axes.ravel(), configurations):
        tracked, valid, _ = track_points_forward_backward(
            previous_gray,
            current_gray,
            points,
            win_size=window,
            max_level=levels,
            maximum_roundtrip_error=2.0,
        )
        displacement = tracked[valid] - points[valid]
        endpoint_error = (
            np.linalg.norm(displacement - truth, axis=1)
            if len(displacement)
            else np.empty(0)
        )
        ax.imshow(frames[current_index])
        for start, end in zip(points[valid], tracked[valid]):
            color = "#2E8B57" if np.linalg.norm((end - start) - truth) < 3 else "#F04E45"
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=1)
            ax.scatter(*end, s=10, color=color)
        median_error = np.median(endpoint_error) if len(endpoint_error) else np.nan
        ax.set_title(
            f"window={window[0]}, levels={levels}\n{valid.sum()} kept · median error={median_error:.1f}px",
            fontsize=9,
        )
        ax.axis("off")
        results[(window, levels)] = (valid, endpoint_error)
    for ax in axes.ravel()[len(configurations) :]:
        ax.axis("off")
    fig.suptitle(
        f"One large jump: true displacement = {tuple(map(int, truth))} pixels"
    )
    plt.tight_layout()
    plt.show()
    return results


def plot_robot_tracks(frames, robot_masks, start_index=11, end_index=17):
    """Track initial robot points through several frames and show their histories."""
    points = detect_corners(
        _gray(frames[start_index]),
        mask=_robot_roi(robot_masks[start_index]),
        max_corners=35,
        quality_level=0.05,
        min_distance=4,
    )
    trajectories = [[point.copy()] for point in points]
    active_ids = np.arange(len(points))
    active_points = points.copy()

    for index in range(start_index + 1, end_index + 1):
        tracked, valid, _ = track_points_forward_backward(
            _gray(frames[index - 1]),
            _gray(frames[index]),
            active_points,
            maximum_roundtrip_error=1.5,
        )
        for point_id, point, keep in zip(active_ids, tracked, valid):
            if keep:
                trajectories[point_id].append(point.copy())
        active_ids = active_ids[valid]
        active_points = tracked[valid]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, index in zip(axes, (start_index, (start_index + end_index) // 2, end_index)):
        ax.imshow(frames[index])
        ax.set_title(f"Frame {index}")
        ax.axis("off")
    for trajectory in trajectories:
        path = np.asarray(trajectory)
        if len(path) > 1:
            axes[-1].plot(path[:, 0], path[:, 1], linewidth=1.3, alpha=0.8)
            axes[-1].scatter(*path[-1], s=12)
    axes[-1].set_title(
        f"Frame {end_index}: {len(active_points)}/{len(points)} points survived"
    )
    fig.suptitle("Points have careers; some end behind the tree")
    plt.tight_layout()
    plt.show()
    return trajectories, active_ids


def plot_real_video_tracks(
    frames,
    start_index=40,
    end_index=75,
    maximum_points=80,
):
    """Detect moving corners and track them through a real fixed-camera sequence."""
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=28, detectShadows=False
    )
    masks = [subtractor.apply(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) for frame in frames[: end_index + 1]]
    start_mask = cv2.morphologyEx(
        masks[start_index],
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    points = detect_corners(
        _gray(frames[start_index]),
        mask=start_mask,
        max_corners=maximum_points,
        quality_level=0.04,
        min_distance=6,
    )
    trajectories = [[point.copy()] for point in points]
    active_ids = np.arange(len(points))
    active_points = points.copy()
    snapshots = {start_index: (active_points.copy(), active_ids.copy())}

    for index in range(start_index + 1, end_index + 1):
        tracked, valid, _ = track_points_forward_backward(
            _gray(frames[index - 1]),
            _gray(frames[index]),
            active_points,
            maximum_roundtrip_error=1.5,
        )
        for point_id, point, keep in zip(active_ids, tracked, valid):
            if keep:
                trajectories[point_id].append(point.copy())
        active_ids = active_ids[valid]
        active_points = tracked[valid]
        if index in ((start_index + end_index) // 2, end_index):
            snapshots[index] = (active_points.copy(), active_ids.copy())

    shown = (start_index, (start_index + end_index) // 2, end_index)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, index in zip(axes, shown):
        current_points, _ = snapshots[index]
        ax.imshow(frames[index])
        if len(current_points):
            ax.scatter(
                current_points[:, 0], current_points[:, 1],
                s=13, facecolors="none", edgecolors="#FFE066", linewidths=0.8,
            )
        ax.set_title(f"Frame {index} · {len(current_points)} active")
        ax.axis("off")
    for point_id in snapshots[end_index][1]:
        path = np.asarray(trajectories[point_id])
        axes[-1].plot(path[:, 0], path[:, 1], linewidth=1, alpha=0.75)
    fig.suptitle("Real points tracked without learning anyone's name")
    plt.tight_layout()
    plt.show()
    return trajectories, active_ids

"""Plotting and presentation helpers for notebook 03."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from wave_part_dual.classical_cv import calibrate_camera, look_at_world_to_camera


def plot_homogeneous_transformations(shape, transformations, title):
    """Plot a 2D shape before and after named homogeneous transformations."""
    shape = np.asarray(shape, dtype=float)
    shape_h = np.column_stack([shape, np.ones(len(shape))])
    colors = ["#D1495B", "#2E8B57", "#8E44AD", "#F39C12"]
    transformed_shapes = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    all_shapes = [("original", shape, "#276FBF")]
    for (label, transformation), color in zip(transformations.items(), colors):
        transformed_h = (np.asarray(transformation) @ shape_h.T).T
        transformed = transformed_h[:, :2] / transformed_h[:, 2:3]
        transformed_shapes[label] = transformed
        all_shapes.append((label, transformed, color))

    for label, points, color in all_shapes:
        closed = np.vstack([points, points[0]])
        ax.plot(closed[:, 0], closed[:, 1], marker="o", linewidth=2.5, label=label, color=color)
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()
    return transformed_shapes


def plot_homogeneous_w_limit(
    translation,
    base_xy,
    w_values,
    title="As w approaches zero, the point loses its address politely",
):
    """Visualize finite homogeneous points approaching a direction at ``w = 0``."""
    columns = 3
    rows = int(np.ceil(len(w_values) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10, 3.5 * rows))
    axes = np.asarray(axes, dtype=object).reshape(-1)

    for ax, w in zip(axes, w_values):
        point_h = np.array([base_xy[0], base_xy[1], w])
        translated_h = translation @ point_h
        ax.axhline(0, color="gray", linewidth=1)
        ax.axvline(0, color="gray", linewidth=1)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, linestyle=":", color="gray", label="unit distance"))
        if w > 0:
            point = point_h[:2] / point_h[2]
            translated = translated_h[:2] / translated_h[2]
            limit = max(3.0, translated.max() * 1.18)
            ax.plot([0, point[0]], [0, point[1]], color="#276FBF", alpha=0.45)
            ax.scatter(*point, s=70, color="#276FBF", label="before translation")
            ax.scatter(*translated, s=70, color="#D1495B", label="after translation")
            ax.plot([point[0], translated[0]], [point[1], translated[1]], color="#D1495B", linestyle="--")
            ax.set_xlim(-0.3, limit)
            ax.set_ylim(-0.3, limit * 0.72)
            ax.set_title(f"w = {w:g}  →  point = ({point[0]:.1f}, {point[1]:.1f})")
        else:
            direction = np.asarray(base_xy) / np.linalg.norm(base_xy)
            ax.arrow(0, 0, *(direction * 2.2), width=0.025, color="#276FBF", length_includes_head=True)
            ax.set_xlim(-0.3, 2.5)
            ax.set_ylim(-0.3, 1.8)
            ax.set_title("w = 0  →  direction only; no address")
            ax.text(0.1, 1.55, "translation changes nothing", color="#D1495B")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    for ax in axes[len(w_values):]:
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_cube_world_and_image(
    cube,
    cube_edges,
    camera_position,
    target,
    cube_pixels,
    image_size=(640, 480),
    image_title="Image frame: the cube becomes pixels",
):
    """Show a cube-camera world setup beside the resulting image projection."""
    fig = plt.figure(figsize=(10, 4.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    for first, second in cube_edges:
        ax3d.plot(*cube[[first, second]].T, color="#276FBF", linewidth=2)
    ax3d.scatter(*camera_position, color="#D1495B", s=90, label="camera")
    ax3d.plot(*np.vstack([camera_position, target]).T, color="#D1495B", linestyle="--")
    ax3d.set(xlabel="world X", ylabel="world Y", zlabel="world Z", title="World frame: cube and camera")
    ax3d.legend()

    ax2d = fig.add_subplot(1, 2, 2)
    for first, second in cube_edges:
        ax2d.plot(*cube_pixels[[first, second]].T, color="#276FBF", linewidth=2)
    ax2d.scatter(cube_pixels[:, 0], cube_pixels[:, 1], color="#D1495B")
    ax2d.set_xlim(0, image_size[0])
    ax2d.set_ylim(image_size[1], 0)
    ax2d.set_aspect("equal")
    ax2d.set(xlabel="u [pixels]", ylabel="v [pixels]", title=image_title)
    ax2d.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def project_cube(camera_position, target, cube, camera_matrix):
    """Project a fixed world-space cube into one camera image."""
    rotation, translation = look_at_world_to_camera(camera_position, target)
    cube_camera = (rotation @ cube.T).T + translation
    if np.any(cube_camera[:, 2] <= 0):
        raise ValueError("Part of the cube is behind the camera.")
    normalized = cube_camera[:, :2] / cube_camera[:, 2:3]
    return (camera_matrix[:2, :2] @ normalized.T).T + camera_matrix[:2, 2]


def draw_cube_projection(ax, cube_pixels, cube_edges, title):
    """Draw one projected cube using fixed image coordinates."""
    for first, second in cube_edges:
        ax.plot(*cube_pixels[[first, second]].T, color="#276FBF", linewidth=1.8)
    ax.scatter(cube_pixels[:, 0], cube_pixels[:, 1], color="#D1495B", s=20)
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_xticks([0, 320, 640])
    ax.set_yticks([0, 240, 480])


def plot_multi_camera_cube_views(
    camera_positions,
    target,
    cube,
    cube_edges,
    camera_matrix,
    world_title,
    views_title,
):
    """Show many camera poses in the world and the cube image from each pose."""
    colors = plt.cm.viridis(np.linspace(0, 1, len(camera_positions)))
    world_figure = plt.figure(figsize=(6, 6))
    ax_world = world_figure.add_subplot(1, 1, 1, projection="3d")
    for first, second in cube_edges:
        ax_world.plot(*cube[[first, second]].T, color="#276FBF", linewidth=1.8)
    for index, (camera_position, color) in enumerate(zip(camera_positions, colors)):
        ax_world.scatter(*camera_position, color=color, s=32, alpha=0.7)
        ax_world.plot(*np.vstack([camera_position, target]).T, color=color, alpha=0.35)
        ax_world.text(*camera_position, str(index), fontsize=8)
    ax_world.set(xlabel="X", ylabel="Y", zlabel="Z", title=world_title)
    plt.tight_layout()
    plt.show()

    columns = 3
    rows = int(np.ceil(len(camera_positions) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9, 3 * rows))
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for index, camera_position in enumerate(camera_positions):
        pixels = project_cube(camera_position, target, cube, camera_matrix)
        position_text = np.array2string(camera_position, precision=1, separator=",")
        draw_cube_projection(
            axes[index],
            pixels,
            cube_edges,
            f"camera {index}: {position_text}",
        )
    for ax in axes[len(camera_positions):]:
        ax.axis("off")
    fig.suptitle(views_title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_pixel_rays(
    selected_pixels,
    rays_camera,
    image_size=(640, 480),
    ray_title="K⁻¹ turns each pixel into a direction",
):
    """Show selected image pixels beside their camera-frame viewing rays."""
    fig = plt.figure(figsize=(10, 5))
    ax_image = fig.add_subplot(1, 2, 1)
    ax_image.scatter(selected_pixels[:, 0], selected_pixels[:, 1], s=90, c=np.arange(len(selected_pixels)), cmap="viridis")
    for index, (u, v) in enumerate(selected_pixels):
        ax_image.annotate(str(index), (u, v), xytext=(7, 7), textcoords="offset points")
    ax_image.set_xlim(0, image_size[0])
    ax_image.set_ylim(image_size[1], 0)
    ax_image.set_aspect("equal")
    ax_image.set(xlabel="u [pixels]", ylabel="v [pixels]", title="Five selected pixels")

    ax_ray = fig.add_subplot(1, 2, 2, projection="3d")
    colors = plt.cm.viridis(np.linspace(0, 1, len(rays_camera)))
    for index, (ray, color) in enumerate(zip(rays_camera, colors)):
        endpoint = ray * 3.0
        ax_ray.plot([0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]], color=color, linewidth=2.5, label=f"pixel {index}")
    ax_ray.set(xlabel="camera x (right)", ylabel="camera y (down)", zlabel="camera z (forward)", title=ray_title)
    ax_ray.legend()
    plt.tight_layout()
    plt.show()


def plot_calibration_corners(
    ideal_image_points,
    distorted_image_points,
    image_points,
    image_size,
    sample_views=(0, 7, 14),
):
    """Show ideal, distorted, and noisy corners for selected synthetic views."""
    fig, axes = plt.subplots(1, len(sample_views), figsize=(4 * len(sample_views), 4))
    axes = np.atleast_1d(axes)
    for panel, (ax, view_index) in enumerate(zip(axes, sample_views)):
        ideal = ideal_image_points[view_index]
        distorted = distorted_image_points[view_index]
        observed = image_points[view_index].reshape(-1, 2)
        for ideal_corner, distorted_corner in zip(ideal, distorted):
            ax.plot(
                [ideal_corner[0], distorted_corner[0]],
                [ideal_corner[1], distorted_corner[1]],
                color="#D1495B",
                alpha=0.35,
                linewidth=0.8,
            )
        ax.scatter(ideal[:, 0], ideal[:, 1], facecolors="none", edgecolors="#276FBF", s=28, label="ideal pinhole")
        ax.scatter(distorted[:, 0], distorted[:, 1], marker="x", color="#D1495B", s=20, alpha=0.75, label="after lens distortion")
        ax.scatter(observed[:, 0], observed[:, 1], color="black", s=7, alpha=0.55, label="noisy observation")
        ax.set_xlim(0, image_size[0])
        ax.set_ylim(image_size[1], 0)
        ax.set_aspect("equal")
        max_lens_shift = np.linalg.norm(distorted - ideal, axis=1).max()
        ax.set_title(f"view {view_index}: max lens shift {max_lens_shift:.1f} px")
        ax.set_xlabel("u [pixels]")
        if panel == 0:
            ax.set_ylabel("v [pixels]")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Known corners: ideal → distorted → noisy evidence", fontsize=14)
    plt.tight_layout()
    plt.show()


def calibrate_and_plot_results(
    object_points,
    image_points,
    image_size,
    K_true,
    dist_true,
    calibration_flags=cv2.CALIB_FIX_K3,
):
    """Estimate camera parameters and display truth, estimates, and errors."""
    result = calibrate_camera(object_points, image_points, image_size, flags=calibration_flags)
    rms = result["rms"]
    matrix_estimated = result["K"]
    distortion_estimated = result["distortion"]
    view_errors = result["view_errors"]

    names = ["fx", "fy", "cx", "cy"]
    true_values = [K_true[0, 0], K_true[1, 1], K_true[0, 2], K_true[1, 2]]
    estimated_values = [matrix_estimated[0, 0], matrix_estimated[1, 1], matrix_estimated[0, 2], matrix_estimated[1, 2]]
    positions = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].bar(positions - 0.18, true_values, 0.36, label="secret truth")
    axes[0].bar(positions + 0.18, estimated_values, 0.36, label="recovered")
    axes[0].set_xticks(positions, names)
    axes[0].set_ylabel("pixels")
    axes[0].set_title("Intrinsic parameters")
    axes[0].legend()

    distortion_names = ["k1", "k2", "p1", "p2"]
    axes[1].bar(positions - 0.18, dist_true[:4], 0.36, label="secret truth")
    axes[1].bar(positions + 0.18, distortion_estimated.ravel()[:4], 0.36, label="recovered")
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set_xticks(positions, distortion_names)
    axes[1].set_title("Distortion coefficients")
    axes[1].legend()

    axes[2].bar(np.arange(len(view_errors)), view_errors, color="#276FBF")
    axes[2].axhline(rms, color="#D1495B", linestyle="--", label=f"overall RMS = {rms:.3f} px")
    axes[2].set_xlabel("checkerboard view")
    axes[2].set_ylabel("reprojection RMS [pixels]")
    axes[2].set_title("How well recovered camera explains observations")
    axes[2].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()

    print("Secret K:\n", K_true)
    print("\nRecovered K:\n", matrix_estimated)
    print(f"\nOverall reprojection RMS: {rms:.3f} pixels")
    return result


def plot_camera_side_view(camera_height=1.6, pitch_degrees=15.0):
    """Show a hovering robot and its pitched camera above flat ground."""
    camera_yz = np.array([0.0, camera_height])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="#566573", linewidth=3)
    ax.fill_between([-0.7, 6.5], -0.12, 0, color="#A9DFBF", alpha=0.45)
    ax.text(5.0, 0.08, "flat ground: Z = 0", color="#566573")
    ax.add_patch(Rectangle((-0.42, 1.48), 0.34, 0.24, color="#5D6D7E", zorder=4))
    ax.add_patch(Circle((-0.25, 1.79), 0.13, color="#85929E", zorder=4))
    ax.plot([-0.38, -0.58], [1.52, 1.35], color="#5D6D7E", linewidth=4)
    ax.plot([-0.12, 0.00], [1.52, 1.35], color="#5D6D7E", linewidth=4)
    ax.add_patch(Circle(tuple(camera_yz), 0.055, color="#D1495B", zorder=6))
    ax.annotate(f"camera at {camera_height} m", camera_yz, xytext=(-0.05, 2.15), arrowprops=dict(arrowstyle="->", color="#D1495B"), ha="center")
    ax.annotate("hovering:\nno ground contact required", (-0.28, 1.48), xytext=(-0.62, 0.72), arrowprops=dict(arrowstyle="->", color="#566573"))
    ax.plot([0, 2.0], [camera_height, camera_height], linestyle="--", color="#7F8C8D", label="horizontal")
    angle = np.deg2rad(pitch_degrees)
    axis_end = camera_yz + 4.8 * np.array([np.cos(angle), -np.sin(angle)])
    ax.add_patch(FancyArrowPatch(camera_yz, axis_end, arrowstyle="-|>", mutation_scale=14, linewidth=2.5, color="#D1495B", label="optical axis"))
    ax.add_patch(Arc(camera_yz, 1.2, 1.2, theta1=-pitch_degrees, theta2=0, color="#D1495B", linewidth=2))
    ax.text(0.72, 1.48, f"pitch = {pitch_degrees:.0f}° down", color="#D1495B")
    for ray_angle, color in zip((8, 18, 32), ("#4C78A8", "#59A14F", "#F28E2B")):
        ray_angle = np.deg2rad(ray_angle)
        ground_y = camera_height / np.tan(ray_angle)
        ax.plot([0, ground_y], [camera_height, 0], color=color, linewidth=1.8, alpha=0.9)
        ax.scatter(ground_y, 0, color=color, s=55, zorder=5)
    ax.text(3.8, 1.02, "pixel rays", color="#566573")
    ax.set(xlim=(-0.75, 6.5), ylim=(-0.15, 2.45), xlabel="forward distance Y [m]", ylabel="height Z [m]")
    ax.set_title("Side view: a pitched camera looking for the flat ground")
    ax.set_aspect("equal")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_pixel_to_ground_results(
    camera_matrix,
    ground_pixels,
    ground_points,
    pitch_degrees=15.0,
    image_size=(640, 480),
    ground_xlim=(-1.5, 1.5),
    ground_ylim=(-0.2, 6.0),
):
    """Show pixels beside the places where their rays meet the ground."""
    ground_pixels = np.asarray(ground_pixels)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = np.arange(len(ground_pixels))
    axes[0].scatter(ground_pixels[:, 0], ground_pixels[:, 1], c=colors, cmap="viridis", s=100)
    for index, pixel in enumerate(ground_pixels):
        axes[0].annotate(str(index), pixel, xytext=(7, 7), textcoords="offset points")
    horizon_v = camera_matrix[1, 2] - camera_matrix[1, 1] * np.tan(np.deg2rad(pitch_degrees))
    axes[0].axhline(horizon_v, color="#D1495B", linestyle="--", label="horizon")
    axes[0].set(xlim=(0, image_size[0]), ylim=(image_size[1], 0), xlabel="u [pixels]", ylabel="v [pixels]", title="Selected image pixels")
    axes[0].set_aspect("equal")
    axes[0].legend()
    axes[1].scatter(ground_points[:, 0], ground_points[:, 1], c=colors, cmap="viridis", s=100)
    axes[1].scatter(0, 0, marker="^", color="#D1495B", s=120, label="camera")
    for index, (point, color_position) in enumerate(zip(ground_points, np.linspace(0, 1, len(ground_points)))):
        axes[1].plot([0, point[0]], [0, point[1]], color=plt.cm.viridis(color_position), alpha=0.7)
        axes[1].annotate(str(index), point[:2], xytext=(7, 7), textcoords="offset points")
    axes[1].set_aspect("equal")
    axes[1].set(xlim=ground_xlim, ylim=ground_ylim, xlabel="world X: sideways [m]", ylabel="world Y: forward [m]", title="Where each pixel ray meets the ground")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def print_pixel_to_ground_results(ground_pixels, ground_points):
    """Print the numeric pixel-to-ground mapping displayed by the plot."""
    print("pixel (u, v)  ->  ground point (X, Y)")
    for pixel, point in zip(np.asarray(ground_pixels).astype(int), ground_points):
        print(f"{tuple(pixel)} -> ({point[0]:6.2f} m, {point[1]:6.2f} m)")

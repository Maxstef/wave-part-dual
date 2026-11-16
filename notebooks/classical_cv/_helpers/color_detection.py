"""Synthetic scenes and plotting helpers for notebook 01."""

from collections.abc import Callable, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from wave_part_dual.classical_cv import mask_iou


def make_scene(brightness: float = 1.0, seed: int = 7) -> tuple[NDArray, NDArray]:
    """Create the notebook's textured scene and target ground-truth mask."""
    rng = np.random.default_rng(seed)
    height, width = 280, 420
    image = np.full((height, width, 3), (45, 70, 95), dtype=np.float32)
    image += rng.normal(0, 4, image.shape)

    ground_truth = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(ground_truth, (210, 145), 67, 255, thickness=-1)
    image[ground_truth > 0] = (245, 125, 28)
    cv2.rectangle(image, (32, 52), (112, 128), (205, 45, 52), thickness=-1)
    for _ in range(45):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        cv2.circle(image, (x, y), int(rng.integers(1, 4)), (235, 120, 30), -1)
    for center in [(145, 42), (335, 62), (362, 205), (82, 225), (315, 145)]:
        cv2.circle(image, center, 5, (235, 120, 30), thickness=-1)
    for center in [(190, 128), (225, 158), (210, 180)]:
        cv2.circle(image, center, 6, (55, 75, 95), thickness=-1)

    illumination = np.linspace(0.45, 1.15, width, dtype=np.float32)
    image *= illumination[None, :, None] * brightness
    return np.clip(image, 0, 255).astype(np.uint8), ground_truth


def evaluate_brightness(
    levels: Iterable[float],
    rgb_lower: NDArray,
    rgb_upper: NDArray,
    hsv_lower: NDArray,
    hsv_upper: NDArray,
    scene_factory: Callable[..., tuple[NDArray, NDArray]] = make_scene,
) -> tuple[NDArray, NDArray]:
    """Measure RGB and HSV mask IoU across global brightness levels."""
    rgb_scores, hsv_scores = [], []
    for level in levels:
        test_rgb, truth = scene_factory(brightness=level)
        rgb_mask = cv2.inRange(test_rgb, rgb_lower, rgb_upper)
        hsv = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2HSV)
        hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        rgb_scores.append(mask_iou(rgb_mask, truth))
        hsv_scores.append(mask_iou(hsv_mask, truth))
    return np.asarray(rgb_scores), np.asarray(hsv_scores)


def plot_brightness_test(levels, rgb_scores, hsv_scores, title: str) -> None:
    """Plot RGB and HSV segmentation quality across brightness levels."""
    plt.figure(figsize=(9, 4.5))
    plt.plot(levels, rgb_scores, "o-", label="RGB interval")
    plt.plot(levels, hsv_scores, "o-", label="HSV interval")
    plt.xlabel("global brightness multiplier")
    plt.ylabel("mask IoU")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.title(title)
    plt.show()


def plot_rgb_threshold_experiments(rgb, ground_truth, experiments) -> None:
    """Compare masks while changing one RGB threshold bound at a time."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input scene")

    for ax, (name, lower, upper) in zip(axes.flat[1:], experiments):
        lower = np.asarray(lower, dtype=np.uint8)
        upper = np.asarray(upper, dtype=np.uint8)
        candidate_mask = cv2.inRange(rgb, lower, upper)
        score = mask_iou(candidate_mask, ground_truth)
        ax.imshow(candidate_mask, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name}\nL={lower.tolist()}  U={upper.tolist()}\nIoU={score:.3f}")

    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("One changed RGB bound at a time", fontsize=14)
    plt.tight_layout()


def plot_morphology_kernel_experiments(
    mask,
    ground_truth,
    kernel_shapes,
    kernel_sizes,
) -> None:
    """Show structuring elements and their opening-then-closing results."""
    experiments = {}
    for shape_name, shape_code in kernel_shapes.items():
        for size in kernel_sizes:
            kernel = cv2.getStructuringElement(shape_code, (size, size))
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            experiments[(shape_name, size)] = (kernel, result)

    fig, axes = plt.subplots(len(kernel_shapes), len(kernel_sizes), figsize=(6, 6))
    for row, shape_name in enumerate(kernel_shapes):
        for column, size in enumerate(kernel_sizes):
            kernel, _ = experiments[(shape_name, size)]
            axes[row, column].imshow(kernel, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[row, column].set_title(f"{shape_name} {size}×{size}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle("Structuring elements", fontsize=14)
    plt.tight_layout()

    fig, axes = plt.subplots(len(kernel_shapes), len(kernel_sizes), figsize=(13, 12))
    for row, shape_name in enumerate(kernel_shapes):
        for column, size in enumerate(kernel_sizes):
            _, result = experiments[(shape_name, size)]
            score = mask_iou(result, ground_truth)
            axes[row, column].imshow(result, cmap="gray", vmin=0, vmax=255)
            axes[row, column].set_title(f"{shape_name} {size}×{size}\nIoU={score:.3f}")
            axes[row, column].axis("off")
    fig.suptitle("Opening → closing with one kernel", fontsize=14)
    plt.tight_layout()


def plot_connectivity_examples() -> None:
    """Visualize how 4- and 8-connectivity label tiny contact patterns."""
    diagonal_touch = np.zeros((9, 9), dtype=np.uint8)
    diagonal_touch[1:3, 1:3] = 255
    diagonal_touch[3:5, 3:5] = 255
    diagonal_touch[5:7, 5:7] = 255

    edge_touch = np.zeros((9, 9), dtype=np.uint8)
    edge_touch[1:3, 1:3] = 255
    edge_touch[1:3, 3:5] = 255
    edge_touch[3:5, 3:5] = 255

    mixed_touch = np.zeros((9, 9), dtype=np.uint8)
    mixed_touch[1:3, 1:3] = 255
    mixed_touch[3:5, 3:5] = 255
    mixed_touch[3:5, 5:7] = 255
    mixed_touch[6:8, 6:8] = 255

    tiny_masks = {
        "Diagonal contacts": diagonal_touch,
        "Edge contacts": edge_touch,
        "Mixed contacts": mixed_touch,
    }
    fig, axes = plt.subplots(len(tiny_masks), 3, figsize=(11, 10))
    for row, (name, mask) in enumerate(tiny_masks.items()):
        axes[row, 0].imshow(mask, cmap="gray", interpolation="nearest")
        axes[row, 0].set_title(f"{name}: mask")
        for column, connectivity in enumerate((4, 8), start=1):
            label_count, labels = cv2.connectedComponents(mask, connectivity=connectivity)
            axes[row, column].imshow(labels, cmap="nipy_spectral", vmin=0, vmax=3, interpolation="nearest")
            axes[row, column].set_title(f"{connectivity}-connectivity: {label_count - 1} component(s)")
            for y, x in np.argwhere(labels > 0):
                axes[row, column].text(x, y, str(labels[y, x]), ha="center", va="center", color="white", fontsize=8)

    for ax in axes.flat:
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.grid(color="lightgray", linewidth=0.6)
        ax.tick_params(labelsize=7)
    plt.tight_layout()

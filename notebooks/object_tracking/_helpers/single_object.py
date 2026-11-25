"""Synthetic scenes and presentation helpers for tracking notebook 03."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from wave_part_dual.tracking import box_iou, expand_box, match_template


def _draw_robot(image, center, scale=1.0, color=(62, 105, 176), badge="R"):
    x, ground_y = center
    body_half = round(18 * scale)
    body_height = round(42 * scale)
    head_radius = round(14 * scale)
    head_y = ground_y - body_height - head_radius + round(3 * scale)
    body_top = ground_y - body_height

    cv2.rectangle(
        image,
        (x - body_half, body_top),
        (x + body_half, ground_y),
        color,
        -1,
    )
    cv2.circle(image, (x, head_y), head_radius, (90, 141, 215), -1)
    eye_offset = max(2, round(7 * scale))
    eye_radius = max(1, round(2 * scale))
    cv2.circle(image, (x - eye_offset, head_y - 2), eye_radius, (245, 245, 245), -1)
    cv2.circle(image, (x + eye_offset, head_y - 2), eye_radius, (245, 245, 245), -1)
    antenna_top = head_y - head_radius - round(10 * scale)
    cv2.line(image, (x, head_y - head_radius), (x, antenna_top), (45, 45, 55), 2)
    cv2.circle(image, (x, antenna_top), max(2, round(3 * scale)), (220, 72, 72), -1)
    wheel_radius = max(4, round(7 * scale))
    for wheel_x in (x - round(12 * scale), x + round(12 * scale)):
        cv2.circle(image, (wheel_x, ground_y + wheel_radius), wheel_radius, (35, 39, 48), -1)
    cv2.putText(
        image,
        badge,
        (x - round(6 * scale), ground_y - round(10 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45 * scale,
        (245, 230, 105),
        max(1, round(scale)),
        cv2.LINE_AA,
    )

    left = x - body_half - wheel_radius
    right = x + body_half + wheel_radius
    top = antenna_top - wheel_radius
    bottom = ground_y + wheel_radius * 2
    return left, top, right - left, bottom - top


def _draw_tree(image, x=300, ground_y=205):
    cv2.rectangle(image, (x - 12, 91), (x + 12, ground_y), (91, 60, 40), -1)
    for center, radius in (((x, 78), 37), ((x - 28, 101), 31), ((x + 28, 101), 31)):
        cv2.circle(image, center, radius, (45, 115, 64), -1)


def make_single_object_sequence(frame_count=24, height=260, width=440, seed=13):
    """Create a moving, growing robot with occlusion and a suspicious twin."""
    rng = np.random.default_rng(seed)
    frames, true_boxes, scales = [], [], []
    for index in range(frame_count):
        image = np.full((height, width, 3), (196, 220, 238), dtype=np.uint8)
        cv2.rectangle(image, (0, 205), (width, height), (91, 145, 85), -1)
        cv2.circle(image, (42, 44), 20, (255, 221, 90), -1)
        cv2.line(image, (0, 204), (width, 204), (71, 118, 69), 2)

        # Small static details keep the universe from becoming perfectly blank.
        cv2.rectangle(image, (16, 176), (31, 204), (125, 82, 51), -1)
        cv2.circle(image, (393, 187), 10, (201, 73, 70), -1)

        scale = 0.85 + 0.025 * index
        x = 54 + 14 * index
        true_box = _draw_robot(image, (x, 190), scale=scale, badge="R")

        # A look-alike appears when the real robot approaches the tree.
        if index >= 13:
            _draw_robot(image, (145, 190), scale=1.0, color=(62, 105, 176), badge="?")

        # Drawn last, so the real robot genuinely passes behind it.
        _draw_tree(image)
        noise = rng.normal(0, 1.8, image.shape)
        frames.append(np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8))
        true_boxes.append(true_box)
        scales.append(scale)
    return frames, true_boxes, np.asarray(scales)


def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _crop(image, box):
    x, y, width, height = box
    return image[y : y + height, x : x + width]


def _draw_box(ax, box, color, label, linewidth=2):
    x, y, width, height = box
    ax.add_patch(
        Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=linewidth)
    )
    ax.text(
        x,
        max(8, y - 4),
        label,
        color=color,
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
    )


def plot_tracking_contract(frames, true_boxes, indices=(0, 7, 15, 22)):
    """Show initialization and later boxes in the synthetic sequence."""
    fig, axes = plt.subplots(1, len(indices), figsize=(14, 3.4))
    for ax, index in zip(axes, indices):
        ax.imshow(frames[index])
        _draw_box(ax, true_boxes[index], "#2E8B57", "true robot")
        ax.set_title("Initialization" if index == indices[0] else f"Frame {index}")
        ax.axis("off")
    fig.suptitle("One identity, one box, increasingly unreasonable circumstances")
    plt.tight_layout()
    plt.show()


def plot_template_match_pair(frames, true_boxes, template_index=0, target_index=9):
    """Visualize a template, correlation response, and best fixed-scale match."""
    template = _crop(_gray(frames[template_index]), true_boxes[template_index])
    target = _gray(frames[target_index])
    response = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    match = match_template(target, template)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].imshow(template, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Template saved at initialization")
    axes[1].imshow(response, cmap="magma", vmin=-1, vmax=1)
    axes[1].scatter(match.box[0], match.box[1], color="#72E5F2", s=35)
    axes[1].set_title("Correlation map\nbright = looks more like the template")
    axes[2].imshow(frames[target_index])
    _draw_box(axes[2], true_boxes[target_index], "#2E8B57", "truth")
    _draw_box(axes[2], match.box, "#F04E45", f"match {match.score:.2f}")
    axes[2].set_title(f"Frame {target_index}: best fixed-size match")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return match


def run_template_tracker(
    frames,
    initial_box,
    search="global",
    search_margin=45,
    scales=(1.0,),
):
    """Track a fixed initial template with global or previous-box-local search."""
    template = _crop(_gray(frames[0]), initial_box)
    boxes, scores, chosen_scales = [initial_box], [1.0], [1.0]
    previous_box = initial_box
    for frame in frames[1:]:
        search_box = (
            None
            if search == "global"
            else expand_box(previous_box, search_margin, frame.shape)
        )
        match = match_template(_gray(frame), template, search_box, scales=scales)
        boxes.append(match.box)
        scores.append(match.score)
        chosen_scales.append(match.scale)
        previous_box = match.box
    return boxes, np.asarray(scores), np.asarray(chosen_scales)


def plot_search_strategies(
    frames,
    true_boxes,
    global_boxes,
    local_boxes,
    frame_indices=(8, 14, 17, 22),
):
    """Compare global and local template searches at selected moments."""
    fig, axes = plt.subplots(2, len(frame_indices), figsize=(14, 6))
    for row, (name, boxes) in enumerate((('Global search', global_boxes), ('Local search', local_boxes))):
        for column, index in enumerate(frame_indices):
            ax = axes[row, column]
            ax.imshow(frames[index])
            _draw_box(ax, true_boxes[index], "#2E8B57", "truth")
            _draw_box(ax, boxes[index], "#F04E45", name)
            ax.set_title(f"Frame {index} · IoU={box_iou(boxes[index], true_boxes[index]):.2f}")
            ax.axis("off")
    fig.suptitle("Search everywhere or trust yesterday's neighborhood?")
    plt.tight_layout()
    plt.show()


def plot_tracker_metrics(true_boxes, tracking_results, occlusion=(16, 20)):
    """Plot similarity confidence and box overlap for several trackers."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for name, (boxes, scores) in tracking_results.items():
        iou = [box_iou(box, truth) for box, truth in zip(boxes, true_boxes)]
        axes[0].plot(scores, marker="o", markersize=3, label=name)
        axes[1].plot(iou, marker="o", markersize=3, label=name)
    for ax in axes:
        ax.axvspan(*occlusion, color="#777777", alpha=0.16, label="tree trouble")
        ax.set_xlabel("frame")
        ax.grid(alpha=0.25)
    axes[0].set(ylabel="template similarity", title="Confidence: how pleased is the tracker?")
    axes[1].set(ylabel="IoU with true box", title="Reality: is the box actually correct?", ylim=(-0.03, 1.03))
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_scale_comparison(
    frames,
    true_boxes,
    fixed_boxes,
    multiscale_boxes,
    multiscale_choices,
    frame_indices=(5, 10, 14),
):
    """Compare fixed-size and multi-scale template matching before occlusion."""
    columns = min(3, len(frame_indices))
    rows_per_tracker = (len(frame_indices) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows_per_tracker * 2,
        columns,
        figsize=(12, 3 * rows_per_tracker * 2),
        squeeze=False,
    )
    for tracker_number, (name, boxes) in enumerate(
        (("Fixed scale", fixed_boxes), ("Multiple scales", multiscale_boxes))
    ):
        row_offset = tracker_number * rows_per_tracker
        for position, index in enumerate(frame_indices):
            row = row_offset + position // columns
            column = position % columns
            ax = axes[row, column]
            ax.imshow(frames[index])
            _draw_box(ax, true_boxes[index], "#2E8B57", "truth")
            _draw_box(ax, boxes[index], "#F04E45", name)
            ax.set_title(
                f"{name} · frame {index}\n"
                f"IoU={box_iou(boxes[index], true_boxes[index]):.2f}"
            )
            ax.axis("off")
        used_in_last_row = len(frame_indices) % columns
        if used_in_last_row:
            for column in range(used_in_last_row, columns):
                axes[row_offset + rows_per_tracker - 1, column].axis("off")
    fig.suptitle("A fixed template remembers the robot before it grew")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()

    fixed_iou = [box_iou(box, truth) for box, truth in zip(fixed_boxes, true_boxes)]
    multi_iou = [box_iou(box, truth) for box, truth in zip(multiscale_boxes, true_boxes)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(fixed_iou, color="#276FBF")
    axes[0].set_title("Fixed-scale IoU")
    axes[1].plot(multi_iou, color="#2E8B57", label="IoU")
    axes[1].plot(
        multiscale_choices,
        color="#F04E45",
        linestyle="--",
        label="chosen scale",
    )
    axes[1].set_title("Multi-scale IoU and chosen scale")
    for ax in axes:
        ax.set_xlabel("frame")
        ax.set_ylim(-0.03, 1.55)
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def track_real_video_mil(frames, initial_box, start_index=40, end_index=75):
    """Track one real-video box using OpenCV's built-in MIL tracker."""
    tracker = cv2.TrackerMIL_create()
    tracker.init(cv2.cvtColor(frames[start_index], cv2.COLOR_RGB2BGR), initial_box)
    boxes = {start_index: tuple(map(int, initial_box))}
    success = {start_index: True}
    for index in range(start_index + 1, end_index + 1):
        ok, box = tracker.update(cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR))
        boxes[index] = tuple(round(value) for value in box)
        success[index] = bool(ok)
    return boxes, success


def plot_real_mil_tracking(
    frames,
    boxes,
    success,
    frame_indices=(40, 50, 60, 75),
):
    """Show selected real frames from an OpenCV MIL tracking run."""
    fig, axes = plt.subplots(1, len(frame_indices), figsize=(15, 4))
    for ax, index in zip(axes, frame_indices):
        ax.imshow(frames[index])
        color = "#2E8B57" if success[index] else "#F04E45"
        _draw_box(ax, boxes[index], color, "MIL box")
        ax.set_title(f"Frame {index}")
        ax.axis("off")
    fig.suptitle("One real pedestrian, one initialized box, no identity documents")
    plt.tight_layout()
    plt.show()

"""Synthetic sequences and presentation helpers for tracking notebook 01."""

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray


OPENCV_VTEST_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
)


def make_robot_sequence(
    frame_count: int = 18,
    height: int = 220,
    width: int = 360,
    noise_sigma: float = 2.5,
    cloud_reveal_frames: int | None = 7,
    seed: int = 7,
) -> tuple[list[NDArray[np.uint8]], list[NDArray[np.uint8]], list[tuple[int, int]]]:
    """Create a fixed-camera scene containing one indecisive robot and weather."""
    rng = np.random.default_rng(seed)
    frames, masks, centers = [], [], []

    for index in range(frame_count):
        image = np.full((height, width, 3), (196, 220, 238), dtype=np.uint8)
        cv2.rectangle(image, (0, 160), (width, height), (92, 145, 86), -1)
        cv2.circle(image, (42, 42), 20, (255, 221, 90), -1)

        # A tree with no tracking ambitions whatsoever.
        cv2.rectangle(image, (285, 82), (299, 164), (92, 61, 41), -1)
        for center in ((292, 68), (272, 88), (312, 88)):
            cv2.circle(image, center, 29, (48, 116, 66), -1)

        # Move, pause for three frames, then continue. The pause is intentional.
        progress = index if index < 8 else (7 if index < 11 else index - 3)
        x = 48 + 14 * progress
        y = 139
        centers.append((x, y))
        mask = np.zeros((height, width), dtype=np.uint8)

        # Robot body, head, wheels, and a tiny antenna for scientific credibility.
        cv2.rectangle(image, (x - 18, y - 27), (x + 18, y + 16), (63, 104, 173), -1)
        cv2.rectangle(mask, (x - 18, y - 27), (x + 18, y + 16), 255, -1)
        cv2.circle(image, (x, y - 38), 14, (90, 139, 211), -1)
        cv2.circle(mask, (x, y - 38), 14, 255, -1)
        cv2.circle(image, (x - 8, y - 40), 2, (245, 245, 245), -1)
        cv2.circle(image, (x + 8, y - 40), 2, (245, 245, 245), -1)
        cv2.line(image, (x, y - 52), (x, y - 60), (45, 45, 55), 2)
        cv2.circle(image, (x, y - 62), 3, (220, 72, 72), -1)
        for wheel_x in (x - 12, x + 12):
            cv2.circle(image, (wheel_x, y + 18), 7, (38, 42, 50), -1)
            cv2.circle(mask, (wheel_x, y + 18), 7, 255, -1)

        if cloud_reveal_frames is not None:
            reveal_step = min(index, cloud_reveal_frames)
            revealed_fraction = reveal_step / max(cloud_reveal_frames, 1)
            brightness_offset = round(-42 * (1.0 - revealed_fraction))
            image = np.clip(
                image.astype(np.int16) + brightness_offset, 0, 255
            ).astype(np.uint8)

            # The cloud stops after uncovering the sun. It has caused enough trouble.
            # Robot speed is 14 px/frame; the cloud drifts at half that speed.
            cloud_x = 58 + 7 * reveal_step
            cloud_color = (142, 151, 160)
            cv2.ellipse(image, (cloud_x, 40), (32, 13), 0, 0, 360, cloud_color, -1)
            cv2.circle(image, (cloud_x - 17, 35), 14, cloud_color, -1)
            cv2.circle(image, (cloud_x + 2, 29), 18, cloud_color, -1)
            cv2.circle(image, (cloud_x + 21, 36), 13, cloud_color, -1)

        noise = rng.normal(0, noise_sigma, image.shape)
        frames.append(np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8))
        masks.append(mask)

    return frames, masks, centers


def motion_mask(
    previous: NDArray[np.uint8],
    current: NDArray[np.uint8],
    threshold: int = 25,
    kernel_size: int = 5,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    """Compute absolute frame difference, a threshold mask, and its cleanup."""
    difference = cv2.absdiff(current, previous)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
    _, raw_mask = cv2.threshold(gray_difference, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    return gray_difference, raw_mask, clean_mask


def foreground_boxes(
    mask: NDArray[np.uint8], minimum_area: int = 80
) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes for sufficiently large connected foreground regions."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area >= minimum_area:
            boxes.append((int(x), int(y), int(width), int(height)))
    return boxes


def plot_sequence_strip(frames, frame_indices=(0, 4, 7, 9, 12, 16)):
    """Show selected moments from the synthetic robot sequence."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    for ax, index in zip(axes.ravel(), frame_indices):
        ax.imshow(frames[index])
        ax.set_title(f"Frame {index}")
        ax.axis("off")
    fig.suptitle("A fixed camera observes one robot reconsidering its life choices")
    plt.show()


def plot_cloud_reveal(frames, frame_indices=(0, 3, 7), threshold=5):
    """Show gradual sunlight recovery and the changes it creates between frames."""
    fig, axes = plt.subplots(2, len(frame_indices), figsize=(10, 5.5))
    for column, index in enumerate(frame_indices):
        axes[0, column].imshow(frames[index])
        axes[0, column].set_title(
            f"Frame {index}\nmean brightness {frames[index].mean():.0f}"
        )
        if index == 0:
            mask = np.zeros(frames[index].shape[:2], dtype=np.uint8)
            mask_title = "No previous frame yet"
        else:
            _, mask, _ = motion_mask(
                frames[index - 1], frames[index], threshold=threshold
            )
            mask_title = f"Change from {index - 1} → {index}\nthreshold {threshold}"
        axes[1, column].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[1, column].set_title(mask_title)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("The cloud leaves quietly; the pixels do not")
    plt.tight_layout()
    plt.show()


def plot_frame_difference(frames, previous_index=5, current_index=6, threshold=25):
    """Visualize the stages of consecutive-frame motion detection."""
    difference, raw_mask, clean_mask = motion_mask(
        frames[previous_index], frames[current_index], threshold=threshold
    )
    images = [
        (f"Previous frame {previous_index}", frames[previous_index], None),
        (f"Current frame {current_index}", frames[current_index], None),
        ("Absolute difference", difference, "magma"),
        (f"Threshold > {threshold}", raw_mask, "gray"),
        ("Opening + closing", clean_mask, "gray"),
        ("Connected motion regions", frames[current_index], None),
    ]
    boxes = foreground_boxes(clean_mask)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (title, image, cmap) in zip(axes.ravel(), images):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    for x, y, width, height in boxes:
        axes.ravel()[-1].add_patch(
            Rectangle((x, y), width, height, fill=False, edgecolor="#F04E45", linewidth=2)
        )
    plt.tight_layout()
    plt.show()
    return {"difference": difference, "raw_mask": raw_mask, "clean_mask": clean_mask}


def plot_threshold_sweep(
    frames,
    thresholds=(10, 25, 50),
    previous_index=5,
    current_index=6,
):
    """Compare raw and cleaned motion masks across several thresholds."""
    fig, axes = plt.subplots(2, len(thresholds), figsize=(10, 5.5), squeeze=False)
    results = {}
    pixel_count = frames[0].shape[0] * frames[0].shape[1]

    for column, threshold in enumerate(thresholds):
        _, raw_mask, clean_mask = motion_mask(
            frames[previous_index],
            frames[current_index],
            threshold=threshold,
        )
        raw_fraction = np.count_nonzero(raw_mask) / pixel_count
        clean_fraction = np.count_nonzero(clean_mask) / pixel_count
        axes[0, column].imshow(raw_mask, cmap="gray", vmin=0, vmax=255)
        axes[0, column].set_title(
            f"Threshold {threshold}: raw\n{raw_fraction:.1%} pixels suspicious"
        )
        axes[1, column].imshow(clean_mask, cmap="gray", vmin=0, vmax=255)
        axes[1, column].set_title(
            f"After morphology\n{clean_fraction:.1%} pixels remain"
        )
        results[threshold] = {"raw_mask": raw_mask, "clean_mask": clean_mask}

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("The same movement observed by increasingly skeptical thresholds")
    plt.tight_layout()
    plt.show()
    return results


def plot_motion_failures(frames, threshold=25, illumination_threshold=5):
    """Compare normal movement, a pause, and gradual illumination change."""
    comparisons = [
        (11, 12, threshold, "Moving: useful evidence"),
        (8, 9, threshold, "Paused: the robot becomes background"),
        (2, 3, illumination_threshold, "Robot + weather: who changed what?"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for column, (previous, current, case_threshold, title) in enumerate(comparisons):
        _, _, mask = motion_mask(
            frames[previous], frames[current], threshold=case_threshold
        )
        axes[0, column].imshow(frames[current])
        axes[0, column].set_title(title, fontsize=10)
        axes[1, column].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[1, column].set_title(
            f"Frames {previous} → {current} · threshold {case_threshold}"
        )
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def running_background_masks(frames, alpha=0.08, threshold=28):
    """Estimate a running-average background and foreground masks."""
    background = frames[0].astype(np.float32)
    backgrounds = [frames[0].copy()]
    masks = [np.zeros(frames[0].shape[:2], dtype=np.uint8)]
    for frame in frames[1:]:
        difference = cv2.absdiff(frame, cv2.convertScaleAbs(background))
        gray = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        masks.append(mask)
        cv2.accumulateWeighted(frame, background, alpha)
        backgrounds.append(cv2.convertScaleAbs(background))
    return backgrounds, masks


def plot_background_learning(frames, alphas=(0.03, 0.25), frame_indices=(6, 10, 16)):
    """Compare slow and fast running-average background adaptation."""
    fig, axes = plt.subplots(2, len(frame_indices) * 2, figsize=(15, 5))
    for row, alpha in enumerate(alphas):
        backgrounds, masks = running_background_masks(frames, alpha=alpha)
        for column, index in enumerate(frame_indices):
            axes[row, column * 2].imshow(backgrounds[index])
            axes[row, column * 2].set_title(
                f"α={alpha:g}\nbackground {index}", fontsize=9
            )
            axes[row, column * 2 + 1].imshow(masks[index], cmap="gray", vmin=0, vmax=255)
            axes[row, column * 2 + 1].set_title(
                f"foreground\nframe {index}", fontsize=9
            )
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def mog2_masks(frames, history=40, var_threshold=24, detect_shadows=True):
    """Apply OpenCV MOG2 sequentially and return its foreground masks."""
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )
    return [subtractor.apply(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) for frame in frames]


def plot_mog2_sequence(frames, history=40, var_threshold=24, detect_shadows=True):
    """Show MOG2 initialization, movement, pause, and illumination response."""
    masks = mog2_masks(frames, history, var_threshold, detect_shadows)
    indices = (0, 5, 9, 13, 16)
    fig, axes = plt.subplots(2, len(indices), figsize=(12, 5))
    for column, index in enumerate(indices):
        axes[0, column].imshow(frames[index])
        axes[0, column].set_title(f"Frame {index}")
        axes[1, column].imshow(masks[index], cmap="gray", vmin=0, vmax=255)
        axes[1, column].set_title("MOG2 mask")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return masks


def ensure_opencv_sample_video(path: str | Path) -> Path:
    """Download OpenCV's public vtest.avi sample when it is not cached locally."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading OpenCV's vtest.avi sample (about 8 MB)...")
        urlretrieve(OPENCV_VTEST_URL, path)
    return path


def read_video_frames(path: str | Path, maximum_frames: int | None = None):
    """Read RGB frames and frames-per-second metadata from a video."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while maximum_frames is None or len(frames) < maximum_frames:
        success, bgr = capture.read()
        if not success:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, fps


def plot_real_video_motion(
    frames,
    frame_indices=(40, 90, 140, 190),
    history=120,
    var_threshold=28,
    minimum_area=180,
):
    """Run MOG2 on real frames and show foreground boxes at selected moments."""
    masks = mog2_masks(
        frames,
        history=history,
        var_threshold=var_threshold,
        detect_shadows=True,
    )
    fig, axes = plt.subplots(2, len(frame_indices), figsize=(14, 6))
    for column, index in enumerate(frame_indices):
        frame = frames[index]
        mask = masks[index]
        foreground = np.where(mask == 255, 255, 0).astype(np.uint8)
        boxes = foreground_boxes(foreground, minimum_area=minimum_area)
        axes[0, column].imshow(frame)
        axes[0, column].set_title(f"Real frame {index}")
        for x, y, width, height in boxes:
            axes[0, column].add_patch(
                Rectangle(
                    (x, y), width, height, fill=False, edgecolor="#F04E45", linewidth=1.5
                )
            )
        axes[1, column].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[1, column].set_title(
            "0 background · 127 shadow · 255 foreground", fontsize=8
        )
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return masks

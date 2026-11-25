"""Small reusable utilities for bounding boxes and template-based tracking."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class TemplateMatch:
    """Best template match found in one frame."""

    box: Box
    score: float
    scale: float


def box_iou(first: Box, second: Box) -> float:
    """Return intersection over union for two ``(x, y, width, height)`` boxes."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def expand_box(box: Box, margin: int, image_shape: tuple[int, ...]) -> Box:
    """Expand a box by a pixel margin and clip it to the image boundary."""
    x, y, width, height = box
    image_height, image_width = image_shape[:2]
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(image_width, x + width + margin)
    bottom = min(image_height, y + height + margin)
    return left, top, right - left, bottom - top


def match_template(
    frame_gray: NDArray[np.uint8],
    template_gray: NDArray[np.uint8],
    search_box: Box | None = None,
    scales: tuple[float, ...] = (1.0,),
) -> TemplateMatch:
    """Find the highest normalized-correlation template match over given scales."""
    if frame_gray.ndim != 2 or template_gray.ndim != 2:
        raise ValueError("frame_gray and template_gray must be two-dimensional.")
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("scales must contain positive values.")

    if search_box is None:
        search_box = (0, 0, frame_gray.shape[1], frame_gray.shape[0])
    x, y, width, height = search_box
    search = frame_gray[y : y + height, x : x + width]
    best = None

    for scale in scales:
        candidate_width = max(1, round(template_gray.shape[1] * scale))
        candidate_height = max(1, round(template_gray.shape[0] * scale))
        if candidate_width > search.shape[1] or candidate_height > search.shape[0]:
            continue
        candidate = cv2.resize(
            template_gray,
            (candidate_width, candidate_height),
            interpolation=cv2.INTER_LINEAR,
        )
        response = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        match = TemplateMatch(
            box=(x + location[0], y + location[1], candidate_width, candidate_height),
            score=float(score),
            scale=float(scale),
        )
        if best is None or match.score > best.score:
            best = match

    if best is None:
        raise ValueError("No scaled template fits inside the search region.")
    return best

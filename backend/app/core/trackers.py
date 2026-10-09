"""Shared runtime types every tracking engine speaks.

Whatever its implementation -- homemade baseline, OpenCV follower, standalone MOT
tracker -- an engine hands back the same tiny structure, so the metrics module,
the explainer and the visualiser can treat them all identically:

    state = engine.update(dets, img)
    state.active      -> list[Track]  (boxes the tracker is currently sure of)
    state.lost_now    -> list[Track]  (tracks that just died this frame)

The engines themselves live in `app.core.plugins`, one plugin class per
tracker; this module holds only what they share.
"""
from __future__ import annotations

from dataclasses import dataclass, field


from contextlib import contextmanager
from typing import Generator, Any

# RFC-5398 style channel-independent object colours (stable across everything).
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (0, 130, 200), (240, 50, 230), (128, 128, 0),
    (250, 190, 212), (35, 35, 220), (170, 255, 195), (0, 0, 128),
]


@dataclass
class Detection:
    """Canonical Detection Data Transfer Object (DTO).
    
    Standard generic interface: [x1, y1, x2, y2, score, class_id] with optional visual features.
    Decouples trackers from specific detector representations (CNN/ViT detectors, SimDetector, etc.).
    """
    box: list[float]  # [x1, y1, x2, y2]
    score: float = 1.0
    cls: int = 0
    feature: Any = None

    @classmethod
    def from_array(cls, arr: Any) -> "Detection":
        """Construct Detection DTO from raw array/slice [x1, y1, x2, y2, score, (optional cls)]."""
        vals = list(arr)
        box = [float(v) for v in vals[:4]]
        score = float(vals[4]) if len(vals) > 4 else 1.0
        cls_id = int(vals[5]) if len(vals) > 5 else 0
        return cls(box=box, score=score, cls=cls_id)

    def to_list(self) -> list[float]:
        return [self.box[0], self.box[1], self.box[2], self.box[3], self.score, float(self.cls)]


@dataclass
class Track:
    id: int
    box: list  # [x1, y1, x2, y2]
    score: float = 1.0
    cls: int = 0
    kind: str = "active"  # active | lost


@dataclass
class TrackerState:
    active: list = field(default_factory=list)
    lost_now: list = field(default_factory=list)


def color_for(track_id: int) -> tuple:
    return _PALETTE[track_id % len(_PALETTE)]


def _iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / uni if uni > 0 else 0.0


def get_compute_device(requested_device: str = "auto") -> str:
    """Safely resolve requested compute device with graceful CPU fallback.
    
    Prevents runtime crashes on non-CUDA / edge environments when torch/cuda is unavailable.
    """
    req = (requested_device or "cpu").lower()
    if req == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@contextmanager
def open_video_writer(path: str, fourcc: int, fps: float, frame_size: tuple[int, int]) -> Generator[Any, None, None]:
    """Context manager for OpenCV VideoWriter ensuring resources are safely closed."""
    import cv2
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    try:
        yield writer
    finally:
        if writer is not None and writer.isOpened():
            writer.release()


@contextmanager
def open_video_capture(source: Any) -> Generator[Any, None, None]:
    """Context manager for OpenCV VideoCapture ensuring camera/file locks are released on exit."""
    import cv2
    cap = cv2.VideoCapture(source)
    try:
        yield cap
    finally:
        if cap is not None and cap.isOpened():
            cap.release()


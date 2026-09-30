"""Shared runtime types every tracking engine speaks.

Whatever its implementation -- homemade baseline, OpenCV follower, ultralytics
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


# RFC-5398 style channel-independent object colours (stable across everything).
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (0, 130, 200), (240, 50, 230), (128, 128, 0),
    (250, 190, 212), (35, 35, 220), (170, 255, 195), (0, 0, 128),
]


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

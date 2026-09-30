"""Report-card arithmetic, with a regression net for the FP/FN double-count.

The bug: the event loops over losses and ghosts each incremented
`total_fp` / `total_fn`, and the MOT accumulation block incremented them again
for the same frames.  Every false positive and every miss was billed twice,
which crushed MOTA for no reason.

A perfect tracker scores 1.0 under both versions, so the test that actually
catches the bug is a *deliberately broken* one, whose `fp`/`fn` must equal the
real unmatched counts.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.metrics import evaluate
from app.core.trackers import Track

TRACKER_ID = "greedy_iou"  # mode="multi", so _eval_multi runs
T = 10
BOX = [10, 10, 40, 40]
GHOST = [500, 500, 540, 540]
GT_TOTAL = T  # one visible object in every frame


class _Scenario:
    def __init__(self, frames: int = T):
        self.meta = {"frames": frames}
        self.gt = [[dict(id=0, box=list(BOX), visible=True, occluded=False)] for _ in range(frames)]


def _state(tracks: list[tuple[int, list]]) -> list[Track]:
    return [Track(i, list(b), score=0.9) for i, b in tracks]


def _evaluate(kind: str):
    """Drive a stub tracker over a scripted clip and score it like the simulator does."""
    track_frames = [_state(_boxes(kind, t)) for t in range(T)]
    dets_frames = [np.array([[*BOX, 0.9]]) for _ in range(T)]
    return evaluate(_Scenario(), TRACKER_ID, track_frames, dets_frames)


def _boxes(kind: str, t: int) -> list[tuple[int, list]]:
    """`kind` decides what the stub reports on frame `t`."""
    if kind == "perfect":
        return [(0, list(BOX))]
    if kind == "blind":
        return []
    if kind == "ghost":
        return [(7, list(GHOST))]
    if kind == "half_dead":
        return [(0, list(BOX))] if t < 5 else []
    if kind == "half_dead_ghost":
        return [(0, list(BOX))] if t < 5 else [(7, list(GHOST))]
    raise AssertionError(kind)


def test_perfect_tracker_scores_a_clean_sheet():
    m = _evaluate("perfect").metrics
    assert m["mota"] == 1.0
    assert m["motp"] == 1.0
    assert m["idsw"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["idf1"] == 1.0
    assert m["gt_total"] == GT_TOTAL


@pytest.mark.parametrize(
    "kind, expected_fp, expected_fn",
    [
        ("blind", 0, GT_TOTAL),        # misses every frame
        ("ghost", GT_TOTAL, GT_TOTAL),  # a wrong box every frame
        ("half_dead", 0, 5),             # tracked for half the clip, then nothing
        ("half_dead_ghost", 5, 5),       # tracked for half, then ghosts
    ],
)
def test_fp_and_fn_count_each_error_exactly_once(kind, expected_fp, expected_fn):
    m = _evaluate(kind).metrics
    assert m["fp"] == expected_fp, f"{kind}: false positives were miscounted"
    assert m["fn"] == expected_fn, f"{kind}: misses were miscounted"
    assert m["gt_total"] == GT_TOTAL
    assert m["mota"] == round(max(0.0, 1.0 - (m["fp"] + m["fn"] + m["idsw"]) / GT_TOTAL), 3)


def test_mota_is_not_double_charged():
    """The headline number: 5 misses out of 10 frames is 0.5, not 0.0."""
    assert _evaluate("half_dead").metrics["mota"] == 0.5


def test_events_still_describe_the_failures():
    """Removing the double-count must not silence the event log."""
    types = [e.type for e in _evaluate("half_dead").events]
    assert "track_loss" in types
    assert "ghost" in [e.type for e in _evaluate("half_dead_ghost").events]

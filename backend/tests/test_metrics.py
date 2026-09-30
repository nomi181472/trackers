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
    def __init__(self, frames: int = T, target: int = 0):
        self.meta = {"frames": frames}
        self.gt = [[dict(id=0, box=list(BOX), visible=True, occluded=False)] for _ in range(frames)]
        self._target = target

    def target_id(self) -> int:
        return self._target


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


# --------------------------------------------------------------------------- #
# Single-object evaluation: longest_correct_run must mean "correct"           #
# --------------------------------------------------------------------------- #

SINGLE_ID = "mil"


def _eval_single(kind: str, gt_visible: list[bool] | None = None, target: int = 0):
    """Score a scripted follower clip. `gt_visible[t]` controls the target."""
    sc = _Scenario()
    if gt_visible is not None:
        sc.gt = [[dict(id=target, box=list(BOX), visible=v, occluded=not v)]
                 for v in gt_visible]
    boxes = {
        "perfect": [list(BOX)] * T,
        "always_wrong": [list(GHOST)] * T,      # reports a box, never on target
        "half_wrong": [list(BOX)] * 5 + [list(GHOST)] * 5,
        "silent": [[]] * T,                      # reports nothing at all
    }[kind]
    track_frames = [[Track(0, list(b), score=0.8)] if b else [] for b in boxes]
    return evaluate(sc, SINGLE_ID, track_frames, [np.zeros((0, 5)) for _ in range(T)])


def test_a_perfect_follower_has_a_perfect_run():
    m = _eval_single("perfect").metrics
    assert m["accuracy"] == 1.0
    assert m["longest_correct_run"] == T
    assert m["lost_frames"] == 0


def test_a_follower_that_is_always_wrong_has_a_run_of_zero():
    """The regression: this used to report `T` and print 'Longest unbroken
    correct run: 90 frames' for a tracker that was never once on target."""
    m = _eval_single("always_wrong").metrics
    assert m["accuracy"] == 0.0
    assert m["correct_frames"] == 0
    assert m["longest_correct_run"] == 0, "a box on the wrong object is not a correct run"


def test_a_follower_that_stops_reporting_also_has_a_run_of_zero():
    m = _eval_single("silent").metrics
    assert m["accuracy"] == 0.0
    assert m["longest_correct_run"] == 0


def test_the_run_counts_only_the_correct_stretch():
    m = _eval_single("half_wrong").metrics
    assert m["correct_frames"] == 5
    assert m["accuracy"] == 0.5
    assert m["longest_correct_run"] == 5, "the trailing wrong stretch must not count"


def test_a_hidden_target_frame_breaks_the_run_and_is_not_a_miss():
    """Unscorable, not wrong: we cannot confirm we are on it."""
    visible = [True] * 4 + [False, False] + [True] * 4
    m = _eval_single("perfect", gt_visible=visible).metrics
    assert m["total_visible"] == 8, "hidden frames stay out of the accuracy denominator"
    assert m["correct_frames"] == 8
    assert m["accuracy"] == 1.0, "hiding the object is not the tracker's fault"
    assert m["longest_correct_run"] == 4, "the hidden gap breaks the streak"


def test_single_eval_scores_the_target_object_not_the_first_visible_one():
    """object 1 is the only visible one, but the follower was seeded on object 0."""
    sc = _Scenario()
    sc.gt = [[dict(id=0, box=list(BOX), visible=False, occluded=True),
              dict(id=1, box=list(GHOST), visible=True, occluded=False)] for _ in range(T)]
    track_frames = [[Track(0, list(BOX), score=0.8)] for _ in range(T)]
    res = evaluate(sc, SINGLE_ID, track_frames, [np.zeros((0, 5)) for _ in range(T)])
    # object 0 never shows, so nothing is scorable at all
    assert res.metrics["total_visible"] == 0
    assert res.metrics["accuracy"] == 0.0
    assert res.metrics["longest_correct_run"] == 0

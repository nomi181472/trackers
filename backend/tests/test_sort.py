"""SORT (Kalman + Hungarian) behaviour, and the claim it exists to support.

SORT is here so that ByteTrack's "second chance for weak detections" becomes
visible: SORT *is* ByteTrack minus that idea.  The tests below pin that contrast
rather than just checking the class builds.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.plugins.custom import (GreedyIoUTracker, SortTracker,
                                      _KalmanBox, _centre)
from app.core.registry import default_params

FAST = dict(min_hits=1, max_age=5, iou_thresh=0.3)
EMPTY = np.zeros((0, 5))


def _engine(**over):
    return SortTracker({**FAST, **over})


def _dets(*boxes, score=0.9):
    """`()` for a frame with no detections; otherwise one box per arg."""
    return np.array([[*b, score] for b in boxes], dtype=np.float64) if boxes \
        else EMPTY


# --------------------------------------------------------------------------- #
# The Kalman filter                                                            #
# --------------------------------------------------------------------------- #

def test_box_conversion_round_trips():
    kf = _KalmanBox.initiate([10, 10, 40, 40])
    assert list(np.round(kf.predict_box(), 6)) == [10.0, 10.0, 40.0, 40.0]


def test_kalman_extrapolates_a_known_velocity():
    """The whole point: after seeing +10px/frame, two predicts land two frames on."""
    kf = _KalmanBox.initiate([0, 0, 20, 20])
    for i in range(6):
        kf.predict()
        kf.update([10 * i, 0, 10 * i + 20, 20])
    kf.predict()
    kf.predict()
    assert kf.predict_box() == pytest.approx([70.0, 0.0, 90.0, 20.0], abs=0.05)


def test_kalman_does_not_explode_on_a_degenerate_box():
    kf = _KalmanBox.initiate([5, 5, 5, 5])  # zero-area box
    kf.predict()
    box = kf.predict_box()
    assert all(np.isfinite(box)) and box[2] > box[0]


# --------------------------------------------------------------------------- #
# Tracking behaviour                                                           #
# --------------------------------------------------------------------------- #

def test_identities_survive_straight_line_motion():
    e = _engine()
    seen = []
    for t in range(10):
        st = e.update(_dets([10 * t, 100, 10 * t + 20, 120],
                            [200 - 4 * t, 100, 220 - 4 * t, 120]))
        seen.append(sorted(tr.id for tr in st.active))
    assert all(ids == [0, 1] for ids in seen), f"ids were not stable: {seen}"


def test_prediction_extrapolates_beyond_the_last_box():
    """The mechanism, isolated: with a learned velocity the filter's prediction
    leads the tracker's last observed box in the direction of travel."""
    e = _engine()
    speed = 10
    for t in range(6):
        e.update(_dets([speed * t, 0, speed * t + 20, 20]))
    last_centre = _centre(e.tracks[0]["box"])[0]
    kf = e._kalman[0]
    kf.predict()
    kf.predict()
    assert _centre(kf.predict_box())[0] == pytest.approx(last_centre + 2 * speed, abs=1.0)


def test_it_keeps_one_identity_across_an_occlusion_that_greedy_cannot():
    """The advantage that actually matters, and the one the default preset exercises.

    A track that vanishes for 5 frames reappears 50px from where it was last
    seen. Greedy IoU matches against the *stale* box, so it finds nothing and
    mints a new id; the Kalman has been predicting the whole way, so it lands
    straight back on identity 0.
    """
    hidden = range(8, 13)

    def frames(cls, **over):
        eng = cls(over)
        seen = []
        for t in range(20):
            dets = EMPTY if t in hidden else _dets([10 * t, 0, 10 * t + 20, 20])
            seen.append([tr.id for tr in eng.update(dets).active])
        return seen

    sort_ids = frames(SortTracker, min_hits=1, max_age=20, iou_thresh=0.3)
    greedy_ids = frames(GreedyIoUTracker, iou_thresh=0.3, max_age=20, keep_last_pos=False)

    assert sort_ids[15] == [0], f"SORT should hold id 0, got {sort_ids[15]}"
    assert greedy_ids[15] != [0], (
        "if greedy also holds id 0, this test is not discriminating")
    assert all(ids == [0] for ids in sort_ids[13:]), "SORT must not have drifted off id 0"
    assert greedy_ids[15][0] != 0, "greedy should have been forced to spend a new id"


def test_min_hits_delays_reporting_a_new_id():
    e = _engine(min_hits=3)
    st1 = e.update(_dets([0, 0, 20, 20]))
    st2 = e.update(_dets([10, 0, 30, 20]))
    assert not st1.active, "unconfirmed tracks must not be reported"
    assert not st2.active
    st3 = e.update(_dets([20, 0, 40, 20]))
    assert len(st3.active) == 1, "the third sighting confirms it"
    assert st3.active[0].id == 0


def test_a_single_missed_frame_costs_the_id_when_memory_is_short():
    """This is precisely the weakness ByteTrack's second chance exists to fix."""
    e = _engine(max_age=1, min_hits=1)
    e.update(_dets([0, 0, 20, 20]))
    e.update(_dets([5, 0, 25, 20]))
    e.update(_dets())  # one blank frame
    st = e.update(_dets([15, 0, 35, 20]))
    assert st.active, "with max_age=1 the track should already be gone"
    assert st.active[0].id != 0, "a new id was minted -- that is the ByteTrack gap"


def test_a_generous_max_age_bridges_the_same_gap():
    e = _engine(max_age=20, min_hits=1)
    e.update(_dets([0, 0, 20, 20]))
    e.update(_dets([5, 0, 25, 20]))
    e.update(_dets())
    st = e.update(_dets([15, 0, 35, 20]))
    assert st.active and st.active[0].id == 0, "prediction should have re-found it"


def test_lost_tracks_are_reported_once_then_go_quiet():
    e = _engine(max_age=20, min_hits=1)
    e.update(_dets([0, 0, 20, 20]))
    e.update(_dets([5, 0, 25, 20]))
    st = e.update(_dets())
    assert len(st.lost_now) == 1, "the UI needs to know the track died"
    assert st.lost_now[0].id == 0
    assert not e.update(_dets()).lost_now, "and only once"


def test_an_empty_frame_keeps_no_active_tracks():
    e = _engine()
    e.update(_dets([0, 0, 20, 20]))
    assert not e.update(_dets()).active


def test_duplicate_detections_do_not_collapse_into_one_track():
    e = _engine(min_hits=1)
    st = e.update(_dets([0, 0, 20, 20], [1, 0, 21, 20]))
    assert len(st.active) == 2, "assignment is one-to-one; both must survive"


def test_it_needs_no_third_party_tracker_library():
    """SORT is the only published algorithm implemented here, so it must not lean
    on third-party tracker frameworks or cv2 -- those would make the 'always available' claim false."""
    import inspect

    import app.core.plugins.custom as custom
    src = inspect.getsource(custom)
    for forbidden in ("external_tracker", "import cv2", "torch"):
        assert forbidden not in src, f"custom.py must not import {forbidden}"


def test_default_params_are_the_papers_vocabulary():
    params = default_params("sort")
    assert set(params) == {"iou_thresh", "min_hits", "max_age"}
    assert params["min_hits"] >= 1
    assert params["max_age"] >= 1


@pytest.mark.parametrize("n_dets", [0, 1, 2, 5])
def test_it_never_raises_regardless_of_input(n_dets):
    e = _engine()
    dets = EMPTY if not n_dets else _dets(*[(10 * i, 0, 20 + 10 * i, 20)
                                             for i in range(n_dets)])
    for _ in range(3):
        e.update(dets)

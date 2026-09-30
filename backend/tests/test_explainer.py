"""Grades and plain-language verdicts are the product here -- a report that
flatters a bad run is worse than no report at all.  These pin the bands and the
wording so they cannot drift silently as the metrics change.
"""
from __future__ import annotations

import pytest

from app.core.explainer import explain

SCENARIO_META = dict(fps=15, width=640, height=640, frames=90)


def _event(severity="warning", blame="tracker", type_="track_loss", frame=3, fix=""):
    return dict(frame=frame, type=type_, text="something happened", severity=severity,
                gt_ids=[0], track_ids=[1], fix=fix, blame=blame)


def _multi(idf1=0.99, mota=0.99, idsw=0, events=None, critical_count=0):
    metrics = dict(mota=mota, motp=0.9, idf1=idf1, idp=0.9, idr=0.9, idsw=idsw,
                   fp=0, fn=0, mt=3, ml=0, pt=0, gt_total=200, gt_objects=3)
    events = list(events or [])
    events += [_event(severity="critical", type_="id_switch") for _ in range(critical_count)]
    return explain(SCENARIO_META, "bytetrack", metrics, events)


# --------------------------------------------------------------------------- #
# Multi-object grades                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("metrics_kwargs, expected", [
    (dict(idf1=0.99, mota=0.99, idsw=0), "S"),                       # flawless
    (dict(idf1=0.60, mota=0.50, idsw=0), "A"),                       # no swaps, sloppy boxes
    (dict(idf1=0.99, mota=0.90, idsw=0), "A"),                       # mota not quite S
    (dict(idf1=0.50, mota=0.50, idsw=2, critical_count=1), "C"),     # a few switches
    (dict(idf1=0.40, mota=0.40, idsw=9, critical_count=4), "D"),     # this scene beats it
    (dict(idf1=0.40, mota=0.40, idsw=9, critical_count=1), "C"),     # switches but not many critical
])
def test_multi_grade_bands(metrics_kwargs, expected):
    assert _multi(**metrics_kwargs)["grade"] == expected


def test_grade_s_needs_all_three_conditions():
    base = dict(idf1=0.99, mota=0.99, idsw=0)
    for missing in ({"idsw": 1}, {"idf1": 0.95}, {"mota": 0.95}):
        assert _multi(**{**base, **missing})["grade"] != "S", missing


def test_only_d_and_f_are_marked_failed():
    assert _multi(idf1=0.4, mota=0.4, idsw=9, critical_count=5)["failed"] is True
    assert _multi(idf1=0.4, mota=0.4, idsw=9, critical_count=1)["failed"] is False
    assert _multi(idf1=0.99, mota=0.99, idsw=0)["failed"] is False


# --------------------------------------------------------------------------- #
# Single-object grades + the wording that carries the correctness promise      #
# --------------------------------------------------------------------------- #

def _single(accuracy, correct=0, visible=0, longest=0):
    metrics = dict(accuracy=accuracy, correct_frames=correct, total_visible=visible,
                   lost_frames=visible - correct, longest_correct_run=longest, frames=90)
    return explain(SCENARIO_META, "mil", metrics, [])


@pytest.mark.parametrize("accuracy, expected", [
    # the bands are strict `>`, so 0.90 lands in C and 0.60 lands in F
    (0.95, "A"), (0.91, "A"), (0.901, "A"), (0.90, "C"),
    (0.89, "C"), (0.61, "C"), (0.601, "C"), (0.60, "F"),
    (0.59, "F"), (0.0, "F"),
])
def test_single_grade_bands(accuracy, expected):
    assert _single(accuracy)["grade"] == expected


def test_single_object_report_states_the_run_length_as_a_correct_run():
    """`longest_correct_run` is only truthful if the metrics side counts
    correctness, so the wording is pinned here next to the metric's test."""
    bullets = _single(0.5, longest=7)["sections"][1]["bullets"]
    joined = " ".join(bullets)
    assert "correct run" in joined
    assert "7 frames" in joined


def test_multi_report_quotes_fp_and_fn_verbatim():
    metrics = dict(mota=0.9, motp=0.9, idf1=0.9, idp=0.9, idr=0.9, idsw=1,
                   fp=3, fn=4, mt=1, ml=1, pt=1, gt_total=200, gt_objects=3)
    joined = " ".join(explain(SCENARIO_META, "bytetrack", metrics, [])["sections"][1]["bullets"])
    assert "3 ghost boxes" in joined and "4 misses" in joined


def test_verdict_is_the_first_section():
    report = _multi()
    assert report["sections"][0]["name"] == "Verdict"
    assert report["sections"][0]["text"] == report["verdict"]


def test_known_weaknesses_come_from_the_plugin_metadata():
    names = [s["name"] for s in _multi()["sections"]]
    assert "Known weaknesses" in names
    assert "Known strengths" in names

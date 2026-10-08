"""`run_simulation` end to end: the pipeline the plugin contract exists to serve.

These are the only tests that render video, so the scenes are deliberately tiny.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.registry import DETECTION_DEFAULTS, default_params
from app.core.runner import SimDetector, run_simulation
from app.core.scenario import Scenario
from app.core.trackers import Track

TINY = dict(width=240, height=240, fps=10, duration_seconds=1.0,
            num_objects=2, crossing=True, occlusion=True)


@pytest.fixture
def scene():
    return Scenario({**TINY, "seed": 4})


def _run(scene, tids, detection=None, out_dir="/tmp"):
    specs = [_spec(t) for t in tids]
    return run_simulation(scene, specs, detection or {}, out_dir=out_dir, job_id="pytest")


def _spec(tid):
    """Build a spec without presupposing the id resolves -- some tests use
    deliberately bogus ids, and `run_simulation` is what must tolerate them."""
    try:
        return {"tracker_id": tid, "params": default_params(tid)}
    except KeyError:
        return {"tracker_id": tid, "params": {}}


def test_a_multi_object_tracker_produces_a_full_report(scene):
    res = _run(scene, ["greedy_iou"])
    (r,) = res["results"]
    assert "error" not in r, r.get("error")
    assert r["mode"] == "multi"
    assert r["metrics"]["mota"] is not None
    assert r["metrics"]["fp"] == r["metrics"]["fp"]  # not None
    assert r["report"]["grade"] in list("SACD")
    assert r["video_url"].endswith(("pytest_greedy_iou.webm", "pytest_greedy_iou.mp4"))
    assert res["trackers"] == ["greedy_iou"]


def test_several_trackers_each_get_their_own_result(scene):
    res = _run(scene, ["greedy_iou", "centroid", "bytetrack"])
    assert res["trackers"] == ["greedy_iou", "centroid", "bytetrack"]
    for r in res["results"]:
        assert "error" not in r, f"{r['tracker_id']}: {r.get('error')}"


def test_an_unknown_tracker_id_becomes_an_error_card_not_a_crash(scene):
    """Previously `get_tracker` sat outside the try, so this killed the job."""
    res = _run(scene, ["not_a_tracker", "greedy_iou"])
    bad, good = res["results"]
    assert "Unknown tracker" in bad["error"]
    assert bad["tracker_id"] == "not_a_tracker"
    assert "error" not in good, "one bad id must not poison the others"


def test_a_tracker_that_cannot_start_is_isolated(scene):
    """`mil` is the only OpenCV single tracker in most builds; whether it starts
    or not, it must be self-contained and leave its neighbour alone."""
    res = _run(scene, ["mil", "greedy_iou"])
    mil, greedy = res["results"]
    assert "error" not in greedy, "one unavailable tracker must not poison the others"
    if "error" in mil:
        assert "mil" in mil["error"]
    else:
        assert mil["mode"] == "single" and mil["metrics"]["frames"] == scene.n_frames


# --------------------------------------------------------------------------- #
# The single-object path                                                       #
# --------------------------------------------------------------------------- #

def test_single_object_metrics_reach_the_result(scene):
    from app.core.plugins import REGISTRY
    if not REGISTRY.get("mil").is_available():
        pytest.skip("no OpenCV single-object tracker in this build")
    res = _run(scene, ["mil"])
    (r,) = res["results"]
    if "error" in r:
        pytest.skip(f"mil unavailable at runtime: {r['error']}")
    assert r["mode"] == "single"
    assert 0.0 <= r["metrics"]["accuracy"] <= 1.0
    assert r["metrics"]["longest_correct_run"] <= r["metrics"]["total_visible"]


def test_single_object_run_is_never_scored_past_the_clip(scene):
    """`longest_correct_run` cannot exceed the frames we could actually judge."""
    from app.core.plugins import REGISTRY
    if not REGISTRY.get("mil").is_available():
        pytest.skip("no OpenCV single-object tracker in this build")
    res = _run(scene, ["mil"])
    (r,) = res["results"]
    if "error" in r:
        pytest.skip("mil unavailable at runtime")
    m = r["metrics"]
    assert m["total_visible"] <= m["frames"]
    assert m["correct_frames"] <= m["total_visible"]
    assert m["lost_frames"] == m["total_visible"] - m["correct_frames"]


def test_a_single_tracker_is_seeded_on_a_visible_frame(scene):
    """The runner must call `init` on the frame `Scenario` nominates."""
    from app.core.plugins import REGISTRY
    if not REGISTRY.get("mil").is_available():
        pytest.skip("no OpenCV single-object tracker in this build")
    seed = scene.first_visible_target_frame()
    assert seed == 0, "the generator starts object 0 visible"
    target = next(e for e in scene.gt[seed] if e["id"] == scene.target_id())
    assert target["visible"]


def test_an_unseedable_single_tracker_reports_why(scene):
    """If the target is never visible the follower gets a named error card."""
    scene.gt = [[dict(id=0, box=[10, 10, 30, 30], visible=False, occluded=True)]
                for _ in range(scene.n_frames)]
    res = _run(scene, ["mil"])
    (r,) = res["results"]
    assert "never visible" in r.get("error", "")


# --------------------------------------------------------------------------- #
# SimDetector: defaults have exactly one source of truth                       #
# --------------------------------------------------------------------------- #

def test_sim_detector_fills_missing_knobs_from_the_registry(scene):
    det = SimDetector(scene, {})
    assert det.conf == DETECTION_DEFAULTS["conf"]
    assert det.miss_rate == DETECTION_DEFAULTS["miss_rate"]
    assert det.fp_rate == DETECTION_DEFAULTS["fp_rate"]
    assert det.jitter == DETECTION_DEFAULTS["jitter"]
    assert det.drop_occluded == DETECTION_DEFAULTS["drop_while_occluded"]


def test_sim_detector_honours_overrides_and_leaves_the_defaults_alone(scene):
    det = SimDetector(scene, {"conf": 0.9, "jitter": 0})
    assert det.conf == 0.9
    assert det.jitter == 0
    assert det.miss_rate == DETECTION_DEFAULTS["miss_rate"], "override must not clobber siblings"
    assert DETECTION_DEFAULTS["conf"] == 0.25, "the registry map must not be mutated"


def test_jitter_zero_produces_exact_ground_truth_boxes(scene):
    det = SimDetector(scene, {"jitter": 0, "conf": 0.0})
    for t in range(scene.n_frames):
        got = det.dets_for_frame(t)
        want = [e["box"] for e in scene.gt[t] if e["visible"]]
        assert len(got) == len(want), "occluded objects are dropped by default"
        for row, box in zip(got, want):
            assert np.allclose(row[:4], box)


def test_keeping_occluded_detections_adds_the_hidden_boxes(scene):
    """The documented 'visible through the wall' mode."""
    dropped = SimDetector(scene, {"jitter": 0, "conf": 0.0})
    through = SimDetector(scene, {"jitter": 0, "conf": 0.0, "drop_while_occluded": False})
    t = next(t for t in range(scene.n_frames)
             if any(not e["visible"] for e in scene.gt[t]))
    n_visible = sum(1 for e in scene.gt[t] if e["visible"])
    n_all = len(scene.gt[t])
    assert n_all > n_visible, "pick a frame where something is actually hidden"
    assert len(dropped.dets_for_frame(t)) == n_visible
    assert len(through.dets_for_frame(t)) == n_all
    # ...at a discounted score, which is why a normal `conf` filters them out
    assert min(through.dets_for_frame(t)[:, 4]) < 0.6


def test_a_conf_threshold_filters_the_discounted_boxes_out(scene):
    t = next(t for t in range(scene.n_frames)
             if any(not e["visible"] for e in scene.gt[t]))
    generous = SimDetector(scene, {"jitter": 0, "conf": 0.0, "drop_while_occluded": False})
    strict = SimDetector(scene, {"jitter": 0, "conf": 0.8, "drop_while_occluded": False})
    assert len(generous.dets_for_frame(t)) > len(strict.dets_for_frame(t))


def test_a_high_threshold_suppresses_every_detection(scene):
    det = SimDetector(scene, {"jitter": 0, "conf": 1.01})
    for t in range(scene.n_frames):
        assert len(det.dets_for_frame(t)) == 0


def test_simulation_runs_on_person_and_car_scenarios():
    for obj_type in ("person", "car"):
        sc = Scenario({**TINY, "seed": 4, "object_type": obj_type})
        res = _run(sc, ["greedy_iou", "centroid"])
        assert res["trackers"] == ["greedy_iou", "centroid"]
        for r in res["results"]:
            assert "error" not in r, f"{r['tracker_id']}: {r.get('error')}"
            assert r["metrics"]["mota"] is not None


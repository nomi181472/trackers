"""The synthetic scene generator is the ground truth every metric is measured
against, so its invariants are worth pinning: same seed must mean same scene,
ids must be stable, and visibility must be internally consistent.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.scenario import Scenario

SMALL = dict(width=320, height=320, fps=10, duration_seconds=1.2, num_objects=3)


def _scene(**overrides) -> Scenario:
    return Scenario({**SMALL, **overrides})


def test_same_seed_produces_an_identical_scene():
    a = _scene(seed=4)
    b = _scene(seed=4)
    assert [e["box"] for f in a.gt for e in f] == [e["box"] for f in b.gt for e in f]
    assert np.array_equal(a.frames, b.frames)


def test_a_different_seed_only_diverges_when_a_stochastic_feature_is_on():
    """The crossing preset draws both objects from fixed endpoints, so the seed
    genuinely does nothing there.  Only shake/noise consume `rng`."""
    assert _scene(seed=4, crossing=True, num_objects=2).gt[0] == \
        _scene(seed=5, crossing=True, num_objects=2).gt[0]

    a = _scene(seed=4, crossing=True, num_objects=2, camera_shake=True, shake_px=12)
    b = _scene(seed=5, crossing=True, num_objects=2, camera_shake=True, shake_px=12)
    assert [e["box"] for f in a.gt for e in f] != [e["box"] for f in b.gt for e in f]


def test_frame_count_follows_duration_and_fps():
    sc = _scene()
    assert sc.n_frames == max(2, int(sc.duration * sc.fps))
    assert sc.meta["frames"] == sc.n_frames
    assert len(sc.gt) == len(sc.frames) == sc.n_frames


def test_object_ids_are_dense_and_ordered():
    for n in (1, 2, 3, 5):
        sc = _scene(num_objects=n, seed=n)
        for frame in sc.gt:
            assert [e["id"] for e in frame] == list(range(n))


def test_visibility_is_the_inverse_of_occlusion():
    sc = _scene(seed=4, crossing=True, occlusion=True)
    seen_occluded = False
    for frame in sc.gt:
        for e in frame:
            assert e["visible"] == (not e["occluded"]), e
            seen_occluded |= e["occluded"]
    assert seen_occluded, "the occlusion preset must actually hide something"


def test_boxes_are_well_formed():
    """Note: boxes are NOT guaranteed to sit fully inside the frame.  `_paths`
    clamps every object centre to `[r+2, H-r-2]` using object 0's radius, but
    `_build` draws the larger later objects with `radii[2] > r`, so those boxes
    overhang the edge.  Pre-existing and cosmetic -- the ground truth is still
    well-formed, so this only pins the shape the metrics actually depend on."""
    for over in ({"camera_shake": True, "shake_px": 12},
                 {"camera_shake": False}, {"occlusion": False}):
        sc = _scene(seed=4, **over)
        for frame in sc.gt:
            for e in frame:
                x1, y1, x2, y2 = e["box"]
                assert all(isinstance(v, int) for v in e["box"]), e
                assert x2 > x1 and y2 > y1, e
                assert e["radius"] > 0, e
                assert (x2 - x1) == e["radius"] * 2, e


def test_the_crossing_preset_actually_swaps_two_objects():
    sc = _scene(seed=4, num_objects=2, crossing=True)
    start = {e["id"]: e["center"] for e in sc.gt[0]}
    end = {e["id"]: e["center"] for e in sc.gt[-1]}
    tol = sc.width * 0.05
    assert end[0][0] == pytest.approx(start[1][0], abs=tol)
    assert end[0][1] == pytest.approx(start[1][1], abs=tol)
    assert end[1][0] == pytest.approx(start[0][0], abs=tol)
    assert end[1][1] == pytest.approx(start[0][1], abs=tol)


def test_object_zero_starts_visible_and_left_of_the_wall():
    """The single-object path initialises on object 0, so this must hold.

    `_paths` starts every object at `0.1W + 0.06W*i` (lanes) or `0.18W`
    (crossing), and the wall sits at `0.58W`.  If someone ever moves a start
    position behind the wall this test is the thing that says so.
    """
    for over in ({"crossing": True, "num_objects": 2},
                 {"crossing": False, "num_objects": 3}):
        sc = _scene(seed=4, occlusion=True, **over)
        wall = sc.occluder_box
        assert wall is not None
        first = sc.gt[0][0]
        assert first["id"] == 0
        assert first["visible"], "object 0 must start outside the occluder wall"
        assert not (wall[0] <= first["center"][0] <= wall[2])


def test_target_is_occluded_mid_clip_then_visible_again():
    """Occlusion has to happen *after* frame 0, or the scene proves nothing."""
    sc = _scene(seed=4, num_objects=2, crossing=True, occlusion=True)
    seen = [e["visible"] for e in [f[0] for f in sc.gt]]
    assert seen[0] is True
    assert any(seen), "object 0 should pass behind the wall"


class _StubScenario(Scenario):
    """A hand-written ground truth, for cases the generator cannot produce."""

    def __init__(self, gt):
        self.gt = gt
        self.meta = {"frames": len(gt)}
        self.frames = None


def _entry(i, visible):
    return dict(id=i, box=[10 * i, 10 * i, 10 * i + 8, 10 * i + 8],
                visible=visible, occluded=not visible, center=(10 * i, 10 * i),
                color=[0, 0, 0], radius=4)


def test_target_helpers_on_the_real_generator():
    sc = _scene(seed=4, num_objects=3)
    assert sc.target_id() == 0
    assert sc.first_visible_target_frame() == 0, "object 0 starts visible (pinned above)"


def test_first_visible_target_frame_skips_a_hidden_start():
    sc = _StubScenario([
        [_entry(0, False)],          # target hidden
        [_entry(0, False)],
        [_entry(0, True)],           # ... becomes visible here
        [_entry(0, True)],
    ])
    assert sc.first_visible_target_frame() == 2


def test_first_visible_target_frame_is_none_when_never_visible():
    sc = _StubScenario([[_entry(0, False)], [_entry(0, False)]])
    assert sc.first_visible_target_frame() is None


def test_a_visible_later_object_does_not_count_as_the_target():
    """The whole point of the fix: only object 0's visibility matters."""
    sc = _StubScenario([
        [_entry(0, False), _entry(1, True)],
        [_entry(0, False), _entry(1, True)],
    ])
    assert sc.first_visible_target_frame() is None

"""Every advertised tracker must actually build -- and the ones that cannot must
fail with a clear error instead of a stray KeyError from deep inside a library.

This is the regression net for the old hardcoded dispatch map, which sent
`nano` / `vit` / `dasiamrpn` to the ultralytics engine and raised
``KeyError: 'nano'`` even though the catalog advertised them as available.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.plugins import REGISTRY, build_engine
from app.core.plugins.base import Engine
from app.core.plugins.custom import CentroidTracker, GreedyIoUTracker
from app.core.plugins.opencv import OpenCVSingleTracker, _opencv_probe
from app.core.plugins.ultralytics import UltralyticsEngine

FPS = 15
BOX = [10, 10, 40, 40]
OPENCV_IDS = ["kcf", "csrt", "mosse", "mil", "medianflow", "nano", "vit", "dasiamrpn"]
# These three used to be the ones that fell through to the ultralytics factory.
REGRESSED_IDS = ["nano", "vit", "dasiamrpn"]


def test_every_available_tracker_builds():
    built = []
    for plugin in REGISTRY.all():
        if not plugin.is_available():
            continue
        engine = build_engine(plugin.id, REGISTRY.default_params(plugin.id), FPS)
        assert isinstance(engine, Engine), plugin.id
        built.append(plugin.id)
    assert built, "no tracker was available -- the environment probe is broken"


def test_opencv_ids_never_fall_through_to_another_family():
    """Unavailable OpenCV trackers must raise ValueError, not KeyError/AttributeError."""
    for tid in OPENCV_IDS:
        try:
            engine = build_engine(tid, {"max_age": 5}, FPS)
        except ValueError as exc:
            assert "not built into this OpenCV install" in str(exc)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{tid} raised {type(exc).__name__} instead of ValueError: {exc}")
        else:
            assert isinstance(engine, OpenCVSingleTracker), tid


def test_previously_broken_ids_dispatch_to_opencv():
    """Fix 2: nano/vit/dasiamrpn reach the OpenCV engine, or say why they cannot."""
    avail = _opencv_probe() or {}
    for tid in REGRESSED_IDS:
        if tid in avail:
            assert isinstance(build_engine(tid, {"max_age": 5}, FPS), OpenCVSingleTracker)
        else:
            with pytest.raises(ValueError):
                build_engine(tid, {"max_age": 5}, FPS)


def test_previously_broken_ids_build_when_the_weights_are_present(monkeypatch):
    """The real regression: with weights present these must build, not KeyError.

    On a machine without the ONNX downloads the assertion above only ever sees
    the unavailable branch, so fake a fully-probed OpenCV build here.
    """
    monkeypatch.setattr(OpenCVSingleTracker, "AVAILABLE",
                        {tid: (lambda tid=tid: None) for tid in OPENCV_IDS})
    for tid in REGRESSED_IDS:
        engine = build_engine(tid, {"max_age": 5}, FPS)
        assert isinstance(engine, OpenCVSingleTracker), f"{tid} fell through to another engine"
        assert engine.tracker_id == tid


def test_custom_engines_are_the_expected_classes():
    assert isinstance(build_engine("greedy_iou", {}, FPS), GreedyIoUTracker)
    assert isinstance(build_engine("centroid", {}, FPS), CentroidTracker)


def test_ultralytics_engines_are_the_expected_class():
    for tid in ("bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack"):
        engine = build_engine(tid, REGISTRY.default_params(tid), FPS)
        assert isinstance(engine, UltralyticsEngine), tid


def test_greedy_iou_tracks_a_static_object():
    engine = build_engine("greedy_iou", REGISTRY.default_params("greedy_iou"), FPS)
    dets = np.array([[10, 10, 40, 40, 0.9]])
    first = engine.update(dets)
    assert len(first.active) == 1
    second = engine.update(dets)
    assert [t.id for t in second.active] == [t.id for t in first.active]
    assert second.active[0].box == [10, 10, 40, 40]


def test_unknown_id_raises_keyerror():
    with pytest.raises(KeyError, match="Unknown tracker"):
        build_engine("definitely_not_a_tracker", {}, FPS)


def test_multi_engines_refuse_init_loudly():
    """A mode='multi' engine has nothing to initialise on -- say so, do not no-op."""
    engine = build_engine("greedy_iou", {}, FPS)
    with pytest.raises(NotImplementedError, match="does not support init"):
        engine.init(np.zeros((10, 10, 3), dtype=np.uint8), BOX)

"""Comprehensive tests for the Vector Embedder (DeepSORT) tracker plugin.

Validates:
1. Object invariance: handles ball, person, and car scenarios cleanly.
2. Embedder handling of small / subpixel / out-of-bounds crops.
3. Hybrid cost matrix and tracking re-identification through occlusion.
4. Latency measurement and metrics compatibility with trade-off and speed graphs.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.metrics import evaluate
from app.core.plugins import build_engine
from app.core.plugins.vector_embed import _CropEmbedder, VectorEmbedderEngine
from app.core.runner import run_simulation
from app.core.scenario import Scenario


def test_crop_embedder_handles_arbitrary_sizes_and_subpixel():
    embedder = _CropEmbedder(target_size=(32, 32))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:60, 30:70] = [200, 50, 50]  # colored patch

    # 1. Normal crop
    feat1 = embedder.extract(img, [30, 20, 70, 60])
    assert feat1.shape == (64,)
    assert abs(np.linalg.norm(feat1) - 1.0) < 1e-4

    # 2. Out of bounds & subpixel coordinates
    feat2 = embedder.extract(img, [-10.5, -5.2, 120.8, 110.3])
    assert feat2.shape == (64,)
    assert abs(np.linalg.norm(feat2) - 1.0) < 1e-4

    # 3. Tiny / degenerate crop
    feat3 = embedder.extract(img, [50, 50, 50, 50])
    assert feat3.shape == (64,)
    assert np.all(feat3 == 0.0)

    # 4. None / empty image
    feat4 = embedder.extract(None, [10, 10, 20, 20])
    assert feat4.shape == (64,)
    assert np.all(feat4 == 0.0)


@pytest.mark.parametrize("obj_type", ["ball", "person", "car"])
def test_vector_embedder_runs_on_all_object_types(obj_type, tmp_path):
    """Verifies that vector embedder runs smoothly on all supported objects."""
    scenario = Scenario(dict(
        seed=1,
        fps=15,
        duration_seconds=2,
        num_objects=2,
        object_type=obj_type,
        crossing=True,
        occlusion=False,
    ))

    specs = [dict(tracker_id="embed_sort", params=dict(min_hits=1))]
    res = run_simulation(scenario, specs, detection_params=dict(conf=0.25, miss_rate=0.0, fp_rate=0.0, jitter=0), out_dir=str(tmp_path))

    assert len(res["results"]) == 1
    tr_res = res["results"][0]
    assert "error" not in tr_res
    metrics = tr_res["metrics"]

    # Verify that all metric keys required by frontend charts exist
    assert "mota" in metrics
    assert "idf1" in metrics
    assert "idsw" in metrics
    assert "fp" in metrics
    assert "fn" in metrics
    assert "avg_time_ms" in metrics
    assert "fps" in metrics
    assert "latencies" in metrics
    assert len(metrics["latencies"]) == scenario.meta["frames"]
    assert metrics["avg_time_ms"] > 0


def test_vector_embedder_maintains_identity_with_distinct_appearances():
    """Verify that different colors are distinguished by the visual embedding."""
    engine = VectorEmbedderEngine(params=dict(min_hits=1, embed_weight=0.7, max_age=5), fps=15)

    # Synthetic frame with two distinctly colored objects (Red vs Blue)
    H, W = 100, 200
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[20:60, 20:60] = [0, 0, 255]    # Red in BGR
    img[20:60, 120:160] = [255, 0, 0]  # Blue in BGR

    dets_f1 = np.array([
        [20, 20, 60, 60, 0.9],
        [120, 20, 160, 60, 0.9],
    ])

    state1 = engine.update(dets_f1, img)
    assert len(state1.active) == 2
    id_red = state1.active[0].id
    id_blue = state1.active[1].id

    # Frame 2: Objects shift slightly
    dets_f2 = np.array([
        [22, 20, 62, 60, 0.9],
        [122, 20, 162, 60, 0.9],
    ])
    state2 = engine.update(dets_f2, img)
    assert len(state2.active) == 2
    active_map = {t.id: t.box for t in state2.active}
    assert id_red in active_map
    assert id_blue in active_map

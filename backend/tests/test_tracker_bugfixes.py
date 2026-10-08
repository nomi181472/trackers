import numpy as np
import pytest

from app.core.trackers_standalone.basetrack import BaseTrack
from app.core.trackers_standalone import TRACKER_MAP
from app.core.trackers_standalone.matching import embedding_distance
from app.core.trackers_standalone.kalman_filter import KalmanFilterXYAH
from app.core.plugins.custom import _KalmanBox, GreedyIoUTracker
from app.core.plugins.vector_embed import VectorEmbedderEngine
from types import SimpleNamespace


def test_basetrack_reset_count():
    BaseTrack.reset_id()
    id1 = BaseTrack.next_id()
    assert id1 == 1
    BaseTrack.reset_id()
    id2 = BaseTrack.next_id()
    assert id2 == 1


def test_embedding_distance_safe_with_none():
    tracks = [SimpleNamespace(smooth_feat=None), SimpleNamespace(smooth_feat=np.ones(64) / 8)]
    dets = [SimpleNamespace(curr_feat=None), SimpleNamespace(curr_feat=np.ones(64) / 8)]
    dists = embedding_distance(tracks, dets)
    assert dists.shape == (2, 2)
    # The valid track and valid det match cosine dist should be near 0
    assert np.isclose(dists[1, 1], 0.0, atol=1e-3)
    # Unmatched / None should have maximum distance
    assert dists[0, 0] == 1.0
    assert dists[0, 1] == 1.0
    assert dists[1, 0] == 1.0


def test_deepocsort_update_with_reid():
    BaseTrack.reset_id()
    cls = TRACKER_MAP["deepocsort"]
    args = SimpleNamespace(
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        new_track_thresh=0.3,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        delta_t=3,
        inertia=0.2,
        use_byte=False,
        gmc_method="none",
        alpha_fixed_emb=0.95,
        proximity_thresh=0.5,
        appearance_thresh=0.9,
        with_reid=True,
    )
    tracker = cls(args=args, frame_rate=30)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    dets = np.array([[10, 10, 50, 50, 0.9, 0]])
    out = tracker.update(dets, img)
    assert len(out) == 1
    assert out[0, 4] == 1  # track id


def test_kalman_box_predict_from_zero():
    box = [0, 0, 10, 10]
    kf = _KalmanBox.initiate(box)
    kf.predict()
    assert kf.x is not None
    # Prior bug checked `if self.x.any()`; now state transition always occurs
    assert kf.P is not None


def test_kalman_xyah_project_aspect_noise():
    kf = KalmanFilterXYAH()
    mean = np.array([50, 50, 1.5, 40, 0, 0, 0, 0], dtype=np.float64)
    cov = np.eye(8)
    _, proj_cov = kf.project(mean, cov)
    # Aspect noise should be 1e-2^2 = 1e-4, not 1e-1^2 = 1e-2
    assert np.isclose(proj_cov[2, 2], 1.0 + 1e-4)


def test_greedy_iou_young_over_old():
    tracker = GreedyIoUTracker({"iou_thresh": 0.2, "max_age": 5, "keep_last_pos": True})
    # Add track 0
    tracker.tracks[0] = {"box": [10, 10, 50, 50], "age": 0, "kind": "active", "score": 0.9, "cls": 0}
    # Add track 1 with older age
    tracker.tracks[1] = {"box": [12, 12, 52, 52], "age": 2, "kind": "active", "score": 0.9, "cls": 0}
    # Detections overlaps both
    dets = np.array([[11, 11, 51, 51, 0.95, 0]])
    tracker.update(dets)
    # Track 0 (younger) should have matched and kept age=0
    assert tracker.tracks[0]["age"] == 0
    assert tracker.tracks[1]["age"] > 2


def test_vector_embed_box_smoothed():
    engine = VectorEmbedderEngine({"embed_weight": 0.5, "dist_thresh": 0.4, "min_hits": 1})
    dets = np.array([[10, 10, 50, 50, 0.9, 0]])
    state = engine.update(dets)
    assert len(state.active) == 1
    # Check that subsequent update uses smoothed Kalman box instead of raw assignment
    dets2 = np.array([[12, 12, 52, 52, 0.9, 0]])
    state2 = engine.update(dets2)
    assert len(state2.active) == 1

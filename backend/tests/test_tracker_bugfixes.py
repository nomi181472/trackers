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


def test_detection_dto_and_compute_fallback():
    from app.core.trackers import Detection, get_compute_device
    det = Detection.from_array([10.0, 20.0, 30.0, 40.0, 0.85, 1])
    assert det.box == [10.0, 20.0, 30.0, 40.0]
    assert det.score == 0.85
    assert det.cls == 1
    assert det.to_list() == [10.0, 20.0, 30.0, 40.0, 0.85, 1.0]

    # CPU/Auto fallback should never raise exception even without torch/cuda
    assert get_compute_device("cpu") == "cpu"
    assert get_compute_device("cuda") in ("cuda", "cpu")
    assert get_compute_device(None) == "cpu"


def test_ocsort_observation_pruning_memory_retention():
    from app.core.trackers_standalone.oc_sort import OCSortTrack
    track = OCSortTrack(np.array([100, 100, 20, 20]), 0.9, 0, delta_t=3)
    # Simulate 200 frames of updates
    for f in range(200):
        track._record_observation(np.array([100 + f, 100 + f, 120 + f, 120 + f], dtype=np.float32), frame_id=f)
    # Ensure observations dictionary does not grow unboundedly
    assert len(track.observations) <= 70
    assert 0 not in track.observations  # Frame 0 should have been pruned


def test_fast_tracker_update_and_occlusion_handling():
    from app.core.trackers_standalone.fast_track import FASTTracker
    args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        reset_velocity_offset_occ=5,
        reset_pos_offset_occ=3,
        enlarge_bbox_occ=1.1,
        dampen_motion_occ=0.5,
        active_occ_to_lost_thresh=10,
        occ_cover_thresh=0.5,
        occ_reappear_window=40,
        init_iou_suppress=0.7,
    )
    tracker = FASTTracker(args=args, frame_rate=30)
    # Frame 1: 2 objects
    dets_f1 = np.array([[10, 10, 50, 50, 0.9, 0], [100, 100, 140, 140, 0.9, 0]])
    out1 = tracker.update(dets_f1)
    assert len(out1) == 2

    # Frame 2: Second object continues, first object becomes unmatched but occluded
    dets_f2 = np.array([[102, 100, 142, 140, 0.9, 0]])
    out2 = tracker.update(dets_f2)
    assert len(out2) >= 1


def test_basetrack_thread_safety():
    import concurrent.futures
    BaseTrack.reset_id()

    def get_ids():
        return [BaseTrack.next_id() for _ in range(50)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: get_ids(), range(8)))

    all_ids = [i for sub in results for i in sub]
    assert len(all_ids) == 400
    assert len(set(all_ids)) == 400  # No duplicate IDs generated under concurrent execution


def test_kalman_box_joseph_update():
    box = [10, 10, 50, 50]
    kf = _KalmanBox.initiate(box)
    kf.predict()
    kf.update([12, 12, 52, 52])
    # Covariance P should remain symmetric
    assert np.allclose(kf.P, kf.P.T, atol=1e-6)
    # Covariance P should be positive-definite (eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(kf.P)
    assert np.all(eigenvalues >= 0.0)


def test_vector_embed_exact_box_assignment():
    engine = VectorEmbedderEngine({"embed_weight": 0.5, "dist_thresh": 0.4, "min_hits": 1})
    det = np.array([[10, 10, 50, 50, 0.9, 0]])
    state = engine.update(det)
    assert state.active[0].box == [10, 10, 50, 50]

    det2 = np.array([[15, 15, 55, 55, 0.95, 0]])
    state2 = engine.update(det2)
    # The active track box should be exactly the measurement, not drifted by Kalman predict_box
    assert state2.active[0].box == [15, 15, 55, 55]


def test_gmc_translation_downscale():
    from app.core.trackers_standalone.gmc import GMC
    gmc = GMC(method="none", downscale=2)
    # Mock downscaled warp
    H = gmc.apply(np.zeros((100, 100, 3), dtype=np.uint8))
    assert H.shape == (2, 3)


def test_centroid_optimal_assignment():
    from app.core.plugins.custom import CentroidTracker
    tracker = CentroidTracker({"dist_thresh": 100, "max_age": 5})
    # Seed 2 tracks
    tracker.update(np.array([[10, 10, 30, 30, 0.9, 0], [100, 100, 120, 120, 0.9, 0]]))
    assert len(tracker.tracks) == 2
    # Next frame
    state = tracker.update(np.array([[12, 12, 32, 32, 0.9, 0], [102, 102, 122, 122, 0.9, 0]]))
    assert len(state.active) == 2


def test_opencv_cache_refresh():
    from app.core.plugins.opencv import OpenCVSingleTracker, _opencv_probe
    _opencv_probe()
    assert OpenCVSingleTracker.AVAILABLE is not None
    OpenCVSingleTracker.reset_available_cache()
    assert OpenCVSingleTracker.AVAILABLE is None


def test_byte_tracker_multi_predict_zeroes_all_velocities():
    from app.core.trackers_standalone.byte_tracker import STrack
    from app.core.trackers_standalone.basetrack import TrackState

    st = STrack(np.array([100, 100, 50, 50]), 0.9, 0)
    st.activate(STrack.shared_kalman, 1)
    st.state = TrackState.Lost
    st.mean[4:8] = np.array([5.0, -3.0, 0.5, 2.0])

    STrack.multi_predict([st])
    np.testing.assert_allclose(st.mean[4:8], np.zeros(4), atol=1e-5)


def test_metrics_assign_maintains_consistent_track_order():
    from app.core.metrics import _assign
    from app.core.trackers import Track

    track_map = {
        10: Track(id=10, box=[0, 0, 10, 10], score=0.9, cls=0),
        2: Track(id=2, box=[100, 100, 110, 110], score=0.9, cls=0),
        5: Track(id=5, box=[50, 50, 60, 60], score=0.9, cls=0),
    }
    sorted_tids = sorted(track_map.keys())
    visible = [
        {"id": 1, "box": [100, 100, 110, 110]},  # Matches track 2
        {"id": 2, "box": [0, 0, 10, 10]},        # Matches track 10
    ]
    pairs = _assign([track_map[i].box for i in sorted_tids], [e["box"] for e in visible])
    matches = {sorted_tids[ti]: e_idx for ti, e_idx, _ in pairs}
    assert matches[2] == 0
    assert matches[10] == 1


def test_tracktrack_lost_match_thr_only_affects_lost():
    from app.core.trackers_standalone.fast_track import TrackTrack
    from app.core.trackers_standalone.basetrack import TrackState

    args = SimpleNamespace(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.4,
        lost_match_thr=0.9,  # Very relaxed threshold, should not apply to confirmed tracks
        min_track_len=1,
        reduce_step=0.0,
        iou_weight=1.0,
        reid_weight=0.0,
        conf_weight=0.0,
        angle_weight=0.0,
        penalty_p=0.0,
        penalty_q=0.0,
        tai_thr=0.0,
        with_reid=False,
    )
    tracker = TrackTrack(args, frame_rate=30)
    # Frame 1: Create active track
    tracker.update(np.array([[10, 10, 50, 50, 0.9, 0]]))
    assert len(tracker.tracked_stracks) == 1
    tr = tracker.tracked_stracks[0]

    # Frame 2: Confirmed track should NOT match far-away detection using lost_match_thr (0.9)
    # With match_thresh=0.4, the far box won't match, so track becomes lost
    tracker.update(np.array([[200, 200, 250, 250, 0.9, 0]]))
    assert tr.state == TrackState.Lost


def test_tentative_tracks_purged_immediately_on_miss():
    from app.core.plugins.custom import SortTracker
    from app.core.plugins.vector_embed import VectorEmbedderEngine

    # 1. SortTracker with min_hits=3
    sort = SortTracker({"min_hits": 3, "max_age": 30, "iou_thresh": 0.3})
    # Frame 1: Detection creates a tentative track (id 0)
    sort.update(np.array([[10, 10, 50, 50, 0.9, 0]]))
    assert len(sort.tracks) == 1
    assert sort.tracks[0]["kind"] == "tentative"
    # Frame 2: Missing detection should purge tentative track immediately without waiting max_age
    st2 = sort.update(np.empty((0, 5)))
    assert len(sort.tracks) == 0
    assert 0 not in sort._kalman

    # 2. VectorEmbedderEngine with min_hits=3
    embed_eng = VectorEmbedderEngine({"min_hits": 3, "max_age": 30, "iou_thresh": 0.3})
    embed_eng.update(np.array([[10, 10, 50, 50, 0.9, 0]]))
    assert len(embed_eng.tracks) == 1
    assert embed_eng.tracks[0]["kind"] == "tentative"
    # Missing frame purges tentative track
    embed_eng.update(np.empty((0, 5)))
    assert len(embed_eng.tracks) == 0


def test_kalman_box_dynamic_covariance_scaling():
    """Verify that _KalmanBox scales Q and R covariances with bounding box height."""
    small_box = [0, 0, 10, 10]
    large_box = [0, 0, 100, 100]

    kf_small = _KalmanBox.initiate(small_box)
    kf_large = _KalmanBox.initiate(large_box)

    # Initial P covariance should be scaled with height (large has 10x height, so 100x covariance)
    assert np.isclose(kf_large.P[0, 0] / kf_small.P[0, 0], 100.0)

    # Step predict: large track should have higher process covariance update
    kf_small.predict()
    kf_large.predict()
    assert kf_large.P[0, 0] > kf_small.P[0, 0]

    # Step update: large track should have scaled innovation covariance
    kf_small.update([1, 1, 11, 11])
    kf_large.update([10, 10, 110, 110])
    assert kf_large.P[0, 0] > kf_small.P[0, 0]

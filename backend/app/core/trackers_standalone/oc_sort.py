"""Standalone OC-SORT and DeepOC-SORT in pure NumPy."""
from __future__ import annotations
from typing import Any
import numpy as np

from .basetrack import BaseTrack, TrackState
from .byte_tracker import BYTETracker, STrack
from .kalman_filter import KalmanFilterXYAH
from .gmc import GMC
from .embedder import SimulatorCropEmbedder, smooth_feature
from . import matching
from .stracks import joint_stracks, merge_track_pools, remove_duplicate_stracks, sub_stracks, xywh2ltwh


class OCSortTrack(STrack):
    def __init__(self, xywh: np.ndarray, score: float, cls: Any, delta_t: int = 3):
        super().__init__(xywh, score, cls)
        self.last_observation = np.array([-1, -1, -1, -1], dtype=np.float32)
        self.observations: dict[int, np.ndarray] = {}
        self.velocity: np.ndarray | None = None
        self.delta_t = delta_t
        self._saved_mean: np.ndarray | None = None
        self._saved_covariance: np.ndarray | None = None

    def activate(self, kalman_filter, frame_id: int):
        super().activate(kalman_filter, frame_id)
        self._record_observation(self.xyxy.astype(np.float32), frame_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()

    def update(self, new_track: STrack, frame_id: int):
        obs = new_track.xyxy.astype(np.float32)
        self._record_observation(obs, frame_id)
        super().update(new_track, frame_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        obs = new_track.xyxy.astype(np.float32)
        self._record_observation(obs, frame_id)
        super().re_activate(new_track, frame_id, new_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()

    def _record_observation(self, obs: np.ndarray, frame_id: int):
        self.observations[frame_id] = obs
        self.last_observation = obs
        prev_fid = frame_id - self.delta_t
        if prev_fid in self.observations:
            prev_obs = self.observations[prev_fid]
            cx_curr = (obs[0] + obs[2]) / 2
            cy_curr = (obs[1] + obs[3]) / 2
            cx_prev = (prev_obs[0] + prev_obs[2]) / 2
            cy_prev = (prev_obs[1] + prev_obs[3]) / 2
            self.velocity = np.array([cx_curr - cx_prev, cy_curr - cy_prev], dtype=np.float32)


class OCSORT(BYTETracker):
    def __init__(self, args, frame_rate: int = 30):
        super().__init__(args, frame_rate)
        self.delta_t = getattr(args, "delta_t", 3)
        self.inertia = getattr(args, "inertia", 0.2)
        self.use_byte = getattr(args, "use_byte", False)

    def update(self, dets: np.ndarray, img: np.ndarray = None) -> np.ndarray:
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(dets) == 0:
            scores = np.empty(0, dtype=np.float32)
            bboxes = np.empty((0, 4), dtype=np.float32)
            classes = np.empty(0, dtype=int)
            indices = np.empty(0, dtype=int)
        else:
            dets = np.asarray(dets, dtype=np.float64)
            scores = dets[:, 4]
            bboxes = dets[:, :4]
            classes = dets[:, 5] if dets.shape[1] > 5 else np.zeros(len(dets), dtype=int)
            indices = np.arange(len(dets))

        remain_inds = scores >= self.track_high_thresh
        inds_low = scores > self.track_low_thresh
        inds_high = scores < self.track_high_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_first = bboxes[remain_inds]
        dets_second = bboxes[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_first = classes[remain_inds]
        cls_second = classes[inds_second]
        idx_first = indices[remain_inds]
        idx_second = indices[inds_second]

        def _to_xywh_with_idx(boxes, idxs):
            if len(boxes) == 0:
                return np.empty((0, 5), dtype=np.float32)
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            cx = boxes[:, 0] + w / 2
            cy = boxes[:, 1] + h / 2
            return np.stack([cx, cy, w, h, idxs], axis=1)

        detections = [
            OCSortTrack(box, score, c, self.delta_t)
            for (box, score, c) in zip(_to_xywh_with_idx(dets_first, idx_first), scores_keep, cls_first)
        ]

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        # Cost matrix: IoU with inertia penalization for direction flips
        dists = matching.iou_distance(strack_pool, detections)
        if self.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.inertia > 0 and len(strack_pool) and len(detections):
            for i, tr in enumerate(strack_pool):
                if tr.velocity is not None and np.linalg.norm(tr.velocity) > 1e-3:
                    v_dir = tr.velocity / np.linalg.norm(tr.velocity)
                    for j, det in enumerate(detections):
                        cx_t = (tr.xyxy[0] + tr.xyxy[2]) / 2
                        cy_t = (tr.xyxy[1] + tr.xyxy[3]) / 2
                        cx_d = (det.xyxy[0] + det.xyxy[2]) / 2
                        cy_d = (det.xyxy[1] + det.xyxy[3]) / 2
                        d_vec = np.array([cx_d - cx_t, cy_d - cy_t])
                        d_norm = np.linalg.norm(d_vec)
                        if d_norm > 1e-3:
                            cos_sim = np.dot(v_dir, d_vec / d_norm)
                            if cos_sim < 0:  # moving opposite to historical velocity
                                dists[i, j] += self.inertia * (-cos_sim)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if self.use_byte:
            detections_second = [
                OCSortTrack(box, score, c, self.delta_t)
                for (box, score, c) in zip(_to_xywh_with_idx(dets_second, idx_second), scores_second, cls_second)
            ]
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            matches, u_track_second, _ = matching.linear_assignment(dists, thresh=0.5)

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            for it in u_track_second:
                track = r_tracked_stracks[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            for it in u_track:
                track = strack_pool[it]
                if track.state != TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

        # Unconfirmed
        detections_rem = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections_rem)
        if self.fuse_score:
            dists = matching.fuse_score(dists, detections_rem)
        matches, u_unconfirmed, u_detection_rem = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_rem[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # New
        for inew in u_detection_rem:
            track = detections_rem[inew]
            if track.score >= self.new_track_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        merge_track_pools(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        out = []
        for t in output_stracks:
            box = t.xyxy
            out.append([box[0], box[1], box[2], box[3], t.track_id, t.score, t.cls, t.idx])
        return np.asarray(out, dtype=np.float64) if len(out) else np.empty((0, 8), dtype=np.float64)


class DeepOCSortTrack(OCSortTrack):
    def __init__(self, xywh: np.ndarray, score: float, cls: Any, delta_t: int = 3, feat=None, alpha_fixed_emb=0.95):
        super().__init__(xywh, score, cls, delta_t)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha_fixed_emb = alpha_fixed_emb
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray):
        feat = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat /= norm
        self.curr_feat = feat
        self.smooth_feat = smooth_feature(self.smooth_feat, feat, self.alpha_fixed_emb)

    def update(self, new_track: DeepOCSortTrack, frame_id: int):
        super().update(new_track, frame_id)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    def re_activate(self, new_track: DeepOCSortTrack, frame_id: int, new_id: bool = False):
        super().re_activate(new_track, frame_id, new_id)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)


class DeepOCSORT(OCSORT):
    def __init__(self, args, frame_rate: int = 30):
        super().__init__(args, frame_rate)
        self.with_reid = getattr(args, "with_reid", False)
        self.proximity_thresh = getattr(args, "proximity_thresh", 0.5)
        self.appearance_thresh = getattr(args, "appearance_thresh", 0.9)
        self.alpha_fixed_emb = getattr(args, "alpha_fixed_emb", 0.95)
        self.gmc = GMC(method=getattr(args, "gmc_method", "none"))
        self.embedder = SimulatorCropEmbedder() if self.with_reid else None

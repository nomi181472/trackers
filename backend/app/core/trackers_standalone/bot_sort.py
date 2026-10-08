"""Standalone BoT-SORT implementation in pure NumPy with GMC and visual embeddings."""
from __future__ import annotations
from typing import Any
import numpy as np

from .basetrack import BaseTrack, TrackState
from .byte_tracker import BYTETracker, STrack
from .kalman_filter import KalmanFilterXYWH
from .gmc import GMC
from .embedder import SimulatorCropEmbedder, smooth_feature
from . import matching
from .stracks import joint_stracks, merge_track_pools, remove_duplicate_stracks, sub_stracks, xywh2ltwh


class BOTrack(STrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, xywh: np.ndarray, score: float, cls: int, feat: np.ndarray | None = None):
        super().__init__(xywh, score, cls)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray):
        feat = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat /= norm
        self.curr_feat = feat
        self.smooth_feat = smooth_feature(self.smooth_feat, feat, self.alpha)

    def update(self, new_track: BOTrack, frame_id: int):
        super().update(new_track, frame_id)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    def re_activate(self, new_track: BOTrack, frame_id: int, new_id: bool = False):
        super().re_activate(new_track, frame_id, new_id)
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    def convert_coords(self, tlwh):
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks: list[BOTrack]):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov


class BOTSORT(BYTETracker):
    def __init__(self, args, frame_rate: int = 30):
        super().__init__(args, frame_rate)
        self.proximity_thresh = getattr(args, "proximity_thresh", 0.5)
        self.appearance_thresh = getattr(args, "appearance_thresh", 0.8)
        self.with_reid = getattr(args, "with_reid", False)
        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))
        self.embedder = SimulatorCropEmbedder() if self.with_reid else None
        self.kalman_filter = KalmanFilterXYWH()

    def update(self, dets: np.ndarray, img: np.ndarray = None) -> np.ndarray:
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. Apply GMC camera motion compensation to kalman states
        if img is not None and self.gmc.method not in (None, "none"):
            warp = self.gmc.apply(img)
            R = warp[:2, :2]
            t = warp[:2, 2]
            for tr in joint_stracks(self.tracked_stracks, self.lost_stracks):
                if tr.mean is not None:
                    tr.mean[:2] = np.dot(R, tr.mean[:2]) + t
                    tr.mean[4:6] = np.dot(R, tr.mean[4:6])

        # 2. Extract features if with_reid
        features = None
        if len(dets) > 0 and self.with_reid and img is not None:
            features = [self.embedder.extract(img, d[:4]) for d in dets]

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

        feats_first = [features[i] for i in np.where(remain_inds)[0]] if features else [None] * len(dets_first)
        feats_second = [features[i] for i in np.where(inds_second)[0]] if features else [None] * len(dets_second)

        detections = [
            BOTrack(box, score, c, f)
            for (box, score, c, f) in zip(_to_xywh_with_idx(dets_first, idx_first), scores_keep, cls_first, feats_first)
        ]

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        BOTrack.multi_predict(strack_pool)

        # First match with IoU + optional ReID
        dists = matching.iou_distance(strack_pool, detections)
        if self.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.with_reid and len(strack_pool) and len(detections):
            emb_dists = matching.embedding_distance(strack_pool, detections)
            # ReID gating: only consider appearance when proximate
            raw_ious = 1.0 - dists
            mask = raw_ious < self.proximity_thresh
            dists = np.where(mask, np.minimum(dists, emb_dists), dists)

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

        # Second match with low score pool
        detections_second = [
            BOTrack(box, score, c, f)
            for (box, score, c, f) in zip(_to_xywh_with_idx(dets_second, idx_second), scores_second, cls_second, feats_second)
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

        # Unconfirmed tracks
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

        # New tracks
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

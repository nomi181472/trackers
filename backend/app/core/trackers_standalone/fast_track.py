"""Standalone FastTracker and TrackTrack in pure NumPy."""
from __future__ import annotations
from collections import deque
from typing import Any
import numpy as np

from .basetrack import BaseTrack, TrackState
from .byte_tracker import BYTETracker, STrack
from .kalman_filter import KalmanFilterXYAH, KalmanFilterXYWH
from .gmc import GMC
from .embedder import SimulatorCropEmbedder, smooth_feature
from . import matching
from .stracks import joint_stracks, merge_track_pools, remove_duplicate_stracks, sub_stracks, xywh2ltwh


class FastSTrack(STrack):
    def __init__(self, xywh: np.ndarray, score: float, cls: Any, history_len: int = 16):
        super().__init__(xywh, score, cls)
        self.mean_history: deque = deque(maxlen=history_len)
        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
        self.last_occluded_frame = -1
        self.was_recently_occluded = False

    def activate(self, kalman_filter, frame_id: int):
        super().activate(kalman_filter, frame_id)
        self.mean_history.append((self.mean.copy(), self.covariance.copy()))

    def update(self, new_track: STrack, frame_id: int):
        super().update(new_track, frame_id)
        self.mean_history.append((self.mean.copy(), self.covariance.copy()))
        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0

    def mark_lost(self):
        super().mark_lost()
        self.not_matched += 1
        if self.is_occluded:
            self.was_recently_occluded = True


class FASTTracker(BYTETracker):
    def __init__(self, args, frame_rate: int = 30):
        super().__init__(args, frame_rate)
        self.reset_velocity_offset_occ = getattr(args, "reset_velocity_offset_occ", 5)
        self.reset_pos_offset_occ = getattr(args, "reset_pos_offset_occ", 3)
        self.enlarge_bbox_occ = getattr(args, "enlarge_bbox_occ", 1.1)
        self.dampen_motion_occ = getattr(args, "dampen_motion_occ", 0.5)
        self.active_occ_to_lost_thresh = getattr(args, "active_occ_to_lost_thresh", 10)
        self.occ_cover_thresh = getattr(args, "occ_cover_thresh", 0.7)
        self.occ_reappear_window = getattr(args, "occ_reappear_window", 40)
        self.init_iou_suppress = getattr(args, "init_iou_suppress", 0.7)


class TrackTrackSTrack(STrack):
    def __init__(self, xywh: np.ndarray, score: float, cls: Any, feat: np.ndarray | None = None):
        super().__init__(xywh, score, cls)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9
        self.tracklet_len = 0
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray):
        feat = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat /= norm
        self.curr_feat = feat
        self.smooth_feat = smooth_feature(self.smooth_feat, feat, self.alpha)

    def activate(self, kalman_filter, frame_id: int):
        super().activate(kalman_filter, frame_id)
        self.tracklet_len = 1

    def update(self, new_track: STrack, frame_id: int):
        super().update(new_track, frame_id)
        self.tracklet_len += 1
        if hasattr(new_track, "curr_feat") and new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        super().re_activate(new_track, frame_id, new_id)
        self.tracklet_len += 1
        if hasattr(new_track, "curr_feat") and new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)


class TrackTrack(BYTETracker):
    def __init__(self, args, frame_rate: int = 30):
        super().__init__(args, frame_rate)
        self.min_track_len = getattr(args, "min_track_len", 3)
        self.reduce_step = getattr(args, "reduce_step", 0.05)
        self.iou_weight = getattr(args, "iou_weight", 0.5)
        self.reid_weight = getattr(args, "reid_weight", 0.5)
        self.conf_weight = getattr(args, "conf_weight", 0.1)
        self.angle_weight = getattr(args, "angle_weight", 0.05)
        self.penalty_p = getattr(args, "penalty_p", 0.2)
        self.penalty_q = getattr(args, "penalty_q", 0.4)
        self.lost_match_thr = getattr(args, "lost_match_thr", 0.0)
        self.tai_thr = getattr(args, "tai_thr", 0.55)
        self.with_reid = getattr(args, "with_reid", False)
        self.embedder = SimulatorCropEmbedder() if self.with_reid else None
        gmc_method = getattr(args, "gmc_method", "sparseOptFlow")
        self.gmc = GMC(method=gmc_method)

    def _calc_multi_cue_cost(self, tracks: list[TrackTrackSTrack], dets: list[TrackTrackSTrack],
                            is_lost: bool = False, penalty: float = 0.0) -> np.ndarray:
        if len(tracks) == 0 or len(dets) == 0:
            return np.empty((len(tracks), len(dets)), dtype=np.float64)

        iou_dists = matching.iou_distance(tracks, dets)
        cost = self.iou_weight * iou_dists

        if self.with_reid:
            reid_dists = matching.embedding_distance(tracks, dets)
            cost += self.reid_weight * reid_dists

        # Detection confidence cue (higher confidence -> lower association cost)
        det_scores = np.array([d.score for d in dets], dtype=np.float64)
        conf_cost = 1.0 - det_scores[None, :]
        cost += self.conf_weight * conf_cost

        # Corner geometry / aspect angle cue
        if self.angle_weight > 0:
            tr_boxes = np.array([t.xyxy for t in tracks], dtype=np.float64)
            dt_boxes = np.array([d.xyxy for d in dets], dtype=np.float64)
            tr_ar = (tr_boxes[:, 2] - tr_boxes[:, 0]) / np.maximum(1e-6, tr_boxes[:, 3] - tr_boxes[:, 1])
            dt_ar = (dt_boxes[:, 2] - dt_boxes[:, 0]) / np.maximum(1e-6, dt_boxes[:, 3] - dt_boxes[:, 1])
            ar_diff = np.abs(tr_ar[:, None] - dt_ar[None, :]) / np.maximum(1e-6, tr_ar[:, None] + dt_ar[None, :])
            cost += self.angle_weight * np.clip(ar_diff, 0.0, 1.0)

        # Normalize weights
        total_w = self.iou_weight + (self.reid_weight if self.with_reid else 0.0) + self.conf_weight + self.angle_weight
        if total_w > 0:
            cost /= total_w

        if is_lost and penalty == 0.0 and self.penalty_q > 0:
            penalty = self.penalty_q

        if penalty > 0:
            cost = cost + penalty

        return cost

    def _iterative_assignment(self, tracks: list[TrackTrackSTrack], dets: list[TrackTrackSTrack],
                              base_thresh: float, reduce_step: float,
                              is_lost: bool = False, penalty: float = 0.0):
        if len(tracks) == 0 or len(dets) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(tracks))), list(range(len(dets)))

        cost_matrix = self._calc_multi_cue_cost(tracks, dets, is_lost=is_lost, penalty=penalty)
        curr_thresh = base_thresh
        matched_tracks = set()
        matched_dets = set()
        all_matches = []

        step = max(0.01, reduce_step) if reduce_step > 0 else 0.0
        n_rounds = 4 if reduce_step > 0 else 1

        for _ in range(n_rounds):
            avail_t = [i for i in range(len(tracks)) if i not in matched_tracks]
            avail_d = [j for j in range(len(dets)) if j not in matched_dets]
            if not avail_t or not avail_d:
                break

            sub_cost = cost_matrix[np.ix_(avail_t, avail_d)]
            matches, _, _ = matching.linear_assignment(sub_cost, thresh=curr_thresh)
            for r_idx, c_idx in matches:
                orig_t = avail_t[r_idx]
                orig_d = avail_d[c_idx]
                matched_tracks.add(orig_t)
                matched_dets.add(orig_d)
                all_matches.append((orig_t, orig_d))

            curr_thresh += step

        u_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        u_dets = [j for j in range(len(dets)) if j not in matched_dets]
        return np.asarray(all_matches, dtype=int) if len(all_matches) else np.empty((0, 2), dtype=int), u_tracks, u_dets

    def update(self, dets: np.ndarray, img: np.ndarray = None) -> np.ndarray:
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # GMC camera motion compensation
        if img is not None:
            warp = self.gmc.apply(img, dets)
            if warp is not None:
                R = warp[:2, :2]
                t = warp[:2, 2]
                for tr in self.tracked_stracks + self.lost_stracks:
                    if tr.mean is not None:
                        tr.mean[:2] = np.dot(R, tr.mean[:2]) + t
                        tr.mean[4:6] = np.dot(R, tr.mean[4:6])

        # Feature extraction if ReID enabled
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
            TrackTrackSTrack(box, score, c, f)
            for (box, score, c, f) in zip(_to_xywh_with_idx(dets_first, idx_first), scores_keep, cls_first, feats_first)
        ]

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Distinguish confirmed tracks vs young tracks
        confirmed_tracked = [t for t in tracked_stracks if getattr(t, "tracklet_len", 0) >= self.min_track_len]
        unconfirmed_tracked = [t for t in tracked_stracks if getattr(t, "tracklet_len", 0) < self.min_track_len]

        strack_pool = joint_stracks(confirmed_tracked, self.lost_stracks)
        TrackTrackSTrack.multi_predict(strack_pool)
        TrackTrackSTrack.multi_predict(unconfirmed_tracked)

        base_thresh = self.lost_match_thr if (self.lost_match_thr > 0 and len(self.lost_stracks) > 0) else self.match_thresh
        matches, u_track, u_detection = self._iterative_assignment(
            strack_pool, detections,
            base_thresh=base_thresh,
            reduce_step=self.reduce_step
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 2: Second-chance matching for low-confidence detections with penalty_p
        detections_second = [
            TrackTrackSTrack(box, score, c, f)
            for (box, score, c, f) in zip(_to_xywh_with_idx(dets_second, idx_second), scores_second, cls_second, feats_second)
        ]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        matches, u_track_second, _ = self._iterative_assignment(
            r_tracked_stracks, detections_second,
            base_thresh=0.5,
            reduce_step=self.reduce_step,
            penalty=self.penalty_p
        )

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

        # Step 3: Match remaining unconfirmed / young tracks with remaining high-conf detections
        detections_rem = [detections[i] for i in u_detection]
        young_pool = unconfirmed + unconfirmed_tracked
        if young_pool and detections_rem:
            dists = matching.iou_distance(young_pool, detections_rem)
            matches_young, u_young, u_detection_rem_idx = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches_young:
                young_pool[itracked].update(detections_rem[idet], self.frame_id)
                activated_stracks.append(young_pool[itracked])
            for it in u_young:
                track = young_pool[it]
                if track in unconfirmed:
                    track.mark_removed()
                    removed_stracks.append(track)
            detections_new = [detections_rem[i] for i in u_detection_rem_idx]
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed_stracks.append(track)
            detections_new = detections_rem

        # Step 4: Track-Aware-Init (TAI) & new tracks initialization
        for inew_track in detections_new:
            if inew_track.score < self.new_track_thresh:
                continue
            # Suppress if heavily overlaps with existing active tracks
            if self.tai_thr > 0 and len(activated_stracks) > 0:
                ious = matching.bbox_ious([inew_track.xyxy], [t.xyxy for t in activated_stracks])
                if np.any(ious > self.tai_thr):
                    continue
            inew_track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(inew_track)

        # Step 5: Clean up old lost tracks
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


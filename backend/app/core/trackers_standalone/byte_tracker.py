"""Standalone ByteTrack implementation in pure NumPy.
No external deep learning or C-extension requirements.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilterXYAH
from . import matching
from .stracks import joint_stracks, merge_track_pools, remove_duplicate_stracks, sub_stracks, xywh2ltwh


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: np.ndarray, score: float, cls: Any):
        super().__init__()
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = float(score)
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1] if len(xywh) > 4 else 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[4:8] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[STrack]):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i, 4:8] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

    def update(self, new_track: STrack, frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(1e-6, ret[3])
        return ret


class BYTETracker:
    def __init__(self, args, frame_rate: int = 30):
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.args = args
        self.track_high_thresh = getattr(args, "track_high_thresh", 0.25)
        self.track_low_thresh = getattr(args, "track_low_thresh", 0.1)
        self.new_track_thresh = getattr(args, "new_track_thresh", 0.25)
        self.match_thresh = getattr(args, "match_thresh", 0.8)
        self.fuse_score = getattr(args, "fuse_score", True)
        self.max_time_lost = int(frame_rate / 30.0 * getattr(args, "track_buffer", 30))
        self.kalman_filter = KalmanFilterXYAH()

    def update(self, dets: np.ndarray, img: np.ndarray = None) -> np.ndarray:
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Parse detections (N, 5) -> [x1, y1, x2, y2, score, (cls), idx]
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

        dets_second = bboxes[inds_second]
        dets_first = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_first = classes[remain_inds]
        cls_second = classes[inds_second]
        idx_first = indices[remain_inds]
        idx_second = indices[inds_second]

        # Convert xyxy to xywh center-size for STrack
        def _to_xywh_with_idx(boxes, idxs):
            if len(boxes) == 0:
                return np.empty((0, 5), dtype=np.float32)
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            cx = boxes[:, 0] + w / 2
            cy = boxes[:, 1] + h / 2
            return np.stack([cx, cy, w, h, idxs], axis=1)

        detections = [
            STrack(box, score, c)
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

        # Step 1: Match high-confidence detections
        dists = matching.iou_distance(strack_pool, detections)
        if self.fuse_score:
            dists = matching.fuse_score(dists, detections)
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

        # Step 2: Match second-chance low-confidence detections
        detections_second = [
            STrack(box, score, c)
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

        # Step 3: Match unconfirmed tracks with remaining high-conf detections
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

        # Step 4: Initialize new tracks for unmatched high-confidence detections
        for inew in u_detection_rem:
            track = detections_rem[inew]
            if track.score >= self.new_track_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        # Step 5: Mark old lost tracks as removed
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        merge_track_pools(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks)

        # Format output rows [x1, y1, x2, y2, track_id, score, cls, idx]
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        out = []
        for t in output_stracks:
            box = t.xyxy
            out.append([box[0], box[1], box[2], box[3], t.track_id, t.score, t.cls, t.idx])
        return np.asarray(out, dtype=np.float64) if len(out) else np.empty((0, 8), dtype=np.float64)

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


class TrackTrack(BYTETracker):
    def __init__(self, args, frame_rate: int = 30):
        super().__init__(args, frame_rate)
        self.min_track_len = getattr(args, "min_track_len", 3)
        self.reduce_step = getattr(args, "reduce_step", 0.05)
        self.iou_weight = getattr(args, "iou_weight", 0.5)
        self.reid_weight = getattr(args, "reid_weight", 0.5)
        self.with_reid = getattr(args, "with_reid", False)
        self.embedder = SimulatorCropEmbedder() if self.with_reid else None

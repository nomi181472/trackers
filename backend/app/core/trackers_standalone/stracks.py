"""Track manipulation utilities and coordinate conversions."""
from __future__ import annotations
import numpy as np
from .basetrack import TrackState
from . import matching


def xywh2ltwh(x):
    """Convert xywh [cx, cy, w, h] to ltwh [x1, y1, w, h]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    return y


def xyxy2ltwh(x):
    """Convert xyxy [x1, y1, x2, y2] to ltwh [x1, y1, w, h]."""
    y = np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def xyxy2xywh(x):
    """Convert xyxy [x1, y1, x2, y2] to xywh [cx, cy, w, h]."""
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def joint_stracks(tlista: list, tlistb: list) -> list:
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: list, tlistb: list) -> list:
    tracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        tid = t.track_id
        if tracks.get(tid, 0):
            del tracks[tid]
    return list(tracks.values())


def remove_duplicate_stracks(stracksa: list, stracksb: list) -> tuple[list, list]:
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


def merge_track_pools(
    tracker,
    activated: list,
    refind: list,
    lost: list,
    removed: list,
    removed_buffer: int = 1000,
) -> None:
    tracker.tracked_stracks = [t for t in tracker.tracked_stracks if t.state == TrackState.Tracked]
    tracker.tracked_stracks = joint_stracks(tracker.tracked_stracks, activated)
    tracker.tracked_stracks = joint_stracks(tracker.tracked_stracks, refind)
    tracker.lost_stracks = sub_stracks(tracker.lost_stracks, tracker.tracked_stracks)
    tracker.lost_stracks.extend(lost)
    tracker.lost_stracks = sub_stracks(tracker.lost_stracks, tracker.removed_stracks)
    tracker.tracked_stracks, tracker.lost_stracks = remove_duplicate_stracks(
        tracker.tracked_stracks, tracker.lost_stracks
    )
    tracker.removed_stracks_frame = removed
    tracker.removed_stracks.extend(removed)
    if len(tracker.removed_stracks) > removed_buffer:
        tracker.removed_stracks = tracker.removed_stracks[-removed_buffer:]

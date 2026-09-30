"""The homemade baselines: the 'before' picture every clever tracker improves on.

Both engines are pure numpy and depend on nothing, so they always run.  They
exist so the user can SEE what motion models, memory and appearance add.
"""
from __future__ import annotations

import itertools

import numpy as np

from app.core.params import BOOL, FLOAT, INT, _d
from app.core.plugins.base import Engine, TrackerPlugin
from app.core.plugins.registry import register
from app.core.trackers import Track, TrackerState, _iou


class CustomTrackerBase(Engine):
    _ids: itertools.count

    def __init__(self, params: dict):
        self._ids = itertools.count(0)
        self.tracks: dict[int, dict] = {}

    def _fresh(self, box, score, cls) -> int:
        tid = next(self._ids)
        self.tracks[tid] = {"box": list(box), "age": 0, "kind": "active", "score": float(score), "cls": int(cls)}
        return tid

    def _predict_box(self, tid):
        return self.tracks[tid]["box"]

    def state_from(self):
        active, lost = [], []
        for tid, t in self.tracks.items():
            tr = Track(id=tid, box=[int(v) for v in t["box"]], score=float(t["score"]),
                       cls=t["cls"], kind=t["kind"])
            if t["kind"] == "active":
                active.append(tr)
            elif t["age"] == 1:
                lost.append(tr)
        return TrackerState(active=active, lost_now=lost)


class GreedyIoUTracker(CustomTrackerBase):
    """Match each detection to the existing track with the biggest box overlap.

    The raw idea every MOT tracker refines: 'same thing = boxes that overlap
    from one frame to the next'.  No motion model, no appearance, no memory.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.iou_thresh = float(params.get("iou_thresh", 0.30))
        self.max_age = int(params.get("max_age", 1))
        self.keep_last = bool(params.get("keep_last_pos", False))

    def update(self, dets, img=None):
        dets = np.asarray(dets, dtype=np.float64)
        rows = [d for d in dets] if len(dets) else []
        remaining = list(range(len(rows)))
        for k in self.tracks:
            self.tracks[k]["age"] += 1
            if self.tracks[k]["age"] > self.max_age:
                self.tracks[k]["kind"] = "removed"
        self.tracks = {k: v for k, v in self.tracks.items() if v["kind"] != "removed"}

        for k, t in sorted(self.tracks.items(), key=lambda kv: -kv[1]["age"]):
            if t["kind"] != "active" and not self.keep_last:
                continue
            if not remaining:
                break
            pred = self._predict_box(k)
            bs, best_i = -1.0, None
            for i in remaining:
                iou = _iou(pred, rows[i][:4])
                if iou > bs:
                    bs, best_i = iou, i
            if best_i is not None and bs >= self.iou_thresh:
                t["box"] = rows[best_i][:4]
                t["age"] = 0
                t["kind"] = "active"
                t["score"] = rows[best_i][4]
                remaining.remove(best_i)
            elif t["kind"] == "active":
                t["kind"] = "lost"
        for i in remaining:
            self._fresh(rows[i][:4], rows[i][4], rows[i][5] if rows[i].shape[0] > 5 else 0)
        return self.state_from()


class CentroidTracker(CustomTrackerBase):
    """Link detections to the nearest track centre within a distance budget."""

    def __init__(self, params: dict):
        super().__init__(params)
        self.dist = float(params.get("dist_thresh", 60))
        self.max_age = int(params.get("max_age", 5))

    def update(self, dets, img=None):
        dets = np.asarray(dets, dtype=np.float64)
        rows = [d for d in dets] if len(dets) else []
        remaining = list(range(len(rows)))
        for k in self.tracks:
            self.tracks[k]["age"] += 1
            if self.tracks[k]["age"] > self.max_age:
                self.tracks[k]["kind"] = "removed"
        self.tracks = {k: v for k, v in self.tracks.items() if v["kind"] != "removed"}

        for k, t in sorted(self.tracks.items(), key=lambda kv: -kv[1]["age"]):
            if t["kind"] != "active" or not remaining:
                continue
            pc = _centre(self._predict_box(k))
            best, best_i = 1e9, None
            for i in remaining:
                c = _centre(rows[i][:4])
                dist = float(np.hypot(c[0] - pc[0], c[1] - pc[1]))
                if dist < best:
                    best, best_i = dist, i
            if best_i is not None and best <= self.dist:
                t["box"] = rows[best_i][:4]
                t["age"] = 0
                t["score"] = rows[best_i][4]
                remaining.remove(best_i)
            elif t["kind"] == "active":
                t["kind"] = "lost"
        for i in remaining:
            self._fresh(rows[i][:4], rows[i][4], rows[i][5] if rows[i].shape[0] > 5 else 0)
        return self.state_from()


@register
class GreedyIouPlugin(TrackerPlugin):
    id = "greedy_iou"
    engine = "custom"
    mode = "multi"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "Greedy IoU",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "The dumb baseline: match boxes purely by overlap.",
            "description": (
                "Links each new detection to the existing track whose box overlaps it the most -- and "
                "only when it overlaps into the next frame. No motion model, no appearance model, no "
                "memory of lost objects. It exists so you can SEE what the clever trackers add."
            ),
            "strengths": ["Instant and dependency-free", "Perfect when objects never touch, hide or move oddly"],
            "failure_modes": [
                "Objects crossing → IDs trade because the boxes overlap mid-crossing",
                "Object gone for one frame → track is gone forever (no memory)",
                "Fast object → laps its own previous box → lost",
                "Any box wobble → ids flicker",
            ],
            "params": [
                _d("iou_thresh", "IoU threshold", FLOAT, 0.30, "Minimum box-overlap a detection needs to reuse an existing track.",
                   "Lower = more forgiving of fast movement, but glues different objects together.", 0.0, 0.95, 0.05),
                _d("max_age", "Lost-track memory", INT, 1, "Frames a track with no detection is kept alive before deletion.",
                   "The folk version of ByteTrack's track_buffer.", 0, 60, 1, "frames"),
                _d("keep_last_pos", "Reuse last known position", BOOL, False,
                   "When a track lost its box, keep matching from its old position.",
                   "Turning this ON is the poor-man's motion prediction."),
            ],
        }

    def build(self, params: dict, fps: int, device: str = "cpu") -> GreedyIoUTracker:
        return GreedyIoUTracker(params)


@register
class CentroidPlugin(TrackerPlugin):
    id = "centroid"
    engine = "custom"
    mode = "multi"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "Centroid (distance)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Naive nearest-centre linking, blind to appearance.",
            "description": (
                "Keeps the centre point of every track and links each new detection to the closest "
                "centre within a distance budget. Simple, smooth -- and completely blind to what the "
                "object looks like, so two look-alikes brushing past will swap identities."
            ),
            "strengths": ["Smooth in slow, well-separated scenes", "Feels natural for balls and dots"],
            "failure_modes": [
                "Objects crossing at the same time → the classic ID swap",
                "Camera shake → all distances jump wildly",
                "Stopped object next to the real one → identity theft",
            ],
            "params": [
                _d("dist_thresh", "Max distance", INT, 60, "Max centre-distance between a detection and a track for a match.",
                   "The tracker's guess at 'how far could it move in one frame'.", 5, 300, 5, "px"),
                _d("max_age", "Lost-track memory", INT, 5, "Frames a track survives without detections.", "", 0, 60, 1, "frames"),
            ],
        }

    def build(self, params: dict, fps: int, device: str = "cpu") -> CentroidTracker:
        return CentroidTracker(params)


def _centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

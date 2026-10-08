"""The homemade baselines: the 'before' picture every clever tracker improves on.

All three engines are pure numpy (plus scipy for the assignment solve) and
depend on no external tracker library, so they always run.  They exist so the
user can SEE what motion models, memory and appearance add.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core.params import BOOL, FLOAT, INT, _d
from app.core.plugins.base import Engine, TrackerPlugin
from app.core.plugins.registry import register
from app.core.trackers import Track, TrackerState, _iou


class _KalmanBox:
    """Constant-velocity Kalman filter over `[cx, cy, w, h]`.

    The classic SORT state: measure what the box is, predict where it should be
    next, then correct the prediction with what the detector actually saw.  The
    prediction is the whole point -- it is how SORT re-finds an object that has
    already moved out of its own previous box.

    Noise weights are the values from the paper (1/20 position, 1/160 velocity).
    """

    _NDIM = 4
    _MOTION = np.vstack([np.hstack([np.eye(_NDIM), np.eye(_NDIM)]),
                         np.hstack([np.zeros((_NDIM, _NDIM)), np.eye(_NDIM)])])
    _STD_POS = 1.0 / 20
    _STD_VEL = 1.0 / 160

    def __init__(self, box):
        self._update_mat = np.eye(self._NDIM, 2 * self._NDIM)
        self._Q = np.diag(np.r_[np.full(self._NDIM, self._STD_POS),
                                np.full(self._NDIM, self._STD_VEL)] ** 2)
        self._R = np.eye(self._NDIM) * self._STD_POS**2
        self.x = np.zeros((2 * self._NDIM, 1))
        self.P = np.diag(np.r_[np.full(self._NDIM, 10.0), np.full(self._NDIM, 1000.0)])
        self.x[:self._NDIM, 0] = self._to_z(box)

    @staticmethod
    def _to_z(box) -> np.ndarray:
        """xyxy -> centre-size, the form the filter actually models."""
        x1, y1, x2, y2 = map(float, box)
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float64)

    @staticmethod
    def _to_bbox(mean) -> list:
        cx, cy, w, h = mean[:4]
        w, h = max(float(w), 1e-6), max(float(h), 1e-6)
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    @classmethod
    def initiate(cls, box) -> "_KalmanBox":
        return cls(box)

    def predict(self) -> None:
        self.x = self._MOTION @ self.x
        self.P = self._MOTION @ self.P @ self._MOTION.T + self._Q

    def predict_box(self) -> list:
        return self._to_bbox(self.x[:self._NDIM, 0])

    def update(self, box) -> None:
        z = self._to_z(box).reshape(-1, 1)
        y = z - self._update_mat @ self.x
        S = self._update_mat @ self.P @ self._update_mat.T + self._R
        K = self.P @ self._update_mat.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self._update_mat) @ self.P


class CustomTrackerBase(Engine):
    _ids: itertools.count

    def __init__(self, params: dict):
        self._ids = itertools.count(0)
        self.tracks: dict[int, dict] = {}

    @staticmethod
    def _normalize_dets(dets):
        from app.core.trackers import Detection
        if isinstance(dets, list) and dets and isinstance(dets[0], Detection):
            return np.asarray([d.to_list() for d in dets], dtype=np.float64)
        return np.asarray(dets, dtype=np.float64)

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
        dets = self._normalize_dets(dets)
        rows = [d for d in dets] if len(dets) else []
        remaining = list(range(len(rows)))
        for k in self.tracks:
            self.tracks[k]["age"] += 1
            if self.tracks[k]["age"] > self.max_age:
                self.tracks[k]["kind"] = "removed"
        self.tracks = {k: v for k, v in self.tracks.items() if v["kind"] != "removed"}

        for k, t in sorted(self.tracks.items(), key=lambda kv: (kv[1]["age"], -kv[1].get("score", 0.0))):
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


class SortTracker(CustomTrackerBase):
    """SORT (Bewley et al. 2016): Kalman prediction + optimal one-to-one matching.

    The parent of ByteTrack, and the reference point that makes ByteTrack's
    "second chance" visible: SORT trusts strong detections only, so a single
    weak frame costs it a track.  ByteTrack exists to fix exactly that.

    Two ideas over `GreedyIoUTracker`:
      1. a constant-velocity Kalman filter, so once it has seen an object's speed
         it keeps predicting through a gap that greedy matching cannot bridge, and
      2. Hungarian assignment, which is global -- greedy matching lets an early
         bad grab strand everyone behind it.

    Note the Kalman needs a few frames to estimate a velocity, so it does not
    rescue a track from rest: a sudden jump before it has seen the object move is
    still lost.  It earns its keep on occlusion, which is where the filter has
    had time to learn.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self.iou_thresh = float(params.get("iou_thresh", 0.30))
        self.max_age = int(params.get("max_age", 20))
        self.min_hits = int(params.get("min_hits", 3))
        self._kalman: dict[int, _KalmanBox] = {}

    def _init_track(self, box, score, cls) -> int:
        tid = super()._fresh(box, score, cls)
        t = self.tracks[tid]
        t["hits"] = 1
        # `_fresh` starts a track 'active'; SORT withholds a new id until it has
        # been matched min_hits times, so start it tentative instead.
        t["kind"] = "active" if self.min_hits <= 1 else "tentative"
        self._kalman[tid] = _KalmanBox.initiate(box)
        return tid

    def _predict_box(self, tid):
        kf = self._kalman.get(tid)
        return kf.predict_box() if kf else self.tracks[tid]["box"]

    def state_from(self):
        # Narrower than the base version on purpose: a track only reaches 'lost'
        # from 'active', so restricting the loss report to that kind stops an
        # unconfirmed track that misses one frame from claiming it "died".
        active, lost = [], []
        for tid, t in self.tracks.items():
            if t["kind"] == "active":
                active.append(Track(id=tid, box=[int(v) for v in t["box"]],
                                    score=float(t["score"]), cls=t["cls"]))
            elif t["kind"] == "lost" and t["age"] == 1:
                lost.append(Track(id=tid, box=[int(v) for v in t["box"]],
                                  score=float(t["score"]), cls=t["cls"], kind="lost"))
        return TrackerState(active=active, lost_now=lost)

    def update(self, dets, img=None):
        dets = self._normalize_dets(dets)
        rows = [d for d in dets] if len(dets) else []

        for t in self.tracks.values():
            t["age"] += 1
            if t["age"] > self.max_age:
                t["kind"] = "removed"
        for tid in [k for k, v in self.tracks.items() if v["kind"] == "removed"]:
            self._kalman.pop(tid, None)
        self.tracks = {k: v for k, v in self.tracks.items() if v["kind"] != "removed"}
        for kf in self._kalman.values():
            kf.predict()

        matched_track, matched_det = set(), set()
        pool = list(self.tracks)
        if pool and rows:
            cost = np.zeros((len(pool), len(rows)))
            for i, tid in enumerate(pool):
                pred = self._kalman[tid].predict_box()
                for j, r in enumerate(rows):
                    cost[i, j] = 1.0 - _iou(pred, r[:4])
            # `cost` is a cost, so minimise it directly; anything above the
            # threshold is "no match", which is what the paper does too.
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] > (1.0 - self.iou_thresh):
                    continue  # overlap too low to be the same object
                tid, row = pool[i], rows[j]
                self._kalman[tid].update(row[:4])
                t = self.tracks[tid]
                t["box"] = list(row[:4])
                t["score"] = row[4]
                t["age"] = 0
                t["hits"] += 1
                # only start reporting once SORT has seen it min_hits times
                t["kind"] = "active" if t["hits"] >= self.min_hits else "tentative"
                matched_track.add(tid)
                matched_det.add(int(j))

        for tid, t in self.tracks.items():
            if tid not in matched_track and t["kind"] == "active":
                # a confirmed track that missed goes 'lost'; the Kalman keeps
                # predicting so it can be re-found without spending a new id
                t["kind"] = "lost"

        for j, row in enumerate(rows):
            if j not in matched_det:
                self._init_track(row[:4], row[4],
                                 row[5] if row.shape[0] > 5 else 0)
        return self.state_from()


class CentroidTracker(CustomTrackerBase):
    """Link detections to the nearest track centre within a distance budget."""

    def __init__(self, params: dict):
        super().__init__(params)
        self.dist = float(params.get("dist_thresh", 60))
        self.max_age = int(params.get("max_age", 5))

    def update(self, dets, img=None):
        dets = self._normalize_dets(dets)
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


@register
class SortPlugin(TrackerPlugin):
    id = "sort"
    engine = "custom"
    mode = "multi"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "SORT (Kalman + Hungarian)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "The 2016 original: predict motion, then match optimally.",
            "description": (
                "The tracker everything else is built on. Every frame it (1) asks a constant-velocity Kalman "
                "filter where each track SHOULD be, (2) scores every track-detection pair by overlap, and "
                "(3) solves the whole assignment at once with the Hungarian algorithm instead of greedily. "
                "Run it next to ByteTrack: the only real difference is that ByteTrack gives weak detections a "
                "second chance, which is exactly what you are missing here."
            ),
            "strengths": [
                "Prediction re-finds an object that reappears well outside its last box",
                "Global assignment — one bad greedy match can't strand the rest",
                "Far fewer ID switches than any other tracker here",
                "Fast and dependency-free (no ReID model, no extra downloads)",
            ],
            "failure_modes": [
                "Only assumes constant velocity → turning or accelerating objects drift off the prediction",
                "Needs a few frames to estimate speed, so a sudden jump from rest is still lost",
                "No appearance model, so two look-alikes crossing is invisible to it",
                "Trusts strong detections only: ONE weak frame and the id is gone (this is ByteTrack's whole fix)",
                "Holds a new id back until it has been seen a few times — better IDs, but slower to start",
            ],
            "params": [
                _d("iou_thresh", "Minimum IoU to match", FLOAT, 0.30,
                   "Overlap a predicted track and a detection need to be considered the same object.",
                   "Lower catches fast movers; raise to stop neighbours being swapped.", 0.0, 0.95, 0.05),
                _d("min_hits", "Frames before a track is reported", INT, 3,
                   "How many times a new track must be matched before it is shown as a real id.",
                   "Lower = starts faster and scores a better MOTA, but spends ids on flickering boxes. "
                   "Higher = fewer ID switches, at the cost of missing frames.", 1, 20, 1, "frames"),
                _d("max_age", "Lost-track memory", INT, 20,
                   "Frames a confirmed track keeps being predicted after it misses, before being deleted.",
                   "This is the 'second chance' for a temporarily hidden object — SORT's only one.",
                   1, 120, 1, "frames"),
            ],
        }

    def build(self, params: dict, fps: int, device: str = "cpu") -> SortTracker:
        return SortTracker(params)


def _centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

"""Unified wrapping layer around every tracking engine.

Every tracker -- whatever its implementation -- exposes the same tiny interface
so the simulator, the metrics module and the visualiser can treat them the same:

    engine = build_engine(tracker_id, params, fps, device)
    state = engine.update(dets, img)
    state.active      -> list[Track]  (boxes the tracker is currently sure of)
    state.lost_now    -> list[Track]  (tracks that just died this frame)

Ultralytics 8.4+ trackers are driven *standalone* (no model.predict/track) so
the simulator can feed them synthetic or YOLO detections and read back
per-frame identities for profiling.  Trackers read their full config from the
community YAMLs shipped inside the ultralytics package, so every knob exposed
in the UI maps 1:1 onto a real library parameter.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# RFC-5398 style channel-independent object colours (stable across everything).
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (0, 130, 200), (240, 50, 230), (128, 128, 0),
    (250, 190, 212), (35, 35, 220), (170, 255, 195), (0, 0, 128),
]


@dataclass
class Track:
    id: int
    box: list  # [x1, y1, x2, y2]
    score: float = 1.0
    cls: int = 0
    kind: str = "active"  # active | lost


@dataclass
class TrackerState:
    active: list = field(default_factory=list)
    lost_now: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Detection "results-like" shim                                               #
# --------------------------------------------------------------------------- #

class DetShim:
    """A drop-in stand-in for `ultralytics.engine.results.Boxes`.

    The ultralytics trackers only touch a few attributes of the detection
    object (`conf`, `xywh`/`xywhr`, `cls`, `xyxy`) plus numpy-style boolean
    indexing.  This shim serves our (N,5) [x1,y1,x2,y2,score] or (N,6)
    [.., cls] arrays through exactly that interface.
    """

    def __init__(self, arr: np.ndarray):
        arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        self._arr = arr

    @property
    def xyxy(self):
        return self._arr[:, :4]

    @property
    def xywh(self):
        a = self._arr[:, :4]
        x1, y1, x2, y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    @property
    def conf(self):
        return self._arr[:, 4]

    @property
    def cls(self):
        return self._arr[:, 5] if self._arr.shape[1] > 5 else np.zeros(len(self._arr))

    def cpu(self):
        return self

    def numpy(self):
        return self

    def __getitem__(self, idx):
        return DetShim(self._arr[idx])

    def __len__(self):
        return len(self._arr)

    def __bool__(self):
        return len(self._arr) > 0


# --------------------------------------------------------------------------- #
# Custom baselines (homemade, fully transparent)                              #
# --------------------------------------------------------------------------- #

class CustomTrackerBase:
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


class OpenCVSingleTracker:
    """Wraps one cv2 legacy tracker to follow a *single* object.

    Initialised by the runner on frame 0 with the true box of the object being
    studied, then it simply follows.  When cv2 gives up, the track dies -- no
    re-initialisation, that is part of the lesson.
    """

    AVAILABLE = None  # filled in lazily by probe()

    def __init__(self, tracker_id: str, params: dict):
        self.tracker_id = tracker_id
        self.max_age = int(params.get("max_age", 30))
        self._age = 0
        self._tracker = None
        self._id = 0
        self.active = False

    def init(self, img, box: list):
        import cv2
        factory = (self.AVAILABLE or _opencv_probe()).get(self.tracker_id)
        if factory is None:
            raise ValueError(f"OpenCV tracker '{self.tracker_id}' not available in this build "
                             f"(available: {sorted((_opencv_probe() or {}).keys())})")
        self._tracker = factory()
        x1, y1, x2, y2 = box
        self._tracker.init(img, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        self.active = True
        self._age = 0

    def update(self, dets, img):
        if not self.active or self._tracker is None:
            return TrackerState(active=[], lost_now=[])
        ok, bbox = self._tracker.update(img)
        if ok:
            x, y, w, h = bbox
            ok = w > 0 and h > 0
        if ok:
            self._age = 0
            x, y, w, h = bbox
            return TrackerState(active=[Track(self._id, [int(x), int(y), int(x + w), int(y + h)], score=0.8)])
        self._age += 1
        if self._age >= self.max_age:
            self.active = False
            return TrackerState(active=[], lost_now=[Track(self._id, [0, 0, 0, 0], score=0.0)])
        return TrackerState(active=[])


_OPENCV_FACTORIES = {
    "kcf": "TrackerKCF_create", "csrt": "TrackerCSRT_create",
    "mosse": "TrackerMOSSE_create", "mil": "TrackerMIL_create",
    "medianflow": "TrackerMedianFlow_create",
    "nano": "TrackerNano_create", "vit": "TrackerVit_create",
    "dasiamrpn": "TrackerDaSiamRPN_create",
}


def _opencv_probe() -> dict:
    if OpenCVSingleTracker.AVAILABLE is None:
        import cv2
        avail = {}
        for name, fn in _OPENCV_FACTORIES.items():
            creator = getattr(cv2, fn, None)
            if creator is None:
                creator = getattr(getattr(cv2, "legacy", None), fn, None)
            if creator is None:
                continue
            try:
                creator()  # neural ones validate/download their ONNX weights here
                avail[name] = creator
            except Exception:  # noqa: BLE001  (missing model weights etc.)
                continue
        OpenCVSingleTracker.AVAILABLE = avail
    return OpenCVSingleTracker.AVAILABLE


# --------------------------------------------------------------------------- #
# Ultralytics native trackers                                                 #
# --------------------------------------------------------------------------- #

ULTRALYTICS_TRACKER_MAP = {
    "bytetrack": "BYTETracker", "botsort": "BOTSORT", "ocsort": "OCSORT",
    "deepocsort": "DeepOCSORT", "fasttrack": "FASTTracker", "tracktrack": "TRACKTRACK",
}


class UltralyticsEngine:
    """Adapter around the current ultralytics tracker API (8.4+).

    Loads the tracker's own YAML config inside the package, overlays any user
    hyperparameters, then drives `tracker.update(DetShim(dets), img)` and reads
    back the `(N,8)` rows  [x1,y1,x2,y2,id,score,cls,idx].
    """

    def __init__(self, tracker_id: str, params: dict, fps: int, device: str = "cpu"):
        import ultralytics
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.trackers.track import TRACKER_MAP
        self._tracker_cls = TRACKER_MAP[tracker_id]

        yaml_path = Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers" / f"{tracker_id}.yaml"
        cfg = IterableSimpleNamespace(**YAML.load(str(yaml_path)))
        cfg.device = device
        for k, v in params.items():
            if hasattr(cfg, k) or k in ("track_high_thresh", "track_low_thresh", "new_track_thresh",
                                        "track_buffer", "match_thresh", "fuse_score"):
                setattr(cfg, k, v)
        self.tracker = self._tracker_cls(args=cfg)
        self.reid = bool(getattr(self.tracker, "encoder", None) is not None)

    def update(self, dets, img):
        dets = np.asarray(dets, dtype=np.float64)
        if dets.ndim == 1 and len(dets):
            dets = dets[None]
        if dets.shape[1] < 5:
            dets = np.pad(dets, ((0, 0), (0, 5 - dets.shape[1])))
        out = self.tracker.update(DetShim(dets), np.ascontiguousarray(img) if img is not None else None)
        rows = np.asarray(out, dtype=np.float64).reshape(-1, 8)
        active, lost = [], []
        prev_ids = {t.id for t in getattr(self, "_last_active", [])}
        cur = []
        for r in rows:
            x1, y1, x2, y2 = map(int, r[:4])
            if x2 <= x1 or y2 <= y1:
                continue
            t = Track(id=int(r[4]), box=[x1, y1, x2, y2], score=float(r[5]), cls=int(r[6]))
            cur.append(t)
        active = cur
        for t in getattr(self, "_last_active", []):
            if t.id not in {c.id for c in cur}:
                lost.append(t)
        self._last_active = cur
        return TrackerState(active=active, lost_now=lost)


# --------------------------------------------------------------------------- #
# Factory                                                                    #
# --------------------------------------------------------------------------- #

def build_engine(tracker_id: str, params: dict, fps: int, device: str = "cpu"):
    meta = {
        "greedy_iou": ("custom",), "centroid": ("custom",),
        "kcf": ("opencv",), "csrt": ("opencv",), "mosse": ("opencv",),
        "mil": ("opencv",), "medianflow": ("opencv",),
    }.get(tracker_id, (None,))
    engine = meta[0]
    if engine == "custom":
        return GreedyIoUTracker(params) if tracker_id == "greedy_iou" else CentroidTracker(params)
    if engine == "opencv":
        if tracker_id not in (_opencv_probe() or {}):
            raise ValueError(
                f"OpenCV tracker '{tracker_id}' is not built into this OpenCV install "
                f"(available: {sorted((_opencv_probe() or {}).keys())})")
        return OpenCVSingleTracker(tracker_id, params)
    return UltralyticsEngine(tracker_id, params, fps, device)


def color_for(track_id: int) -> tuple:
    return _PALETTE[track_id % len(_PALETTE)]


def _iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    uni = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / uni if uni > 0 else 0.0


def _centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
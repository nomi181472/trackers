"""Vector embedder-based tracker plugin (DeepSORT / ReID / Visual Embedding).

Combines Kalman motion prediction with visual appearance feature embeddings.
Extracts normalized feature vectors from image crops across all object types
(person, car, ball) and solves multi-object association with a dual-cost
matrix (IoU + Cosine distance) with appearance gating.
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core.params import FLOAT, INT, _d
from app.core.plugins.base import Engine, TrackerPlugin
from app.core.plugins.custom import _KalmanBox
from app.core.plugins.registry import register
from app.core.trackers import Track, TrackerState, _iou


class _CropEmbedder:
    """Fast, robust visual feature extractor for object crops.

    Handles any crop aspect ratio (tall people, wide cars, square balls).
    Combines spatial grid pooling with normalized color histograms to
    generate an L2-normalized vector embedding robust to minor deformation.
    """

    def __init__(self, target_size: tuple[int, int] = (64, 64)):
        self.target_size = target_size

    def extract(self, img: np.ndarray, box: list) -> np.ndarray:
        if img is None or img.size == 0:
            return np.zeros(64, dtype=np.float32)

        H, W = img.shape[:2]
        x1 = max(0, min(int(round(box[0])), W - 1))
        y1 = max(0, min(int(round(box[1])), H - 1))
        x2 = max(x1 + 1, min(int(round(box[2])), W))
        y2 = max(y1 + 1, min(int(round(box[3])), H))

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return np.zeros(64, dtype=np.float32)

        resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_AREA)

        # 1. Spatial 4x4 block mean features (16 blocks x 3 channels = 48 dims)
        blocks = []
        bh, bw = self.target_size[1] // 4, self.target_size[0] // 4
        for r in range(4):
            for c in range(4):
                blk = resized[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
                blocks.extend(blk.mean(axis=(0, 1)))

        # 2. HSV color distribution (16 bins: 8 Hue + 4 Saturation + 4 Value)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [4], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [4], [0, 256]).flatten()
        color_feats = np.concatenate([h_hist, s_hist, v_hist])

        vec = np.concatenate([np.array(blocks, dtype=np.float32), color_feats.astype(np.float32)])
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec


class VectorEmbedderEngine(Engine):
    """DeepSORT-style tracking engine driven by visual embeddings + Kalman motion."""

    def __init__(self, params: dict, fps: int = 30):
        self.params = params
        self.fps = max(1, fps)
        self.embed_weight = float(params.get("embed_weight", 0.50))
        self.dist_thresh = float(params.get("dist_thresh", 0.40))
        self.iou_thresh = float(params.get("iou_thresh", 0.30))
        self.ema_alpha = float(params.get("ema_alpha", 0.85))
        self.max_age = int(params.get("max_age", 30))
        self.min_hits = int(params.get("min_hits", 3))

        self.embedder = _CropEmbedder()
        self._ids = itertools.count(0)
        self.tracks: dict[int, dict] = {}

    def _fresh(self, box, score, cls, embedding: np.ndarray) -> int:
        tid = next(self._ids)
        self.tracks[tid] = {
            "kf": _KalmanBox.initiate(box),
            "box": list(box),
            "embedding": embedding,
            "hits": 1,
            "age": 0,
            "time_since_update": 0,
            "score": float(score),
            "cls": int(cls),
            "kind": "confirmed" if self.min_hits <= 1 else "tentative",
        }
        return tid

    def update(self, dets, img=None) -> TrackerState:
        from app.core.trackers import Detection
        if isinstance(dets, list) and dets and isinstance(dets[0], Detection):
            dets = np.asarray([d.to_list() for d in dets], dtype=np.float64)
        else:
            dets = np.asarray(dets, dtype=np.float64) if len(dets) else np.zeros((0, 5))

        # 1. Kalman prediction for all existing tracks
        for tr in self.tracks.values():
            tr["kf"].predict()
            tr["box"] = tr["kf"].predict_box()
            tr["age"] += 1
            tr["time_since_update"] += 1

        # 2. Extract visual embeddings for current frame detections
        det_embeddings = []
        for d in dets:
            if img is not None:
                feat = self.embedder.extract(img, d[:4])
            else:
                feat = np.zeros(64, dtype=np.float32)
            det_embeddings.append(feat)

        # 3. Association: matching tracks with detections
        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(dets)

        matched_tracks, matched_dets = set(), set()

        if n_tracks > 0 and n_dets > 0:
            cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float64)

            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                tr_box = tr["box"]
                tr_emb = tr["embedding"]

                for j in range(n_dets):
                    det_box = dets[j, :4]
                    det_emb = det_embeddings[j]

                    # IoU cost: [0, 1]
                    iou = _iou(tr_box, det_box)
                    iou_cost = 1.0 - iou

                    # Cosine distance: 1 - cos(theta) in [0, 2]
                    norm_tr = np.linalg.norm(tr_emb)
                    norm_det = np.linalg.norm(det_emb)
                    if norm_tr > 1e-6 and norm_det > 1e-6:
                        cos_sim = float(np.dot(tr_emb, det_emb) / (norm_tr * norm_det))
                        cos_dist = max(0.0, min(2.0, 1.0 - cos_sim))
                    else:
                        cos_dist = 1.0

                    # Hybrid fused cost
                    fused_cost = (1.0 - self.embed_weight) * iou_cost + self.embed_weight * cos_dist

                    # Appearance gating: if appearance is drastically dissimilar and IoU is weak
                    if cos_dist > self.dist_thresh and iou < self.iou_thresh:
                        fused_cost += 10.0

                    cost_matrix[i, j] = fused_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_ind, col_ind):
                tid = track_ids[r]
                cost = cost_matrix[r, c]
                # Reject match if cost is too high (gated by distance and IoU threshold)
                max_cost = (1.0 - self.embed_weight) * (1.0 - self.iou_thresh) + self.embed_weight * max(self.dist_thresh, 1.8)
                if cost > max_cost:
                    continue

                tr = self.tracks[tid]
                det_box = dets[c, :4]
                det_score = dets[c, 4]
                det_cls = dets[c, 5] if dets.shape[1] > 5 else 0

                tr["kf"].update(det_box)
                tr["box"] = list(det_box)
                tr["score"] = float(det_score)
                tr["cls"] = int(det_cls)
                tr["hits"] += 1
                tr["time_since_update"] = 0

                if tr["hits"] >= self.min_hits:
                    tr["kind"] = "confirmed"

                # Update appearance gallery using Exponential Moving Average (EMA)
                det_emb = det_embeddings[c]
                if np.linalg.norm(det_emb) > 1e-6:
                    tr["embedding"] = (
                        self.ema_alpha * tr["embedding"] + (1.0 - self.ema_alpha) * det_emb
                    )
                    norm = np.linalg.norm(tr["embedding"])
                    if norm > 1e-6:
                        tr["embedding"] /= norm

                matched_tracks.add(tid)
                matched_dets.add(c)

        # 4. Create new tentative tracks for unmatched detections
        for j in range(n_dets):
            if j not in matched_dets:
                det_box = dets[j, :4]
                det_score = dets[j, 4]
                det_cls = dets[j, 5] if dets.shape[1] > 5 else 0
                self._fresh(det_box, det_score, det_cls, det_embeddings[j])

        # 5. Handle lost / expired tracks
        active_tracks: list[Track] = []
        lost_now: list[Track] = []

        dead_ids = []
        for tid, tr in self.tracks.items():
            if tr["time_since_update"] > self.max_age:
                dead_ids.append(tid)
                continue
            if tr["time_since_update"] > 0 and tr["kind"] == "tentative":
                dead_ids.append(tid)
                continue

            if tr["time_since_update"] == 0 and tr["kind"] == "confirmed":
                active_tracks.append(
                    Track(
                        id=tid,
                        box=[int(round(v)) for v in tr["box"]],
                        score=float(tr["score"]),
                        cls=tr["cls"],
                        kind="active",
                    )
                )
            elif tr["time_since_update"] == 1 and tr["kind"] == "confirmed":
                lost_now.append(
                    Track(
                        id=tid,
                        box=[int(round(v)) for v in tr["box"]],
                        score=float(tr["score"]),
                        cls=tr["cls"],
                        kind="lost",
                    )
                )

        for tid in dead_ids:
            del self.tracks[tid]

        return TrackerState(active=active_tracks, lost_now=lost_now)


@register
class VectorEmbedderPlugin(TrackerPlugin):
    """DeepSORT / ReID vector embedder tracking plugin."""

    id = "embed_sort"
    engine = "custom"
    mode = "multi"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "Vector Embedder (DeepSORT)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Kalman motion prediction + Visual vector feature embedding re-identification.",
            "description": (
                "Extracts visual vector embeddings from object crops across every frame. "
                "Matches objects using a hybrid cost matrix of Kalman IoU overlap and cosine feature similarity. "
                "Maintains a rolling appearance gallery via EMA, allowing objects (people, cars, balls) "
                "to reliably recover identity after full occlusions or path crossings."
            ),
            "strengths": [
                "Recovers identity after disappearing behind walls or during occlusions",
                "Disambiguates crossing paths between differently-styled or colored objects",
                "Works across all simulated object types (person skeleton, car, ball) and real video",
                "Self-contained zero-external-download visual embedder",
            ],
            "failure_modes": [
                "Higher per-frame compute latency (visible on Trade-off and Latency graphs)",
                "Appearance confusion when objects are identical look-alikes (similar_colors)",
            ],
            "params": [
                _d(
                    "embed_weight",
                    "Embedding vs IoU Weight",
                    FLOAT,
                    0.50,
                    "Weight given to visual appearance (0 = pure Kalman IoU, 1 = pure visual cosine distance).",
                    "Raise for scenes with frequent occlusions or rapid velocity changes.",
                    0.0,
                    1.0,
                    0.05,
                ),
                _d(
                    "dist_thresh",
                    "Cosine Distance Gate",
                    FLOAT,
                    0.40,
                    "Maximum cosine distance to allow visual association.",
                    "Lower = stricter appearance matching; higher = tolerant of rotation and lighting.",
                    0.1,
                    0.9,
                    0.05,
                ),
                _d(
                    "iou_thresh",
                    "Minimum IoU Threshold",
                    FLOAT,
                    0.30,
                    "Minimum box overlap required for spatial verification.",
                    "",
                    0.0,
                    0.95,
                    0.05,
                ),
                _d(
                    "ema_alpha",
                    "Appearance Smoothing (EMA)",
                    FLOAT,
                    0.85,
                    "Exponential moving average weight for track embedding gallery updates.",
                    "Higher = retains historical appearance longer.",
                    0.1,
                    0.99,
                    0.05,
                ),
                _d(
                    "max_age",
                    "ReID Memory Buffer",
                    INT,
                    30,
                    "Frames a lost track is remembered for visual recovery before deletion.",
                    "Longer buffer allows surviving longer occlusions behind walls.",
                    1,
                    120,
                    1,
                    "frames",
                ),
                _d(
                    "min_hits",
                    "Frames to confirm track",
                    INT,
                    3,
                    "Consecutive matches required before an ID is reported.",
                    "Higher reduces false ID switches on flickering boxes.",
                    1,
                    10,
                    1,
                    "frames",
                ),
            ],
        }

    def build(self, params: dict, fps: int, device: str = "cpu") -> VectorEmbedderEngine:
        return VectorEmbedderEngine(params, fps)

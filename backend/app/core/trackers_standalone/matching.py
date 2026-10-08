"""Bipartite matching and linear sum assignment utilities.
Pure NumPy and SciPy implementation without external C-extensions.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = False):
    """Solve linear assignment using scipy.optimize.linear_sum_assignment."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    r, c = linear_sum_assignment(cost_matrix)
    matches = np.asarray([[r[i], c[i]] for i in range(len(r)) if cost_matrix[r[i], c[i]] <= thresh])
    if len(matches) == 0:
        unmatched_a = list(np.arange(cost_matrix.shape[0]))
        unmatched_b = list(np.arange(cost_matrix.shape[1]))
    else:
        unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
        unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def bbox_ious(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute IoU between two arrays of bounding boxes (N, 4) and (M, 4) in xyxy format."""
    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xA = np.maximum(x11, x21.T)
    yA = np.maximum(y11, y21.T)
    xB = np.minimum(x12, x22.T)
    yB = np.minimum(y12, y22.T)

    interArea = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    unionArea = boxAArea + boxBArea.T - interArea
    return interArea / np.clip(unionArea, eps, None)


def bbox_ioa(boxes1: np.ndarray, boxes2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """Intersection over box2 area or IoU."""
    if iou:
        return bbox_ious(boxes1, boxes2, eps)
    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xA = np.maximum(x11, x21.T)
    yA = np.maximum(y11, y21.T)
    xB = np.minimum(x12, x22.T)
    yB = np.minimum(y12, y22.T)
    interArea = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    boxBArea = (x22 - x21) * (y22 - y21)
    return interArea / np.clip(boxBArea.T, eps, None)


def iou_distance(atracks, btracks) -> np.ndarray:
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xyxy for track in atracks]
        btlbrs = [track.xyxy for track in btracks]
    _ious = bbox_ious(atlbrs, btlbrs)
    return 1 - _ious


def embedding_distance(tracks, detections, metric: str = "cosine") -> np.ndarray:
    """Compute cosine distance between track embeddings and detection embeddings."""
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([d.curr_feat for d in detections], dtype=np.float64)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, 1.0 - np.dot(track_features, det_features.T))
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections) -> np.ndarray:
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1.0 - fuse_sim

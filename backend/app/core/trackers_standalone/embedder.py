"""Appearance embedding extractor for standalone trackers without PyTorch or external model weights."""
from __future__ import annotations
import cv2
import numpy as np


class SimulatorCropEmbedder:
    """Extracts a normalized 64-dimensional appearance feature vector from an image bounding box crop."""

    def __init__(self, target_size: tuple[int, int] = (64, 64)):
        self.target_size = target_size

    def extract(self, img: np.ndarray, box: list | np.ndarray) -> np.ndarray:
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

        # 1. Spatial 4x4 block mean features (48 dims)
        blocks = []
        bh, bw = self.target_size[1] // 4, self.target_size[0] // 4
        for r in range(4):
            for c in range(4):
                blk = resized[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
                blocks.extend(blk.mean(axis=(0, 1)))

        # 2. HSV color distribution (16 bins)
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


def smooth_feature(old_feat: np.ndarray | None, new_feat: np.ndarray | None, alpha: float = 0.9) -> np.ndarray:
    if old_feat is None:
        return new_feat
    if new_feat is None:
        return old_feat
    feat = alpha * old_feat + (1.0 - alpha) * new_feat
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 1e-6 else feat

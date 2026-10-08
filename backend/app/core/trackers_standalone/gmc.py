"""Camera motion compensation (GMC) in pure OpenCV / NumPy without deep learning."""
from __future__ import annotations
import copy
import cv2
import numpy as np


class GMC:
    """Generalized Motion Compensation class supporting OpenCV-based alignment."""

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
        self.method = method
        self.downscale = max(1, downscale)
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

        if self.method == "orb":
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.method == "sift" and hasattr(cv2, "SIFT_create"):
            self.detector = cv2.SIFT_create()
            self.extractor = self.detector
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.method == "ecc":
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

    def apply(self, raw_frame: np.ndarray, detections=None) -> np.ndarray:
        if raw_frame is None or self.method in (None, "none"):
            return np.eye(2, 3)

        height, width = raw_frame.shape[:2]
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        H = np.eye(2, 3)
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H

        try:
            if self.method == "sparseOptFlow":
                H = self._apply_sparse_opt_flow(frame)
            elif self.method in ("orb", "sift"):
                H = self._apply_features(frame)
            elif self.method == "ecc":
                H = self._apply_ecc(frame)
        except Exception:
            H = np.eye(2, 3)

        self.prevFrame = frame.copy()
        return H

    def _apply_sparse_opt_flow(self, frame: np.ndarray) -> np.ndarray:
        prev_pts = cv2.goodFeaturesToTrack(self.prevFrame, maxCorners=1000, qualityLevel=0.01, minDistance=8)
        if prev_pts is None or len(prev_pts) < 4:
            return np.eye(2, 3)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, prev_pts, None)
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]
        if len(good_prev) < 4:
            return np.eye(2, 3)
        H, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
        return H if H is not None else np.eye(2, 3)

    def _apply_features(self, frame: np.ndarray) -> np.ndarray:
        kp = self.detector.detect(frame)
        if kp is None or len(kp) < 4:
            return np.eye(2, 3)
        kp, desc = self.extractor.compute(frame, kp)
        if desc is None:
            return np.eye(2, 3)
        if self.prevDescriptors is None:
            self.prevKeyPoints = kp
            self.prevDescriptors = desc
            return np.eye(2, 3)
        matches = self.matcher.knnMatch(self.prevDescriptors, desc, k=2)
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                good.append(m[0])
        if len(good) < 4:
            return np.eye(2, 3)
        prev_pts = np.float32([self.prevKeyPoints[m.queryIdx].pt for m in good])
        curr_pts = np.float32([kp[m.trainIdx].pt for m in good])
        H, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        self.prevKeyPoints = kp
        self.prevDescriptors = desc
        return H if H is not None else np.eye(2, 3)

    def _apply_ecc(self, frame: np.ndarray) -> np.ndarray:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        _, warp_matrix = cv2.findTransformECC(self.prevFrame, frame, warp_matrix, self.warp_mode, self.criteria)
        return warp_matrix

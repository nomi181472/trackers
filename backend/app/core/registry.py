"""Hyperparameters that are NOT per-tracker, plus thin accessors into the
plugin registry.

Per-tracker metadata lives with the tracker itself: each `TrackerPlugin` class
carries its own name, explanation and parameter list, and
`app.core.plugins.REGISTRY` is the single source of truth.  What remains here
is the shared detector / scenario-detection parameter surface, which every
tracker consumes identically.
"""
from __future__ import annotations

from app.core.params import BOOL, FLOAT, INT, MODEL_OPTIONS, SELECT, _d
from app.core.plugins import REGISTRY

# --------------------------------------------------------------------------- #
# Shared detector + scenario-level detection hyperparameters                  #
# --------------------------------------------------------------------------- #

DETECTOR_PARAMS = [
    _d("model", "Detector model", SELECT, "yolov8n", "The object-detector that 'sees' boxes before the tracker links them.",
       "Small models miss small/fast objects more often.", options=MODEL_OPTIONS),
    _d("conf", "Detection confidence", FLOAT, 0.25, "Minimum confidence for a detection to count.",
       "Raise to remove junk, but you'll cut weak-but-real detections (occluded objects).", 0.01, 0.95, 0.01),
    _d("iou", "NMS IoU", FLOAT, 0.7, "Non-max suppression overlap (real-model mode).", "", 0.05, 0.95, 0.05),
    _d("imgsz", "Input size", INT, 640, "Image resolution fed to the detector (real-model mode).",
       "Higher resolution helps small objects but is slower.", 256, 1280, 32, "px"),
    _d("half", "FP16 inference", BOOL, False, "Half-precision detector inference (faster, slightly less accurate)."),
    _d("device", "Device", SELECT, "cpu", "Where the detector runs.", options={"cpu": "CPU", "0": "GPU 0"}),
]

SCENARIO_DETECTION_PARAMS = [
    _d("drop_while_occluded", "Drop detections during occlusion", BOOL, True,
       "When an object hides behind the wall the detector realistically fails. Off = pure tracker behaviour without detector noise.",
       "That's the detector's fault, not the tracker's — compare."),
    _d("miss_rate", "Random detection miss rate", FLOAT, 0.0, "Probability the detector randomly fails on a visible object in a given frame.",
       "Wiggle to simulate sensor noise; kills trackers that need a hit every frame.", 0.0, 0.6, 0.02),
    _d("fp_rate", "Ghost-box rate", FLOAT, 0.0, "Probability of a random detection box spawning from thin air.",
       "Simulates clutter/mirrors; trackers that trust every box hallucinate tracks.", 0.0, 0.5, 0.02),
    _d("jitter", "Box jitter", INT, 2, "Random pixels added to each detection box edge.",
       "Detectors wobble; this stresses trackers that assume the box is exact.", 0, 25, 1, "px"),
]


# --------------------------------------------------------------------------- #
# Thin accessors over the plugin registry                                     #
# --------------------------------------------------------------------------- #

def tracker_ids() -> list[str]:
    return REGISTRY.ids()


def get_tracker(tid: str) -> dict:
    return REGISTRY.get(tid).meta()


def default_params(tid: str) -> dict:
    return REGISTRY.default_params(tid)

"""The classic single-object followers, wrapped as plugins.

All eight share one engine: `OpenCVSingleTracker`, which is initialised on the
first frame with the true box and then just follows.  When OpenCV gives up the
track dies -- no re-initialisation, that is part of the lesson.

Availability is probed lazily and cached on the engine class, because the
factory constructors double as a "can I actually run this?" test (the neural
ones validate -- and download -- their ONNX weights on creation).
"""
from __future__ import annotations

from app.core.params import FLOAT, INT, _d
from app.core.plugins.base import Engine, TrackerPlugin
from app.core.plugins.registry import register
from app.core.trackers import Track, TrackerState


class OpenCVSingleTracker(Engine):
    """Wraps one cv2 legacy tracker to follow a *single* object."""

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


class _OpenCvSinglePlugin(TrackerPlugin):
    """Shared behaviour: one engine, availability straight from the probe."""

    engine = "opencv"
    mode = "single"

    @classmethod
    def is_available(cls) -> bool:
        return cls.id in (_opencv_probe() or {})

    def build(self, params: dict, fps: int, device: str = "cpu") -> OpenCVSingleTracker:
        avail = _opencv_probe() or {}
        if self.id not in avail:
            raise ValueError(
                f"OpenCV tracker '{self.id}' is not built into this OpenCV install "
                f"(available: {sorted(avail.keys())})")
        return OpenCVSingleTracker(self.id, params)


@register
class KcfPlugin(_OpenCvSinglePlugin):
    id = "kcf"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "KCF (single-object)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Correlation-filter follower — the classic lightweight single-object tracker.",
            "description": (
                "Learns a tiny correlation filter around the object in the first frame, then hunts for "
                "the spot that 'correlates' the most in every later frame. Extremely fast, but has no "
                "idea what a person is and happily locks onto the wrong thing. Great for the 'follow "
                "one ball' experiments."
            ),
            "strengths": ["Hundreds of FPS on a laptop", "Follows a clearly visible, mostly-static-shaped object"],
            "failure_modes": ["Full occlusion → loses it and freezes on whatever emerges", "Abrupt fast motion → search window can't keep up", "Identical nearby object → jumps onto it"],
            "params": [
                _d("max_age", "Lost-track memory", INT, 30, "Frames it keeps 'searching' after confidence collapses before declaring loss.", "", 1, 120, 1, "frames"),
            ],
        }


@register
class CsrtPlugin(_OpenCvSinglePlugin):
    id = "csrt"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "CSRT (single-object)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Discriminative filter + colour histogram = harder to lose.",
            "description": (
                "KCF's smarter sibling: besides the filter it keeps a colour histogram and a spatial "
                ". Instead of a hard yes/no correlation peak it watches a peak-to-sidelobe ratio, so "
                "it drifts much less and can often ride out short occlusions. The honest pick for "
                "'one object, real world'."
            ),
            "strengths": ["Best accuracy among classic single-object trackers", "Tolerates some occlusion and appearance change"],
            "failure_modes": ["Long/full occlusion still loses it (no memory of what it can't see)", "Identical colours → histogram is ambiguous"],
            "params": [
                _d("psr_threshold", "Peak-to-sidelobe ratio", FLOAT, 0.1, "Confidence gate on the correlation peak; below it the tracker is 'unsure'.", "Raise it to fail loudly instead of tracking garbage.", 0.0, 0.5, 0.01),
                _d("max_age", "Lost-track memory", INT, 45, "Frames it keeps searching after losing confidence.", "", 1, 120, 1, "frames"),
            ],
        }


@register
class MossePlugin(_OpenCvSinglePlugin):
    id = "mosse"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "MOSSE (single-object)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "The original correlation filter — the founding father.",
            "description": (
                "The minimum-output-squared-error filter from 2010. Insanely fast, famously simple, "
                "and does NOT adapt: once the object rotates, scales or re-lights, it starts drifting. "
                "Perfect for teaching WHY people invented multi-filter trackers (CPU-only builds)."
            ),
            "strengths": ["Essentially free computationally", "Fine when object and background never change"],
            "failure_modes": ["Drifts on rotation/scale", "Loses on occlusion, never recovers", "Very sensitive to camera shake"],
            "params": [
                _d("max_age", "Lost-track memory", INT, 10, "Frames it keeps running after confidence collapses.", "", 1, 120, 1, "frames"),
            ],
        }


@register
class MilPlugin(_OpenCvSinglePlugin):
    id = "mil"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "MIL (single-object)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Multiple-Instance-Learning tracker — learns from bags of patches.",
            "description": (
                "Trains on a *bag* of slightly-shifted windows around the object instead of one exact "
                "positive patch, so noisy boundary boxes don't wreck it. A genuinely different idea "
                "worth watching fail gracefully on occlusion."
            ),
            "strengths": ["Robust to jitter / noisy init boxes", "No single-pixel drift snowball"],
            "failure_modes": ["Slower than pure filters", "Still a patch-matcher → true occlusion and scale change beat it"],
            "params": [
                _d("max_age", "Lost-track memory", INT, 30, "Frames it keeps searching after loss.", "", 1, 120, 1, "frames"),
            ],
        }


@register
class MedianFlowPlugin(_OpenCvSinglePlugin):
    id = "medianflow"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "MedianFlow (single-object)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Point-motion tracker: the median of many tiny trackers.",
            "description": (
                "Spreads dozens of small points over the object, tracks each with optical flow, and "
                "uses the MEDIAN of their motions so outliers can't hijack it. Its forward-backward "
                "error check literally asks 'did we get back where we started?' -- a beautiful way to "
                "spot when tracking has failed."
            ),
            "strengths": ["Honest built-in failure detection", "Robust to a few occluded points, adaptive"],
            "failure_modes": ["Needs texture: a plain same-colour ball has few trackable points", "All-points occlusion kills it", "Rough on scale changes"],
            "params": [
                _d("max_age", "Lost-track memory", INT, 20, "Frames the point cloud keeps living without confident motion.", "", 1, 120, 1, "frames"),
            ],
        }


@register
class NanoPlugin(_OpenCvSinglePlugin):
    id = "nano"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "Nano (single-object)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "OpenCV's tiny neural follower.",
            "description": (
                "A lightweight neural single-object tracker shipped with recent OpenCV. A compact "
                "network trained to re-locate whatever box you hand it -- much harder to shake than "
                "the old correlation filters, at a still tiny cost. Mirrors how real products track "
                "'this one person' on a phone."
            ),
            "strengths": ["Neural robustness beats classic filters", "Good for fast, well-textured objects"],
            "failure_modes": ["Very long / total occlusion still loses the target", "First-use downloads model files"],
            "params": [
                _d("max_age", "Lost-track memory", INT, 60, "Frames it keeps re-searching after confidence collapse.", "", 1, 180, 1, "frames"),
            ],
        }


@register
class VitPlugin(_OpenCvSinglePlugin):
    id = "vit"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "ViT (single-object)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Vision-Transformer single-object tracker.",
            "description": (
                "OpenCV's newest single-object tracker: a Vision Transformer that re-locates the target "
                "with attention over the whole frame. State-of-the-art for single-object follow on "
                "tricky scenes, showing how deep 'everything-to-everything' matching beats local "
                "patches. First run downloads the ViT weights."
            ),
            "strengths": ["Best single-object robustness here", "Attention-based long-range re-search"],
            "failure_modes": ["Slowest single-object option", "Weights download + VRAM on first run"],
            "params": [
                _d("max_age", "Lost-track memory", INT, 90, "Frames it keeps re-searching.", "", 1, 240, 1, "frames"),
            ],
        }


@register
class DaSiamRpnPlugin(_OpenCvSinglePlugin):
    id = "dasiamrpn"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "DaSiamRPN (single-object)",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Siamese tracker — 'find where this patch went' via a neural template matcher.",
            "description": (
                "A siamese-region-proposal single-object tracker. It memorises a template of the object "
                "and uses a tiny RPN head to regress 'where is this template now' each frame. The "
                "neural ancestor of the appearance-matching idea that modern MOT trackers (BoT-SORT "
                "ReID) reuse at scale. First run downloads the ONNX model."
            ),
            "strengths": ["Refreshes templates → adapts to change", "Solid under partial occlusion"],
            "failure_modes": ["Full occlusion → template gets stale", "Needs model files on first use"],
            "params": [
                _d("max_age", "Lost-track memory", INT, 40, "Frames it keeps searching after confidence loss.", "", 1, 120, 1, "frames"),
            ],
        }

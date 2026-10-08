"""Standalone native MOT trackers (ByteTrack, BoT-SORT, OC-SORT, DeepOC-SORT, FastTrack, TrackTrack).

These MOT trackers are implemented natively in standalone pure NumPy/SciPy without
PyTorch or Ultralytics model weights. They operate directly on synthetic detection streams
or standardized coordinate bounding boxes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.params import BOOL, FLOAT, GMC_OPTIONS, INT, SELECT, _d
from app.core.plugins.base import Engine, TrackerPlugin
from app.core.plugins.registry import register
from app.core.trackers import Track, TrackerState


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


from types import SimpleNamespace
from app.core.trackers_standalone import TRACKER_MAP


class UltralyticsEngine(Engine):
    """Adapter around standalone MOT trackers (ByteTrack, BoT-SORT, OC-SORT, DeepOC-SORT, FastTrack, TrackTrack).

    Runs self-contained pure-NumPy tracker implementations directly without needing PyTorch or external model weights.
    """

    def __init__(self, tracker_id: str, params: dict, fps: int, device: str = "cpu"):
        from app.core.trackers_standalone.basetrack import BaseTrack
        BaseTrack.reset_id()
        self._tracker_cls = TRACKER_MAP[tracker_id]
        cfg = SimpleNamespace(**params)
        cfg.device = device
        self.tracker = self._tracker_cls(args=cfg, frame_rate=fps)
        self.reid = bool(getattr(self.tracker, "with_reid", False))

    def update(self, dets, img):
        from app.core.trackers import Detection
        if isinstance(dets, list) and dets and isinstance(dets[0], Detection):
            dets = np.asarray([d.to_list() for d in dets], dtype=np.float64)
        else:
            dets = np.asarray(dets, dtype=np.float64)
        if dets.ndim == 1 and len(dets):
            dets = dets[None]
        if dets.ndim == 2 and dets.shape[1] < 5:
            dets = np.pad(dets, ((0, 0), (0, 5 - dets.shape[1])))
        out = self.tracker.update(dets, np.ascontiguousarray(img) if img is not None else None)
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


class _UltralyticsPlugin(TrackerPlugin):
    """Shared behaviour: standalone tracker execution in pure NumPy/SciPy."""

    engine = "ultralytics"
    mode = "multi"

    def build(self, params: dict, fps: int, device: str = "cpu") -> UltralyticsEngine:
        return UltralyticsEngine(self.id, params, fps, device)


@register
class ByteTrackPlugin(_UltralyticsPlugin):
    id = "bytetrack"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "ByteTrack",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "The workhorse: 'every detection counts' (BYTE strategy).",
            "description": (
                "Most trackers only trust strong detections. ByteTrack keeps TWO pools -- strong boxes "
                "matched first, weak ones matched to still-claimed tracks afterwards. That 'second "
                "chance' makes it survive blurry frames and brief self-occlusion, while a Kalman "
                "filter predicts where each track should be next."
            ),
            "strengths": ["Very robust to missing/weak detections", "Fast (no ReID network)", "Positions well under clutter"],
            "failure_modes": [
                "Long full occlusion → no appearance; can't re-find what it never saw re-emerge",
                "Identical objects → motion alone can't tell them apart",
            ],
            "params": [
                _d("track_high_thresh", "High-confidence threshold", FLOAT, 0.25, "Detection score that counts as 'strong' and gets matched first.",
                   "Lower it when your objects genuinely score low (small, distant, blurry).", 0.05, 0.95, 0.05),
                _d("track_low_thresh", "Low-confidence threshold", FLOAT, 0.1, "Second-chance pool: detections between low and high thresholds keep existing tracks alive.",
                   "The heart of BYTE. Raise track_high_thresh while keeping this low to lean on it more.", 0.0, 0.6, 0.01),
                _d("new_track_thresh", "New-track threshold", FLOAT, 0.25, "Confidence a detection needs to start a brand-new track when nothing matched.",
                   "Raise to stop shadows spawning ghost tracks (you'll wait longer for real ids).", 0.1, 0.99, 0.05),
                _d("track_buffer", "Lost track buffer", INT, 30, "How long a track with no detections is kept in limbo before deletion.",
                   "Bigger = more forgiving of occlusion, but ghosts linger and steal boxes.", 1, 120, 1, "frames"),
                _d("match_thresh", "Association IoU threshold", FLOAT, 0.8, "Overlap required between a detection and a track's Kalman prediction to be the same object.",
                   "Lower it for fast objects / shake; raise it to stop gluing neighbours together.", 0.1, 0.99, 0.01),
                _d("fuse_score", "Fuse detection score", BOOL, True, "Blend detection confidence into the association cost.",
                   "Off = trusts geometry only."),
            ],
        }


@register
class BotSortPlugin(_UltralyticsPlugin):
    id = "botsort"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "BoT-SORT",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "ByteTrack + appearance (ReID) + camera-motion compensation.",
            "description": (
                "Takes ByteTrack's association and bolts on two superpowers: (1) a ReID embedding "
                "model that remembers WHAT each object looks like and can re-match it after it fully "
                "disappears and reappears, and (2) GMC (Global Motion Compensation) that undoes "
                "camera shake before matching frames. The old default for busy real-world footage."
            ),
            "strengths": ["Survives full long occlusions via appearance re-matching", "Steady under camera shake (GMC)", "Strong ID consistency"],
            "failure_modes": [
                "Look-alikes (same shirt) → appearance is useless when everyone looks the same",
                "ReID adds latency; needs a network to download on first use",
                "Violent camera motion can outrun GMC",
            ],
            "params": [
                _d("track_high_thresh", "High-confidence threshold", FLOAT, 0.25, "Detection score that counts as strong for first association.", "", 0.05, 0.95, 0.05),
                _d("track_low_thresh", "Low-confidence threshold", FLOAT, 0.1, "Lower boundary of the second-chance (BYTE) pool.", "", 0.0, 0.6, 0.01),
                _d("new_track_thresh", "New-track threshold", FLOAT, 0.25, "Score needed to start a new track.", "", 0.1, 0.99, 0.05),
                _d("track_buffer", "Lost track buffer", INT, 30, "Frames a lost track is kept. Appearance matching can rescue it inside this window.",
                   "Longer buffer + with_reid ON = survives long disappearances.", 1, 120, 1, "frames"),
                _d("match_thresh", "Association IoU threshold", FLOAT, 0.8, "Overlap required between detection and prediction for a match.", "", 0.1, 0.99, 0.01),
                _d("fuse_score", "Fuse detection score", BOOL, True, "Blend confidence into matching.", ""),
                _d("gmc_method", "Camera compensation (GMC)", SELECT, "sparseOptFlow",
                   "How consecutive frames are aligned to cancel camera motion before matching.", options=GMC_OPTIONS),
                _d("proximity_thresh", "Proximity threshold", FLOAT, 0.5, "Min IoU to consider tracks 'proximate' enough for ReID.",
                   "Higher = stricter neighbourhood for appearance checks.", 0.0, 1.0, 0.05),
                _d("appearance_thresh", "Appearance threshold", FLOAT, 0.8, "Min appearance similarity to re-link by ReID. Raise to avoid swaps.",
                   "Lower = more tolerant of real appearance change (rotation, lighting).", 0.0, 1.0, 0.05),
                _d("with_reid", "Enable ReID appearance model", BOOL, False,
                   "Remember object appearance via an embedding model. Off ≈ ByteTrack + GMC.",
                   "Yes for long-occlusion tests; needs compute (and a model download)."),
                _d("model", "ReID model", SELECT, "auto",
                   "'auto' reuses detector features when available (falls back to no encoding in the pure simulator).",
                   options={"auto": "auto — detector features", "osnet_x0_25_msmt17.pt": "osnet_x0_25_msmt17.pt — classic ReID weights"}),
            ],
        }


@register
class OcSortPlugin(_UltralyticsPlugin):
    id = "ocsort"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "OC-SORT",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Observation-Centric SORT: trust the box, fix the velocity.",
            "description": (
                "Classic SORT-style filtering has a known failure: its Kalman 'state' drifts during "
                "occlusion and then re-associates wrong. OC-SORT keeps the observation-centric trick "
                "-- during the gap it reconstructs velocity from real detections (OCM) and rejects "
                "impossible direction flips (orientation consistency) so handovers stop stealing IDs."
            ),
            "strengths": ["Best-in-class ID consistency under occlusion among motion-only trackers", "No ReID needed on well-textured scenes"],
            "failure_modes": [
                "Camera shake → velocity guesses go haywire (no GMC)",
                "Identical objects with parallel motion → velocity can't separate them",
            ],
            "params": [
                _d("track_high_thresh", "High-confidence threshold", FLOAT, 0.25, "", "", 0.05, 0.95, 0.05),
                _d("track_low_thresh", "Low-confidence threshold", FLOAT, 0.1, "Second-stage pool (only used when use_byte is on).", "", 0.0, 0.6, 0.01),
                _d("new_track_thresh", "New-track threshold", FLOAT, 0.25, "", "", 0.1, 0.99, 0.05),
                _d("track_buffer", "Lost track buffer", INT, 30, "", "", 1, 120, 1, "frames"),
                _d("match_thresh", "Association IoU threshold", FLOAT, 0.8, "", "", 0.1, 0.99, 0.01),
                _d("fuse_score", "Fuse detection score", BOOL, True, "", ""),
                _d("delta_t", "Velocity window", INT, 3, "Frames back used to estimate a track's direction of travel.",
                   "Bigger = smoother velocity, slower reaction, 0 is okay too.", 0, 15, 1, "frames"),
                _d("inertia", "Direction-change penalty", FLOAT, 0.2, "How hard to penalise a track suddenly reversing its direction.",
                   "Higher = stops direction flips, but misses genuinely turning objects.", 0.0, 1.0, 0.05),
                _d("use_byte", "Enable Byte second-pass", BOOL, False, "Also match weak detections (ByteTrack style) in the second pass.",
                   "Helps with blurry frames; costs a little speed."),
            ],
        }


@register
class DeepOcSortPlugin(_UltralyticsPlugin):
    id = "deepocsort"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "DeepOCSORT",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "OC-SORT + ReID + camera compensation: the walled-off-completer.",
            "description": (
                "Adds appearance embeddings and GMC on top of OC-SORT's observation-centric matching. "
                "When occlusion breaks pure motion reasoning, appearance says 'wait, that Object is "
                "the same colour/shape', and GMC asks the camera shake to stand still. The strongest "
                "all-rounder in the family."
            ),
            "strengths": ["Excels at long occlusions and busy scenes", "GMC + ReID + motion stacked deliberately"],
            "failure_modes": [
                "Identical appearance still fools it (embeddings are trained on humans, balls are hard)",
                "Slowest of the six (embedding + flow every frame)",
            ],
            "params": [
                _d("track_high_thresh", "High-confidence threshold", FLOAT, 0.3, "", "", 0.05, 0.95, 0.05),
                _d("track_low_thresh", "Low-confidence threshold", FLOAT, 0.1, "", "", 0.0, 0.6, 0.01),
                _d("new_track_thresh", "New-track threshold", FLOAT, 0.3, "With ReID, it can start tracks later and more safely.", "", 0.1, 0.99, 0.05),
                _d("track_buffer", "Lost track buffer", INT, 30, "", "", 1, 120, 1, "frames"),
                _d("match_thresh", "Association IoU threshold", FLOAT, 0.8, "", "", 0.1, 0.99, 0.01),
                _d("fuse_score", "Fuse detection score", BOOL, True, "", ""),
                _d("delta_t", "Velocity window", INT, 3, "", "", 0, 15, 1, "frames"),
                _d("inertia", "Direction-change penalty", FLOAT, 0.2, "", "", 0.0, 1.0, 0.05),
                _d("use_byte", "Enable Byte second-pass", BOOL, False, "", ""),
                _d("gmc_method", "Camera compensation (GMC)", SELECT, "none", "Align frames to cancel camera motion.", options=GMC_OPTIONS),
                _d("alpha_fixed_emb", "Appearance memory decay", FLOAT, 0.95, "How slowly the remembered appearance of a track is updated (EMA).",
                   "High = very stable identity but slow to adapt to appearance change.", 0.5, 0.999, 0.01),
                _d("proximity_thresh", "Proximity threshold", FLOAT, 0.5, "", "", 0.0, 1.0, 0.05),
                _d("appearance_thresh", "Appearance threshold", FLOAT, 0.9, "Min appearance similarity to accept a ReID match.", "", 0.0, 1.0, 0.05),
                _d("with_reid", "Enable ReID appearance model", BOOL, False, "Appearance matching via an embedding model.", ""),
                _d("model", "ReID model", SELECT, "auto",
                   "'auto' reuses detector features when available (no-encoding fallback in pure simulator).",
                   options={"auto": "auto — detector features", "osnet_x0_25_msmt17.pt": "osnet_x0_25_msmt17.pt"}),
            ],
        }


@register
class FastTrackPlugin(_UltralyticsPlugin):
    id = "fasttrack"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "FastTracker",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Occlusion-aware ByteTrack with Kalman rollback.",
            "description": (
                "A 2025 lineage-tuning of ByteTrack built around ONE emergency: occlusion. The moment "
                "a box starts covering another, FastTracker saves its Kalman state, widens its search "
                "area, dampens velocity, and when the covered object re-emerges it ROLLS BACK the "
                "filter to before the confusion and re-links cleanly. Its whole parameter set is about "
                "that single manoeuvre."
            ),
            "strengths": ["Excellent occlusion-to-reappearance handovers", "Very good ID stability per compute cost"],
            "failure_modes": [
                "Occlusions lasting past active_occ_to_lost_thresh still kill tracks",
                "Boxes that look like one covering the other trigger the machinery on innocent scenes",
            ],
            "params": [
                _d("track_high_thresh", "High-confidence threshold", FLOAT, 0.25, "", "", 0.05, 0.95, 0.05),
                _d("track_low_thresh", "Low-confidence threshold", FLOAT, 0.1, "", "", 0.0, 0.6, 0.01),
                _d("new_track_thresh", "New-track threshold", FLOAT, 0.25, "", "", 0.1, 0.99, 0.05),
                _d("track_buffer", "Lost track buffer", INT, 30, "", "", 1, 120, 1, "frames"),
                _d("match_thresh", "Association IoU threshold", FLOAT, 0.8, "", "", 0.1, 0.99, 0.01),
                _d("fuse_score", "Fuse detection score", BOOL, True, "", ""),
                _d("reset_velocity_offset_occ", "Velocity rollback window", INT, 5, "Frames back to restore the Kalman velocity from when occlusion started.",
                   "Too big = stale velocity; too small = dropped momentum.", 1, 20, 1, "frames"),
                _d("reset_pos_offset_occ", "Position rollback window", INT, 3, "Frames back to restore the Kalman position.", "", 1, 20, 1, "frames"),
                _d("enlarge_bbox_occ", "Search-area enlarge", FLOAT, 1.1, "Scale the occluded track's box to widen where it keeps looking.",
                   "Upper bound of how far the object might crawl while hidden.", 1.0, 2.0, 0.05),
                _d("dampen_motion_occ", "Velocity dampening", FLOAT, 0.5, "How much to slow the Kalman velocity while occluded (0-1).",
                   "Less dampening → the track keeps flying in a straight line.", 0.0, 1.0, 0.05),
                _d("active_occ_to_lost_thresh", "Occlusion → lost cutoff", INT, 10, "Max consecutive occluded frames before the track is dropped.",
                   "Tune with your longest realistic occlusion.", 1, 60, 1, "frames"),
                _d("occ_cover_thresh", "Coverage to declare occlusion", FLOAT, 0.7, "Fraction of the track's area another box must cover to call it 'occluded'.", "", 0.3, 1.0, 0.05),
                _d("occ_reappear_window", "Re-find window", INT, 40, "How many frames a recently-occluded lost track stays re-findable.", "", 1, 120, 1, "frames"),
                _d("init_iou_suppress", "Init suppression", FLOAT, 0.7, "Suppress new tracks whose box overlaps an active track ≥ this (1 = off).",
                   "Stops the tracker double-counting one object as two.", 0.0, 1.0, 0.05),
            ],
        }


@register
class TrackTrackPlugin(_UltralyticsPlugin):
    id = "tracktrack"

    @classmethod
    def meta(cls) -> dict:
        return {
            "id": cls.id,
            "name": "TrackTrack",
            "engine": cls.engine,
            "mode": cls.mode,
            "tagline": "Multi-cue fusion: IoU + ReID + confidence + corner geometry.",
            "description": (
                "A CVPR-2025 tracker that fuses FOUR cues in a learned-ish cost (HMIoU, appearance, "
                "detection confidence, corner angles), then runs iterative assignment that relaxes "
                "its threshold a little each round instead of hard-batching. Tracks must be 'confirmed' "
                "by history before they can claim detections (that kills flicker-born ghosts)."
            ),
            "strengths": ["Very strong identity precision when detections are good", "Confirms tracks before trusting them"],
            "failure_modes": [
                "Needs decent detection scores (high new_track_thresh) → slow to start ids",
                "Six weights to juggle — easiest to mis-tune by a layperson",
            ],
            "params": [
                _d("track_high_thresh", "High-confidence threshold", FLOAT, 0.6, "Detections above this are 'strong'.", "", 0.1, 0.99, 0.01),
                _d("track_low_thresh", "Low-confidence threshold", FLOAT, 0.25, "Second-pass score window.", "", 0.0, 0.99, 0.01),
                _d("new_track_thresh", "New-track threshold", FLOAT, 0.7, "Score needed to START a track (TrackTrack is picky).", "", 0.1, 0.99, 0.01),
                _d("track_buffer", "Lost track buffer", INT, 30, "", "", 1, 120, 1, "frames"),
                _d("match_thresh", "Initial match threshold", FLOAT, 0.7, "Starting similarity needed; relaxed each iteration.", "", 0.1, 0.99, 0.01),
                _d("lost_match_thr", "Lost-track rebind", FLOAT, 0.0, "Relaxed threshold to rebind still-Lost tracks (0 disables). > match_thresh helps long occlusions.", "", 0.0, 1.0, 0.05),
                _d("iou_weight", "IoU weight", FLOAT, 0.5, "How much box-overlap (HMIoU) drives matching.", "", 0.0, 1.0, 0.05),
                _d("reid_weight", "Appearance weight", FLOAT, 0.5, "How much appearance drives matching (falls back to HMIoU without ReID).", "", 0.0, 1.0, 0.05),
                _d("conf_weight", "Confidence weight", FLOAT, 0.1, "How much detection score drives matching.", "", 0.0, 1.0, 0.05),
                _d("angle_weight", "Angle weight", FLOAT, 0.05, "How much corner angles drive matching.", "", 0.0, 1.0, 0.05),
                _d("penalty_p", "Weak-detection penalty", FLOAT, 0.2, "Cost penalty for low-confidence detections in a round.", "", 0.0, 1.0, 0.05),
                _d("penalty_q", "Recovered-detection penalty", FLOAT, 0.4, "Extra cost for detections that were deleted/recovered.", "", 0.0, 1.0, 0.05),
                _d("reduce_step", "Threshold relaxation", FLOAT, 0.05, "How much the match threshold relaxes per iteration.", "", 0.0, 0.5, 0.01),
                _d("tai_thr", "Init NMS threshold", FLOAT, 0.55, "IoU threshold for Track-Aware-Init suppression.", "", 0.0, 1.0, 0.05),
                _d("min_track_len", "Track confirmation", INT, 3, "Minimum frames of history before a track can claim detections.", "", 1, 30, 1, "frames"),
                _d("gmc_method", "Camera compensation (GMC)", SELECT, "sparseOptFlow", "Align frames to cancel camera motion.", options=GMC_OPTIONS),
                _d("with_reid", "Enable ReID appearance", BOOL, False, "Add appearance embeddings to the cue mix.", ""),
                _d("model", "ReID model", SELECT, "auto", "'auto' reuses detector features when available.", options={"auto": "auto — detector features", "osnet_x0_25_msmt17.pt": "osnet_x0_25_msmt17.pt"}),
            ],
        }

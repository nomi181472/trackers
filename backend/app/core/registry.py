"""Central registry of every tracker the simulator can run, together with all of
their tunable hyperparameters and -- crucially -- *layman explanations* of what
each knob does and when it matters.

Tracker families covered:
  * custom baseline     : greedy IoU, centroid (the 'before' picture)
  * ultralytics native  : ByteTrack, BoT-SORT, OC-SORT, DeepOCSORT, FastTracker,
                          TrackTrack  (their full config surfaces 1:1 here)
  * opencv classic      : KCF, CSRT, MOSSE, MIL, MedianFlow (single-object)
"""
from __future__ import annotations

INT = "int"
FLOAT = "number"
BOOL = "bool"
SELECT = "select"

MODEL_OPTIONS = {
    "yolov8n": "yolov8n — smallest/fastest, weakest accuracy",
    "yolov8s": "yolov8s — small, good balance",
    "yolov8m": "yolov8m — medium",
    "yolov8l": "yolov8l — large",
    "yolov8x": "yolov8x — biggest/best, slowest",
}
GMC_OPTIONS = {
    "sparseOptFlow": "sparseOptFlow — sparse optical flow (well-rounded default)",
    "orb": "orb — ORB keypoint matching (fast)",
    "sift": "sift — SIFT keypoint matching (needs the non-free license)",
    "ecc": "ecc — intensity-based alignment (robust, slower)",
    "none": "none — assume a fixed camera (cheapest)",
}


def _d(k, label, typ, default, tooltip, hint="", minv=None, maxv=None, step=None, unit="",
       options=None):
    p = dict(key=k, label=label, type=typ, default=default, tooltip=tooltip, hint=hint)
    if type is not None and typ in (INT, FLOAT):
        p.update(min=minv, max=maxv, step=step)
    if unit:
        p["unit"] = unit
    if options:
        p["options"] = options
    return p


# --------------------------------------------------------------------------- #
# Individual trackers                                                         #
# --------------------------------------------------------------------------- #

REGISTRY: list[dict] = [
    {
        "id": "greedy_iou",
        "name": "Greedy IoU",
        "engine": "custom",
        "mode": "multi",
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
    },
    {
        "id": "centroid",
        "name": "Centroid (distance)",
        "engine": "custom",
        "mode": "multi",
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
    },
    {
        "id": "bytetrack",
        "name": "ByteTrack",
        "engine": "ultralytics",
        "mode": "multi",
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
    },
    {
        "id": "botsort",
        "name": "BoT-SORT",
        "engine": "ultralytics",
        "mode": "multi",
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
    },
    {
        "id": "ocsort",
        "name": "OC-SORT",
        "engine": "ultralytics",
        "mode": "multi",
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
    },
    {
        "id": "deepocsort",
        "name": "DeepOCSORT",
        "engine": "ultralytics",
        "mode": "multi",
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
    },
    {
        "id": "fasttrack",
        "name": "FastTracker",
        "engine": "ultralytics",
        "mode": "multi",
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
    },
    {
        "id": "tracktrack",
        "name": "TrackTrack",
        "engine": "ultralytics",
        "mode": "multi",
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
    },
    {
        "id": "kcf",
        "name": "KCF (single-object)",
        "engine": "opencv",
        "mode": "single",
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
    },
    {
        "id": "csrt",
        "name": "CSRT (single-object)",
        "engine": "opencv",
        "mode": "single",
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
    },
    {
        "id": "mosse",
        "name": "MOSSE (single-object)",
        "engine": "opencv",
        "mode": "single",
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
    },
    {
        "id": "mil",
        "name": "MIL (single-object)",
        "engine": "opencv",
        "mode": "single",
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
    },
    {
        "id": "medianflow",
        "name": "MedianFlow (single-object)",
        "engine": "opencv",
        "mode": "single",
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
    },
{
        "id": "nano",
        "name": "Nano (single-object)",
        "engine": "opencv",
        "mode": "single",
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
    },
    {
        "id": "vit",
        "name": "ViT (single-object)",
        "engine": "opencv",
        "mode": "single",
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
    },
    {
        "id": "dasiamrpn",
        "name": "DaSiamRPN (single-object)",
        "engine": "opencv",
        "mode": "single",
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
    },
]

TRACKER_MAP = {t["id"]: t for t in REGISTRY}
ULTRALYTICS_TRACKERS = [t["id"] for t in REGISTRY if t["engine"] == "ultralytics"]
SINGLE_OBJECT_TRACKERS = [t["id"] for t in REGISTRY if t["mode"] == "single"]

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


def tracker_ids() -> list[str]:
    return [t["id"] for t in REGISTRY]


def get_tracker(tid: str) -> dict:
    if tid not in TRACKER_MAP:
        raise KeyError(f"Unknown tracker '{tid}'. Known: {tracker_ids()}")
    return TRACKER_MAP[tid]


def default_params(tid: str) -> dict:
    return {p["key"]: p["default"] for p in TRACKER_MAP[tid]["params"]}
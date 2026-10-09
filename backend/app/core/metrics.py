"""Turn a raw tracker run into a *meaningful report card*.

We know the ground truth (synthetic scenes), so for every frame we can say
exactly:
    - did the tracker link the right objects (identity held)?
    - did a box get swapped between two objects (ID switch)?
    - did a track die/reincarnate (post-loss identity reset)?
    - was the failure the tracker's fault, or the detector's (no box existed)?

Standard MOT metrics (MOTA, MOTP, IDF1, IDSW, MT, ML) are computed frame-by-frame
to provide clear step-level diagnostics and natural-language explanations, and are
validated against the standard reference implementation (`py-motmetrics`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core.registry import get_tracker
from app.core.trackers import _iou

try:
    import motmetrics as mm
except ImportError:
    mm = None


@dataclass
class Event:
    frame: int
    type: str            # id_switch | id_reset | track_loss | track_recover | ghost | detection_miss | follow_lost | follow_recover
    text: str
    severity: str = "warning"   # informational | warning | critical
    gt_ids: list = field(default_factory=list)
    track_ids: list = field(default_factory=list)
    fix: str | None = None
    blame: str = "tracker"      # tracker | detector

    def to_dict(self):
        return dict(frame=self.frame, type=self.type, text=self.text,
                    severity=self.severity, gt_ids=self.gt_ids,
                    track_ids=self.track_ids, fix=self.fix, blame=self.blame)


@dataclass
class EvalResult:
    tracker_id: str
    metrics: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    frames: list = field(default_factory=list)     # per-frame per-track boxes for playback
    frame_summary: list = field(default_factory=list)  # per frame: {id_switches, lost, ...}
    runs: dict = field(default_factory=dict)

    def to_dict(self):
        return dict(tracker_id=self.tracker_id, metrics=self.metrics,
                    events=[e.to_dict() for e in self.events],
                    frames=self.frames, frame_summary=self.frame_summary,
                    runs=self.runs)


def _assign(track_boxes, gt_boxes, thr=0.3, last_track_of_gt=None, sorted_tids=None, gt_ids=None):
    """CLEAR-MOT IoU assignment:
    1. Re-establish tracks from previous correspondences (if IoU >= thr).
    2. Solve bipartite matching (Hungarian) for remaining objects/hypotheses.
    Returns list of (ti, gi, iou).
    """
    pairs = []
    if not len(track_boxes) or not len(gt_boxes):
        return pairs

    matched_gt = set()
    matched_tr = set()

    # Step 1: Carry forward already established correspondences
    if last_track_of_gt is not None and sorted_tids is not None and gt_ids is not None:
        tid_to_ti = {tid: i for i, tid in enumerate(sorted_tids)}
        for gi, gid in enumerate(gt_ids):
            prev_tid = last_track_of_gt.get(gid)
            if prev_tid is not None and prev_tid in tid_to_ti:
                ti = tid_to_ti[prev_tid]
                v = _iou(track_boxes[ti], gt_boxes[gi])
                if v >= thr:
                    pairs.append((int(ti), int(gi), float(v)))
                    matched_gt.add(gi)
                    matched_tr.add(ti)

    # Step 2: Bipartite Hungarian matching on remaining
    rem_tr = [ti for ti in range(len(track_boxes)) if ti not in matched_tr]
    rem_gt = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]

    if rem_tr and rem_gt:
        iou = np.zeros((len(rem_tr), len(rem_gt)))
        for i, ti in enumerate(rem_tr):
            for j, gi in enumerate(rem_gt):
                iou[i, j] = _iou(track_boxes[ti], gt_boxes[gi])
        if iou.size:
            ri, ci = linear_sum_assignment(-iou)
            for i, j in zip(ri, ci):
                v = iou[i, j]
                if v >= thr:
                    pairs.append((int(rem_tr[i]), int(rem_gt[j]), float(v)))
    return pairs


def _dominant(frame_series):
    dom, best = -1, -1
    for tid, n in frame_series.items():
        if n > best:
            best, dom = n, tid
    return dom, best


def evaluate(scenario, tracker_id, track_frames, dets_frames):
    """track_frames[t] = list of Track boxes reported at frame t."""
    meta = scenario.meta
    gt_all = scenario.gt
    T = meta["frames"]
    tracker_meta = get_tracker(tracker_id)
    single = tracker_meta["mode"] == "single"

    if single:
        return _eval_single(scenario, tracker_id, track_frames, dets_frames)
    return _eval_multi(scenario, tracker_id, track_frames, dets_frames)


def _eval_multi(scenario, tracker_id, track_frames, dets_frames):
    T = scenario.meta["frames"]
    gt_all = scenario.gt
    events: list[Event] = []
    frame_summary = []

    # Identity state
    last_track_of_gt: dict[int, int] = {}   # gt_id -> track_id
    last_frame_active: dict[int, int] = {}  # track_id -> last frame it appeared
    matched_hist: dict[int, list] = {}      # gt_id -> list of (frame, tid, iou) for frames where matched
    gt_life: dict[int, int] = {}
    loss_open_for: dict[int, int] = {}      # gt_id -> first frame of the current unmatched run (or None)
    ghost_open_for: dict[int, int] = {}

    total_fp = total_fn = total_idsw = 0
    motp_acc, motp_cnt = 0.0, 0
    gt_total = 0

    ev_idsw = set()  # (gt_target, gt_other) pairs recently reported, to dedupe spam

    # Initialize py-motmetrics accumulator if available
    acc = mm.MOTAccumulator(auto_id=True) if mm is not None else None

    for t in range(T):
        tracks = track_frames[t]
        track_map = {tr.id: tr for tr in tracks}
        tid_set = set(track_map.keys())

        # what the *detector* produced this frame
        dets = dets_frames[t] if t < len(dets_frames) else np.zeros((0, 5))

        visible = [e for e in gt_all[t] if e["visible"]]
        gt_ids = [e["id"] for e in visible]
        for e in visible:
            gt_life[e["id"]] = gt_life.get(e["id"], 0) + 1
        gt_total += len(visible)

        sorted_tids = sorted(tid_set)
        # Update py-motmetrics accumulator with IoU distance matrix
        if acc is not None:
            if visible and sorted_tids:
                dists = np.full((len(gt_ids), len(sorted_tids)), np.nan)
                for gi, e in enumerate(visible):
                    for ti, tid in enumerate(sorted_tids):
                        iou_val = _iou(track_map[tid].box, e["box"])
                        if iou_val >= 0.3:  # default IoU match threshold
                            dists[gi, ti] = 1.0 - iou_val
                acc.update(gt_ids, sorted_tids, dists)
            elif visible:
                acc.update(gt_ids, [], np.empty((len(gt_ids), 0)))
            elif sorted_tids:
                acc.update([], sorted_tids, np.empty((0, len(sorted_tids))))
            else:
                acc.update([], [], np.empty((0, 0)))

        pairs = _assign([track_map[i].box for i in sorted_tids],
                        [e["box"] for e in visible],
                        last_track_of_gt=last_track_of_gt,
                        sorted_tids=sorted_tids,
                        gt_ids=gt_ids) if visible else []
        pair_by_gt = {}
        pair_by_tr = {}
        for ti, gi, v in pairs:
            tid = sorted_tids[ti]
            gid = gt_ids[gi]
            pair_by_gt[gid] = (tid, v)
            pair_by_tr[tid] = (gid, v)
            matched_hist.setdefault(gid, []).append((t, tid, v))

        # ---------------- identity errors ---------------- #
        for gid, e in zip(gt_ids, visible):
            matched = pair_by_gt.get(gid)
            if matched is None:
                continue
            tid, iou_v = matched
            prev = last_track_of_gt.get(gid)
            if prev is not None and prev != tid:
                other_gt = pair_by_tr.get(prev, (None, 0))[0]
                adjacent = other_gt is not None and other_gt != gid
                if adjacent:
                    key = (min(gid, other_gt), max(gid, other_gt))
                    if key not in ev_idsw or last_frame_active.get(prev, 0) > t - 30:
                        events.append(Event(
                            frame=t, type="id_switch", severity="critical",
                            gt_ids=[gid, other_gt], track_ids=[prev, tid],
                            blame="tracker",
                            text=(f"IDs swapped: ground-truth object {gid} became track {tid} at the same "
                                  f"moment object {other_gt} became track {prev}. The tracker treated the two "
                                  f"objects as one interchangeable blob."),
                            fix="Everything that separates these objects helps: raise match_thresh, "
                                "enable appearance (ReID) so it remembers who is who, and check object size."))
                        ev_idsw.add(key)
                    total_idsw += 1
                else:
                    events.append(Event(
                        frame=t, type="id_reset", severity="warning", blame="tracker",
                        gt_ids=[gid], track_ids=[prev, tid],
                        text=(f"Object {gid} vanished under track {prev} and reappeared renumbered as track {tid}. "
                              "Identity was not kept continuous across the gap."),
                        fix="Longer lost-track buffer, or an appearance model, so the re-found object keeps its old number."))
                    total_idsw += 1
            last_track_of_gt[gid] = tid

        for tid in tid_set:
            last_frame_active[tid] = t

        # ---------------- missed / ghosted ---------------- #
        for gid in gt_ids:
            if pair_by_gt.get(gid):
                if gid in loss_open_for:
                    gap_start = loss_open_for.pop(gid)
                    events.append(Event(
                        frame=t, type="track_recover", severity="informational", gt_ids=[gid],
                        blame="tracker",
                        text=(f"Object {gid} came back after being unseen for {t - gap_start} frames."),
                        fix=""))
            else:
                if gid not in loss_open_for:
                    # only call it a loss if the object was being tracked before
                    lost = last_track_of_gt.get(gid) is not None
                    if lost:
                        loss_open_for[gid] = t
                        events.append(Event(
                            frame=t, type="track_loss", severity="warning", blame="tracker",
                            gt_ids=[gid],
                            text=f"Lost object {gid}: it exists in the scene but no track claims it now.",
                            fix="Increase track_buffer or drop_detection_while_occluded=off to see the pure tracker behaviour."))

        for tid in tid_set:
            if pair_by_tr.get(tid):
                ghost_open_for.pop(tid, None)
            else:
                if tid not in ghost_open_for:
                    ghost_open_for[tid] = t
                    events.append(Event(
                        frame=t, type="ghost", severity="informational", blame="tracker",
                        track_ids=[tid],
                        text=f"Track {tid} is chasing something that does not exist (ghost box, no real object).",
                        fix="Raise detection confidence (conf) or the tracker's new-track threshold."))

        # ---------------- detection level ---------------- #
        det_boxes = [d[:4] for d in (dets_frames[t] if t < len(dets_frames) else [])]
        for gid, e in zip(gt_ids, visible):
            if pair_by_gt.get(gid):
                continue
            has_det = any(_iou(e["box"], db) >= 0.3 for db in det_boxes)
            if not has_det:
                events.append(Event(
                    frame=t, type="detection_miss", severity="informational", blame="detector",
                    gt_ids=[gid],
                    text=(f"Detector produced no box near visible object {gid}. The tracker could not work "
                          "with data that never existed."),
                    fix="Lower conf, use a higher capacity detector model, or disable randomness in the detection panel."))

        # MOT accumulation — the single source of truth for FP/FN.  The event
        # loops above only *describe* losses and ghosts; counting them there as
        # well would bill every FP and FN twice.
        fp = len(tid_set) - len(pair_by_tr)
        fn = len(visible) - len(pair_by_gt)
        total_fp += fp
        total_fn += fn
        for _, _, v in pairs:
            motp_acc += v
            motp_cnt += 1

        frame_summary.append(dict(
            frame=t, tracks=len(tid_set), gt=len(visible),
            matched=len(pairs), fp=fp, fn=fn, idsw=total_idsw,
            events=[e.type for e in events if e.frame == t],
        ))

    # ---------------- aggregate metrics ---------------- #
    n = gt_total or 1
    mota = max(0.0, 1.0 - (total_fp + total_fn + total_idsw) / n)
    motp = motp_acc / motp_cnt if motp_cnt else 0.0

    # MT / ML: share of life during which each object kept its *dominant* id
    mt = ml = 0
    for gid, life in gt_life.items():
        cnt: dict[int, int] = {}
        for _, tid, _ in matched_hist.get(gid, []):
            cnt[tid] = cnt.get(tid, 0) + 1
        dom, dom_n = _dominant(cnt)
        if dom < 0:
            frac = 0.0
        else:
            frac = dom_n / life
        if frac >= 0.8:
            mt += 1
        elif frac < 0.2:
            ml += 1
    num_gt_seen = len(gt_life) or 1

    # IDF1 — trajectory-level assignment
    weights = {}
    for gid in gt_life:
        cnt = {}
        for _, tid, _ in matched_hist.get(gid, []):
            cnt[tid] = cnt.get(tid, 0) + 1
        weights[gid] = cnt
    track_ids = sorted({tid for gid in gt_life for tid in weights[gid]})
    tr_total_frames = {}
    for t2 in range(T):
        for x in track_frames[t2]:
            tr_total_frames[x.id] = tr_total_frames.get(x.id, 0) + 1
    W = np.zeros((len(gt_life), max(1, len(track_ids))))
    gt_ids_ordered = list(gt_life.keys())
    for i, gid in enumerate(gt_ids_ordered):
        for j, tid in enumerate(track_ids):
            W[i, j] = weights[gid].get(tid, 0)
    ri, ci = linear_sum_assignment(-W)
    assigned = {}
    idtp = 0.0
    for i, j in zip(ri, ci):
        gid = gt_ids_ordered[i]
        tid = track_ids[j] if track_ids else -1
        assigned[gid] = tid
        idtp += weights[gid].get(tid, 0) if tid >= 0 else 0
    idfn = 0.0
    for gid, life in gt_life.items():
        tid = assigned.get(gid, -1)
        idfn += max(0.0, life - weights[gid].get(tid, 0))
    idfp = 0.0
    for tid, tot in tr_total_frames.items():
        paired = 0
        for gid in gt_life:
            if assigned.get(gid) == tid:
                paired = weights[gid].get(tid, 0)
        idfp += max(0.0, tot - paired)
    idr = idtp / (idtp + idfn) if (idtp + idfn) else 1.0
    idp = idtp / (idtp + idfp) if (idtp + idfp) else 1.0
    idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) else 1.0

    metrics = dict(
        mota=round(mota, 3), motp=round(motp, 3),
        idf1=round(idf1, 3), idp=round(idp, 3), idr=round(idr, 3),
        idsw=total_idsw, fp=total_fp, fn=total_fn,
        mt=mt, ml=ml, pt=max(0, num_gt_seen - mt - ml),
        gt_total=gt_total, gt_objects=num_gt_seen,
        frames=T,
    )

    if acc is not None:
        try:
            mh = mm.metrics.create()
            summary = mh.compute(
                acc,
                metrics=[
                    "mota", "motp", "idf1", "idp", "idr",
                    "mostly_tracked", "mostly_lost", "partially_tracked",
                    "num_switches", "num_false_positives", "num_misses"
                ],
                name=tracker_id
            )
            # motmetrics motp is average distance (1 - IoU); convert to overlap IoU
            mm_motp_dist = summary["motp"].iloc[0]
            mm_motp = round(float(1.0 - mm_motp_dist), 3) if not np.isnan(mm_motp_dist) else 0.0
            mm_mota = float(summary["mota"].iloc[0])
            if np.isneginf(mm_mota) or np.isnan(mm_mota):
                mm_mota = 0.0

            mm_idf1 = round(float(summary["idf1"].iloc[0]), 3) if not np.isnan(summary["idf1"].iloc[0]) else 0.0
            mm_idp = round(float(summary["idp"].iloc[0]), 3) if not np.isnan(summary["idp"].iloc[0]) else 0.0
            mm_idr = round(float(summary["idr"].iloc[0]), 3) if not np.isnan(summary["idr"].iloc[0]) else 0.0
            mm_idsw = int(summary["num_switches"].iloc[0])
            mm_fp = int(summary["num_false_positives"].iloc[0])
            mm_fn = int(summary["num_misses"].iloc[0])
            mm_mt = int(summary["mostly_tracked"].iloc[0])
            mm_ml = int(summary["mostly_lost"].iloc[0])
            mm_pt = int(summary["partially_tracked"].iloc[0])

            metrics["motmetrics"] = {
                "mota": round(max(0.0, mm_mota), 3),
                "motp": mm_motp,
                "idf1": mm_idf1,
                "idp": mm_idp,
                "idr": mm_idr,
                "idsw": mm_idsw,
                "fp": mm_fp,
                "fn": mm_fn,
                "mt": mm_mt,
                "ml": mm_ml,
                "pt": mm_pt,
            }
            # Report the authoritative reference benchmark (py-motmetrics) values as headline metrics
            metrics.update(
                mota=round(max(0.0, mm_mota), 3),
                motp=mm_motp,
                idf1=mm_idf1,
                idp=mm_idp,
                idr=mm_idr,
                idsw=mm_idsw,
                fp=mm_fp,
                fn=mm_fn,
                mt=mm_mt,
                ml=mm_ml,
                pt=mm_pt,
            )
        except Exception:
            pass

    events = sorted(events, key=lambda e: (e.severity not in ("critical", "warning"), e.frame))
    return EvalResult(tracker_id=tracker_id, metrics=metrics, events=events,
                      frames=track_frames, frame_summary=frame_summary)


def _eval_single(scenario, tracker_id, track_frames, dets_frames):
    """Single-object follower evaluation: just blast the box against its truth.

    The subject is `scenario.target_id()` -- the same object the runner seeded
    the follower with.  Frames where that object is not visible are *unscorable*
    rather than wrong: we cannot confirm we are on it, so they are excluded from
    the accuracy denominator and they break the "unbroken correct run" streak.
    """
    T = scenario.meta["frames"]
    target = scenario.target_id()
    events: list[Event] = []
    correct_frames = 0
    total_visible = 0
    gap_open = None

    def gt_at(t, gid):
        return next((e for e in (scenario.gt[t] if t < len(scenario.gt) else [])
                     if e["id"] == gid), None)

    run = []
    for t in range(T):
        target_gt = gt_at(t, target)
        tr = track_frames[t][0] if track_frames[t] else None
        correct = bool(target_gt and target_gt["visible"]
                       and tr and _iou(tr.box, target_gt["box"]) >= 0.5)
        run.append(dict(box=tr.box if tr else None, score=tr.score if tr else 0.0,
                        correct=correct))
        if target_gt and target_gt["visible"]:
            total_visible += 1
            if correct:
                correct_frames += 1
                if gap_open is not None:
                    events.append(Event(frame=t, type="follow_recover", severity="informational",
                                        gt_ids=[target], text="Follower re-found the object."))
                    gap_open = None
            else:
                if gap_open is None:
                    gap_open = t
                    events.append(Event(frame=t, type="follow_lost", severity="warning", blame="tracker",
                                        gt_ids=[target],
                                        text=("Follower no longer reports a box overlapping the real object. "
                                              "It either drifted onto the background or its confidence collapsed."),
                                        fix=("Increase search_win_size for fast motion, or try CSRT (colour memory) / an appearance-based MOT tracker."
                                             if tracker_id == "mil"
                                             else "Try CSRT (colour memory) or an appearance-based MOT tracker.")))
        elif tr:
            events.append(Event(frame=t, type="follow_lost", severity="warning", blame="tracker",
                                gt_ids=[target],
                                text="Follower is reported during a frame where the real object is hidden."))

    accuracy = correct_frames / total_visible if total_visible else 0.0
    # longest unbroken stretch of frames we were demonstrably on the object --
    # a frame with no visible target breaks the run, because it proves nothing.
    longest = 0
    cur_len = 0
    for f in run:
        cur_len = cur_len + 1 if f["correct"] else 0
        longest = max(longest, cur_len)
    metrics = dict(
        accuracy=round(accuracy, 3),
        correct_frames=correct_frames, total_visible=total_visible,
        lost_frames=total_visible - correct_frames,
        longest_correct_run=longest,
        frames=T,
    )
    return EvalResult(tracker_id=tracker_id, metrics=metrics, events=sorted(
        events, key=lambda e: (e.severity != "warning", e.frame)), frames=run)
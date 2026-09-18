"""The simulation engine: run every requested tracker over a scenario and
produce a full report card (metrics + timeline + annotated video + thumbnails).

All trackers under comparison see the *identical* detection stream -- the only
variable is the tracker itself. That is what makes side-by-side profiling fair.
"""
from __future__ import annotations

import os

import numpy as np

from app.core.explainer import explain
from app.core.metrics import evaluate
from app.core.registry import get_tracker
from app.core.trackers import build_engine, color_for, _iou


class SimDetector:
    """Synthetic detector: ground-truth boxes with controllable imperfection.

    Lets the user isolate *tracker* failures from *detector* failures by
    fiddling only with the detector knobs (miss rate, ghost rate, jitter).
    """

    def __init__(self, scenario, params: dict):
        self.scenario = scenario
        self.conf = float(params.get("conf", 0.25))
        self.miss_rate = float(params.get("miss_rate", 0.0))
        self.fp_rate = float(params.get("fp_rate", 0.0))
        self.jitter = int(params.get("jitter", 2))
        self.drop_occluded = bool(params.get("drop_while_occluded", True))
        self.rng = np.random.default_rng(scenario.seed + 999)
        self.blur = bool(scenario.blur)

    def dets_for_frame(self, t: int):
        W, H = self.scenario.width, self.scenario.height
        dets = []
        for e in self.scenario.gt[t]:
            if e["occluded"] and self.drop_occluded:
                continue
            # simulated detector noise on score
            score = 0.90 + self.rng.normal(0, 0.04)
            if self.blur:
                score *= 0.62
            if e["occluded"]:  # visible-through-wall option
                score *= 0.55
            if self.rng.random() < self.miss_rate:
                continue
            if score < self.conf:
                continue
            box = np.array(e["box"], dtype=np.float64)
            if self.jitter:
                box += self.rng.uniform(-self.jitter, self.jitter, 4)
            dets.append([box[0], box[1], box[2], box[3], max(0.05, score)])
        # false positives / ghosts
        for _ in range(int(self.rng.random() < self.fp_rate)):
            w = self.rng.uniform(18, 40)
            h = w * self.rng.uniform(0.8, 1.2)
            x = self.rng.uniform(0, W - w)
            y = self.rng.uniform(0, H - h)
            dets.append([x, y, x + w, y + h, self.rng.uniform(self.conf + 0.05, 0.7)])
        return np.array(dets) if dets else np.zeros((0, 5))


def write_video(frames, path: str, fps: int):
    """Best-effort H.264 writer (avc1 -> ffmpeg transcode -> mp4v fallback)."""
    import cv2
    video = None
    started = None
    for fc in ("avc1", "avc3", "mp4v", "XVID"):
        v = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fc), fps, (frames[0].shape[1], frames[0].shape[0]))
        if v.isOpened():
            video, started = v, fc
            break
    if video is None:
        raise RuntimeError("No working video codec available")
    for f in frames:
        video.write(f)
    video.release()
    try:
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg and started != "avc1":
            tmp = path + ".raw.mp4"
            os.replace(path, tmp)
            code = os.system(
                f'"{ffmpeg}" -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p -an "{path}"')
            if code != 0:
                os.replace(tmp, path)
            else:
                os.remove(tmp)
                started = "avc1(ffmpeg)"
    except Exception:  # noqa: BLE001
        pass
    return str(started)


def render_event_thumb(scenario, t: int, tracks, event_text: str, out_path: str, highlight: bool = True):
    import cv2
    img = scenario.render_annotated_frame(t, tracks, draw_gt=True, labels=True)
    if highlight:
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 38), -1)
        cv2.putText(img, event_text[:82], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 90), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
    return out_path


def _attach_preview(scenario, job_id: str):
    """Write a plain scenario preview clip and expose it on scenario.meta."""
    from app import config
    preview_path = config.SCENARIOS_DIR / f"{job_id}_preview.mp4"
    codec = write_video(scenario.frames, str(preview_path), scenario.fps)
    scenario.meta["preview_url"] = f"/api/media/{preview_path.name}"
    scenario.meta["preview_codec"] = codec
    return scenario.meta["preview_url"]


def run_simulation(scenario, specs: list[dict], detection_params: dict,
                   progress=None, out_dir: str = "data/jobs", job_id: str = "local") -> dict:
    """specs: [{"tracker_id":..., "params": {...}}]"""
    os.makedirs(out_dir, exist_ok=True)
    T = scenario.meta["frames"]

    def tick(msg, frac, detail=""):
        if progress:
            progress(msg, frac, detail)

    tick("Building scenario data", 0.03)
    det = SimDetector(scenario, detection_params)
    dets_frames = [det.dets_for_frame(t) for t in range(T)]

    n_total = len(specs) * (T + 3)
    done = 0
    results = []

    for si, spec in enumerate(specs):
        tid = spec["tracker_id"]
        meta = get_tracker(tid)
        tick(f"Starting {meta['name']}", 0.05 + 0.9 * (si / max(1, len(specs))))
        try:
            engine = build_engine(tid, spec["params"], scenario.fps,
                                  device=detection_params.get("device", "cpu"))
        except Exception as e:  # noqa: BLE001
            results.append(dict(tracker_id=tid, name=meta["name"], tagline=meta["tagline"],
                                mode=meta["mode"], error=str(e)))
            continue

        track_frames = []
        for t in range(T):
            img = scenario.frames[t]
            if meta["mode"] == "single":
                if t == 0:
                    target = scenario.gt[0][0]
                    engine.init(img, target["box"])
                state = engine.update(np.zeros((0, 5)), img)
            else:
                state = engine.update(dets_frames[t], img)
            track_frames.append(state.active)
            done += 1
            if done % max(1, T // 5) == 0:
                tick(f"{meta['name']}: frame {t + 1}/{T}",
                     0.1 + 0.8 * ((si * T + t) / max(1, n_total)))

        tick(f"Scoring {meta['name']}", 0.9 + 0.05 * si)
        eval_res = evaluate(scenario, tid, track_frames, dets_frames)
        report = explain(scenario.meta, tid, eval_res.metrics, [e.to_dict() for e in eval_res.events])

        # ---- render annotated video ---- #
        annotated = []
        for t in range(T):
            annotated.append(scenario.render_annotated_frame(t, track_frames[t], draw_gt=True, labels=True))
        vid_name = f"{job_id}_{tid}.mp4"
        vid_path = os.path.join(out_dir, vid_name)
        codec = write_video(annotated, vid_path, scenario.fps)

        # ---- event thumbnails ---- #
        thumbs = []
        for e in eval_res.events[:6]:
            if e.severity == "informational":
                continue
            tn = f"{job_id}_{tid}_e{e.frame}.jpg"
            tp = os.path.join(out_dir, tn)
            tracks_at = track_frames[e.frame] if e.frame < len(track_frames) else []
            render_event_thumb(scenario, e.frame, tracks_at, e.text, tp)
            thumbs.append(dict(frame=e.frame, type=e.type, severity=e.severity,
                               url=f"/api/media/{tn}"))

        verdict = report["grade"]
        results.append(dict(
            tracker_id=tid, name=meta["name"], tagline=meta["tagline"], mode=meta["mode"],
            metrics=eval_res.metrics, events=[e.to_dict() for e in eval_res.events],
            frame_summary=eval_res.frame_summary, report=report,
            video_url=f"/api/media/{vid_name}", codec=codec,
            thumbnails=thumbs,
        ))
        tick(f"Done with {meta['name']}", 0.9 + 0.1 * (len(results) / max(1, len(specs))))

    try:
        _attach_preview(scenario, job_id)
    except Exception as e:  # noqa: BLE001
        scenario.meta["preview_warning"] = str(e)
    tick("Finished", 1.0)
    return dict(scenario=scenario.meta, detection=detection_params,
                trackers=[r["tracker_id"] for r in results], results=results)
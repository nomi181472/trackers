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
from app.core.plugins import build_engine
from app.core.registry import DETECTION_DEFAULTS, get_tracker
from app.core.trackers import Track, TrackerState


class SimDetector:
    """Synthetic detector: ground-truth boxes with controllable imperfection.

    Lets the user isolate *tracker* failures from *detector* failures by
    fiddling only with the detector knobs (miss rate, ghost rate, jitter).
    """

    def __init__(self, scenario, params: dict):
        params = {**DETECTION_DEFAULTS, **(params or {})}
        self.scenario = scenario
        self.conf = float(params["conf"])
        self.miss_rate = float(params["miss_rate"])
        self.fp_rate = float(params["fp_rate"])
        self.jitter = int(params["jitter"])
        self.drop_occluded = bool(params["drop_while_occluded"])
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
            low_score = min(float(self.conf + 0.05), 0.68)
            dets.append([x, y, x + w, y + h, self.rng.uniform(low_score, max(low_score + 0.01, 0.70))])
        return np.array(dets) if dets else np.zeros((0, 5))


def get_ffmpeg_exe() -> str | None:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def video_to_data_url(path: str) -> str | None:
    import base64
    from pathlib import Path
    p = Path(path)
    # Always prioritize WebM for HTML5 browser compatibility
    candidates = [p.with_suffix(".webm"), p]
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            mime = "video/webm" if cand.suffix == ".webm" else "video/mp4"
            try:
                b64 = base64.b64encode(cand.read_bytes()).decode("ascii")
                return f"data:{mime};base64,{b64}"
            except Exception:
                pass
    return None


def img_to_data_url(path: str) -> str | None:
    import base64
    from pathlib import Path
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        try:
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            pass
    return None


def write_video(frames, path: str, fps: int):
    """Write browser-native WebM video (VP80) for 100% universal HTML5 playback.
    Also ensures the target path exists if an .mp4 path was requested.
    """
    import cv2
    import gc
    from pathlib import Path

    frames_iter = iter(frames)
    try:
        first_frame = next(frames_iter)
    except StopIteration:
        return "empty"

    h, w = first_frame.shape[0], first_frame.shape[1]
    p = Path(path)
    webm_path = str(p.with_suffix(".webm"))
    from app.core.trackers import open_video_writer

    started = None
    frame_list = []
    need_frames = (p.suffix == ".mp4")

    # 1. Always generate WebM with VP80 (guaranteed browser playback)
    with open_video_writer(webm_path, cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h)) as v_webm:
        if not v_webm.isOpened():
            with open_video_writer(webm_path, cv2.VideoWriter_fourcc(*"VP90"), fps, (w, h)) as v_webm_alt:
                if v_webm_alt.isOpened():
                    started = "vp8(webm)"
                    v_webm_alt.write(first_frame)
                    if need_frames:
                        frame_list.append(first_frame)
                    for f in frames_iter:
                        v_webm_alt.write(f)
                        if need_frames:
                            frame_list.append(f)
                else:
                    frame_list = [first_frame] + list(frames_iter)
        else:
            started = "vp8(webm)"
            v_webm.write(first_frame)
            if need_frames:
                frame_list.append(first_frame)
            for f in frames_iter:
                v_webm.write(f)
                if need_frames:
                    frame_list.append(f)

    # 2. If target path is .mp4, also write .mp4
    if p.suffix == ".mp4":
        ffmpeg = get_ffmpeg_exe()
        if ffmpeg:
            tmp = path + ".raw.mp4"
            with open_video_writer(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) as v:
                if v.isOpened():
                    for f in frame_list:
                        v.write(f)
            if os.path.exists(tmp):
                code = os.system(
                    f'"{ffmpeg}" -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p -an "{path}"')
                if code == 0:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                else:
                    os.replace(tmp, path)
        elif not os.path.exists(path) and os.path.exists(webm_path):
            import shutil
            shutil.copyfile(webm_path, path)

    gc.collect()
    return str(started or "webm")


def render_event_thumb(scenario, t: int, tracks, event_text: str, out_path: str, highlight: bool = True):
    import cv2
    # no corner HUD here: the event banner below owns the top strip
    img = scenario.render_annotated_frame(t, tracks, draw_gt=True, labels=True, show_frame_no=False)
    if highlight:
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 38), -1)
        cv2.putText(img, event_text[:82], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 90), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
    return out_path


def _attach_preview(scenario, job_id: str):
    """Write a plain scenario preview clip and expose it on scenario.meta."""
    from app import config
    import gc
    preview_path = config.SCENARIOS_DIR / f"{job_id}_preview.webm"
    def _preview_gen():
        for t in range(scenario.meta["frames"]):
            yield scenario.render_annotated_frame(t, tracks=[], draw_gt=True)
    codec = write_video(_preview_gen(), str(preview_path), scenario.fps)
    scenario.meta["preview_url"] = f"/api/media/{preview_path.name}"
    scenario.meta["preview_codec"] = codec
    data_url = video_to_data_url(str(preview_path))
    if data_url:
        scenario.meta["preview_data_url"] = data_url
    gc.collect()
    return scenario.meta["preview_url"]


def run_simulation(scenario, specs: list[dict], detection_params: dict,
                   progress=None, out_dir: str | None = None, job_id: str = "local") -> dict:
    """specs: [{"tracker_id":..., "params": {...}}]"""
    from app import config
    out_dir = str(out_dir) if out_dir is not None else str(config.JOBS_DIR)
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
        try:
            meta = get_tracker(tid)
        except KeyError as e:  # noqa: PERF203  (one bad id must not kill the job)
            results.append(dict(tracker_id=tid, name=tid, tagline="", mode="multi",
                                error=str(e.args[0] if e.args else e)))
            continue
        tick(f"Starting {meta['name']}", 0.05 + 0.9 * (si / max(1, len(specs))))
        try:
            engine = build_engine(tid, spec["params"], scenario.fps,
                                  device=detection_params.get("device", "cpu"))
        except Exception as e:  # noqa: BLE001
            results.append(dict(tracker_id=tid, name=meta["name"], tagline=meta["tagline"],
                                mode=meta["mode"], error=str(e)))
            continue

        # A follower needs one box to latch onto, and that box has to belong to
        # the object the scorer will later measure against -- `Scenario` owns
        # that choice so the two can never drift apart.  Seeding it on a frame
        # where the object is hidden behind the wall would lose it immediately.
        seed_frame = None
        if meta["mode"] == "single":
            seed_frame = scenario.first_visible_target_frame()
            if seed_frame is None:
                results.append(dict(
                    tracker_id=tid, name=meta["name"], tagline=meta["tagline"], mode=meta["mode"],
                    error=(f"Ground-truth object {scenario.target_id()} is never visible in this "
                           f"scenario, so there is nothing to initialise the follower on.")))
                continue

        track_frames = []
        latencies_ms = []
        try:
            for t in range(T):
                # Check for user-requested cancellation at every frame tick
                if progress and hasattr(progress, "check_cancelled"):
                    progress.check_cancelled()
                img = scenario.frames[t]
                import time
                if meta["mode"] == "single":
                    if t < seed_frame:
                        t0 = time.perf_counter()
                        state = TrackerState(active=[])
                        latencies_ms.append(round((time.perf_counter() - t0) * 1000.0, 2))
                    elif t == seed_frame:
                        target = next(e for e in scenario.gt[t]
                                      if e["id"] == scenario.target_id())
                        t0 = time.perf_counter()
                        engine.init(img, target["box"])
                        latencies_ms.append(round((time.perf_counter() - t0) * 1000.0, 2))
                        state = TrackerState(active=[Track(getattr(engine, "_id", 0), list(target["box"]), score=1.0)])
                    else:
                        t0 = time.perf_counter()
                        state = engine.update(np.zeros((0, 5)), img)
                        latencies_ms.append(round((time.perf_counter() - t0) * 1000.0, 2))
                else:
                    t0 = time.perf_counter()
                    state = engine.update(dets_frames[t], img)
                    latencies_ms.append(round((time.perf_counter() - t0) * 1000.0, 2))
                track_frames.append(state.active)
                done += 1
                if done % max(1, T // 5) == 0:
                    tick(f"{meta['name']}: frame {t + 1}/{T}",
                         0.1 + 0.8 * ((si * T + t) / max(1, n_total)))
        except Exception as e:  # noqa: BLE001  (one bad tracker must not kill the job)
            from app.core.jobs import JobCancelledError
            if isinstance(e, JobCancelledError):
                raise  # re-raise so _run() sets status=cancelled, not status=error
            results.append(dict(tracker_id=tid, name=meta["name"], tagline=meta["tagline"],
                                mode=meta["mode"], error=str(e)))
            continue

        tick(f"Scoring {meta['name']}", 0.9 + 0.05 * si)
        eval_res = evaluate(scenario, tid, track_frames, dets_frames)
        if latencies_ms:
            avg_ms = float(np.mean(latencies_ms))
            eval_res.metrics["avg_time_ms"] = round(avg_ms, 2)
            eval_res.metrics["fps"] = round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0
            eval_res.metrics["latencies"] = latencies_ms
        report = explain(scenario.meta, tid, eval_res.metrics, [e.to_dict() for e in eval_res.events])

        # ---- render annotated video ---- #
        vid_name = f"{job_id}_{tid}.mp4"
        vid_path = os.path.join(out_dir, vid_name)
        def _annotated_gen():
            for t in range(T):
                yield scenario.render_annotated_frame(t, track_frames[t], draw_gt=True,
                                                      labels=True,
                                                      tracker_name=meta["name"])
        codec = write_video(_annotated_gen(), vid_path, scenario.fps)
        vid_data_url = video_to_data_url(vid_path)

        # ---- event thumbnails ---- #
        thumbs = []
        for e in eval_res.events[:6]:
            if e.severity == "informational":
                continue
            tn = f"{job_id}_{tid}_e{e.frame}.jpg"
            tp = os.path.join(out_dir, tn)
            tracks_at = track_frames[e.frame] if e.frame < len(track_frames) else []
            render_event_thumb(scenario, e.frame, tracks_at, e.text, tp)
            thumb_info = dict(frame=e.frame, type=e.type, severity=e.severity,
                              url=f"/api/media/{tn}")
            t_data = img_to_data_url(tp)
            if t_data:
                thumb_info["data_url"] = t_data
            thumbs.append(thumb_info)

        verdict = report["grade"]
        res_item = dict(
            tracker_id=tid, name=meta["name"], tagline=meta["tagline"], mode=meta["mode"],
            metrics=eval_res.metrics, events=[e.to_dict() for e in eval_res.events],
            frame_summary=eval_res.frame_summary, report=report,
            video_url=f"/api/media/{vid_name}", codec=codec,
            thumbnails=thumbs,
        )
        if vid_data_url:
            res_item["video_data_url"] = vid_data_url
        results.append(res_item)
        tick(f"Done with {meta['name']}", 0.9 + 0.1 * (len(results) / max(1, len(specs))))

    # ---- Generate and save comparison chart PNG ---- #
    chart_name = f"{job_id}_chart.png"
    chart_path = os.path.join(out_dir, chart_name)
    chart_url = None
    try:
        _render_simulation_chart(results, chart_path)
        chart_url = f"/api/media/{chart_name}"
    except Exception as e:
        scenario.meta["chart_warning"] = str(e)

    try:
        _attach_preview(scenario, job_id)
    except Exception as e:  # noqa: BLE001
        scenario.meta["preview_warning"] = str(e)
    tick("Finished", 1.0)
    return dict(
        scenario=scenario.meta,
        detection=detection_params,
        trackers=[r["tracker_id"] for r in results],
        results=results,
        chart_url=chart_url,
    )


def _render_simulation_chart(results: list[dict], out_path: str):
    """Renders a high-res comparison chart (Trade-off & Errors) and saves as PNG."""
    if not results:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0f172a")

    # Colors for trackers
    palette = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f87171", "#e879f9", "#2dd4bf"]

    # 1. Left Subplot: Latency vs MOTA trade-off
    ax1.set_facecolor("#1e293b")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    ax1.set_title("Speed vs Accuracy Trade-off", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Latency (ms / frame) — Lower is Faster", color="#94a3b8", fontsize=10, labelpad=8)
    ax1.set_ylabel("MOTA (%) — Higher is Better", color="#94a3b8", fontsize=10, labelpad=8)
    ax1.tick_params(colors="#94a3b8")

    # Sweet spot highlight box (top-left)
    ax1.axhspan(50, 100, xmin=0, xmax=0.5, color="#34d399", alpha=0.07)

    for idx, r in enumerate(results):
        m = r.get("metrics", {})
        mota = (m.get("mota", 0.0) or 0.0) * 100
        latency = m.get("avg_time_ms", 1.0) or 1.0
        color = palette[idx % len(palette)]
        total_err = (m.get("fp", 0) or 0) + (m.get("fn", 0) or 0) + (m.get("idsw", 0) or 0)
        bubble_size = max(80, min(800, 120 + total_err * 20))

        ax1.scatter(latency, mota, s=bubble_size, color=color, alpha=0.75, edgecolors="#ffffff", linewidth=1.5, label=r["name"])
        ax1.annotate(r["name"], (latency, mota), textcoords="offset points", xytext=(0, 10),
                     ha="center", color="#f8fafc", fontsize=9, fontweight="bold")

    ax1.set_ylim(-10, 105)

    # 2. Right Subplot: Error Breakdown (FP, FN, ID Switches)
    ax2.set_facecolor("#1e293b")
    ax2.grid(True, linestyle="--", alpha=0.25, color="#64748b", axis="y")
    ax2.set_title("Tracking Error Breakdown", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Total Occurrences", color="#94a3b8", fontsize=10, labelpad=8)
    ax2.tick_params(colors="#94a3b8")

    names = [r["name"] for r in results]
    fps = [r.get("metrics", {}).get("fp", 0) or 0 for r in results]
    fns = [r.get("metrics", {}).get("fn", 0) or 0 for r in results]
    idsws = [r.get("metrics", {}).get("idsw", 0) or 0 for r in results]

    x = np.arange(len(names))
    width = 0.25

    ax2.bar(x - width, fps, width, label="False Positives (FP)", color="#f87171", alpha=0.85)
    ax2.bar(x, fns, width, label="Missed Frames (FN)", color="#fbbf24", alpha=0.85)
    ax2.bar(x + width, idsws, width, label="ID Switches", color="#38bdf8", alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, color="#f8fafc", fontsize=9.5, rotation=15, ha="right")
    ax2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
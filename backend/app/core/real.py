"""Real-video mode: run an actual YOLO detector + a chosen tracker on a
user-supplied video, with heuristic event spotting and an annotated H.264 clip.

There is no ground truth here, so instead of MOT scores we surface things a
layperson can *see*: identity churn, ids running purely on tracker memory,
ghost tracks chasing nothing, and frames where the detector delivered nothing.
"""
from __future__ import annotations

import os

import numpy as np


def _box_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def run_real(model, cap, tracker_id: str, tracker_params: dict, det_params: dict,
             progress=None, out_dir: str = "data/videos", job_id: str = "real") -> dict:
    import cv2
    from app.core.plugins import build_engine
    from app.core.registry import get_tracker
    from app.core.trackers import color_for

    fps0 = max(1, int(cap.get(cv2.CAP_PROP_FPS)) or 15)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1)
    stride = max(1, int(np.ceil(total / 600))) if total > 1 else 1

    meta = get_tracker(tracker_id)
    if meta["mode"] == "single":
        raise RuntimeError(
            f"'{meta['name']}' is a single-object follower: it needs one known box to latch onto "
            f"and cannot follow a whole video. Use a multi-object tracker in real mode.")
    engine = build_engine(tracker_id, tracker_params, fps0)
    conf = float(det_params.get("conf", 0.25))
    imgsz = int(det_params.get("imgsz", 640))
    half = bool(det_params.get("half", False))
    device = det_params.get("device", "cpu")

    events, frame_log = [], []
    latencies_ms = []
    last_box_of_track: dict[int, np.ndarray] = {}
    last_frame_of_track: dict[int, int] = {}
    loss_open: dict[int, int] = {}
    spawn_frame: dict[int, int] = {}
    last_active = set()

    out_path = os.path.join(out_dir, f"{job_id}_{tracker_id}.mp4")
    writer = _open_writer(out_path, fps0 / stride, (W, H))
    codec = writer[2] if isinstance(writer, (list, tuple)) else "mp4v"
    vw = writer[0]

    first_out = False
    raw_frames_consumed = 0
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t % stride:
            t += 1
            continue

        dets = model.predict(source=frame, verbose=False, conf=conf, imgsz=imgsz,
                             half=half, device=device)[0]
        arr = []
        if dets.boxes is not None and len(dets.boxes):
            xyxy = dets.boxes.xyxy.cpu().numpy()
            sc = dets.boxes.conf.cpu().numpy()
            for b, s in zip(xyxy, sc):
                arr.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)])
        det_arr = np.array(arr) if arr else np.zeros((0, 5))

        import time
        t0 = time.perf_counter()
        state = engine.update(det_arr.copy(), frame)
        latencies_ms.append(round((time.perf_counter() - t0) * 1000.0, 2))
        cur = {tr.id: np.array(tr.box, dtype=np.float32) for tr in state.active}

        # --- identity heuristics -----------------------------------
        vanished = last_active - set(cur.keys())
        appeared = set(cur.keys()) - last_active
        for v in vanished:
            if v in last_box_of_track:
                pv = last_box_of_track[v]
                cv_ = _centre(pv)
                h = pv[3] - pv[1] + 1
                for a in appeared:
                    if a in cur:
                        ca = _centre(cur[a])
                        if np.hypot(ca[0] - cv_[0], ca[1] - cv_[1]) < 0.45 * h:
                            events.append(dict(
                                frame=t, type="possible_id_swap", severity="warning", blame="tracker",
                                track_ids=[v, a],
                                text=(f"Track {v} vanished and track {a} appeared almost exactly where "
                                      f"{v} left off — likely identity handed over mid-scene."),
                                fix="Enable/strengthen appearance (ReID) or tighten the association threshold."))
        for a in appeared:
            spawn_frame[a] = t
            if a in loss_open:
                events.append(dict(frame=t, type="track_recover", severity="informational", blame="tracker",
                                   track_ids=[a],
                                   text=f"Track {a} reappeared after {t - loss_open.pop(a)} frames.",
                                   fix=""))
        for tr in state.lost_now:
            if tr.id not in loss_open and tr.id in last_box_of_track:
                loss_open[tr.id] = t
                events.append(dict(frame=t, type="track_loss", severity="warning", blame="tracker",
                                   track_ids=[tr.id],
                                   text=f"Track {tr.id} ran out of detections and was dropped.",
                                   fix="Raise the lost-track buffer so it survives the gap."))

        # --- memory-only tracks (box with no detection under it) ----
        # keep it simple: report once when NO detections exist at all
        if len(state.active) and len(det_arr) == 0:
            events.append(dict(frame=t, type="memory_only", severity="informational", blame="tracker",
                               track_ids=[tr.id for tr in state.active],
                               text="No detections reached the tracker this frame; active boxes are pure prediction.",
                               fix="Lower conf or use a bigger detector model."))

        # --- annotated frame ---------------------------------------
        for tr in state.active:
            x1, y1, x2, y2 = tr.box
            col = color_for(tr.id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 18)), (min(W, x1 + 86), max(0, y1)), col, -1)
            cv2.putText(frame, f"ID {tr.id}", (x1 + 3, max(13, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"tracker: {meta['name']}  fps ~{fps0 / stride:.0f}",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        if not first_out:
            cv2.putText(frame, f"frame {t}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 230, 230), 1, cv2.LINE_AA)
        vw.write(frame)
        first_out = True

        for tr in state.active:
            last_box_of_track[tr.id] = np.array(tr.box, dtype=np.float32)
            last_frame_of_track[tr.id] = t
        last_active = set(cur.keys())

        frame_log.append(dict(frame=t, tracks=len(state.active), detections=len(det_arr),
                              ids=sorted(cur.keys())))
        raw_frames_consumed += 1
        if raw_frames_consumed % 40 == 0 and progress:
            progress(f"Processed frame {t} (source # {raw_frames_consumed})",
                     0.2 + 0.7 * (raw_frames_consumed / max(1, (total // stride))))
        t += 1

    vw.release()
    cap.release()

    _maybe_transcode(out_path)
    codec = _probe_codec(out_path) or codec
    if progress:
        progress("Scoring clip", 0.93)

    n_spawns = len(spawn_frame)
    avg_life = 0.0
    if n_spawns:
        avg_life = float(np.mean([(last_frame_of_track.get(i, t) - spawn_frame[i] + 1) for i in spawn_frame]))

    id_resets = len([e for e in events if e["type"] in ("possible_id_swap", "track_loss")])
    avg_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    metrics = dict(
        max_concurrent=max((f["tracks"] for f in frame_log), default=0),
        total_tracks_spawned=n_spawns,
        avg_track_life=round(avg_life, 1),
        detection_rate=round(float(np.mean([f["detections"] > 0 for f in frame_log])), 3) if frame_log else 0.0,
        id_resets=id_resets,
        frames_processed=len(frame_log),
        avg_time_ms=round(avg_ms, 2),
        fps=round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
        latencies=latencies_ms,
    )
    lines = [
        f"{metrics['total_tracks_spawned']} track ids were created over the clip "
        f"(max {metrics['max_concurrent']} at once).",
        f"Tracks lived on average ~{metrics['avg_track_life']} frames before an id reset or loss.",
        f"The detector delivered a box in {metrics['detection_rate'] * 100:.0f}% of frames.",
    ]
    if not metrics["id_resets"]:
        verdict, grade = "Clean run: no obvious identity trouble in this clip.", "A"
    elif metrics["id_resets"] < 4:
        verdict, grade = "Mild identity churn — a dropped id or two.", "B"
    else:
        verdict, grade = "Heavy identity churn: ids were dropped, swapped or re-spawned many times.", "D"
    sections = [dict(name="What we saw", bullets=lines)]
    if events:
        sections.append(dict(name="Event log", bullets=[e["text"] for e in events[:8]]))
    report = dict(tracker_id=tracker_id, name=meta["name"], tagline=meta["tagline"],
                  grade=grade, verdict=verdict, sections=sections,
                  failed=grade in ("D", "E", "F"))
    return dict(tracker_id=tracker_id, name=meta["name"], tagline=meta["tagline"],
                mode="real", codec=codec,
                video_url=f"/api/media/{os.path.basename(out_path)}",
                metrics=metrics, events=events[:60], report=report)


def _open_writer(path, fps, size):
    import cv2
    for fc in ("avc1", "avc3", "mp4v", "XVID"):
        v = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fc), fps, size)
        if v.isOpened():
            return v, fc, fc
    raise RuntimeError("No video writer backend available")


def _maybe_transcode(path):
    try:
        from app.core.runner import get_ffmpeg_exe
        ff = get_ffmpeg_exe()
        if not ff:
            return
        tmp = path + ".raw.mp4"
        os.replace(path, tmp)
        code = os.system(f'"{ff}" -y -loglevel error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p -an "{path}"')
        if code != 0:
            os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


def _probe_codec(path: str):
    import shutil
    try:
        import subprocess
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return None
        out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
                             capture_output=True, text=True)
        c = out.stdout.strip()
        return {"h264": "avc1(ffmpeg)", "avc1": "avc1"}.get(c, c) if c else None
    except Exception:  # noqa: BLE001
        return None
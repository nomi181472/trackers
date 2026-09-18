"""Translate raw metrics + events into a plain-language report card.

The whole point of the simulator is *why*.  So every report starts from the
actual events that happened and walks a lay reader through them: what failed,
whose fault it was (detector vs tracker), why the tracker behaved that way,
and which hyperparameter the reader can turn to make it better.
"""
from __future__ import annotations

from app.core.registry import get_tracker

SEVERITY_RANK = {"critical": 0, "warning": 1, "informational": 2}


def _metric_plain(metrics: dict, tracker_meta: dict) -> list[str]:
    out = []
    if tracker_meta["mode"] == "single":
        acc = metrics.get("accuracy", 0)
        out.append(f"Followed the object correctly {metrics.get('correct_frames', 0)} of "
                   f"{metrics.get('total_visible', 0)} visible frames ({acc * 100:.0f}%).")
        out.append(f"Longest unbroken correct run: {metrics.get('longest_correct_run', 0)} frames.")
        return out
    out.append(f"MOTA {metrics.get('mota', 0):.0%} — the headline 'how much went right' score "
               f"(1 - misses, false boxes and ID swaps).")
    out.append(f"IDF1 {metrics.get('idf1', 0):.0%} — how consistently each object kept ONE identity "
               f"(this is identity-ness, not just box-ness).")
    out.append(f"{metrics.get('idsw', 0)} identity switches happened; "
               f"{metrics.get('fp', 0)} ghost boxes and {metrics.get('fn', 0)} misses.")
    out.append(f"Mostly-tracked {metrics.get('mt')} objects, mostly-lost {metrics.get('ml')}.")
    return out


def explain(scenario_meta: dict, tracker_id: str, metrics: dict, events: list[dict]) -> dict:
    meta = get_tracker(tracker_id)
    lines = _metric_plain(metrics, meta)

    critical = [e for e in events if e["severity"] == "critical"]
    warnings = [e for e in events if e["severity"] == "warning"]
    detector_blame = [e for e in events if e.get("blame") == "detector"]

    if meta["mode"] == "multi":
        if metrics.get("idsw", 0) == 0 and metrics.get("idf1", 0) > 0.95 and metrics.get("mota", 0) > 0.95:
            verdict = ("Flawless run. Every object kept its identity from first to last frame — "
                       "nothing to fix here.")
            grade = "S"
        elif metrics.get("idsw", 0) == 0:
            verdict = ("No identities were swapped, but the tracker still let some objects drift. "
                       "A solid, boring run.")
            grade = "A"
        elif len(critical) > 3:
            verdict = ("This scene is exactly what this tracker is bad at. It repeatedly fused "
                       "two objects into one identity, which is the classic falling-down at "
                       "crossings / look-alike objects.")
            grade = "D"
        else:
            verdict = ("A few identity switches. The object WAS followed, but at the moments the "
                       "tracker became unsure, it bet on the wrong object.")
            grade = "C"
    else:
        acc = metrics.get("accuracy", 0)
        if acc > 0.9:
            verdict, grade = "Kept its eye on the right thing the whole time.", "A"
        elif acc > 0.6:
            verdict, grade = "More on than off, but it clearly struggled at some point.", "C"
        else:
            verdict, grade = "Lost the object quickly and never really got it back.", "F"

    sections = [dict(name="Verdict", text=verdict)]
    sections.append(dict(name="How it scored", bullets=lines))

    if critical or warnings:
        sample = (critical or warnings)[:4]
        bullets = []
        for e in sample:
            blame = "detector's" if e.get("blame") == "detector" else "this tracker's"
            bullets.append(f"At frame {e['frame']}: {e['text']} — the {blame} doing.")
        sections.append(dict(name="What went wrong", bullets=bullets))

    if meta["failure_modes"]:
        sections.append(dict(name="Known weaknesses", bullets=meta["failure_modes"][:3]))
    if meta["strengths"]:
        sections.append(dict(name="Known strengths", bullets=meta["strengths"]))

    # give concrete tuning suggestions grounded in the actual events
    suggestions = []
    for e in (critical + warnings)[:5]:
        if e.get("fix"):
            suggestions.append(e["fix"])
    suggestions = _dedupe(suggestions)[:4]
    if suggestions:
        sections.append(dict(name="Try these knobs", bullets=suggestions))

    return dict(tracker_id=tracker_id, name=meta["name"], tagline=meta["tagline"],
                grade=grade, verdict=verdict, sections=sections,
                failed=grade in ("D", "F"),
                cause=(f"objects were swapped/identical (appearance-blindness)"
                       if any(e["type"] == "id_switch" for e in critical)
                       else ("object left the tracker's memory (loss)"
                             if any(e["type"] == "track_loss" for e in warnings)
                             else "detections were missing")))


def _dedupe(items):
    seen = set()
    out = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
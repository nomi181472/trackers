"""Hyperparameter primitives shared by every tracker plugin.

`params.py` deliberately knows nothing about trackers: it only describes the
*shape* of a tunable knob (type, default, tooltip, hint, bounds, unit, options)
so a plugin author can declare one without touching anything else.

A plugin supplies a list of `_d(...)` dicts; the API ships them straight to the
UI and `default_params()` flattens them into a `{key: default}` map.
"""
from __future__ import annotations

INT = "int"
FLOAT = "number"
BOOL = "bool"
SELECT = "select"

MODEL_OPTIONS = {
    "detector_nano": "detector_nano — smallest/fastest, weakest accuracy",
    "detector_small": "detector_small — small, good balance",
    "detector_medium": "detector_medium — medium",
    "detector_large": "detector_large — large",
    "detector_xlarge": "detector_xlarge — biggest/best, slowest",
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

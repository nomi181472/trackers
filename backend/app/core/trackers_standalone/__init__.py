"""Standalone Multi-Object Tracking (MOT) Engines in pure NumPy / SciPy.
Native implementations from original papers without PyTorch or external model weights.
"""
from __future__ import annotations
from types import SimpleNamespace
from .byte_tracker import BYTETracker
from .bot_sort import BOTSORT
from .oc_sort import OCSORT, DeepOCSORT
from .fast_track import FASTTracker, TrackTrack

TRACKER_MAP = {
    "bytetrack": BYTETracker,
    "botsort": BOTSORT,
    "ocsort": OCSORT,
    "deepocsort": DeepOCSORT,
    "fasttrack": FASTTracker,
    "tracktrack": TrackTrack,
}

__all__ = [
    "TRACKER_MAP",
    "BYTETracker",
    "BOTSORT",
    "OCSORT",
    "DeepOCSORT",
    "FASTTracker",
    "TrackTrack",
]

"""Standalone Tracker Base Classes and Structures.
Pure NumPy implementation without external framework dependencies.
"""
from __future__ import annotations
from typing import Any
import numpy as np


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.score = 0.0
        self.start_frame = 0
        self.frame_id = 0

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args: Any) -> None:
        raise NotImplementedError

    def predict(self) -> None:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @staticmethod
    def reset_id() -> None:
        BaseTrack._count = 0

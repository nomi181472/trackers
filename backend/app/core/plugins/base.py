"""The two contracts every tracker plugin is built on.

A plugin is a *self-contained* unit: one class carries its own metadata (name,
plain-language explanation, hyperparameters) *and* the code that builds its
engine.  Nothing else in the app needs to know a tracker exists -- registering
the class is the whole integration step.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.trackers import TrackerState


class Engine(ABC):
    """The runtime object the simulator drives frame by frame.

    Implementations read a detection array and (optionally) the frame, and
    return the identities they currently believe in.  They never see the ground
    truth -- that is the whole point of the exercise.
    """

    @abstractmethod
    def update(self, dets, img=None) -> TrackerState:
        """Consume one frame of detections and return the current state."""

    def init(self, img, box) -> None:
        """Seed a single-object follower on the first frame.

        Raising (rather than silently doing nothing) means a plugin that
        declares ``mode="single"`` but forgets this method fails loudly instead
        of producing a black video that scores a perfect grade.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support init()")


class TrackerPlugin(ABC):
    """Metadata + construction for exactly one tracker.

    Subclasses set the three identity class attributes and implement ``build``.
    ``meta()`` is a classmethod so the catalog can be served without importing
    cv2/ultralytics into the request path more than necessary.
    """

    id: str
    engine: str
    mode: str

    @classmethod
    def meta(cls) -> dict:
        """Everything the API exposes: id/name/engine/mode/tagline/description/
        strengths/failure_modes/params."""
        raise NotImplementedError(f"{cls.__name__} does not implement meta()")

    @classmethod
    def is_available(cls) -> bool:
        """Whether this build can actually run the tracker."""
        return True

    @abstractmethod
    def build(self, params: dict, fps: int, device: str = "cpu") -> Engine:
        """Construct the engine for one run."""

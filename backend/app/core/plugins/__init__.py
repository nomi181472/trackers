"""Tracker plugins: one self-contained class per tracker.

Adding a tracker means writing one `TrackerPlugin` subclass in a module under
this package and importing it here so `register()` runs.  No other file in the
app needs to change -- the API, the simulator and the UI all read the registry.
"""
from __future__ import annotations

from app.core.plugins.base import Engine, TrackerPlugin
from app.core.plugins.registry import REGISTRY, PluginRegistry, build_engine, register

# Importing the concrete modules is what actually populates REGISTRY.  This must
# happen *after* the re-exports above, which is why `registry.py` lives outside
# `__init__.py` -- otherwise every plugin import would re-enter this module.
from app.core.plugins import custom, ultralytics, opencv, vector_embed  # noqa: E402,F401

__all__ = ["Engine", "TrackerPlugin", "PluginRegistry", "REGISTRY", "build_engine", "register"]

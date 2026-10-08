"""Where plugins live once they have declared themselves.

Kept in its own module (rather than inside ``plugins/__init__.py``) so the
concrete plugin modules can call ``register()`` at import time without
re-entering a half-initialised package.
"""
from __future__ import annotations

from app.core.plugins.base import Engine, TrackerPlugin


class PluginRegistry:
    """id -> plugin class, with the catalog shape the HTTP API needs."""

    def __init__(self) -> None:
        self._plugins: dict[str, type[TrackerPlugin]] = {}

    def register(self, plugin: type[TrackerPlugin]) -> type[TrackerPlugin]:
        tid = plugin.id
        if tid in self._plugins:
            raise ValueError(f"Duplicate tracker plugin id '{tid}' "
                             f"(already registered by {self._plugins[tid].__module__})")
        self._plugins[tid] = plugin
        return plugin

    def get(self, tid: str) -> type[TrackerPlugin]:
        if tid not in self._plugins:
            raise KeyError(f"Unknown tracker '{tid}'. Known: {self.ids()}")
        return self._plugins[tid]

    def all(self) -> list[type[TrackerPlugin]]:
        return list(self._plugins.values())

    def ids(self) -> list[str]:
        return list(self._plugins.keys())

    def default_params(self, tid: str) -> dict:
        return {p["key"]: p["default"] for p in self.get(tid).meta()["params"]}

    def catalog(self) -> list[dict]:
        """One entry per tracker: the plugin's meta plus a live `available` flag."""
        out = []
        for plugin in self._plugins.values():
            entry = dict(plugin.meta())
            entry["available"] = plugin.is_available()
            out.append(entry)
        return out


REGISTRY = PluginRegistry()


def register(plugin: type[TrackerPlugin]) -> type[TrackerPlugin]:
    """Class decorator used by every concrete plugin module."""
    return REGISTRY.register(plugin)


def build_engine(tid: str, params: dict, fps: int, device: str = "cpu") -> Engine:
    """The one entry point the simulator uses to get a running engine."""
    from app.core.trackers import get_compute_device
    safe_device = get_compute_device(device)
    return REGISTRY.get(tid)().build(params, fps, safe_device)

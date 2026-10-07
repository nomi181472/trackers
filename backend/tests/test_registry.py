"""The registry is the app's tracker list -- if it drifts, everything drifts."""
from __future__ import annotations

import pytest

from app.core.plugins import REGISTRY, PluginRegistry, build_engine
from app.core.plugins.base import TrackerPlugin
from app.core.plugins.registry import register
from app.core.registry import default_params, get_tracker, tracker_ids

EXPECTED_IDS = {
    "greedy_iou", "centroid", "sort", "embed_sort",
    "bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack",
    "kcf", "csrt", "mosse", "mil", "medianflow", "nano", "vit", "dasiamrpn",
}

# frontend/lib/types.ts -> TrackerMeta
CATALOG_KEYS = {
    "id", "name", "engine", "mode", "tagline", "description",
    "strengths", "failure_modes", "params", "available",
}

OPENCV_IDS = {"kcf", "csrt", "mosse", "mil", "medianflow", "nano", "vit", "dasiamrpn"}
ULTRALYTICS_IDS = {"bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack"}


def test_every_tracker_is_registered():
    """Count is derived from EXPECTED_IDS so adding one does not touch a magic number."""
    assert set(REGISTRY.ids()) == EXPECTED_IDS
    assert len(REGISTRY.ids()) == len(EXPECTED_IDS)


def test_ids_are_unique_and_stable():
    ids = REGISTRY.ids()
    assert len(ids) == len(set(ids))
    assert ids == REGISTRY.ids(), "registry iteration order must be stable across calls"


def test_catalog_order_matches_display_order():
    """The UI renders the catalog as-is, so the familiar grouping must survive:
    homemade baselines first, then the library trackers, then the single-object ones."""
    assert REGISTRY.ids() == [
        "greedy_iou", "centroid", "sort",
        "bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack",
        "kcf", "csrt", "mosse", "mil", "medianflow", "nano", "vit", "dasiamrpn",
        "embed_sort",
    ]


def test_register_rejects_duplicate_ids():
    class Clashing(TrackerPlugin):
        id = "greedy_iou"
        engine = "custom"
        mode = "multi"

    with pytest.raises(ValueError, match="Duplicate tracker plugin id 'greedy_iou'"):
        register(Clashing)
    assert REGISTRY.ids().count("greedy_iou") == 1


def test_get_unknown_id_raises_keyerror_with_known_ids():
    with pytest.raises(KeyError) as exc:
        REGISTRY.get("not_a_tracker")
    assert "Unknown tracker 'not_a_tracker'" in str(exc.value)
    assert "greedy_iou" in str(exc.value)


def test_every_meta_has_the_full_param_shape():
    for plugin in REGISTRY.all():
        meta = plugin.meta()
        assert set(meta) == CATALOG_KEYS - {"available"}, plugin.id
        assert meta["id"] == plugin.id, plugin.id
        assert meta["engine"] == plugin.engine, plugin.id
        assert meta["mode"] == plugin.mode, plugin.id
        for field in ("name", "tagline", "description"):
            assert isinstance(meta[field], str) and meta[field].strip(), f"{plugin.id}.{field}"
        for field in ("strengths", "failure_modes"):
            assert isinstance(meta[field], list) and meta[field], f"{plugin.id}.{field}"
        assert plugin.mode in ("multi", "single"), plugin.id

        seen = set()
        for p in meta["params"]:
            for key in ("key", "label", "type", "default"):
                assert key in p, f"{plugin.id} param missing {key}: {p}"
            assert p["type"] in ("int", "number", "bool", "select"), f"{plugin.id}/{p['key']}"
            assert p["key"] not in seen, f"{plugin.id} declares {p['key']} twice"
            seen.add(p["key"])


def test_default_params_keys_match_declared_params():
    for tid in REGISTRY.ids():
        declared = [p["key"] for p in REGISTRY.get(tid).meta()["params"]]
        assert list(REGISTRY.default_params(tid)) == declared
        assert default_params(tid) == REGISTRY.default_params(tid)


def test_catalog_returns_exactly_the_frontend_keys():
    catalog = REGISTRY.catalog()
    assert len(catalog) == len(EXPECTED_IDS)
    for entry in catalog:
        assert set(entry) == CATALOG_KEYS, entry.get("id")
        assert isinstance(entry["available"], bool)
        assert entry["id"] in EXPECTED_IDS


def test_engine_families_are_classified_consistently():
    for plugin in REGISTRY.all():
        if plugin.id in OPENCV_IDS:
            assert plugin.engine == "opencv" and plugin.mode == "single", plugin.id
        elif plugin.id in ULTRALYTICS_IDS:
            assert plugin.engine == "ultralytics" and plugin.mode == "multi", plugin.id
        else:
            assert plugin.engine == "custom" and plugin.mode == "multi", plugin.id


def test_legacy_accessors_delegate_to_the_plugin_registry():
    assert set(tracker_ids()) == EXPECTED_IDS
    for tid in REGISTRY.ids():
        assert get_tracker(tid) == REGISTRY.get(tid).meta()
        assert default_params(tid) == REGISTRY.default_params(tid)
    with pytest.raises(KeyError):
        get_tracker("nope")


def test_build_engine_routes_through_the_registry():
    with pytest.raises(KeyError):
        build_engine("nope", {}, 15)
    assert isinstance(REGISTRY, PluginRegistry)

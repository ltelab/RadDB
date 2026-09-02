"""Tests for :mod:`raddb.viz.interactive` — the ipyleaflet AOI selector.

The module splits into a **pure** dispatch layer (a drawn GeoJSON feature to the matching
``RadDB`` method) and a widget wrapper.  The dispatch is where the behavior lives and is
tested directly with hand-written features; the widget is driven by calling its callbacks
rather than by simulating clicks, which needs no Jupyter frontend.

The map is WGS-84, so every dispatched call must carry ``crs=4326``.  Reading those
degrees as archive meters would put the AOI thousands of kilometers away.
"""

from __future__ import annotations

import json

import pytest

from raddb.tests.conftest import FI_SITE, RADAR
from raddb.viz.interactive import (
    AOISelector,
    _crop_from_feature,
    _feature_collection,
    _is_axis_aligned_box,
)

pytest.importorskip("ipyleaflet")
pytest.importorskip("ipywidgets")


def _feature(geometry: dict) -> dict:
    """Wrap a geometry the way ipyleaflet's draw control hands it over."""
    return {"type": "Feature", "properties": {}, "geometry": geometry}


def _box(lon, lat, half=0.05):
    """An axis-aligned rectangle ring around ``(lon, lat)``, as a drawn rectangle."""
    return [
        [lon - half, lat - half],
        [lon + half, lat - half],
        [lon + half, lat + half],
        [lon - half, lat + half],
        [lon - half, lat - half],
    ]


# ---------------------------------------------------------------------------
# _is_axis_aligned_box — how a rectangle is told from a polygon
# ---------------------------------------------------------------------------


def test_is_axis_aligned_box_accepts_a_drawn_rectangle():
    """Four corners over two distinct longitudes and two latitudes."""
    assert _is_axis_aligned_box(_box(*FI_SITE))


def test_is_axis_aligned_box_accepts_an_unclosed_ring():
    """Some producers omit the repeated closing vertex."""
    assert _is_axis_aligned_box(_box(*FI_SITE)[:-1])


def test_is_axis_aligned_box_rejects_a_rotated_quad():
    """A rotated rectangle has four distinct longitudes, so it is a polygon."""
    assert not _is_axis_aligned_box([[0, 1], [1, 2], [2, 1], [1, 0], [0, 1]])


def test_is_axis_aligned_box_rejects_a_triangle():
    """Anything other than four corners is a polygon."""
    assert not _is_axis_aligned_box([[0, 0], [1, 0], [0, 1], [0, 0]])


def test_is_axis_aligned_box_rejects_an_empty_ring():
    """A degenerate ring must return ``False``, not raise."""
    assert not _is_axis_aligned_box([])


# ---------------------------------------------------------------------------
# _crop_from_feature — the dispatch table
# ---------------------------------------------------------------------------


def test_a_marker_dispatches_to_crop_around_point(rdb):
    """A drawn marker crops a radius, using the widget's distance."""
    kind, out = _crop_from_feature(rdb, _feature({"type": "Point", "coordinates": list(FI_SITE)}), distance_m=8_000)

    assert kind == "point"
    assert 0 < len(out) < len(rdb)


def test_a_rectangle_dispatches_to_crop_by_bbox(rdb):
    """An axis-aligned polygon is recognized as a bbox crop."""
    kind, out = _crop_from_feature(rdb, _feature({"type": "Polygon", "coordinates": [_box(*FI_SITE)]}))

    assert kind == "bbox"
    assert len(out) > 0


def test_a_rotated_polygon_dispatches_to_crop_by_polygon(rdb):
    """Not axis-aligned, so the full polygon path runs instead."""
    ring = [
        [FI_SITE[0], FI_SITE[1] + 0.06],
        [FI_SITE[0] + 0.06, FI_SITE[1]],
        [FI_SITE[0], FI_SITE[1] - 0.06],
        [FI_SITE[0] - 0.06, FI_SITE[1]],
        [FI_SITE[0], FI_SITE[1] + 0.06],
    ]

    kind, out = _crop_from_feature(rdb, _feature({"type": "Polygon", "coordinates": [ring]}))

    assert kind == "polygon"
    assert len(out) > 0


def test_a_polyline_dispatches_to_extract_cross_section(rdb):
    """Only the first and last vertex are used — a section is defined by two points."""
    line = {"type": "LineString", "coordinates": [[26.9, 62.0], [27.0, 62.0], [27.1, 62.0]]}

    kind, out = _crop_from_feature(rdb, _feature(line))

    assert kind == "cross_section"
    assert "cs_polygon" in out.data.columns


def test_a_bare_geometry_is_accepted(rdb):
    """The draw control sometimes emits a geometry without the Feature wrapper."""
    kind, _ = _crop_from_feature(rdb, {"type": "Point", "coordinates": list(FI_SITE)}, distance_m=8_000)

    assert kind == "point"


def test_an_unsupported_geometry_is_refused(rdb):
    """A circle has no crop equivalent; the message names the type."""
    with pytest.raises(ValueError, match="unsupported drawn geometry"):
        _crop_from_feature(rdb, _feature({"type": "GeometryCollection", "geometries": []}))


def test_the_drawn_coordinates_are_read_as_lonlat(rdb):
    """A marker at the radar site must land on the radar, not 2600 km away."""
    _, at_site = _crop_from_feature(
        rdb,
        _feature({"type": "Point", "coordinates": list(FI_SITE)}),
        distance_m=8_000,
    )
    _, elsewhere = _crop_from_feature(
        rdb,
        _feature({"type": "Point", "coordinates": [0.0, 0.0]}),
        distance_m=8_000,
    )

    assert len(at_site) > 0
    assert len(elsewhere) == 0


# ---------------------------------------------------------------------------
# _feature_collection — saving the drawn AOI
# ---------------------------------------------------------------------------


def test_feature_collection_wraps_a_feature():
    """An already-wrapped feature is reused, not double-wrapped."""
    feat = _feature({"type": "Point", "coordinates": list(FI_SITE)})

    fc = _feature_collection(feat)

    assert fc["type"] == "FeatureCollection"
    assert fc["features"] == [feat]


def test_feature_collection_wraps_a_bare_geometry():
    """A bare geometry gains the Feature envelope GeoJSON readers expect."""
    fc = _feature_collection({"type": "Point", "coordinates": list(FI_SITE)})

    assert fc["features"][0]["type"] == "Feature"
    assert fc["features"][0]["geometry"]["type"] == "Point"


def test_a_saved_collection_reloads_as_an_aoi(tmp_path, rdb):
    """The round trip the docstring promises: save, then ``crop_by_polygon(path)``."""
    ring = _box(*FI_SITE)
    fc = _feature_collection(_feature({"type": "Polygon", "coordinates": [ring]}))
    path = tmp_path / "aoi.geojson"
    path.write_text(json.dumps(fc))

    assert len(rdb.crop_by_polygon(str(path))) > 0


# ---------------------------------------------------------------------------
# AOISelector — the widget
# ---------------------------------------------------------------------------


def test_AOISelector(rdb):
    """Construction builds a map, a draw control and the four controls."""
    sel = AOISelector(rdb)

    assert sel.map is not None
    assert sel.draw is not None
    assert (sel.feature, sel.result, sel.kind) == (None, None, None)


def test_AOISelector_init(rdb):
    """The map centers on the radar sites and marks each one."""
    sel = AOISelector(rdb, point_radius_m=8_000)

    lat, lon = sel.map.center
    assert (round(lat, 3), round(lon, 3)) == (FI_SITE[1], FI_SITE[0])
    assert sel.radius.value == pytest.approx(8_000.0)


def test_the_center_falls_back_to_switzerland_without_sites(rdb):
    """An unknown radar yields no site, so the map still opens somewhere sensible."""
    sel = AOISelector(rdb, radars=["ZZZZ"])

    assert tuple(sel.map.center) == (46.82, 8.23)


def test_an_explicit_center_wins(rdb):
    """``center=`` overrides the derived one."""
    assert tuple(AOISelector(rdb, center=(35.3, -97.3)).map.center) == (35.3, -97.3)


def test_a_missing_radar_does_not_break_the_map(rdb):
    """A radar without info is skipped; the map is decoration, not a gate."""
    sel = AOISelector(rdb, radars=[RADAR, "ZZZZ"])

    assert sel._radar_sites([RADAR, "ZZZZ"]) == {RADAR: (FI_SITE[1], FI_SITE[0])}


def test_the_draw_callback_keeps_the_last_shape(rdb):
    """``created`` and ``edited`` update the stored feature; nothing else does."""
    sel = AOISelector(rdb)
    first = _feature({"type": "Point", "coordinates": list(FI_SITE)})
    second = _feature({"type": "Point", "coordinates": [FI_SITE[0] + 0.1, FI_SITE[1] + 0.1]})

    sel._on_draw(None, "created", first)
    assert sel.feature is first

    sel._on_draw(None, "edited", second)
    assert sel.feature is second

    sel._on_draw(None, "deleted", first)
    assert sel.feature is second


def test_apply_without_a_shape_asks_for_one(rdb, capsys):
    """Clicking Apply on an empty map explains what to do rather than raising."""
    sel = AOISelector(rdb)

    sel._apply()

    assert sel.result is None
    assert "Draw a shape" in capsys.readouterr().out


def test_apply_runs_the_crop_and_stores_the_result(rdb):
    """The whole point of the widget: ``.result`` holds a cropped RadDB."""
    sel = AOISelector(rdb, point_radius_m=8_000)
    sel.feature = _feature({"type": "Point", "coordinates": list(FI_SITE)})

    sel._apply()

    assert sel.kind == "point"
    assert 0 < len(sel.result) < len(rdb)


def test_apply_reports_the_sweep_count_from_the_gate_ids(rdb, capsys):
    """``sweep`` is a LUT column, so it is decoded from ``gate_id``, not read off."""
    sel = AOISelector(rdb, point_radius_m=8_000)
    sel.feature = _feature({"type": "Point", "coordinates": list(FI_SITE)})

    sel._apply()

    text = capsys.readouterr().out
    assert "2 sweeps" in text
    assert "? sweeps" not in text


def test_apply_surfaces_a_failure_in_the_widget(rdb, capsys):
    """A bad shape is reported instead of killing the kernel."""
    sel = AOISelector(rdb)
    sel.feature = _feature({"type": "GeometryCollection", "geometries": []})

    sel._apply()

    assert sel.result is None
    assert "crop failed" in capsys.readouterr().out


def test_save_without_a_shape_says_so(rdb, capsys):
    """Nothing drawn, nothing written — and no traceback."""
    sel = AOISelector(rdb)

    sel._save()

    assert "Nothing to save" in capsys.readouterr().out


def test_save_writes_a_reloadable_geojson(tmp_path, rdb):
    """The saved file is a FeatureCollection that ``crop_by_polygon`` accepts."""
    sel = AOISelector(rdb)
    sel.feature = _feature({"type": "Polygon", "coordinates": [_box(*FI_SITE)]})
    sel.save_path.value = str(tmp_path / "aoi.geojson")

    sel._save()

    saved = json.loads((tmp_path / "aoi.geojson").read_text())
    assert saved["type"] == "FeatureCollection"
    assert len(rdb.crop_by_polygon(sel.save_path.value)) > 0


def test_AOISelector_display(rdb):
    """``display()`` renders the widget and returns the selector for chaining."""
    pytest.importorskip("IPython")
    sel = AOISelector(rdb)

    assert sel.display() is sel


def test_the_selector_renders_itself_in_a_notebook(rdb):
    """``_ipython_display_`` is what makes a bare selector show up in a cell."""
    pytest.importorskip("IPython")
    sel = AOISelector(rdb)

    sel._ipython_display_()  # must not raise

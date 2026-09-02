"""Tests for :mod:`raddb.viz.plot` — the four plots, the quicklook and the scatter grid.

Each plot draws **one plot into one Axes** and returns the matplotlib artist, so the
caller composes panels by passing ``ax=``.  All four read gate geometry from the LUT
lattices and join on ``gate_id``, which means a filtered, ``sel``-ed or cropped input
draws exactly the gates it still holds — nothing is reindexed onto a full
azimuth x range grid.

Geometry follows the input: a RadDB or frame reads the stored lattices, a raw
``xr.DataTree`` computes them from its own coordinates and needs no archive, and a
GeoDataFrame is treated as a frame (its own geometry column is ignored).

There is exactly **one** geometry path — the exact frustum.  No plot takes a
``beamwidth_deg``: for archive-backed data the beamwidth was applied when ``v_plane`` was
generated, and for a DataTree it is inferred from the file exactly as LUT generation does.
"""

from __future__ import annotations

import copy
import inspect
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest
import shapely

import raddb.viz.plot as vp
from raddb.lut import DEFAULT_BEAMWIDTH_DEG, cappi_chords, gate_corner_table
from raddb.main import RadDB
from raddb.tests.conftest import (
    FMI_EPSG,
    PLOT_GEOMETRY,
    PLOT_RADAR,
    RADAR,
    RADAR_B,
    US_EPSG,
    US_SITE,
    build_datatree,
)
from raddb.viz.plot import (
    PLOT_DEFAULTS,
    _beamwidth,
    _KmFormatter,
    _line_endpoints,
    _resolve_chord_overlap,
    plot_aoi_quicklook,
    plot_cappi,
    plot_cross_section,
    plot_ppi,
    plot_rhi,
    plot_vcs,
)

N_AZ = PLOT_GEOMETRY["n_az"]
N_RNG = PLOT_GEOMETRY["n_rng"]
N_SWEEPS = PLOT_GEOMETRY["n_sweeps"]

VOL_TIMES = [pd.Timestamp("2024-08-01 12:00:00"), pd.Timestamp("2024-08-02 06:30:00")]


def _n_polys(artist) -> int:
    """Number of gate polygons an artist drew."""
    return len(artist.get_paths())


def _line_across(site, half=12_000):
    """A diagonal section line through the radar, in the archive's own meters."""
    return ((site[0] - half, site[1] - half), (site[0] + half, site[1] + half))


@pytest.fixture(scope="session")
def plot_dtree():
    """The same volume the plotting archive was built from, unarchived."""
    return build_datatree(**PLOT_GEOMETRY)


@pytest.fixture(scope="session")
def plot_gdf(plot_archive_dir):
    """The plotting archive as a GeoDataFrame."""
    pytest.importorskip("geopandas")
    return RadDB(archive_dir=str(plot_archive_dir), crs=FMI_EPSG).open(radars=PLOT_RADAR).to_geopandas()


# ---------------------------------------------------------------------------
# plot_ppi
# ---------------------------------------------------------------------------


def test_plot_ppi(plot_rdb):
    """One sweep, one PolyCollection, one polygon per surviving gate."""
    from matplotlib.collections import PolyCollection

    artist = plot_ppi(plot_rdb, sweep=1)

    assert isinstance(artist, PolyCollection)
    assert _n_polys(artist) == N_AZ * N_RNG


def test_plot_ppi_draws_into_the_supplied_axes(plot_rdb):
    """``ax=`` is how a caller composes a panel; the plot never makes its own figure."""
    _, ax = plt.subplots()

    artist = plot_ppi(plot_rdb, sweep=1, ax=ax)

    assert artist.axes is ax
    assert len(ax.collections) == 1


def test_a_multi_panel_figure_composes(plot_rdb):
    """One plot per Axes — the user builds the panel, not the plot function."""
    _, axes = plt.subplots(2, 2)

    for ax, variable in zip(axes.ravel(), ["DBZH", "ZDR", "RHOHV", "PHIDP"], strict=False):
        plot_ppi(plot_rdb, sweep=1, variable=variable, ax=ax)

    assert all(len(ax.collections) == 1 for ax in axes.ravel())


def test_save_writes_a_file(plot_rdb, tmp_path):
    """``save=`` writes the figure the artist was drawn into."""
    out = tmp_path / "ppi.png"

    plot_ppi(plot_rdb, sweep=1, save=str(out))

    assert out.exists()
    assert out.stat().st_size > 0


def test_the_title_and_colorbar_are_optional(plot_rdb):
    """A composed panel usually wants neither."""
    artist = plot_ppi(plot_rdb, sweep=1, add_colorbar=False, title="custom")

    assert artist.axes.get_title() == "custom"


def test_a_filtered_frame_draws_only_its_surviving_gates(plot_rdb, plot_archive_dir):
    """Nothing is reindexed onto a full grid, so the polygon count is the gate count."""
    sub = plot_rdb.filter({"var": "DBZH", "logic": ">", "threshold": 20})
    lut = RadDB(archive_dir=str(plot_archive_dir)).get_lut(PLOT_RADAR)
    on_sweep_1 = (
        sub.data.select("gate_id")
        .join(lut.filter(pl.col("sweep") == 1).select("gate_id"), on="gate_id", how="semi")
        .height
    )

    assert 0 < len(sub) < len(plot_rdb)
    assert _n_polys(plot_ppi(sub, sweep=1)) == on_sweep_1


def test_a_cropped_frame_plots(plot_rdb, plot_site):
    """A crop is a smaller frame, nothing more."""
    crop = plot_rdb.crop_around_point(plot_site, distance=8_000)

    assert 0 < len(crop) < len(plot_rdb)
    assert _n_polys(plot_ppi(crop, sweep=1)) > 0


def test_a_sel_frame_plots(plot_rdb):
    """``sel`` narrows a range window and the plot follows."""
    narrowed = plot_rdb.sel(range=slice(2_000, 12_000))

    assert 0 < len(narrowed) < len(plot_rdb)
    assert _n_polys(plot_ppi(narrowed, sweep=1)) > 0


def test_an_empty_selection_raises(plot_rdb):
    """Nothing to draw is an error, not a blank Axes."""
    empty = plot_rdb.filter({"var": "DBZH", "logic": ">", "threshold": 1e9})

    with pytest.raises(ValueError):
        plot_ppi(empty, sweep=1)


def test_an_unknown_variable_raises(plot_rdb):
    """A typo must not silently plot DBZH."""
    with pytest.raises(KeyError):
        plot_ppi(plot_rdb, sweep=1, variable="NOT_A_VAR")


def test_a_missing_sweep_raises(plot_rdb):
    """Sweep 99 does not exist in a six-sweep volume."""
    with pytest.raises(ValueError):
        plot_ppi(plot_rdb, sweep=99)


def test_a_bare_frame_needs_an_archive(plot_rdb):
    """Geometry lives in the LUT; a frame alone cannot say where its gates are."""
    with pytest.raises(ValueError, match="archive"):
        plot_ppi(plot_rdb.data, sweep=1)


def test_a_bare_frame_with_an_archive_plots(plot_rdb, plot_archive_dir):
    """Given the archive, a bare polars frame is as good as a RadDB."""
    assert _n_polys(plot_ppi(plot_rdb.data, sweep=1, archive_dir=plot_archive_dir)) > 0


def test_a_multi_radar_frame_needs_the_radar_named(tmp_path, make_datatree):
    """A PPI fixes one sweep of one radar; two radars is ambiguous."""
    db = RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG)
    db.archive(datatree={RADAR: [make_datatree(24, 20)], RADAR_B: [make_datatree(24, 20)]})

    with pytest.raises(ValueError, match="radars"):
        plot_ppi(db.open(), sweep=1)


# ---------------------------------------------------------------------------
# plot_rhi
# ---------------------------------------------------------------------------


def test_plot_rhi(plot_rdb):
    """One azimuth, stacked across every sweep."""
    assert _n_polys(plot_rhi(plot_rdb, azimuth=0)) == N_RNG * N_SWEEPS


def test_rhi_height_reference_shifts_the_axis(plot_rdb, plot_archive_dir):
    """``asl`` is ``rel`` plus the site altitude, exactly."""
    altitude = RadDB(archive_dir=str(plot_archive_dir)).get_radar_info(PLOT_RADAR)["altitude"]

    asl = plot_rhi(plot_rdb, azimuth=0, height="asl").axes.get_ylim()[0]
    rel = plot_rhi(plot_rdb, azimuth=0, height="rel").axes.get_ylim()[0]

    assert asl - rel == pytest.approx(altitude, abs=1.0)


def test_rhi_rejects_an_unknown_height_reference(plot_rdb):
    """Only ``asl`` and ``rel`` exist."""
    with pytest.raises(ValueError):
        plot_rhi(plot_rdb, azimuth=0, height="furlongs")


def test_rhi_beyond_the_tolerance_raises(plot_rdb):
    """No ray within ``az_tol`` means there is no RHI to draw."""
    with pytest.raises(ValueError, match="no sweep has a ray within"):
        plot_rhi(plot_rdb, azimuth=2.5, az_tol=0.1)


def test_rhi_picks_a_ray_per_sweep_when_azimuths_jitter(tmp_path):
    """A real antenna's azimuths differ between sweeps.

    Matching one azimuth *value* across the whole LUT would select a single sweep and
    collapse the RHI, so each sweep needs its own nearest ray.
    """
    import xarray as xr

    dt = build_datatree(n_az=36, n_rng=20, n_sweeps=4)
    jittered = {}
    for i, (name, node) in enumerate(dt.children.items()):
        ds = node.to_dataset()
        jittered[name] = ds.assign_coords(azimuth=ds["azimuth"].to_numpy() + 0.13 * i)
    dt = xr.DataTree.from_dict(jittered)

    db = RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG)
    db.archive(datatree={PLOT_RADAR: [dt]})
    assert db.get_lut(PLOT_RADAR)["azimuth"].n_unique() == 36 * 4, "fixture should have per-sweep jitter"

    assert _n_polys(plot_rhi(db.open(radars=PLOT_RADAR), azimuth=90.0, az_tol=1.0)) == 20 * 4


# ---------------------------------------------------------------------------
# plot_cappi
# ---------------------------------------------------------------------------


def test_plot_cappi(plot_rdb):
    """A constant-altitude slice draws the chords that reach it."""
    assert _n_polys(plot_cappi(plot_rdb, altitude=1200)) > 0


def test_a_higher_slice_draws_fewer_gates(plot_rdb):
    """Fewer beams reach higher, so the slice shrinks."""
    assert _n_polys(plot_cappi(plot_rdb, altitude=1400)) < _n_polys(plot_cappi(plot_rdb, altitude=1100))


def test_overlap_nearest_draws_fewer_gates_than_all(plot_rdb):
    """``nearest`` resolves the double coverage that ``all`` keeps."""
    assert _n_polys(plot_cappi(plot_rdb, altitude=1200, overlap="nearest")) < _n_polys(
        plot_cappi(plot_rdb, altitude=1200, overlap="all"),
    )


def test_resolve_chord_overlap_leaves_no_double_coverage(plot_archive_dir):
    """The resolved chords must partition the ground-distance axis."""
    resolved = _resolve_chord_overlap(cappi_chords(PLOT_RADAR, plot_archive_dir, 1200.0))

    intervals = np.sort(
        np.stack([resolved["d_near"].to_numpy(), resolved["d_far"].to_numpy()], axis=1),
        axis=0,
    )

    assert (intervals[1:, 0] >= intervals[:-1, 1] - 1e-3).all()


def test_fill_lowest_extends_the_far_field(plot_rdb):
    """Beyond the lowest beam's reach the slice is extended, never shrunk."""
    assert _n_polys(plot_cappi(plot_rdb, altitude=1200, fill_lowest=True)) >= _n_polys(
        plot_cappi(plot_rdb, altitude=1200, fill_lowest=False),
    )


def test_an_altitude_above_every_beam_raises(plot_rdb):
    """There is no slice at 100 km; say so rather than draw nothing."""
    with pytest.raises(ValueError, match="reaches"):
        plot_cappi(plot_rdb, altitude=99_999.0)


def test_cappi_rejects_an_unknown_overlap_mode(plot_rdb):
    """Only ``nearest`` and ``all`` exist."""
    with pytest.raises(ValueError):
        plot_cappi(plot_rdb, altitude=1200, overlap="sometimes")


def test_slice_polygons_sit_inside_the_full_footprints(plot_rdb, plot_archive_dir):
    """The constant-z cut trims gates along the beam; it never grows them."""
    drawn = np.array([path.vertices[:4] for path in plot_cappi(plot_rdb, altitude=1200, overlap="all").get_paths()])
    tbl = gate_corner_table(PLOT_RADAR, plot_archive_dir, kind="h_plane")
    full = np.stack(
        [np.stack([tbl[f"x_{k}"].to_numpy(), tbl[f"y_{k}"].to_numpy()], axis=1) for k in range(1, 5)],
        axis=1,
    )

    assert shapely.area(shapely.polygons(drawn)).sum() <= shapely.area(shapely.polygons(full)).sum()


# ---------------------------------------------------------------------------
# plot_vcs — the section has to be defined
# ---------------------------------------------------------------------------


def test_plot_vcs(plot_rdb, plot_site):
    """A line cuts the section and then draws it, in one call."""
    assert _n_polys(plot_vcs(plot_rdb, line=_line_across(plot_site))) > 0


def test_vcs_accepts_a_linestring(plot_rdb, plot_site):
    """A shapely LineString is the same thing as a point pair."""
    p1, p2 = _line_across(plot_site)

    assert _n_polys(plot_vcs(plot_rdb, line=shapely.LineString([p1, p2]))) > 0


def test_vcs_accepts_a_precut_frame(plot_rdb, plot_site):
    """A frame that already carries ``cs_polygon`` is drawn directly."""
    p1, p2 = _line_across(plot_site)
    cs = plot_rdb.extract_cross_section(p1, p2)

    assert _n_polys(plot_vcs(cs)) == len(cs)


def test_vcs_without_a_section_raises(plot_rdb):
    """No line and no ``cs_polygon`` leaves the section undefined."""
    with pytest.raises(ValueError, match="cross-section"):
        plot_vcs(plot_rdb)


def test_vcs_with_both_a_line_and_a_precut_frame_is_ambiguous(plot_rdb, plot_site):
    """Two definitions of the same section; refuse rather than pick one."""
    p1, p2 = _line_across(plot_site)
    cs = plot_rdb.extract_cross_section(p1, p2)

    with pytest.raises(ValueError, match="ambiguous"):
        plot_vcs(cs, line=(plot_site, (plot_site[0] + 5_000, plot_site[1])))


def test_an_area_cropped_frame_has_no_section(plot_rdb, plot_site):
    """The common mistake: an AOI crop selects an area, not a line."""
    crop = plot_rdb.crop_around_point(plot_site, distance=10_000)

    with pytest.raises(ValueError, match="no 'cs_polygon'"):
        plot_vcs(crop)


def test_vcs_refuses_a_datatree():
    """The section path is ``gate_id``-keyed, and a DataTree has none until archived."""
    with pytest.raises(TypeError, match="Archive the volume first"):
        plot_vcs(build_datatree(24, 20), line=((0, 0), (1, 1)))


def test_a_geojson_line_honors_its_own_crs(plot_rdb, plot_site, tmp_path):
    """A GeoJSON is lon/lat by RFC 7946, not archive meters.

    Reading those degrees as projected meters would put the section thousands of km away.
    """
    import pyproj

    from raddb.aoi import _to_pyproj_crs

    transformer = pyproj.Transformer.from_crs(_to_pyproj_crs(FMI_EPSG), _to_pyproj_crs(4326), always_xy=True)
    p1, p2 = _line_across(plot_site)
    a, b = transformer.transform(*p1), transformer.transform(*p2)
    path = tmp_path / "section.geojson"
    path.write_text(json.dumps({"type": "LineString", "coordinates": [list(a), list(b)]}))

    from_file = _n_polys(plot_vcs(plot_rdb, line=str(path)))
    from_meters = _n_polys(plot_vcs(plot_rdb, line=(p1, p2)))

    assert from_file > 0
    assert abs(from_file - from_meters) <= 0.02 * from_meters


def test_a_shapefile_line_is_read(plot_rdb, plot_site, tmp_path):
    """Pyshp reads the ``.shp``; the ``.prj`` (absent here) would declare the CRS."""
    shapefile = pytest.importorskip("shapefile")

    p1, p2 = _line_across(plot_site)
    writer = shapefile.Writer(str(tmp_path / "sec"))
    writer.field("id", "N")
    writer.line([[list(p1), list(p2)]])
    writer.record(1)
    writer.close()

    assert _n_polys(plot_vcs(plot_rdb, line=str(tmp_path / "sec.shp"))) > 0


@pytest.mark.parametrize("kind", ["polars", "pandas", "geopandas"])
def test_a_line_works_from_a_bare_frame_with_an_archive(plot_rdb, plot_site, plot_archive_dir, kind):
    """``plot_vcs`` must not be stricter than ``plot_ppi``: gate_id plus the LUT is enough."""
    data = {"polars": plot_rdb.data, "pandas": plot_rdb.to_pandas(), "geopandas": plot_rdb.to_geopandas()}[kind]
    line = _line_across(plot_site)

    assert _n_polys(plot_vcs(data, line=line, archive_dir=plot_archive_dir)) == _n_polys(
        plot_vcs(plot_rdb, line=line),
    )


def test_a_line_from_a_bare_frame_without_an_archive_raises(plot_rdb, plot_site):
    """The section is cut against the LUT, so the archive is required."""
    with pytest.raises(ValueError, match="archive"):
        plot_vcs(plot_rdb.data, line=(plot_site, (plot_site[0] + 5_000, plot_site[1])))


def test_a_precut_frame_survives_a_pandas_or_gdf_round_trip(plot_rdb, plot_site, plot_archive_dir):
    """``cs_polygon`` holds shapely objects; they must WKB-encode into polars."""
    p1, p2 = _line_across(plot_site)
    cs = plot_rdb.extract_cross_section(p1, p2)
    expected = _n_polys(plot_vcs(cs))

    assert _n_polys(plot_vcs(cs.to_pandas(), archive_dir=plot_archive_dir)) == expected
    assert _n_polys(plot_vcs(cs.to_geopandas(), archive_dir=plot_archive_dir)) == expected


def test_line_endpoints_reports_the_source_crs(tmp_path):
    """``_line_endpoints`` returns ``(p1, p2, src_crs)`` so a file's CRS can win."""
    path = tmp_path / "section.geojson"
    path.write_text(json.dumps({"type": "LineString", "coordinates": [[26.9, 62.0], [27.1, 62.0]]}))

    p1, p2, src_crs = _line_endpoints(str(path))

    assert src_crs == 4326
    assert p1 == pytest.approx((26.9, 62.0))
    assert p2 == pytest.approx((27.1, 62.0))


def test_line_endpoints_from_a_point_pair_declares_no_crs():
    """Hand-typed coordinates are in whatever frame the caller says — here, none."""
    p1, p2, src_crs = _line_endpoints(((0.0, 0.0), (1.0, 1.0)))

    assert (p1, p2, src_crs) == ((0.0, 0.0), (1.0, 1.0), None)


# ---------------------------------------------------------------------------
# plot_cross_section — the deprecated alias
# ---------------------------------------------------------------------------


def test_plot_cross_section(plot_rdb, plot_site):
    """The standalone renderer: it draws a frame that already carries ``cs_polygon``.

    Unlike ``plot_vcs`` it never cuts the section itself, and it returns
    ``(fig, ax, PolyCollection)`` rather than a bare artist.
    """
    p1, p2 = _line_across(plot_site)
    cs = plot_rdb.extract_cross_section(p1, p2)

    fig, ax, artist = plot_cross_section(cs.to_pandas())

    assert _n_polys(artist) == len(cs)
    assert artist.axes is ax
    assert ax.figure is fig


def test_plot_cross_section_honors_a_supplied_axes(plot_rdb, plot_site):
    """It composes like the other four, through ``ax=``."""
    p1, p2 = _line_across(plot_site)
    cs = plot_rdb.extract_cross_section(p1, p2)
    _, ax = plt.subplots()

    _, got_ax, artist = plot_cross_section(cs.to_pandas(), ax=ax)

    assert got_ax is ax
    assert artist.axes is ax


def test_the_raddb_alias_for_plot_cross_section_is_deprecated(plot_rdb, plot_site):
    """``RadDB.plot_cross_section`` warns and delegates to ``plot_vcs``."""
    p1, p2 = _line_across(plot_site)
    cs = plot_rdb.extract_cross_section(p1, p2)

    with pytest.deprecated_call():
        assert _n_polys(cs.plot_cross_section()) > 0


# ---------------------------------------------------------------------------
# plot_aoi_quicklook — the crop/section backdrop
# ---------------------------------------------------------------------------


def test_plot_aoi_quicklook(plot_archive_dir, plot_site):
    """Draws the AOI outline and returns the Axes it drew into."""
    aoi = shapely.Point(*plot_site).buffer(10_000)

    _fig, ax = plot_aoi_quicklook(aoi, radars=[PLOT_RADAR], base_path=plot_archive_dir, epsg=FMI_EPSG)

    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    assert x0 <= plot_site[0] <= x1
    assert y0 <= plot_site[1] <= y1


def test_the_quicklook_is_framed_on_the_archive_not_on_a_fixed_country(us_archive_dir):
    """A view framed on a fixed home country used to put a US AOI 5,855 km off-map."""
    from raddb.aoi import _reproject_to_aoi

    site = _reproject_to_aoi(shapely.Point(*US_SITE), 4326, US_EPSG)

    _fig, ax = plot_aoi_quicklook(site.buffer(20_000), radars=[RADAR], base_path=us_archive_dir, epsg=US_EPSG)

    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    assert x0 <= site.x <= x1
    assert y0 <= site.y <= y1


def test_the_quicklook_can_show_the_selected_gates(plot_archive_dir, plot_site, plot_rdb):
    """``show_gates=True`` samples the selection on top of the outline."""
    aoi = shapely.Point(*plot_site).buffer(8_000)
    selected = plot_rdb.crop_around_point(plot_site, distance=8_000)

    _fig, ax = plot_aoi_quicklook(
        aoi,
        selected=selected.data,
        radars=[PLOT_RADAR],
        base_path=plot_archive_dir,
        epsg=FMI_EPSG,
        show_gates=True,
    )

    assert ax.collections or ax.lines


def test_the_quicklook_saves_to_a_file(plot_archive_dir, plot_site, tmp_path):
    """``save_path=`` writes the figure out."""
    out = tmp_path / "quicklook.png"

    plot_aoi_quicklook(
        shapely.Point(*plot_site).buffer(10_000),
        radars=[PLOT_RADAR],
        base_path=plot_archive_dir,
        epsg=FMI_EPSG,
        save_path=str(out),
    )

    assert out.exists()
    assert out.stat().st_size > 0


def test_the_quicklook_context_can_be_dropped(plot_archive_dir, plot_site):
    """``context=None`` draws the AOI with no country outline behind it."""
    _fig, ax = plot_aoi_quicklook(
        shapely.Point(*plot_site).buffer(10_000),
        radars=[PLOT_RADAR],
        base_path=plot_archive_dir,
        epsg=FMI_EPSG,
        context=None,
    )

    assert ax is not None


# ---------------------------------------------------------------------------
# Coordinate frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coords", ["xy", "cartesian", "lonlat", "geo", "projected", 3067])
def test_every_accepted_coordinate_frame_plots(plot_rdb, coords):
    """Projection and basemap are separate; ``coords`` only chooses the frame."""
    assert _n_polys(plot_ppi(plot_rdb, sweep=1, coords=coords)) > 0


def test_an_unstyled_variable_still_plots(plot_rdb):
    """No entry in PLOT_DEFAULTS is not an error — matplotlib's own defaults apply."""
    assert "VRADH" in plot_rdb.columns(), "the fixture should carry an extra moment"

    assert _n_polys(plot_ppi(plot_rdb, sweep=1, variable="VRADH")) > 0


def test_PLOT_DEFAULTS_is_the_extension_point(plot_rdb):
    """A package with its own moments registers them; RadDB ships none of its own.

    The discrete-classification machinery stays here because it is not specific to any
    network — FMI's ``HCLASS`` and a MeteoSwiss ``HC_MCH`` both need it — but the class
    tables themselves belong to whoever produces the data.
    """
    assert not {"HC_MCH", "HC_PYART", "HZT", "TEMP"} & set(PLOT_DEFAULTS), "MCH styling must not ship in raddb"

    PLOT_DEFAULTS["VRADH"] = {
        "discrete": True,
        "classes": ["a", "b", "c"],
        "colors": ["red", "green", "blue"],
        "label": "Registered elsewhere",
    }
    try:
        artist = plot_ppi(plot_rdb, sweep=1, variable="VRADH")

        cbar_axis = artist.axes.figure.axes[-1].yaxis
        assert [tick.get_text() for tick in cbar_axis.get_ticklabels()] == ["a", "b", "c"]
    finally:
        del PLOT_DEFAULTS["VRADH"]


@pytest.mark.parametrize("alias", ["swiss", "lv95", "2056"])
def test_the_named_lv95_aliases_resolve_to_a_projected_frame(alias):
    """``coords="swiss"`` is a spelling of ``coords=2056``.

    Checked on the alias table rather than on a plot: an archive only carries the
    ``x_<epsg>`` columns of the CRS it was written with, and the fixtures are TM35FIN.
    """
    assert vp._resolve_coords(alias, None) == ("projected", 2056)


def test_lonlat_axes_are_in_degrees(plot_rdb):
    """``lonlat`` must not leave meters on the axis."""
    artist = plot_ppi(plot_rdb, sweep=1, coords="lonlat")

    assert -180 <= artist.axes.get_xlim()[0] <= 180
    assert -90 <= artist.axes.get_ylim()[0] <= 90


def test_projected_axes_are_tm35fin_meters(plot_rdb):
    """``coords=3067`` uses the LUT's own ``x_3067``/``y_3067`` columns.

    The site sits on TM35FIN's central meridian, so its easting is the 500 km false
    origin and the whole 20 km volume stays inside 470-530 km.
    """
    assert 4.7e5 < plot_ppi(plot_rdb, sweep=1, coords=3067).axes.get_xlim()[0] < 5.0e5


def test_xy_is_centered_on_the_radar(plot_rdb):
    """``xy`` is meters from the radar, so the origin is inside the view."""
    x0, x1 = plot_ppi(plot_rdb, sweep=1, coords="xy").axes.get_xlim()

    assert x0 < 0 < x1


def test_an_unknown_coordinate_frame_raises(plot_rdb):
    """The message names the argument so the typo is findable."""
    with pytest.raises(ValueError, match="coords"):
        plot_ppi(plot_rdb, sweep=1, coords="banana")


def test_projected_resolves_the_archives_own_crs(plot_archive_dir):
    """Reading needs no CRS: the archive records the one it was written with."""
    plain = RadDB(archive_dir=str(plot_archive_dir)).open(radars=PLOT_RADAR)

    assert plain._crs is None
    assert plain.crs().to_epsg() == FMI_EPSG
    assert _n_polys(plot_ppi(plain, sweep=1, coords="projected")) > 0


def test_projected_raises_when_nothing_declares_a_crs(plot_rdb):
    """A bare frame with no archive has nothing to resolve from."""
    with pytest.raises((ValueError, KeyError)):
        plot_ppi(plot_rdb.data, sweep=1, coords="projected", archive_dir=None)


# ---------------------------------------------------------------------------
# Volume selection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def multi_volume_rdb(tmp_path_factory):
    """A two-volume archive, so the volume-selection arguments have something to pick."""
    base = tmp_path_factory.mktemp("multi_vol")
    db = RadDB(archive_dir=str(base), crs=FMI_EPSG)
    db.archive(
        datatree={str(t): build_datatree(24, 20, n_sweeps=2, vol_time=t) for t in VOL_TIMES},
        radar=PLOT_RADAR,
    )
    return db.open(radars=PLOT_RADAR)


def test_several_volumes_without_a_timestep_raises(multi_volume_rdb):
    """A PPI draws one volume; which one must be said."""
    with pytest.raises(ValueError, match="volumes"):
        plot_ppi(multi_volume_rdb, sweep=1)


def test_a_timestep_picks_the_nearest_volume(multi_volume_rdb):
    """``timestep=`` snaps to the closest recorded volume time."""
    assert _n_polys(plot_ppi(multi_volume_rdb, sweep=1, timestep=VOL_TIMES[0])) > 0


def test_a_time_window_narrows_to_one_volume(multi_volume_rdb):
    """``start_time``/``end_time`` are the alternative to ``timestep``."""
    assert _n_polys(plot_ppi(multi_volume_rdb, sweep=1, start_time="2024-08-01", end_time="2024-08-01 23:59")) > 0


def test_a_window_excluding_everything_raises(multi_volume_rdb):
    """An empty window is an error, not an empty plot."""
    with pytest.raises(ValueError):
        plot_ppi(multi_volume_rdb, sweep=1, start_time="1999-01-01", end_time="1999-12-31")


# ---------------------------------------------------------------------------
# DataTree input — no archive needed
# ---------------------------------------------------------------------------


def test_a_datatree_needs_no_archive(plot_dtree):
    """The whole point: a DataTree is self-describing."""
    assert _n_polys(plot_ppi(plot_dtree, sweep=1, variable="DBZH", archive_dir=None)) == N_AZ * N_RNG


def test_datatree_geometry_matches_the_stored_lattice(plot_dtree, plot_archive_dir):
    """Corners computed from the tree's own coords must equal the ones generation stored."""
    drawn = np.array([q.vertices[:4] for q in plot_ppi(plot_dtree, sweep=1, variable="DBZH").get_paths()])
    tbl = gate_corner_table(PLOT_RADAR, plot_archive_dir, kind="h_plane", sweep=1)
    stored = np.stack(
        [np.stack([tbl[f"x_{k}"].to_numpy(), tbl[f"y_{k}"].to_numpy()], axis=1) for k in range(1, 5)],
        axis=1,
    )

    # The lattices are float32, so ~1e-7 relative — a few centimeters at 200 km range.
    # Anything tighter would be testing parquet, not geometry.
    assert np.abs(np.sort(drawn, axis=0) - np.sort(stored, axis=0)).max() < 5e-2


def test_an_rhi_from_a_datatree(plot_dtree):
    """Vertical faces are computed on the fly from the declared beamwidth."""
    assert _n_polys(plot_rhi(plot_dtree, azimuth=0, variable="DBZH")) == N_RNG * N_SWEEPS


def test_a_cappi_from_a_datatree_matches_the_lut_path(plot_dtree, plot_rdb):
    """Both paths must select the same chords at the same altitude."""
    assert _n_polys(plot_cappi(plot_dtree, altitude=1200, variable="DBZH")) == _n_polys(
        plot_cappi(plot_rdb, altitude=1200),
    )


def test_a_datatree_with_an_unknown_variable_raises(plot_dtree):
    """Same contract as the archive path."""
    with pytest.raises(KeyError):
        plot_ppi(plot_dtree, sweep=1, variable="NOPE")


def test_a_datatree_with_a_missing_sweep_raises(plot_dtree):
    """Same contract as the archive path."""
    with pytest.raises(ValueError, match="sweep"):
        plot_ppi(plot_dtree, sweep=99, variable="DBZH")


# ---------------------------------------------------------------------------
# GeoDataFrame input — treated as a frame
# ---------------------------------------------------------------------------


def test_a_geodataframe_goes_through_the_lut(plot_gdf, plot_archive_dir):
    """A gdf is not special-cased: geometry always comes from the LUT."""
    assert _n_polys(plot_ppi(plot_gdf, sweep=1, archive_dir=plot_archive_dir)) > 0


def test_a_geodataframe_without_an_archive_raises(plot_gdf):
    """Its own geometry column is ignored, so the LUT is still required."""
    with pytest.raises(ValueError, match="archive"):
        plot_ppi(plot_gdf, sweep=1)


def test_a_geodataframe_cappi_matches_a_frame(plot_gdf, plot_archive_dir, plot_rdb):
    """Same geometry source, same result."""
    assert _n_polys(plot_cappi(plot_gdf, altitude=1200, archive_dir=plot_archive_dir)) == _n_polys(
        plot_cappi(plot_rdb, altitude=1200),
    )


def test_the_geometry_column_is_not_required(plot_gdf, plot_archive_dir):
    """Dropping it changes nothing, which is exactly the claim."""
    plain = plot_gdf.drop(columns=plot_gdf.geometry.name)

    assert _n_polys(plot_ppi(plain, sweep=1, archive_dir=plot_archive_dir)) > 0


# ---------------------------------------------------------------------------
# Beamwidth is a LUT-generation parameter, not a plot argument
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["plot_ppi", "plot_rhi", "plot_cappi"])
def test_no_plot_takes_a_beamwidth(name):
    """For archive-backed data it was applied when ``v_plane`` was generated."""
    assert "beamwidth_deg" not in inspect.signature(getattr(vp, name)).parameters


def test_beamwidth_comes_from_the_datatree_or_the_default(plot_dtree):
    """No archive to bake it in, so it is inferred as LUT generation does."""
    dt = copy.deepcopy(plot_dtree)
    src = type("S", (), {"kind": "datatree", "dtree": dt})()

    assert _beamwidth(src) == DEFAULT_BEAMWIDTH_DEG

    dt.attrs["radar_beam_width_h"] = 0.5
    assert _beamwidth(src) == 0.5


def test_a_wider_declared_beam_reaches_a_cappi_over_more_bins(plot_dtree):
    """Only ``v_plane``/``corners`` depend on beamwidth — and this is how it shows."""
    narrow = copy.deepcopy(plot_dtree)
    narrow.attrs["radar_beam_width_h"] = 0.5
    wide = copy.deepcopy(plot_dtree)
    wide.attrs["radar_beam_width_h"] = 2.0

    assert _n_polys(plot_cappi(wide, altitude=1200, variable="DBZH")) > _n_polys(
        plot_cappi(narrow, altitude=1200, variable="DBZH"),
    )


# ---------------------------------------------------------------------------
# _KmFormatter — data in meters, axes labeled in km
# ---------------------------------------------------------------------------


def _tick_labels(lo, hi, offset=0.0):
    """Render a y-axis through ``_KmFormatter`` and return ``(value, label)`` pairs."""
    fig, ax = plt.subplots()
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(_KmFormatter(offset))
    fig.canvas.draw()
    locs = np.asarray(ax.yaxis.get_majorticklocs(), dtype=float)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    plt.close(fig)
    return [(v, lab) for v, lab in zip(locs, labels, strict=False) if lab and lo <= v <= hi]


@pytest.mark.parametrize(
    ("lo", "hi", "offset"),
    [
        (1400, 5900, 0.0),  # the reported case: 500 m steps
        (0, 20000, 0.0),  # matplotlib picks 2.5 km steps
        (0, 13000, 0.0),
        (0, 500, 0.0),
        (0, 200, 0.0),  # 25 m steps -> 3 decimals
        (0, 60, 0.0),
        (0, 250000, 0.0),
        (2_600_000, 2_760_000, 2e6),  # LV95 easting
        (1_050_000, 1_150_000, 1e6),  # LV95 northing
    ],
)
def test_km_labels_are_unique_and_exact(lo, hi, offset):
    """Decimals come from the ticks matplotlib chose, not from ``log10(step)``.

    A fixed ``.0f`` collapses labels below a 1 km step, and deriving decimals from the
    step's magnitude misses matplotlib's 2.5x10**n steps: 2500 m would print
    ``0, 2, 5, 8, 10`` — distinct, so a duplicate check passes, but wrong.
    """
    pairs = _tick_labels(lo, hi, offset)
    labels = [lab for _, lab in pairs]

    assert len(labels) == len(set(labels)), f"repeated labels: {labels}"
    for value, label in pairs:
        km = (value - offset) / 1e3
        assert abs(float(label) - km) < 1e-6 * max(1.0, abs(km)), f"label {label!r} does not equal {km}"


def test_the_reported_km_label_regression():
    """A 1.4-5.9 km section used to read 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6."""
    assert [lab for _, lab in _tick_labels(1400, 5900)] == [
        "1.5",
        "2.0",
        "2.5",
        "3.0",
        "3.5",
        "4.0",
        "4.5",
        "5.0",
        "5.5",
    ]


def test_real_plots_have_no_duplicate_ticks(plot_rdb, plot_site):
    """Every plot, both axes."""
    artists = [
        plot_ppi(plot_rdb, sweep=1),
        plot_ppi(plot_rdb, sweep=1, coords="projected"),
        plot_rhi(plot_rdb, azimuth=0),
        plot_cappi(plot_rdb, altitude=1200),
        plot_vcs(plot_rdb, line=_line_across(plot_site)),
    ]

    for artist in artists:
        artist.axes.figure.canvas.draw()
        for axis in (artist.axes.xaxis, artist.axes.yaxis):
            labels = [t.get_text() for t in axis.get_ticklabels() if t.get_text() and t.get_visible()]
            assert len(labels) == len(set(labels)), f"repeated ticks: {labels}"

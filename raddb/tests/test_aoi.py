"""Tests for :mod:`raddb.aoi` — AOI selection, crop geometry and cross-sections.

The module has only two public callables, but it is where the package's most expensive
bug lived: ``aoi.py`` used to hardcode EPSG:2056, so a "50 km" crop at a US radar reached
about 46 km — a silent 17% error that looked entirely normal on a map.

So the theme throughout is: **every AOI runs in the archive's own CRS**, resolved from
``{radar}_info.yaml`` (or recovered from the LUT's ``x_<epsg>`` columns), never defaulted.
The Finnish and US archives are tested side by side for exactly that reason.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pyproj
import pytest
import shapely

from raddb.aoi import (
    SWISS_EPSG,
    _apply_gate_ids,
    _crs_to_spec,
    _geojson_crs,
    _load_aoi_polygon,
    _lut_centroids,
    _prj_crs,
    _radars_from_gate_ids,
    _read_geometry_file,
    _reproject_to_aoi,
    _resolve_aoi_centroids,
    _resolve_context,
    _resolve_gate_ids,
    _to_pyproj_crs,
    aoi_epsg,
    aoi_epsg_for,
)
from raddb.main import RadDB
from raddb.tests.conftest import FI_SITE, FMI_EPSG, RADAR, RADAR_B, US_EPSG, US_SITE, relocate

# ---------------------------------------------------------------------------
# aoi_epsg — the archive's own frame
# ---------------------------------------------------------------------------


def test_aoi_epsg(archive_dir, us_archive_dir):
    """The frame comes from the archive that was written, not from a default."""
    assert aoi_epsg(archive_dir, RADAR) == FMI_EPSG
    assert aoi_epsg(us_archive_dir, RADAR) == US_EPSG


def test_aoi_epsg_is_recovered_from_the_lut_when_info_has_no_crs_block(archive_dir):
    """Archives predating the ``crs`` block still resolve, from ``x_<epsg>`` columns."""
    import yaml

    info_path = archive_dir / RADAR / "LUT" / f"{RADAR}_info.yaml"
    info = yaml.safe_load(info_path.read_text())
    info.pop("crs", None)
    info_path.write_text(yaml.safe_dump(info))

    assert aoi_epsg(archive_dir, RADAR) == FMI_EPSG


def test_aoi_epsg_refuses_an_unprojected_archive(tmp_path, archive_dir):
    """With neither an info CRS nor projected columns there is nothing to guess from."""
    import yaml

    info_path = archive_dir / RADAR / "LUT" / f"{RADAR}_info.yaml"
    info = yaml.safe_load(info_path.read_text())
    info.pop("crs", None)
    info_path.write_text(yaml.safe_dump(info))

    lut_path = archive_dir / RADAR / "LUT" / f"{RADAR}_LUT.parquet"
    lut = pl.read_parquet(lut_path)
    lut.drop([c for c in lut.columns if c[:2] in ("x_", "y_") and c[2:].isdigit()]).write_parquet(lut_path)

    with pytest.raises(ValueError, match="no projected coordinates"):
        aoi_epsg(archive_dir, RADAR)


def test_aoi_epsg_names_the_way_out_in_its_error(archive_dir):
    """The refusal must tell the user what to pass, not merely complain."""
    import yaml

    info_path = archive_dir / RADAR / "LUT" / f"{RADAR}_info.yaml"
    info = yaml.safe_load(info_path.read_text())
    info.pop("crs", None)
    info_path.write_text(yaml.safe_dump(info))
    lut_path = archive_dir / RADAR / "LUT" / f"{RADAR}_LUT.parquet"
    lut = pl.read_parquet(lut_path)
    lut.drop([c for c in lut.columns if c[:2] in ("x_", "y_") and c[2:].isdigit()]).write_parquet(lut_path)

    with pytest.raises(ValueError, match=r"aoi_crs="):
        aoi_epsg(archive_dir, RADAR)


# ---------------------------------------------------------------------------
# aoi_epsg_for — one frame for a multi-radar AOI
# ---------------------------------------------------------------------------


def test_aoi_epsg_for(archive_dir_two_radars):
    """Radars written in the same CRS share it without argument."""
    assert aoi_epsg_for(archive_dir_two_radars, [RADAR, RADAR_B]) == FMI_EPSG


def test_mixed_crs_radars_refuse_a_shared_aoi(tmp_path, make_datatree):
    """No silent reprojection: the user must name the common frame."""
    base = tmp_path / "mixed"
    RadDB(archive_dir=str(base), crs=FMI_EPSG).archive(
        datatree={RADAR: [make_datatree(n_az=24, n_rng=20, n_sweeps=2)]},
    )
    RadDB(archive_dir=str(base), crs=US_EPSG).archive(
        datatree={RADAR_B: [relocate(make_datatree(n_az=24, n_rng=20, n_sweeps=2), *US_SITE)]},
    )

    assert aoi_epsg_for(base, [RADAR]) == FMI_EPSG
    assert aoi_epsg_for(base, [RADAR_B]) == US_EPSG
    with pytest.raises(ValueError, match="different CRSs"):
        aoi_epsg_for(base, [RADAR, RADAR_B])


def test_an_override_wins_over_the_archive(archive_dir):
    """``aoi_crs=`` names a common frame explicitly."""
    assert aoi_epsg_for(archive_dir, [RADAR], override=32635) == 32635


def test_an_override_is_validated_against_every_site(us_archive_dir):
    """An override still has to be valid where the radar actually is."""
    with pytest.raises(ValueError, match="distorts distance"):
        aoi_epsg_for(us_archive_dir, [RADAR], override=FMI_EPSG)


# ---------------------------------------------------------------------------
# _lut_centroids / _resolve_gate_ids — the selection itself
# ---------------------------------------------------------------------------


def test_lut_centroids_returns_a_fixed_column_set(archive_dir):
    """Callers never have to know the EPSG: the projected pair is renamed x/y."""
    centroids = _lut_centroids(archive_dir, [RADAR])

    assert {"gate_id", "radar", "sweep", "x", "y", "z", "altitude"} <= set(centroids.columns)
    assert centroids["radar"].unique().to_list() == [RADAR]
    assert not centroids.is_empty()


def test_lut_centroids_concatenates_radars(archive_dir_two_radars):
    """A multi-radar AOI sees one table spanning both."""
    centroids = _lut_centroids(archive_dir_two_radars, [RADAR, RADAR_B])

    assert sorted(centroids["radar"].unique().to_list()) == sorted([RADAR, RADAR_B])


def test_lut_centroids_projects_on_the_fly_for_an_override(archive_dir):
    """An ``aoi_crs`` the LUT does not store is computed from latitude/longitude."""
    centroids = _lut_centroids(archive_dir, [RADAR], epsg=32635)

    assert {"x", "y"} <= set(centroids.columns)
    assert np.isfinite(centroids["x"].to_numpy()).all()


def test_lut_centroids_raises_on_a_missing_lut(tmp_path):
    """A radar with no LUT cannot be cropped; say so rather than return nothing."""
    with pytest.raises((FileNotFoundError, ValueError)):
        _lut_centroids(tmp_path, ["Z"])


def test_resolve_gate_ids_selects_only_gates_inside(archive_dir):
    """A tight buffer around the site keeps a strict subset of the gates."""
    centroids = _lut_centroids(archive_dir, [RADAR])
    site = _reproject_to_aoi(shapely.Point(*FI_SITE), 4326, FMI_EPSG)

    ids = _resolve_gate_ids(centroids, site.buffer(5_000))

    assert 0 < len(ids) < centroids.height
    assert ids.dtype == np.int64


def test_resolve_gate_ids_is_empty_outside_the_radar(archive_dir):
    """An AOI nowhere near the radar selects nothing, and does not raise."""
    centroids = _lut_centroids(archive_dir, [RADAR])
    far = shapely.Point(0.0, 0.0).buffer(1_000)

    assert len(_resolve_gate_ids(centroids, far)) == 0


def test_resolve_aoi_centroids_returns_rows_not_just_ids(archive_dir):
    """Callers clip by altitude afterwards, so the rows must survive the filter."""
    centroids = _lut_centroids(archive_dir, [RADAR])
    site = _reproject_to_aoi(shapely.Point(*FI_SITE), 4326, FMI_EPSG)

    sub = _resolve_aoi_centroids(centroids, site.buffer(5_000))

    assert set(sub.columns) == set(centroids.columns)
    assert sub.height == len(_resolve_gate_ids(centroids, site.buffer(5_000)))


def test_resolve_aoi_centroids_on_an_empty_table(archive_dir):
    """An empty input yields an empty output with the schema intact."""
    empty = _lut_centroids(archive_dir, [RADAR]).clear()

    assert _resolve_aoi_centroids(empty, shapely.Point(0, 0).buffer(1)).is_empty()


def test_a_footprint_selects_the_whole_vertical_column(archive_dir):
    """Intersection runs over every sweep, so one footprint takes the column above it."""
    centroids = _lut_centroids(archive_dir, [RADAR])
    site = _reproject_to_aoi(shapely.Point(*FI_SITE), 4326, FMI_EPSG)

    sub = _resolve_aoi_centroids(centroids, site.buffer(8_000))

    assert sorted(sub["sweep"].unique().to_list()) == sorted(centroids["sweep"].unique().to_list())


def test_a_crop_radius_is_true_meters(us_archive_dir):
    """The 17% bug, in one assertion: a 10 km crop must select a 10 km radius.

    Truth is a WGS-84 geodesic from the radar to every gate; the projected selection
    must agree to within 1%.
    """
    lut = RadDB(archive_dir=str(us_archive_dir)).get_lut(RADAR)
    lon = lut["longitude"].to_numpy()
    lat = lut["latitude"].to_numpy()
    _, _, ground = pyproj.Geod(ellps="WGS84").inv(
        np.full(lon.size, US_SITE[0]),
        np.full(lat.size, US_SITE[1]),
        lon,
        lat,
    )

    centroids = _lut_centroids(us_archive_dir, [RADAR])
    site = _reproject_to_aoi(shapely.Point(*US_SITE), 4326, US_EPSG)
    for radius in (10_000, 15_000):
        truth = int((ground <= radius).sum())
        got = len(_resolve_gate_ids(centroids, site.buffer(radius)))
        assert abs(got - truth) <= 0.01 * truth, f"{radius / 1000:.0f} km crop took {got} gates, truth {truth}"


# ---------------------------------------------------------------------------
# _apply_gate_ids — the semi-join onto dynamic data
# ---------------------------------------------------------------------------


def test_apply_gate_ids_selects_rows_without_widening():
    """A semi-join: the LUT geometry stays in its own table."""
    df = pl.DataFrame({"gate_id": [1, 2, 3], "DBZH": [10.0, 20.0, 30.0]})

    out = _apply_gate_ids(df, np.array([1, 3], dtype=np.int64))

    assert out.columns == df.columns
    assert out["gate_id"].to_list() == [1, 3]


def test_apply_gate_ids_with_an_empty_selection():
    """No gates in the AOI yields an empty frame, not the whole input."""
    df = pl.DataFrame({"gate_id": [1, 2], "DBZH": [10.0, 20.0]})

    assert _apply_gate_ids(df, np.empty(0, dtype=np.int64)).is_empty()


def test_apply_gate_ids_requires_the_join_key():
    """Without ``gate_id`` there is no way to apply an AOI; fail loudly."""
    with pytest.raises(KeyError, match="gate_id"):
        _apply_gate_ids(pl.DataFrame({"DBZH": [1.0]}), np.array([1], dtype=np.int64))


def test_radars_from_gate_ids(archive_dir):
    """The radar name is embedded in every ``gate_id``, so no registry is needed."""
    ids = _lut_centroids(archive_dir, [RADAR])["gate_id"].to_numpy()

    assert _radars_from_gate_ids(ids) == [RADAR]


# ---------------------------------------------------------------------------
# _to_pyproj_crs / _reproject_to_aoi — the frame conversions
# ---------------------------------------------------------------------------


def test_to_pyproj_crs_accepts_the_four_spellings():
    """int, proj4 string, EPSG string and a ready CRS all resolve."""
    from pyproj import CRS

    assert _to_pyproj_crs(2056).is_projected
    assert _to_pyproj_crs("+proj=longlat +datum=WGS84 +no_defs").is_geographic
    assert _to_pyproj_crs("EPSG:32614").is_projected
    crs = CRS.from_epsg(2056)
    assert _to_pyproj_crs(crs) is crs


def test_known_frames_resolve_without_the_proj_database():
    """2056 and 4326 go through proj4 strings so a broken PROJ db cannot break them."""
    assert _to_pyproj_crs(2056).to_proj4()
    assert _to_pyproj_crs(4326).is_geographic


def test_reproject_to_aoi_is_a_no_op_when_the_frames_match():
    """``None`` and every spelling of the AOI's own EPSG return the geometry untouched."""
    geom = shapely.Point(2_600_000, 1_200_000)

    for crs in (None, 2056, "2056", "EPSG:2056", "epsg:2056"):
        assert _reproject_to_aoi(geom, crs, 2056) is geom


def test_reproject_to_aoi_moves_lonlat_into_meters():
    """Degrees in, meters out — the failure mode is a section thousands of km away."""
    projected = _reproject_to_aoi(shapely.Point(*FI_SITE), 4326, FMI_EPSG)

    assert 400_000 < projected.x < 600_000
    assert 6_700_000 < projected.y < 7_000_000


def test_reproject_to_aoi_round_trips():
    """There and back lands within a millimeter."""
    geom = shapely.Point(*FI_SITE)

    there = _reproject_to_aoi(geom, 4326, FMI_EPSG)
    back = _reproject_to_aoi(there, FMI_EPSG, 4326)

    assert back.distance(geom) < 1e-7


def test_crs_to_spec_reduces_to_an_epsg_int():
    """A pyproj CRS with a known EPSG becomes an int; ``None`` stays ``None``."""
    assert _crs_to_spec(pyproj.CRS.from_epsg(2056)) == 2056
    assert _crs_to_spec(None) is None


# ---------------------------------------------------------------------------
# _resolve_context — the quicklook backdrop
# ---------------------------------------------------------------------------


def test_resolve_context_none_is_none():
    """No context means no context, in any frame."""
    assert _resolve_context(None, aoi_epsg=SWISS_EPSG) is None


def test_resolve_context_rejects_an_unknown_name():
    """Only ``'switzerland'`` and its aliases are named contexts."""
    with pytest.raises(ValueError, match="unknown context"):
        _resolve_context("atlantis", aoi_epsg=SWISS_EPSG)


def test_resolve_context_takes_a_bare_geometry_as_already_in_frame():
    """A shapely geometry carries no CRS, so it is trusted as-is."""
    geom = shapely.Point(2_600_000, 1_200_000).buffer(50_000)

    assert _resolve_context(geom, aoi_epsg=SWISS_EPSG).equals(geom)


def test_resolve_context_reprojects_a_geodataframe_into_the_archive_frame(us_archive_dir):
    """A caller's context must not be reprojected to LV95 on a US archive.

    This is the fix for a backdrop drawn 5,855 km off-map at KTLX.
    """
    gpd = pytest.importorskip("geopandas")

    box = shapely.box(US_SITE[0] - 1, US_SITE[1] - 1, US_SITE[0] + 1, US_SITE[1] + 1)
    gdf = gpd.GeoDataFrame(geometry=[box], crs="EPSG:4326")

    got = _resolve_context(gdf, aoi_epsg=US_EPSG)

    assert got.distance(_reproject_to_aoi(box, 4326, US_EPSG)) < 1.0
    assert got.contains(_reproject_to_aoi(shapely.Point(*US_SITE), 4326, US_EPSG))


def test_resolve_context_survives_a_broken_geodataframe():
    """A context is decoration; a failure drops it rather than killing the plot."""

    class Broken:
        crs = "EPSG:4326"

        def union_all(self):
            raise RuntimeError("no geometry")

    assert _resolve_context(Broken(), aoi_epsg=SWISS_EPSG) is None


def test_resolve_context_ignores_an_unsupported_type():
    """Anything without a ``.crs`` and not a geometry yields ``None``."""
    assert _resolve_context(42, aoi_epsg=SWISS_EPSG) is None


# ---------------------------------------------------------------------------
# Geometry file loading — a file's declared CRS wins
# ---------------------------------------------------------------------------


def _write_geojson(path, geometry, crs_name=None):
    """Write a bare-geometry GeoJSON, optionally with a legacy ``crs`` member."""
    data = dict(geometry)
    if crs_name is not None:
        data["crs"] = {"type": "name", "properties": {"name": crs_name}}
    path.write_text(json.dumps(data))
    return path


def test_geojson_defaults_to_wgs84_per_rfc_7946():
    """A GeoJSON without a ``crs`` member is lon/lat, not archive meters."""
    assert _geojson_crs({"type": "Polygon"}) == 4326


def test_geojson_honors_a_legacy_crs_member():
    """The pre-RFC ``crs`` member is still read when present."""
    member = {"crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2056"}}}

    assert _geojson_crs(member) == SWISS_EPSG


def test_read_geometry_file_reads_a_polygon(tmp_path):
    """A polygon GeoJSON round-trips, and reports WGS-84."""
    square = {
        "type": "Polygon",
        "coordinates": [[[26.9, 61.9], [27.1, 61.9], [27.1, 62.1], [26.9, 62.1], [26.9, 61.9]]],
    }
    path = _write_geojson(tmp_path / "aoi.geojson", square)

    geom, crs = _read_geometry_file(path)

    assert geom.geom_type == "Polygon"
    assert crs == 4326


def test_read_geometry_file_reads_a_line(tmp_path):
    """Lines matter here: a cross-section is defined by one."""
    line = {"type": "LineString", "coordinates": [[26.9, 62.0], [27.1, 62.0]]}
    path = _write_geojson(tmp_path / "cs.geojson", line)

    geom, _ = _read_geometry_file(path)

    assert geom.geom_type == "LineString"


def test_read_geometry_file_reads_a_feature_collection(tmp_path):
    """The common export shape; features are unioned."""
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [27.0, 62.0]}, "properties": {}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [27.1, 62.1]}, "properties": {}},
        ],
    }
    path = tmp_path / "fc.geojson"
    path.write_text(json.dumps(fc))

    geom, _ = _read_geometry_file(path)

    assert geom.geom_type == "MultiPoint"


def test_read_geometry_file_rejects_an_unknown_type(tmp_path):
    """An unrecognized GeoJSON object is refused rather than half-read."""
    path = tmp_path / "bad.geojson"
    path.write_text(json.dumps({"type": "Topology"}))

    with pytest.raises(ValueError, match="Unrecognized GeoJSON"):
        _read_geometry_file(path)


def test_read_geometry_file_rejects_an_unsupported_suffix(tmp_path):
    """Only ``.shp`` and ``.geojson``/``.json`` are read."""
    path = tmp_path / "aoi.kml"
    path.touch()

    with pytest.raises(ValueError, match="Unsupported geometry file type"):
        _read_geometry_file(path)


def test_read_geometry_file_raises_on_a_missing_file(tmp_path):
    """A path typo must not be read as "no AOI"."""
    with pytest.raises(FileNotFoundError):
        _read_geometry_file(tmp_path / "nope.geojson")


def test_prj_crs_returns_none_without_a_prj(tmp_path):
    """A shapefile with no ``.prj`` declares nothing."""
    assert _prj_crs(tmp_path / "missing.prj") is None


def test_prj_crs_reads_a_wkt(tmp_path):
    """A readable ``.prj`` yields its EPSG code."""
    prj = tmp_path / "aoi.prj"
    prj.write_text(pyproj.CRS.from_epsg(SWISS_EPSG).to_wkt())

    assert _prj_crs(prj) == SWISS_EPSG


def test_prj_crs_ignores_an_empty_prj(tmp_path):
    """An empty file declares nothing rather than raising."""
    prj = tmp_path / "empty.prj"
    prj.write_text("   ")

    assert _prj_crs(prj) is None


# ---------------------------------------------------------------------------
# _load_aoi_polygon — the crop_by_polygon entry point
# ---------------------------------------------------------------------------


def test_load_aoi_polygon_from_a_shapely_geometry():
    """A bare geometry with no ``crs`` is taken as already in the AOI frame."""
    square = shapely.box(490_000, 6_864_000, 510_000, 6_884_000)

    assert _load_aoi_polygon(square, aoi_epsg=FMI_EPSG).equals(square)


def test_load_aoi_polygon_reprojects_a_geojson_from_its_own_crs(tmp_path):
    """A GeoJSON is lon/lat; reading those degrees as meters lands thousands of km away."""
    square = {
        "type": "Polygon",
        "coordinates": [[[26.9, 61.9], [27.1, 61.9], [27.1, 62.1], [26.9, 62.1], [26.9, 61.9]]],
    }
    path = _write_geojson(tmp_path / "aoi.geojson", square)

    geom = _load_aoi_polygon(path, aoi_epsg=FMI_EPSG)

    assert geom.contains(_reproject_to_aoi(shapely.Point(*FI_SITE), 4326, FMI_EPSG))


def test_an_explicit_crs_overrides_the_files_declaration(tmp_path):
    """``crs=`` wins, for a file that declares the wrong thing."""
    square = {
        "type": "Polygon",
        "coordinates": [[[26.9, 61.9], [27.1, 61.9], [27.1, 62.1], [26.9, 62.1], [26.9, 61.9]]],
    }
    path = _write_geojson(tmp_path / "aoi.geojson", square, crs_name="urn:ogc:def:crs:EPSG::3067")

    declared = _load_aoi_polygon(path, aoi_epsg=FMI_EPSG)
    overridden = _load_aoi_polygon(path, crs=4326, aoi_epsg=FMI_EPSG)

    assert not declared.equals(overridden)


def test_load_aoi_polygon_from_a_geodataframe():
    """A GeoDataFrame's own ``.crs`` is honored and its parts dissolved."""
    gpd = pytest.importorskip("geopandas")

    gdf = gpd.GeoDataFrame(geometry=[shapely.box(26.9, 61.9, 27.1, 62.1)], crs="EPSG:4326")

    geom = _load_aoi_polygon(gdf, aoi_epsg=FMI_EPSG)

    assert geom.contains(_reproject_to_aoi(shapely.Point(*FI_SITE), 4326, FMI_EPSG))


def test_load_aoi_polygon_rejects_a_line():
    """A crop needs an area; a line defines a cross-section instead."""
    with pytest.raises(ValueError, match="Polygon/MultiPolygon"):
        _load_aoi_polygon(shapely.LineString([(0, 0), (1, 1)]), aoi_epsg=FMI_EPSG)


def test_load_aoi_polygon_rejects_an_empty_geometry():
    """An empty AOI would select every gate or none; refuse it."""
    with pytest.raises(ValueError, match="empty"):
        _load_aoi_polygon(shapely.Polygon(), aoi_epsg=FMI_EPSG)


def test_load_aoi_polygon_rejects_an_unsupported_type():
    """The error must list what is accepted."""
    with pytest.raises(TypeError, match="shapely"):
        _load_aoi_polygon(42, aoi_epsg=FMI_EPSG)


# ---------------------------------------------------------------------------
# The lattices are the single source of gate geometry
# ---------------------------------------------------------------------------


def test_gate_footprints_come_from_the_h_plane_lattice(archive_dir):
    """``aoi.py`` and the plots must draw the same gate, corner for corner.

    The old planar / nominal-1-degree construction was off by 32 m mean (125 m max) per
    corner on a real radar and mis-sized high-elevation gates by -23% at sweep 20.
    """
    from raddb.aoi import _gate_footprints, _lut_cs_table

    cs_table = _lut_cs_table(archive_dir, [RADAR]).to_pandas().head(300)
    footprints = _gate_footprints(cs_table, np.tan(np.deg2rad(0.5)), base_path=archive_dir, epsg=FMI_EPSG)

    h_plane = RadDB(archive_dir=str(archive_dir)).get_h_plane(RADAR, per_gate=True)
    aligned = pl.DataFrame({"gate_id": cs_table["gate_id"].to_numpy()}).join(
        h_plane,
        on="gate_id",
        how="left",
        maintain_order="left",
    )
    reference = shapely.polygons(
        np.stack(
            [
                np.stack([aligned[f"x_3067_{k}"].to_numpy(), aligned[f"y_3067_{k}"].to_numpy()], axis=1)
                for k in range(1, 5)
            ],
            axis=1,
        ).astype(np.float64),
    )

    assert np.allclose(shapely.get_coordinates(footprints), shapely.get_coordinates(reference))


def test_the_planar_fallback_is_still_available(archive_dir):
    """Archives with no projected lattice keep working on the old approximation."""
    from raddb.aoi import _gate_footprints, _lut_cs_table

    cs_table = _lut_cs_table(archive_dir, [RADAR]).to_pandas().head(50)

    planar = _gate_footprints(cs_table, np.tan(np.deg2rad(0.5)), base_path=None, epsg=None)

    assert shapely.is_valid(planar).all()


def test_a_cross_section_height_follows_the_v_plane(plot_rdb, plot_archive_dir, plot_site):
    """Gate height in the section plane must track the stored beam thickness.

    Uses the realistically-sampled 72 x 60 x 6 archive: the small fixture has only two
    sweeps at 30 degree azimuth spacing, where height and thickness are collinear for
    reasons that have nothing to do with the beam.
    """
    from raddb.main import _decode_geometry
    from raddb.tests.conftest import PLOT_RADAR

    cs = plot_rdb.extract_cross_section(
        (plot_site[0] - 12_000, plot_site[1] - 12_000),
        (plot_site[0] + 12_000, plot_site[1] + 12_000),
    )
    pdf = _decode_geometry(cs.data.to_pandas()).head(200)
    heights = np.array([p.bounds[3] - p.bounds[1] for p in pdf["cs_polygon"]])

    v_plane = RadDB(archive_dir=str(plot_archive_dir)).get_v_plane(PLOT_RADAR, per_gate=True)
    aligned = pl.DataFrame({"gate_id": pdf["gate_id"].to_numpy()}).join(
        v_plane,
        on="gate_id",
        how="left",
        maintain_order="left",
    )
    thickness = 0.5 * (
        np.abs(aligned["z_asl_4"].to_numpy() - aligned["z_asl_1"].to_numpy())
        + np.abs(aligned["z_asl_3"].to_numpy() - aligned["z_asl_2"].to_numpy())
    )

    assert np.corrcoef(heights, thickness)[0, 1] > 0.9


def test_a_crop_matches_a_plain_membership_reference(rdb, archive_dir):
    """The semi-join must select exactly what an ``isin`` filter would."""
    geo = rdb.to_pandas(with_geometry=True)
    cx, cy = float(geo["x_3067"].median()), float(geo["y_3067"].median())
    bounds = (cx - 5000, cy - 5000, cx + 5000, cy + 5000)

    crop = rdb.crop_by_bbox(bounds=bounds)

    centroids = _lut_centroids(archive_dir, [RADAR])
    want = set(_resolve_gate_ids(centroids, shapely.box(*bounds)).tolist())
    assert set(crop.data["gate_id"].to_list()) == set(rdb.data["gate_id"].to_list()) & want

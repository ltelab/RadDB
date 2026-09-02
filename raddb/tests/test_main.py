"""Tests for :mod:`raddb.main` — the :class:`~raddb.RadDB` class.

``RadDB`` is **one class with two roles**.  *Archive-bound* (``RadDB(archive_dir=...)``)
gives ``archive()``, ``open()``, ``inventory()`` and the LUT accessors; *data-carrying* is
what ``open()`` returns, holding a polars frame in ``.data``.  ``filter``/``sel``/``crop_*``/
``extract_cross_section`` each return a **new** ``RadDB``, so calls chain and the receiver
is never mutated.

Two contracts run through everything here.

**The dynamic frame carries no geometry.** ``open()`` returns values only; the static LUT
stays in its own table and is joined on ``gate_id`` by the converters.  A crop
**selects** rows — it must never widen the frame with LUT columns.

**A CRS is mandatory to write and never needed to read.** There is no default, because a
wrong projection is silently wrong: EPSG:3067 outside Finland mis-measures distance
by ~36% and the resulting crops look entirely normal.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
import shapely
import xarray as xr

from raddb.main import RadDB, _format_elapsed_time, _format_size, _iter_days, _normalize_time_period
from raddb.tests.conftest import (
    FI_SITE,
    FMI_EPSG,
    RADAR,
    RADAR_B,
    US_EPSG,
    US_SITE,
    build_datatree,
    relocate,
)

VOL_TIMES = [pd.Timestamp("2024-08-01 12:00:00"), pd.Timestamp("2024-08-02 06:30:00")]

LUT_ONLY_COLUMNS = {
    "latitude",
    "longitude",
    "altitude",
    "x",
    "y",
    "z",
    "x_3067",
    "y_3067",
    "azimuth",
    "range",
    "elevation_angle",
    # Columns that belong to the static LUT and must never appear in a dynamic frame.
}


@pytest.fixture
def two_volume_rdb(tmp_path):
    """A data-carrying RadDB over a two-volume, one-radar archive."""
    db = RadDB(archive_dir=str(tmp_path / "arch"), crs=FMI_EPSG)
    db.archive(datatree={str(t): build_datatree(vol_time=t) for t in VOL_TIMES}, radar=RADAR)
    return db.open(radars=RADAR)


@pytest.fixture
def datatree_dir(tmp_path):
    """A directory of saved DataTree files, not archived."""
    pytest.importorskip("netCDF4")
    directory = tmp_path / "datatrees"
    directory.mkdir()
    for when in VOL_TIMES:
        build_datatree(vol_time=when).to_netcdf(directory / f"{RADAR}_{when:%Y%m%d_%H%M%S}.nc")
    return directory


def _site_xy(db):
    """The radar site in the archive's own projected meters."""
    from raddb.aoi import _reproject_to_aoi

    info = db.get_radar_info(RADAR)
    point = _reproject_to_aoi(shapely.Point(info["longitude"], info["latitude"]), 4326, FMI_EPSG)
    return (point.x, point.y)


# ---------------------------------------------------------------------------
# Construction and the dual role
# ---------------------------------------------------------------------------


def test_RadDB():
    """An archive-bound instance carries no data until ``open()`` is called."""
    db = RadDB(archive_dir="/some/where", crs=FMI_EPSG)

    assert db.archive_dir is not None
    with pytest.raises(ValueError):
        _ = db.data


def test_RadDB_init(archive_dir):
    """The two constructor arguments are remembered; ``network`` is metadata only."""
    db = RadDB(archive_dir=str(archive_dir), crs=FMI_EPSG, network="FMI")

    assert str(db.archive_dir) == str(archive_dir)
    assert db.crs().to_epsg() == FMI_EPSG


def test_a_bare_instance_needs_an_archive_dir_to_read():
    """Every archive-bound method says which argument is missing."""
    with pytest.raises(ValueError):
        RadDB().list_radars()


def test_RadDB_data(rdb):
    """``.data`` is the polars frame the object carries."""
    assert isinstance(rdb.data, pl.DataFrame)
    assert "gate_id" in rdb.data.columns


def test_RadDB_len(rdb):
    """Length is the gate count."""
    assert len(rdb) == rdb.data.height > 0


def test_RadDB_repr(rdb, db):
    """The repr distinguishes the two roles at a glance."""
    assert "gates" in repr(rdb)
    assert repr(db)


def test_the_dynamic_frame_carries_no_lut_columns(rdb):
    """``open()`` returns dynamic values only — geometry stays in its own table."""
    leaked = LUT_ONLY_COLUMNS & set(rdb.columns())

    assert not leaked, f"LUT columns leaked into the dynamic frame: {leaked}"
    assert {"latitude", "longitude"} <= set(rdb.to_pandas(with_geometry=True).columns)


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------


def test_RadDB_archive(tmp_path, datatree):
    """One in-memory volume becomes a LUT plus one POL parquet."""
    out = tmp_path / "arch"

    result = RadDB(archive_dir=str(out), crs=FMI_EPSG).archive(datatree=datatree, radar=RADAR)

    assert (result["n_archived"], result["n_failed"]) == (1, 0)
    assert (out / RADAR / "LUT" / f"{RADAR}_LUT.parquet").exists()
    assert len(list((out / RADAR).rglob("*_POL.parquet"))) == 1


def test_archive_from_a_directory_of_datatrees(tmp_path, datatree_dir):
    """``datatree_dir=`` walks the directory and archives every volume it finds."""
    out = tmp_path / "arch"
    db = RadDB(archive_dir=str(out), crs=FMI_EPSG)

    result = db.archive(datatree_dir=datatree_dir, radar=RADAR)

    assert (result["n_archived"], result["n_failed"]) == (2, 0)
    assert (db.open(radars=RADAR).data["DBZH"] > 0.0).all()


def test_archive_resumes_from_its_checkpoint(tmp_path, datatree_dir):
    """A second run re-attempts nothing, which is what makes a 530-volume run resumable."""
    db = RadDB(archive_dir=str(tmp_path / "arch"), crs=FMI_EPSG)

    assert db.archive(datatree_dir=datatree_dir, radar=RADAR)["n_archived"] == 2
    assert db.archive(datatree_dir=datatree_dir, radar=RADAR)["n_archived"] == 0


def test_archive_honors_a_time_period(tmp_path, datatree_dir):
    """Only volumes whose filename timestamp falls in the window are read."""
    result = RadDB(archive_dir=str(tmp_path / "arch"), crs=FMI_EPSG).archive(
        datatree_dir=datatree_dir,
        radar=RADAR,
        time_period=("2024-08-01 00:00", "2024-08-01 23:59"),
    )

    assert result["n_archived"] == 1


def test_archive_infers_the_radar_from_the_filename(tmp_path, datatree_dir):
    """``radar=None`` reads the name off each file, so a mixed directory works."""
    db = RadDB(archive_dir=str(tmp_path / "arch"), crs=FMI_EPSG)

    db.archive(datatree_dir=datatree_dir)

    assert db.list_radars() == [RADAR]


def test_archive_keeps_a_four_letter_radar_name(tmp_path, datatree):
    """A NEXRAD-style name survives whole; it used to collapse to its last letter."""
    pytest.importorskip("netCDF4")
    src = tmp_path / "input"
    src.mkdir()
    datatree.to_netcdf(src / "KTLX_20240101_120000.nc")
    db = RadDB(archive_dir=str(tmp_path / "arch"), crs=FMI_EPSG)

    result = db.archive(datatree_dir=src)

    assert (result["n_archived"], result["n_failed"]) == (1, 0)
    assert db.list_radars() == ["KTLX"]


def test_archive_skips_an_unusable_radar_name(tmp_path, datatree):
    """``OVERLONG`` is not filed under ``N`` — two sites would overwrite each other."""
    pytest.importorskip("netCDF4")
    src = tmp_path / "input"
    src.mkdir()
    datatree.to_netcdf(src / "OVERLONG_20240101_120000.nc")
    out = tmp_path / "arch"

    result = RadDB(archive_dir=str(out), crs=FMI_EPSG).archive(datatree_dir=src)

    assert result["n_archived"] == 0
    assert not (out / "N").exists()


def test_archive_needs_exactly_one_source(tmp_path):
    """``datatree_dir`` and ``datatree`` are alternatives, not a pair."""
    with pytest.raises(ValueError, match="exactly one"):
        RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG).archive(datatree_dir=tmp_path, datatree=object())


def test_archive_requires_a_crs(tmp_path, datatree):
    """A wrong projection is silently wrong, so there is no default."""
    with pytest.raises(ValueError, match="requires a CRS"):
        RadDB(archive_dir=str(tmp_path)).archive(datatree={RADAR: [datatree]})


def test_a_rejected_crs_aborts_and_writes_nothing(tmp_path, make_datatree):
    """It must not leave POL files behind with no usable LUT."""
    us_volume = relocate(make_datatree(), *US_SITE)

    with pytest.raises(ValueError, match="distorts distance"):
        RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG).archive(datatree={RADAR: [us_volume]})

    assert not list(tmp_path.rglob("*POL.parquet"))


def test_a_rejected_crs_aborts_a_directory_archive_too(tmp_path, make_datatree):
    """The file path used to swallow the refusal and write POL files anyway.

    ``_ensure_lut`` re-raises a rejected CRS on purpose, but the ``datatree_dir=``
    branch caught it under a bare ``except Exception`` and carried on: every volume
    was written against a LUT that had never been generated, and the run still
    reported them as archived.
    """
    pytest.importorskip("netCDF4")
    directory = tmp_path / "datatrees"
    directory.mkdir()
    for when in VOL_TIMES:
        relocate(make_datatree(vol_time=when), *US_SITE).to_netcdf(
            directory / f"{RADAR}_{when:%Y%m%d_%H%M%S}.nc",
        )
    out = tmp_path / "arch"

    with pytest.raises(ValueError, match="distorts distance"):
        RadDB(archive_dir=str(out), crs=FMI_EPSG).archive(datatree_dir=directory, radar=RADAR)

    assert not list(out.rglob("*POL.parquet"))


def test_a_valid_crs_archives_a_us_radar(tmp_path, make_datatree):
    """UTM 14N at KTLX, recorded in the info YAML for every later read."""
    us_volume = relocate(make_datatree(), *US_SITE)

    RadDB(archive_dir=str(tmp_path), crs=US_EPSG).archive(datatree={RADAR: [us_volume]})

    assert RadDB(archive_dir=str(tmp_path)).get_radar_info(RADAR)["crs"]["epsg"] == US_EPSG


def test_an_empty_volume_is_skipped_not_failed(tmp_path, make_datatree):
    """A clear-air volume counts separately from a failure."""
    blank = make_datatree()
    for name in blank.children:
        ds = blank[name].to_dataset()
        ds["DBZH"] = ds["DBZH"].where(False)
        blank[name].dataset = ds

    result = RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG).archive(datatree=blank, radar=RADAR)

    assert (result["n_archived"], result["n_failed"], result["n_skipped"]) == (0, 0, 1)


def test_the_three_counts_sum_to_the_volumes_attempted(tmp_path, make_datatree):
    """One good volume plus one empty: 1 archived, 0 failed, 1 skipped."""
    good = make_datatree(vol_time=VOL_TIMES[0])
    empty = make_datatree(vol_time=VOL_TIMES[1])
    for name in empty.children:
        ds = empty[name].to_dataset()
        ds["DBZH"] = ds["DBZH"].where(False)
        empty[name].dataset = ds

    result = RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG).archive(datatree=[good, empty], radar=RADAR)

    assert (result["n_archived"], result["n_failed"], result["n_skipped"]) == (1, 0, 1)
    assert sum((result["n_archived"], result["n_failed"], result["n_skipped"])) == 2


def test_archive_accepts_a_radar_keyed_dict(tmp_path, make_datatree):
    """The multi-radar form, ``{radar: [volumes]}``."""
    result = RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG).archive(
        datatree={RADAR: [make_datatree()], RADAR_B: [make_datatree()]},
    )

    assert sorted(result["radars"]) == sorted([RADAR, RADAR_B])


# ---------------------------------------------------------------------------
# LUT accessors
# ---------------------------------------------------------------------------


def test_RadDB_get_lut(db):
    """The centroid table for one radar, as polars."""
    lut = db.get_lut(RADAR)

    assert isinstance(lut, pl.DataFrame)
    assert lut.height == 12 * 24 * 2


def test_RadDB_get_radar_info(db):
    """The info YAML as a dict, including the CRS block."""
    info = db.get_radar_info(RADAR)

    assert info["radar"] == RADAR
    assert info["crs"]["epsg"] == FMI_EPSG


def test_RadDB_add_lut_projection(db):
    """Adds an ``x_<epsg>``/``y_<epsg>`` pair the archive does not already store."""
    projected = db.add_lut_projection(RADAR, epsg=32635)

    assert {"x_32635", "y_32635"} <= set(projected.columns)


def test_RadDB_get_h_plane(db):
    """The compact node lattice: one ``(n_az+1) x (n_rng+1)`` grid per sweep."""
    nodes = db.get_h_plane(RADAR)

    assert nodes.height == 2 * 13 * 25
    assert "el_level" not in nodes.columns


def test_get_h_plane_expands_to_four_corners_per_gate(db):
    """``per_gate=True`` is what the PPI draws."""
    table = db.get_h_plane(RADAR, per_gate=True)

    assert table.height == 12 * 24 * 2
    for k in range(1, 5):
        assert {f"x_{k}", f"y_{k}"} <= set(table.columns)
    assert "x_5" not in table.columns


def test_get_h_plane_filters_by_sweep(db):
    """One sweep costs one sweep's worth of rows."""
    nodes = db.get_h_plane(RADAR, sweep=1)

    assert nodes["sweep"].unique().to_list() == [1]
    assert nodes.height == 13 * 25


def test_RadDB_get_v_plane(db):
    """Ground distance and both altitude references, per node."""
    nodes = db.get_v_plane(RADAR)
    site_altitude = db.get_radar_info(RADAR)["altitude"]

    assert np.allclose(nodes["z_asl"].to_numpy() - nodes["z_rel"].to_numpy(), site_altitude, atol=1e-3)


def test_get_v_plane_expands_to_rhi_quads(db):
    """``(d, z_asl, z_rel)`` quads — one RHI ray's worth of geometry."""
    table = db.get_v_plane(RADAR, per_gate=True)

    assert table.height == 12 * 24 * 2
    for k in range(1, 5):
        assert {f"d_{k}", f"z_asl_{k}", f"z_rel_{k}"} <= set(table.columns)


def test_get_v_plane_azimuth_selects_one_ray(db):
    """A single azimuth across every sweep, which is what an RHI needs."""
    table = db.get_v_plane(RADAR, azimuth=0.0, per_gate=True)

    assert table.height == 24 * 2


def test_get_v_plane_azimuth_requires_per_gate(db):
    """A node lattice has no azimuth to select on; the gates do."""
    with pytest.raises(ValueError):
        db.get_v_plane(RADAR, azimuth=90.0)


def test_RadDB_get_corners(db):
    """Eight corners per gate, from the two-level 3-D lattice."""
    table = db.get_corners(RADAR, per_gate=True)

    assert table.height == 12 * 24 * 2
    for k in range(1, 9):
        assert {f"x_{k}", f"y_{k}", f"z_rel_{k}"} <= set(table.columns)
    assert "x_9" not in table.columns


def test_get_corners_returns_the_lattice_by_default(db):
    """Both elevation levels, not yet expanded."""
    nodes = db.get_corners(RADAR)

    assert sorted(nodes["el_level"].unique().to_list()) == [-1, 1]


def test_RadDB_export_h_plane_geoparquet(db, tmp_path):
    """Opt-in GeoParquet with an embedded CRS, for QGIS."""
    gpd = pytest.importorskip("geopandas")
    out = tmp_path / "h_plane.parquet"

    db.export_h_plane_geoparquet(RADAR, out)

    gdf = gpd.read_parquet(out)
    assert len(gdf) == 12 * 24 * 2
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == FMI_EPSG
    assert gdf.geometry.is_valid.all()
    # Corner order is clockwise in storage; GeoParquet prefers counter-clockwise.
    assert shapely.is_ccw(shapely.get_exterior_ring(gdf.geometry.values)).all()


def test_export_h_plane_geoparquet_falls_back_to_wgs84(db, tmp_path):
    """An EPSG the LUT does not carry falls back rather than writing an unlabeled file."""
    gpd = pytest.importorskip("geopandas")
    out = tmp_path / "h_plane_4326.parquet"

    db.export_h_plane_geoparquet(RADAR, out, epsg=9999)

    assert gpd.read_parquet(out).crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# list_radars / inventory
# ---------------------------------------------------------------------------


def test_RadDB_list_radars(archive_dir_two_radars):
    """The radars an archive holds, sorted."""
    assert RadDB(archive_dir=str(archive_dir_two_radars)).list_radars() == sorted([RADAR, RADAR_B])


def test_list_radars_on_an_empty_archive(tmp_path):
    """Nothing archived is an empty list, not an error."""
    assert RadDB(archive_dir=str(tmp_path)).list_radars() == []


def test_RadDB_inventory(two_volume_rdb, capsys):
    """Prints an overview and returns nothing."""
    db = RadDB(archive_dir=str(two_volume_rdb.archive_dir))

    assert db.inventory() is None

    out = capsys.readouterr().out
    assert "archived data" in out
    assert f"\n  {RADAR} " in out
    assert "2024-08-01 12:00:00 .. 2024-08-02 06:30:00" in out


def test_inventory_detailed_adds_lut_columns_and_days(two_volume_rdb, capsys):
    """``detailed=True`` shows the LUT shape and the per-volume moment columns."""
    RadDB(archive_dir=str(two_volume_rdb.archive_dir)).inventory(detailed=True)

    out = capsys.readouterr().out
    assert "LUT:" in out
    assert "sweeps" in out
    assert "DBZH" in out
    assert "2024-08-01" in out
    assert "2024-08-02" in out


def test_inventory_on_an_empty_archive(tmp_path, capsys):
    """Says so rather than printing an empty table."""
    RadDB(archive_dir=str(tmp_path)).inventory()

    assert "nothing archived here yet" in capsys.readouterr().out


def test_inventory_without_an_archive_dir_raises():
    """There is nothing to inventory."""
    with pytest.raises(ValueError):
        RadDB().inventory()


def test_inventory_of_a_datatree_directory(datatree_dir, capsys):
    """The input side: files on disk that have not been archived yet."""
    RadDB().inventory(datatree_dir=str(datatree_dir))

    out = capsys.readouterr().out
    assert "not archived yet" in out
    assert "files     : 2" in out
    assert "2024-08-01 12:00:00 .. 2024-08-02 06:30:00" in out


def test_inventory_does_not_warn_about_a_four_letter_radar(tmp_path, datatree, capsys):
    """A NEXRAD-style name is archivable, so there is nothing to warn about."""
    pytest.importorskip("netCDF4")
    directory = tmp_path / "nexrad"
    directory.mkdir()
    datatree.to_netcdf(directory / "KTLX_20240801_120000.nc")

    RadDB().inventory(datatree_dir=str(directory), detailed=True)

    out = capsys.readouterr().out
    assert "KTLX" in out
    assert "not a usable radar name" not in out


def test_inventory_warns_about_an_unusable_radar_name(tmp_path, datatree, capsys):
    """Better to say so up front than to archive nothing and report success."""
    pytest.importorskip("netCDF4")
    directory = tmp_path / "odd"
    directory.mkdir()
    datatree.to_netcdf(directory / "OVERLONG_20240801_120000.nc")

    RadDB().inventory(datatree_dir=str(directory), detailed=True)

    out = capsys.readouterr().out
    assert "OVERLONG" in out
    assert "not a usable radar name" in out


def test_inventory_of_a_missing_directory_raises(tmp_path):
    """A path typo must not read as "no data"."""
    with pytest.raises(FileNotFoundError):
        RadDB().inventory(datatree_dir=str(tmp_path / "nope"))


def test_inventory_of_an_empty_directory(tmp_path, capsys):
    """No volumes is reported, not raised."""
    RadDB().inventory(datatree_dir=str(tmp_path))

    assert "no .zarr / .nc" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# open
# ---------------------------------------------------------------------------


def test_RadDB_open(db):
    """Returns a data-carrying RadDB over the whole archive."""
    rdf = db.open(radars=RADAR)

    assert isinstance(rdf, RadDB)
    assert len(rdf) > 0
    assert rdf.radars() == [RADAR]


def test_open_needs_no_crs(archive_dir):
    """Reading never restates the CRS; the archive records the one it was written with."""
    plain = RadDB(archive_dir=str(archive_dir)).open(radars=RADAR)

    assert plain._crs is None
    assert plain.crs().to_epsg() == FMI_EPSG


def test_open_narrows_by_time_period(two_volume_rdb):
    """A window selects whole volumes."""
    db = RadDB(archive_dir=str(two_volume_rdb.archive_dir))

    first = db.open(radars=RADAR, time_period=("2024-08-01", "2024-08-01 23:59"))

    assert 0 < len(first) < len(two_volume_rdb)


def test_open_selects_columns(db):
    """``columns=`` is pushed into the parquet reader."""
    rdf = db.open(radars=RADAR, columns=["gate_id", "DBZH"])

    assert "ZDR" not in rdf.columns()


def test_open_applies_filters(db):
    """``filters=`` runs at read time, so filtered rows are never materialized."""
    rdf = db.open(radars=RADAR, filters={"var": "DBZH", "logic": ">", "threshold": 20})

    assert rdf.data["DBZH"].min() > 20


def test_open_spans_several_radars(archive_dir_two_radars):
    """No ``radars=`` means every radar in the archive."""
    rdf = RadDB(archive_dir=str(archive_dir_two_radars)).open()

    assert sorted(rdf.radars()) == sorted([RADAR, RADAR_B])


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_RadDB_filter(rdb):
    """Rows failing the comparison are dropped; a new RadDB comes back."""
    out = rdb.filter({"var": "DBZH", "logic": ">", "threshold": 10})

    assert out is not rdb
    assert 0 < len(out) < len(rdb)
    assert out.data["DBZH"].min() > 10


def test_filter_ands_a_list_of_dicts(rdb):
    """A list is ANDed, which is how a band is expressed."""
    band = rdb.filter([{"var": "DBZH", "logic": ">", "threshold": 10}, {"var": "DBZH", "logic": "<", "threshold": 20}])

    values = band.data["DBZH"].to_numpy()
    assert ((values > 10) & (values < 20)).all()


def test_filter_rejects_a_misspelled_key(rdb):
    """``value`` is not a filter key; ``threshold`` would default to 0 and keep every row."""
    with pytest.raises(KeyError, match="unknown filter key"):
        rdb.filter({"var": "DBZH", "logic": ">", "value": 10})


def test_filter_rejects_an_unknown_column(rdb):
    """Neither a dynamic column nor a LUT one."""
    with pytest.raises(KeyError, match="cannot filter on"):
        rdb.filter({"var": "nope", "logic": ">", "threshold": 0})


def test_filter_can_use_a_lut_column_without_leaking_it(rdb):
    """Vertical subsetting borrows ``altitude`` from the LUT, then drops it again."""
    assert "altitude" not in rdb.columns(), "altitude is LUT data, not dynamic"
    cut = float(rdb.to_pandas(with_geometry=True)["altitude"].median())

    band = rdb.filter({"var": "altitude", "logic": ">", "threshold": cut})

    assert 0 < len(band) < len(rdb)
    assert band.columns() == rdb.columns(), "the borrowed LUT column leaked"
    assert band.to_pandas(with_geometry=True)["altitude"].min() > cut


# ---------------------------------------------------------------------------
# sel
# ---------------------------------------------------------------------------


def test_RadDB_sel(rdb):
    """xarray-style label selection; slice bounds are inclusive on both ends."""
    out = rdb.sel(DBZH=slice(5, 15))
    values = out.data["DBZH"].to_numpy()

    assert len(out) > 0
    assert values.min() >= 5.0
    assert values.max() <= 15.0


def test_open_ended_slices_match_filter(rdb):
    """``slice(10, None)`` is ``>= 10``, and ``slice(None, 10)`` is ``<= 10``."""
    assert len(rdb.sel(DBZH=slice(10, None))) == len(rdb.filter({"var": "DBZH", "logic": ">=", "threshold": 10}))
    assert len(rdb.sel(DBZH=slice(None, 10))) == len(rdb.filter({"var": "DBZH", "logic": "<=", "threshold": 10}))


def test_sel_with_no_arguments_is_a_no_op(rdb):
    """Nothing selected, nothing dropped."""
    assert len(rdb.sel()) == len(rdb)


def test_sel_keywords_are_anded(rdb):
    """Two indexers in one call equal two chained calls."""
    assert len(rdb.sel(DBZH=slice(10, None), ZDR=slice(None, 5))) == len(
        rdb.sel(DBZH=slice(10, None)).sel(ZDR=slice(None, 5)),
    )


def test_sel_on_a_partial_time_string(two_volume_rdb):
    """``"2024-08-01"`` selects the whole day, as xarray does."""
    assert 0 < len(two_volume_rdb.sel(time="2024-08-01")) < len(two_volume_rdb)
    assert len(two_volume_rdb.sel(time="2024-08")) == len(two_volume_rdb)
    assert len(two_volume_rdb.sel(time="1999-01")) == 0


def test_sel_on_a_lut_column_does_not_leak_it(rdb):
    """The borrowed static column is evaluated, then dropped again."""
    out = rdb.sel(range=slice(2_000, 10_000))

    assert 0 < len(out) < len(rdb)
    assert out.columns() == rdb.columns()
    assert "range" not in out.columns()


def test_sel_on_range_matches_the_lut(rdb, db):
    """The selection is exactly what the LUT says, intersected with what is held."""
    lut = db.get_lut(RADAR)
    want = set(lut.filter((pl.col("range") >= 2_000) & (pl.col("range") <= 10_000))["gate_id"].to_list())

    got = set(rdb.sel(range=slice(2_000, 10_000)).data["gate_id"].to_list())

    assert got == set(rdb.data["gate_id"].to_list()) & want


def test_sel_accepts_lat_lon_aliases(rdb):
    """``lon``/``lat`` name the geographic columns; the full extent keeps everything."""
    extent = rdb.geographic_extent()

    out = rdb.sel(lon=slice(extent[0], extent[1]), lat=slice(extent[2], extent[3]))

    assert len(out) == len(rdb)
    assert out.columns() == rdb.columns()


def test_sel_mixes_static_and_dynamic_columns(rdb):
    """One call may span both tables."""
    out = rdb.sel(DBZH=slice(10, None), range=slice(2_000, 10_000), sweep=1)

    assert out.columns() == rdb.columns()
    assert len(out) <= len(rdb)


def test_sel_on_radars(archive_dir_two_radars):
    """``radars=`` narrows a multi-radar frame."""
    both = RadDB(archive_dir=str(archive_dir_two_radars)).open()

    only_a = both.sel(radars=[RADAR])

    assert only_a.radars() == [RADAR]
    assert 0 < len(only_a) < len(both)


def test_sel_keeps_the_lut_synchronized(rdb):
    """The geometry must shrink with the data, gate for gate."""
    out = rdb.sel(range=slice(2_000, 10_000), DBZH=slice(10, None))

    geometry = out._gate_geometry()
    assert len(geometry) < len(rdb._gate_geometry())
    assert set(geometry["gate_id"].to_list()) == set(out.data["gate_id"].to_list())


def test_sel_never_mutates_the_receiver(rdb):
    """Every selection returns a new object; chaining must be safe."""
    before_len, before_columns = len(rdb), rdb.columns()

    out = rdb.sel(DBZH=slice(0, 1), range=slice(2_000, 3_000), sweep=1)

    assert (len(rdb), rdb.columns()) == (before_len, before_columns)
    assert out is not rdb
    assert out.crs() == rdb.crs()
    assert str(out.archive_dir) == str(rdb.archive_dir)


def test_sel_rejects_an_unknown_column(rdb):
    """Neither dynamic nor static."""
    with pytest.raises(KeyError):
        rdb.sel(NOT_A_COLUMN=1)


def test_sel_rejects_a_slice_step(rdb):
    """A step has no meaning on unordered label selection."""
    with pytest.raises(ValueError):
        rdb.sel(DBZH=slice(0, 10, 2))


# ---------------------------------------------------------------------------
# add_feature
# ---------------------------------------------------------------------------


def test_RadDB_add_feature(rdb):
    """A derived column is appended and a new RadDB comes back."""
    out = rdb.add_feature("Z_lin", lambda df: 10 ** (df["DBZH"].to_numpy() / 10.0))

    assert "Z_lin" in out.columns()
    assert "Z_lin" not in rdb.columns(), "add_feature must not mutate the receiver"
    assert len(out) == len(rdb)


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def test_RadDB_to_pandas(rdb):
    """The intentional user-facing converter; geometry is opt-in."""
    plain = rdb.to_pandas()

    assert isinstance(plain, pd.DataFrame)
    assert len(plain) == len(rdb)
    assert "latitude" not in plain.columns


def test_to_pandas_with_geometry_joins_the_lut(rdb):
    """Cartesian position and sweep come from the LUT on request."""
    with_geometry = rdb.to_pandas(with_geometry=True)

    assert {"latitude", "longitude", "altitude", "sweep", "x_3067", "y_3067"} <= set(with_geometry.columns)
    assert len(with_geometry) == len(rdb)


@pytest.mark.parametrize("column", ["range", "azimuth", "elevation_angle"])
def test_polar_coordinates_are_opt_in(rdb, column):
    """They duplicate what the Cartesian columns already say, unless you work in polar."""
    assert column not in rdb.to_pandas(with_geometry=True).columns
    assert column in rdb.to_pandas(with_polar_coords=True).columns


def test_with_polar_coords_implies_with_geometry(rdb):
    """Both come from the same LUT join."""
    assert "latitude" in rdb.to_pandas(with_polar_coords=True).columns


def test_polar_coordinate_values_match_the_lut(rdb, db):
    """They are read off, not recomputed."""
    pdf = rdb.to_pandas(with_polar_coords=True)
    reference = pl.DataFrame({"gate_id": pdf["gate_id"].to_numpy()}).join(
        db.get_lut(RADAR).select(["gate_id", "range", "azimuth", "elevation_angle"]),
        on="gate_id",
        how="left",
        maintain_order="left",
    )

    for column in ("range", "azimuth", "elevation_angle"):
        assert np.allclose(pdf[column].to_numpy(), reference[column].to_numpy())


def test_RadDB_to_geopandas(rdb):
    """A GeoDataFrame of gate centroids, in the archive's own CRS."""
    gpd = pytest.importorskip("geopandas")

    gdf = rdb.to_geopandas()

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == len(rdb)
    assert gdf.geometry.notna().all()


def test_to_geopandas_takes_polar_coordinates_too(rdb):
    """The same opt-in as ``to_pandas``."""
    pytest.importorskip("geopandas")

    assert "azimuth" in rdb.to_geopandas(with_polar_coords=True).columns


def test_RadDB_to_geoarrow(rdb):
    """Point geometry, tagged so lonboard reads it as geometry."""
    pytest.importorskip("pyarrow")

    table = rdb.to_geoarrow(geometry="point")

    metadata = table.schema.field("geometry").metadata
    assert metadata[b"ARROW:extension:name"] == b"geoarrow.point"
    assert b"EPSG:4326" in metadata[b"ARROW:extension:metadata"]


def test_to_geoarrow_polygons_are_closed_wedges(rdb):
    """One ring per gate: four corners plus the closing vertex."""
    pytest.importorskip("pyarrow")

    table = rdb.to_geoarrow(geometry="polygon")

    assert table.schema.field("geometry").metadata[b"ARROW:extension:name"] == b"geoarrow.polygon"
    rings = table.column("geometry").to_pylist()
    assert rings
    assert all(r is not None for r in rings), "every gate should be placed"
    for ring in rings:
        assert len(ring) == 1, "a gate is a single ring"
        assert len(ring[0]) == 5, "4 corners + closing vertex"
        assert ring[0][0] == ring[0][-1], "ring must be closed"


def test_to_geoarrow_wedges_surround_their_own_centroid(rdb, db):
    """The polygon must contain the LUT centroid it was built from."""
    pytest.importorskip("pyarrow")

    table = rdb.to_geoarrow(geometry="polygon")
    polygons = shapely.polygons(np.array([r[0] for r in table.column("geometry").to_pylist()]))
    assert shapely.is_valid(polygons).all()

    reference = pl.DataFrame({"gate_id": table.column("gate_id").to_numpy()}).join(
        db.get_lut(RADAR).select("gate_id", "latitude", "longitude"),
        on="gate_id",
        how="left",
    )
    centers = shapely.centroid(polygons)
    dx = (shapely.get_x(centers) - reference["longitude"].to_numpy()) * 111_320 * np.cos(np.radians(FI_SITE[1]))
    dy = (shapely.get_y(centers) - reference["latitude"].to_numpy()) * 111_320

    assert np.hypot(dx, dy).max() < 20_000 / 24  # one gate length


def test_to_geoarrow_has_a_row_guardrail(rdb):
    """A full archive would blow up a browser; the limit is explicit and overridable."""
    pytest.importorskip("pyarrow")

    with pytest.raises(ValueError, match="max_rows"):
        rdb.to_geoarrow(max_rows=1)
    assert rdb.to_geoarrow(max_rows=None).num_rows == len(rdb)


def test_to_geoarrow_polygons_need_a_single_radar(rdb):
    """Corner arrays are per radar, so a mixed frame is refused."""
    pytest.importorskip("pyarrow")
    mixed = rdb._derive(pl.concat([rdb.data, rdb.data.with_columns(pl.lit("B").alias("radar"))]))

    with pytest.raises(ValueError, match="single radar"):
        mixed.to_geoarrow(geometry="polygon", max_rows=None)


def test_RadDB_to_datatree(rdb):
    """Back to xarray, reindexed onto the full azimuth x range grid."""
    dt = rdb.to_datatree()

    from raddb.helper import list_sweep_names

    assert isinstance(dt, xr.DataTree)
    assert list_sweep_names(dt) == ["sweep_1", "sweep_2"]


def test_to_datatree_needs_the_radar_named_when_several_are_held(archive_dir_two_radars):
    """One tree describes one radar."""
    both = RadDB(archive_dir=str(archive_dir_two_radars)).open()

    assert both.to_datatree(radar=RADAR) is not None


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------


def test_RadDB_head(rdb):
    """A polars frame, not a RadDB — this is for looking, not chaining."""
    assert isinstance(rdb.head(3), pl.DataFrame)
    assert rdb.head(3).height == 3


def test_RadDB_tail(rdb):
    """The other end of the same frame."""
    assert rdb.tail(3).height == 3
    # Compared on gate_id: a row tuple holds NaN moments, and NaN != NaN.
    assert rdb.tail(3)["gate_id"].to_list() == rdb.data["gate_id"].to_list()[-3:]


def test_RadDB_columns(rdb):
    """The dynamic columns the frame holds."""
    assert "DBZH" in rdb.columns()
    assert rdb.columns() == rdb.data.columns


def test_RadDB_radars(rdb):
    """Decoded from the ``gate_id`` values, so it needs no registry."""
    assert rdb.radars() == [RADAR]


def test_RadDB_start_time(two_volume_rdb):
    """The earliest volume time held."""
    assert isinstance(two_volume_rdb.start_time(), datetime.datetime)
    assert two_volume_rdb.start_time().replace(tzinfo=None) == VOL_TIMES[0]


def test_RadDB_end_time(two_volume_rdb):
    """The latest volume time held."""
    assert two_volume_rdb.end_time().replace(tzinfo=None) == VOL_TIMES[1]


def test_RadDB_extent(rdb):
    """``[xmin, xmax, ymin, ymax]`` in the archive's projected meters."""
    extent = rdb.extent()

    assert len(extent) == 4
    assert extent[0] < extent[1]
    assert extent[2] < extent[3]
    assert 4.7e5 < extent[0] < 5.0e5


def test_RadDB_geographic_extent(rdb):
    """The same box in degrees, around the synthetic site."""
    extent = rdb.geographic_extent()

    assert extent[0] < FI_SITE[0] < extent[1]
    assert extent[2] < FI_SITE[1] < extent[3]


def test_RadDB_crs(rdb, archive_dir):
    """Declared or resolved from the archive — either way it answers."""
    assert rdb.crs().to_epsg() == FMI_EPSG
    assert RadDB(archive_dir=str(archive_dir)).open(radars=RADAR).crs().to_epsg() == FMI_EPSG


def test_RadDB_geographic_crs(rdb):
    """The lon/lat frame the converters emit."""
    assert rdb.geographic_crs().to_epsg() == 4326


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------


def test_RadDB_crop_by_bbox(rdb, db):
    """A bounding box in the archive's own meters selects a strict subset."""
    cx, cy = _site_xy(db)

    crop = rdb.crop_by_bbox(bounds=(cx - 5000, cy - 5000, cx + 5000, cy + 5000))

    assert 0 < len(crop) < len(rdb)
    assert crop.columns() == rdb.columns(), f"crop widened the frame: {set(crop.columns()) - set(rdb.columns())}"


def test_crop_by_bbox_accepts_an_extent(rdb, db):
    """``extent=[xmin, xmax, ymin, ymax]`` is the matplotlib ordering."""
    cx, cy = _site_xy(db)

    by_bounds = rdb.crop_by_bbox(bounds=(cx - 5000, cy - 5000, cx + 5000, cy + 5000))
    by_extent = rdb.crop_by_bbox(extent=[cx - 5000, cx + 5000, cy - 5000, cy + 5000])

    assert len(by_bounds) == len(by_extent)


def test_a_crop_matching_nothing_still_converts(rdb):
    """Regression: ``pl.concat([])`` on an empty selection."""
    far = rdb.crop_by_bbox(bounds=(9e6, 9e6, 9e6 + 1000, 9e6 + 1000))

    assert len(far) == 0
    assert far.to_pandas(with_geometry=True).empty


def test_RadDB_crop_by_polygon(rdb, db):
    """An arbitrary shapely polygon, in the archive's frame."""
    cx, cy = _site_xy(db)

    crop = rdb.crop_by_polygon(shapely.Point(cx, cy).buffer(5_000))

    assert 0 < len(crop) < len(rdb)
    assert crop.columns() == rdb.columns()


def test_crop_by_polygon_reads_a_geojson_in_its_own_crs(rdb, tmp_path):
    """A GeoJSON is lon/lat; reading those degrees as meters lands thousands of km away."""
    import json

    square = {
        "type": "Polygon",
        "coordinates": [[[26.9, 61.9], [27.1, 61.9], [27.1, 62.1], [26.9, 62.1], [26.9, 61.9]]],
    }
    path = tmp_path / "aoi.geojson"
    path.write_text(json.dumps(square))

    assert len(rdb.crop_by_polygon(str(path))) > 0


def test_RadDB_crop_around_point(rdb, db):
    """A radius in true meters around a point."""
    crop = rdb.crop_around_point(_site_xy(db), distance=5_000)

    assert 0 < len(crop) < len(rdb)
    assert crop.columns() == rdb.columns()


def test_crop_around_point_accepts_lonlat(rdb):
    """``crs=4326`` says the point is in degrees."""
    assert len(rdb.crop_around_point(FI_SITE, distance=5_000, crs=4326)) > 0


def test_an_aoi_crs_override_is_validated(us_archive_dir):
    """An override still has to be valid where the radar actually is."""
    rdf = RadDB(archive_dir=str(us_archive_dir)).open(radars=RADAR)

    with pytest.raises(ValueError, match="distorts distance"):
        rdf.crop_around_point(US_SITE, distance=10_000, crs=4326, aoi_crs=FMI_EPSG)


def test_a_valid_aoi_crs_override_selects_gates(rdb):
    """A frame the LUT does not store is projected on the fly from latitude/longitude."""
    crop = rdb.crop_around_point(FI_SITE, distance=10_000, crs=4326, aoi_crs=32635)

    assert len(crop) > 0


def test_a_crop_radius_is_true_meters_away_from_the_home_frame(us_archive_dir):
    """The 17% bug end to end: a "10 km" crop used to reach ~8.3 km at a US radar."""
    import pyproj

    db = RadDB(archive_dir=str(us_archive_dir))
    lut = db.get_lut(RADAR)
    lon, lat = lut["longitude"].to_numpy(), lut["latitude"].to_numpy()
    _, _, ground = pyproj.Geod(ellps="WGS84").inv(
        np.full(lon.size, US_SITE[0]),
        np.full(lat.size, US_SITE[1]),
        lon,
        lat,
    )
    truth = int((ground <= 10_000).sum())

    crop = db.open(radars=RADAR).crop_around_point(US_SITE, distance=10_000, crs=4326)

    assert abs(len(crop) - truth) <= 0.02 * truth


def test_a_quicklook_is_framed_on_the_archive(us_archive_dir):
    """A view framed on the archive's home CRS used to push a US AOI off-screen entirely."""
    import matplotlib.pyplot as plt

    from raddb.aoi import _reproject_to_aoi

    rdf = RadDB(archive_dir=str(us_archive_dir)).open(radars=RADAR)
    rdf.crop_around_point(US_SITE, distance=20_000, crs=4326, quicklook=True)

    ax = plt.gcf().axes[0]
    site = _reproject_to_aoi(shapely.Point(*US_SITE), 4326, US_EPSG)
    assert ax.get_xlim()[0] <= site.x <= ax.get_xlim()[1]
    assert ax.get_ylim()[0] <= site.y <= ax.get_ylim()[1]


def test_RadDB_interactive_crop(rdb):
    """Returns the ipyleaflet selector; the crop itself happens on Apply."""
    pytest.importorskip("ipyleaflet")
    from raddb.viz.interactive import AOISelector

    assert isinstance(rdb.interactive_crop(), AOISelector)


# ---------------------------------------------------------------------------
# extract_cross_section
# ---------------------------------------------------------------------------


def test_RadDB_extract_cross_section(rdb, db):
    """A line cuts a vertical section and attaches ``cs_polygon`` per gate."""
    cx, cy = _site_xy(db)

    cs = rdb.extract_cross_section((cx - 10_000, cy - 10_000), (cx + 10_000, cy + 10_000))

    assert 0 < len(cs) <= len(rdb)
    assert "cs_polygon" in cs.data.columns


def test_a_cross_section_measures_true_ground_distance(us_archive_dir):
    """A 91 km section at KTLX in UTM 14N measures to -0.009% of the true geodesic."""
    import pyproj

    rdf = RadDB(archive_dir=str(us_archive_dir)).open(radars=RADAR)
    p1 = (US_SITE[0] - 0.2, US_SITE[1])
    p2 = (US_SITE[0] + 0.2, US_SITE[1])
    truth = pyproj.Geod(ellps="WGS84").inv(p1[0], p1[1], p2[0], p2[1])[2]

    cs = rdf.extract_cross_section(p1=p1, p2=p2, crs=4326)

    assert cs.data.height > 0, "the section selected no gates"
    # Read off the gate footprints: ``d_center`` alone sits half a gate short.
    span = float(shapely.bounds(cs.to_pandas()["cs_polygon"].to_numpy())[:, 2].max())
    assert abs(span - truth) <= 0.005 * truth, f"section spans {span:,.0f} m, true geodesic {truth:,.0f} m"


def test_a_cross_section_quicklook_is_framed_on_the_archive(us_archive_dir):
    """``quicklook=True`` used to frame a US section over the archive's home country."""
    import matplotlib.pyplot as plt

    from raddb.aoi import _reproject_to_aoi

    rdf = RadDB(archive_dir=str(us_archive_dir)).open(radars=RADAR)
    rdf.extract_cross_section(
        p1=(US_SITE[0] - 0.2, US_SITE[1]),
        p2=(US_SITE[0] + 0.2, US_SITE[1]),
        crs=4326,
        quicklook=True,
    )

    ax = plt.gcf().axes[0]
    site = _reproject_to_aoi(shapely.Point(*US_SITE), 4326, US_EPSG)
    assert ax.get_xlim()[0] <= site.x <= ax.get_xlim()[1]
    assert ax.get_ylim()[0] <= site.y <= ax.get_ylim()[1]


# ---------------------------------------------------------------------------
# The plot delegations
# ---------------------------------------------------------------------------


def test_RadDB_plot_ppi(plot_rdb):
    """Delegates to :func:`raddb.viz.plot.plot_ppi` and returns its artist."""
    from matplotlib.collections import PolyCollection

    assert isinstance(plot_rdb.plot_ppi(sweep=1), PolyCollection)


def test_RadDB_plot_rhi(plot_rdb):
    """One azimuth, stacked across every sweep."""
    assert len(plot_rdb.plot_rhi(azimuth=0).get_paths()) > 0


def test_RadDB_plot_cappi(plot_rdb):
    """A constant-altitude slice."""
    assert len(plot_rdb.plot_cappi(altitude=1200).get_paths()) > 0


def test_RadDB_plot_vcs(plot_rdb, plot_site):
    """A vertical cross-section along an arbitrary line."""
    line = ((plot_site[0] - 12_000, plot_site[1] - 12_000), (plot_site[0] + 12_000, plot_site[1] + 12_000))

    assert len(plot_rdb.plot_vcs(line=line).get_paths()) > 0


def test_RadDB_plot_cross_section(plot_rdb, plot_site):
    """The deprecated alias: it warns and delegates to ``plot_vcs``."""
    cs = plot_rdb.extract_cross_section(
        (plot_site[0] - 12_000, plot_site[1] - 12_000),
        (plot_site[0] + 12_000, plot_site[1] + 12_000),
    )

    with pytest.deprecated_call():
        assert len(cs.plot_cross_section().get_paths()) > 0


def test_a_plot_composes_through_ax(plot_rdb):
    """The delegation must forward ``ax=`` so panels still compose."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()

    assert plot_rdb.plot_ppi(sweep=1, ax=ax).axes is ax


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_iter_days():
    """The archive layout is ``{YYYY}/{MM}/{DD}``, so batching walks whole days."""
    days = list(_iter_days(pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-03")))

    assert len(days) == 3


def test_format_elapsed_time():
    """Seconds below a minute, then minutes, then hours."""
    assert _format_elapsed_time(5) == "5s"
    assert "m" in _format_elapsed_time(125)
    assert "h" in _format_elapsed_time(7300)


def test_format_size():
    """Human-readable byte counts for the inventory listing."""
    assert _format_size(512).endswith("B")
    assert "MB" in _format_size(5 * 1024**2)
    assert "GB" in _format_size(3 * 1024**3)


def test_normalize_time_period():
    """A ``(start, end)`` pair becomes tz-aware UTC timestamps."""
    start, end = _normalize_time_period(("2024-08-01", "2024-08-02"))

    assert start.tzinfo is not None
    assert end.tzinfo is not None
    assert start < end


def test_normalize_time_period_passes_none_through():
    """No period means no bounds."""
    assert _normalize_time_period(None) in (None, (None, None))


def test_a_volume_with_only_nat_times_is_skipped(tmp_path, make_datatree):
    """DBZH survives the filter but every ray's time is NaT, so no path can be built."""
    dt = make_datatree()
    for name in dt.children:
        ds = dt[name].to_dataset()
        ds["time"] = xr.full_like(ds["time"], np.datetime64("NaT"))
        dt[name].dataset = ds

    result = RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG).archive(datatree=dt, radar=RADAR)

    assert (result["n_archived"], result["n_failed"], result["n_skipped"]) == (0, 0, 1)
    assert not list((tmp_path / RADAR).rglob("*_POL.parquet"))


def test_a_skipped_volume_does_not_poison_the_archive(tmp_path, make_datatree):
    """The rest of the archive must stay readable."""
    empty = make_datatree(vol_time=VOL_TIMES[1])
    for name in empty.children:
        ds = empty[name].to_dataset()
        ds["DBZH"] = ds["DBZH"].where(False)
        empty[name].dataset = ds

    db = RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG)
    db.archive(datatree=make_datatree(vol_time=VOL_TIMES[0]), radar=RADAR)
    db.archive(datatree=empty, radar=RADAR)

    rdf = RadDB(archive_dir=str(tmp_path)).open(radars=RADAR)
    assert len(rdf) > 0
    assert rdf.radars() == [RADAR]


def test_the_disk_path_checkpoints_a_skip(tmp_path, make_datatree):
    """A skipped volume is checkpointed too, so a resume does not retry it forever."""
    pytest.importorskip("netCDF4")
    src = tmp_path / "trees"
    src.mkdir()
    make_datatree(vol_time=VOL_TIMES[0]).to_netcdf(src / f"{RADAR}_20240801_120000.nc")
    empty = make_datatree(vol_time=VOL_TIMES[1])
    for name in empty.children:
        ds = empty[name].to_dataset()
        ds["DBZH"] = ds["DBZH"].where(False)
        empty[name].dataset = ds
    empty.to_netcdf(src / f"{RADAR}_20240802_063000.nc")
    arch = tmp_path / "arch"

    first = RadDB(archive_dir=str(arch), crs=FMI_EPSG).archive(datatree_dir=str(src), radar=RADAR)
    assert (first["n_archived"], first["n_failed"], first["n_skipped"]) == (1, 0, 1)

    again = RadDB(archive_dir=str(arch), crs=FMI_EPSG).archive(datatree_dir=str(src), radar=RADAR)
    assert (again["n_archived"], again["n_failed"], again["n_skipped"]) == (0, 0, 0)


def test_the_multi_radar_path_counts_a_skip_separately(tmp_path, make_datatree):
    """The ``{radar: [volumes]}`` form keeps the three counts apart too."""
    empty = make_datatree(vol_time=VOL_TIMES[1])
    for name in empty.children:
        ds = empty[name].to_dataset()
        ds["DBZH"] = ds["DBZH"].where(False)
        empty[name].dataset = ds

    result = RadDB(archive_dir=str(tmp_path), crs=FMI_EPSG).archive(
        datatree={RADAR: [make_datatree(vol_time=VOL_TIMES[0]), empty], RADAR_B: [make_datatree()]},
    )

    assert (result["n_archived"], result["n_failed"], result["n_skipped"]) == (2, 0, 1)
    assert sorted(result["radars"]) == sorted([RADAR, RADAR_B])

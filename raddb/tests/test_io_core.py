"""Tests for :mod:`raddb.io_core` — DataTree <-> DataFrame <-> Parquet.

This is the archive write path and the read path back out.  Three contracts drive most of
what is asserted here.

**Volumes must join their LUT.** The LUT stores a radar's *scan strategy*, not one
volume's measured azimuths, and every incoming ray is snapped onto that nominal grid
before its ``gate_id`` is built.  Without it a drifting antenna silently lost 6% of gates
per volume on a 360-ray C-band network and 35% on WSR-88D — invisible except in LUT joins.

**A scan-strategy change is refused, not reconciled.** More rays than the grid, or a
missing sweep, raises; *fewer* rays is a rotation with holes and is accepted.

**Empty is not an error.** A clear-air volume archives zero gates and is *skipped*; it
used to crash the batch on ``pd.NaT.month``.

**A moment is a per-gate variable.** There is no list of known moment names: whatever a
volume measures on both the azimuth and the range dimension is archived, and the per-ray
or scalar metadata beside it (``sweep_mode``, ``prt_mode``, ...) never is.  A hardcoded
allowlist used to archive 5 of an FMI volume's 19 moments and silently drop the rest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from raddb.io_core import (
    _build_polar_dataframe,
    _col,
    _gate_variables,
    _save_polar_parquet,
    _snap_volume_azimuths,
    _to_pandas_frame,
    _to_polars_frame,
    add_feature_to_df,
    add_feature_to_dt,
    archive_multiple_volumes,
    archive_volume,
    archive_volumes_multi_radar,
    dataframe_to_datatree,
    datatree_to_dataframe,
    datatree_to_dataset,
    datatree_to_parquet,
    join_labels_with_lut,
    labels_to_dataframe,
    open_any_datatree,
    parquet_to_dataframe,
    parquet_to_datatree,
    reconstruct_datatree,
    reconstruct_sweep_dataset,
    scan_polar_parquet,
)
from raddb.lut import generate_lut_from_datatree, lut_file_path
from raddb.main import RadDB
from raddb.tests.conftest import ANTENNA_BIAS, FMI_EPSG, RADAR, RADAR_B, build_datatree, retime

# A window wide enough to hold every synthetic volume in this module.
TIME_WINDOW = ("2024-08-01 00:00", "2024-08-02 00:00")


@pytest.fixture
def lut_base(tmp_path, make_datatree):
    """A base path with only radar ``A``'s LUT written — no volumes yet."""
    generate_lut_from_datatree(make_datatree(), radar=RADAR, output_base_path=str(tmp_path), projection_epsg=FMI_EPSG)
    return str(tmp_path)


def _blank_dbzh(dt: xr.DataTree) -> xr.DataTree:
    """Null out DBZH everywhere, so the default ``DBZH > 0`` keeps nothing."""
    for name in dt.children:
        ds = dt[name].to_dataset()
        ds["DBZH"] = ds["DBZH"].where(False)
        dt[name].dataset = ds
    return dt


def _drop_range_index(dt: xr.DataTree) -> xr.DataTree:
    """Reproduce the shape xradar hands back for a raw NEXRAD Level II volume."""
    for name in dt.children:
        dt[name].dataset = dt[name].to_dataset().drop_indexes("range")
    return dt


# ---------------------------------------------------------------------------
# open_any_datatree
# ---------------------------------------------------------------------------


def test_open_any_datatree(tmp_path, datatree):
    """NetCDF round-trips with every sweep and every value intact."""
    pytest.importorskip("netCDF4")
    path = tmp_path / "vol_20240101_120000.nc"
    datatree.to_netcdf(path)

    loaded = open_any_datatree(path)

    assert sorted(g.lstrip("/") for g in loaded.groups if "sweep" in g) == ["sweep_1", "sweep_2"]
    np.testing.assert_allclose(
        loaded["sweep_1"].to_dataset()["DBZH"].values,
        datatree["sweep_1"].to_dataset()["DBZH"].values,
    )


def test_open_any_datatree_reads_zarr(tmp_path, datatree):
    """A Zarr store is a directory, and the engine is sniffed from the suffix."""
    pytest.importorskip("zarr")
    path = tmp_path / "vol_20240101_120000.zarr"
    datatree.to_zarr(path)

    loaded = open_any_datatree(path)

    np.testing.assert_allclose(
        loaded["sweep_1"].to_dataset()["DBZH"].values,
        datatree["sweep_1"].to_dataset()["DBZH"].values,
    )


def test_open_any_datatree_raises_on_a_missing_path(tmp_path):
    """A typo must fail loudly, not return an empty tree."""
    with pytest.raises(FileNotFoundError):
        open_any_datatree(tmp_path / "nope.nc")


# ---------------------------------------------------------------------------
# datatree_to_dataset / datatree_to_dataframe
# ---------------------------------------------------------------------------


def test_datatree_to_dataset(datatree):
    """A sweep is addressable by name or by number."""
    by_name = datatree_to_dataset(datatree, "sweep_1")
    by_number = datatree_to_dataset(datatree, 1)

    assert "DBZH" in by_name
    np.testing.assert_array_equal(by_name["DBZH"].values, by_number["DBZH"].values)


def test_datatree_to_dataframe(datatree):
    """The flattened volume is polars, one row per gate, across every sweep."""
    df = datatree_to_dataframe(datatree)

    assert isinstance(df, pl.DataFrame)
    assert df.height == 12 * 24 * 2
    assert {"azimuth", "range", "sweep", "DBZH"} <= set(df.columns)


def test_datatree_to_dataframe_keeps_true_range_values(datatree, make_datatree):
    """Raw NEXRAD sweeps arrive with ``range`` as a coordinate carrying no index.

    ``to_dataframe`` then indexes that dimension by position and emits the true values
    as a same-named column, which ``reset_index`` refuses to insert.  The flattener
    rebuilds the index first, so meters survive rather than the positions 0..n-1.
    """
    expected = datatree_to_dataframe(datatree)
    got = datatree_to_dataframe(_drop_range_index(make_datatree()))

    assert got.shape == expected.shape
    assert sorted(got["range"].unique().to_list()) == sorted(expected["range"].unique().to_list())
    assert got["range"].min() == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# archive_volume and the batch wrappers
# ---------------------------------------------------------------------------


def test_archive_volume(lut_base, make_datatree):
    """One volume becomes one POL parquet, whose path is returned."""
    path = archive_volume(make_datatree(), radar=RADAR, base_output_path=lut_base)

    assert path is not None
    assert path.endswith("_POL.parquet")
    assert "DBZH" in pl.read_parquet(path).columns


def test_archive_volume_applies_the_clear_sky_filter(lut_base, make_datatree):
    """No-echo gates are dropped at archive time, which is what keeps archives small."""
    dt = make_datatree(dbzh_min=-10.0, dbzh_max=30.0)

    path = archive_volume(dt, radar=RADAR, base_output_path=lut_base, filter_threshold=0.0, filter_logic=">")

    df = pl.read_parquet(path)
    assert (df["DBZH"].to_numpy() > 0).all()
    assert df.height < 12 * 24 * 2


def test_archive_volume_returns_none_for_an_empty_volume(lut_base, make_datatree):
    """A clear-air volume is skipped, not written and not crashed on."""
    assert archive_volume(_blank_dbzh(make_datatree()), radar=RADAR, base_output_path=lut_base) is None


def test_archive_volume_records_timings(lut_base, make_datatree):
    """The ``timer=`` hook is what makes the pipeline profile possible.

    ``archive_volume`` records its three internal stages; the enclosing
    ``archive_volume`` stage is timed one level up, by the batch wrapper.
    """
    from raddb.helper import StageTimer

    timer = StageTimer()
    archive_volume(make_datatree(), radar=RADAR, base_output_path=lut_base, timer=timer, volume="vol_001")

    stages = set(timer.to_dataframe()["stage"])
    assert stages == {"datatree_to_df", "generate_gate_ids", "save_parquet"}
    assert set(timer.to_dataframe()["volume"]) == {"vol_001"}


def test_archive_multiple_volumes(lut_base, make_datatree):
    """A list or dict of volumes archives sequentially and reports one record each."""
    volumes = {f"vol_{i}": make_datatree(vol_time=pd.Timestamp(f"2024-08-01 12:0{i}:00")) for i in range(3)}

    records = archive_multiple_volumes(volumes, radar=RADAR, base_output_path=lut_base, verbose=False)

    assert len(records) == 3
    assert all(r["success"] and not r["skipped"] for r in records)
    assert all(r["n_gates"] > 0 and r["error"] is None for r in records)
    assert [r["label"] for r in records] == ["vol_0", "vol_1", "vol_2"]


def test_archive_multiple_volumes_accepts_a_plain_list(lut_base, make_datatree):
    """The dict keys are labels only; a bare list works the same."""
    volumes = [make_datatree(vol_time=pd.Timestamp(f"2024-08-01 12:0{i}:00")) for i in range(2)]

    assert len(archive_multiple_volumes(volumes, radar=RADAR, base_output_path=lut_base, verbose=False)) == 2


def test_archive_volumes_multi_radar(tmp_path, make_datatree):
    """The ``{radar: [volumes]}`` form keeps each radar's records separate."""
    for radar in (RADAR, RADAR_B):
        generate_lut_from_datatree(
            make_datatree(),
            radar=radar,
            output_base_path=str(tmp_path),
            projection_epsg=FMI_EPSG,
        )

    results = archive_volumes_multi_radar(
        {RADAR: [make_datatree()], RADAR_B: [make_datatree()]},
        base_output_path=str(tmp_path),
        verbose=False,
    )

    assert sorted(results) == sorted([RADAR, RADAR_B])
    assert all(len(v) == 1 for v in results.values())


def test_datatree_to_parquet(lut_base, make_datatree):
    """The single-volume entry point behind ``archive_volume``."""
    path = datatree_to_parquet(make_datatree(), radar=RADAR, base_output_path=lut_base)

    assert path.endswith("_POL.parquet")
    assert pl.read_parquet(path).height > 0


def test_the_output_path_encodes_the_volume_time(lut_base, make_datatree):
    """``{radar}_{YYYYMMDD}_{HHMMSS}_POL.parquet`` under ``{YYYY}/{MM}/{DD}``."""
    dt = make_datatree(vol_time=pd.Timestamp("2024-08-26 02:50:00"))

    path = datatree_to_parquet(dt, radar=RADAR, base_output_path=lut_base)

    assert path.endswith(f"{RADAR}_20240826_025000_POL.parquet")
    assert Path(path).parts[-4:-1] == ("2024", "08", "26")


def test_save_polar_parquet_returns_none_on_an_empty_frame():
    """The guard itself: an empty frame has no volume time to build a path from.

    ``pd.to_datetime(None)`` is ``NaT``, ``pd.NaT.month`` is *nan* — a float — so
    ``f"{...:02d}"`` raised ``Unknown format code 'd'``.  Two real volumes hit this.
    """
    assert _save_polar_parquet(pl.DataFrame({"gate_id": [], "time": []}), RADAR, "/nonexistent") is None


# ---------------------------------------------------------------------------
# Which columns are archived — the per-gate rule
# ---------------------------------------------------------------------------


def test_gate_variables(make_datatree):
    """Per-gate moments in, scan metadata out."""
    assert _gate_variables(make_datatree()) == ["DBZH", "ZDR", "RHOHV", "PHIDP", "VRADH"]


def test_scan_metadata_never_reaches_the_parquet(lut_base, make_datatree):
    """``sweep_mode`` is a scalar string describing the scan, not a measurement.

    It broadcasts to one value per row when the volume is flattened, so an
    exclusion rule based on column names alone would have to know it by name.  The
    dimension rule drops it because it is not carried on azimuth x range — which also
    keeps a string out of a numeric parquet column.
    """
    path = datatree_to_parquet(make_datatree(), radar=RADAR, base_output_path=lut_base)

    assert "sweep_mode" not in pl.read_parquet(path).columns


def test_a_moment_raddb_never_heard_of_is_archived(lut_base, make_datatree):
    """The point of dropping the allowlist: a network records what it likes.

    ``SPAM`` is in no list anywhere in the package, and it must still survive.
    """
    dt = make_datatree()
    for name in dt.children:
        node = dt[name].to_dataset()
        dt[name] = node.assign(SPAM=(("azimuth", "range"), np.full(node["DBZH"].shape, 7.0, np.float32)))

    path = datatree_to_parquet(dt, radar=RADAR, base_output_path=lut_base)
    out = pl.read_parquet(path)

    assert "SPAM" in out.columns
    assert out["SPAM"].drop_nulls().unique().to_list() == [7.0]


def test_variables_narrows_what_is_archived(lut_base, make_datatree):
    """``variables=`` is the knob for a lean archive; ``gate_id``/``time`` always stay."""
    path = datatree_to_parquet(
        make_datatree(),
        radar=RADAR,
        base_output_path=lut_base,
        variables=["DBZH", "ZDR"],
    )

    assert pl.read_parquet(path).columns == ["gate_id", "time", "DBZH", "ZDR"]


def test_archive_volume_honors_variables(lut_base, make_datatree):
    """The same knob on the batch entry point, not only the single-volume one."""
    path = archive_volume(make_datatree(), radar=RADAR, base_output_path=lut_base, variables=["DBZH"])

    assert pl.read_parquet(path).columns == ["gate_id", "time", "DBZH"]


def test_float_columns_are_narrowed_to_float32(lut_base, make_datatree):
    """Decided by dtype, not by a list of names, so an unknown moment is narrowed too."""
    dt = make_datatree()
    for name in dt.children:
        node = dt[name].to_dataset()
        dt[name] = node.assign(SPAM=(("azimuth", "range"), np.full(node["DBZH"].shape, 7.0, np.float64)))

    out = pl.read_parquet(datatree_to_parquet(dt, radar=RADAR, base_output_path=lut_base))

    assert out["SPAM"].dtype == pl.Float32
    assert out["gate_id"].dtype == pl.Int64, "the join key must stay int64"


# ---------------------------------------------------------------------------
# The nominal azimuth grid — why a volume joins its LUT
# ---------------------------------------------------------------------------


@pytest.fixture
def drifting_archive(tmp_path):
    """One LUT-defining volume plus four later ones with drifting azimuths."""
    base = tmp_path / "drift"
    db = RadDB(archive_dir=str(base), crs=FMI_EPSG)
    first = build_datatree(n_az=360, n_rng=40, n_sweeps=3)
    db.archive(datatree=first, radar=RADAR)
    rng = np.random.default_rng(7)
    for k in range(1, 5):
        when = pd.Timestamp("2024-08-01 12:00:00") + pd.Timedelta(minutes=5 * k)
        db.archive(datatree=retime(first, when, rng, ANTENNA_BIAS), radar=RADAR)
    return base


def test_every_volume_joins_its_lut_completely(drifting_archive):
    """The bug this was written for: 6% of gates per volume used to vanish."""
    lut = pl.read_parquet(drifting_archive / RADAR / "LUT" / f"{RADAR}_LUT.parquet", columns=["gate_id"])
    pols = sorted((drifting_archive / RADAR).rglob("*_POL.parquet"))

    assert len(pols) == 5
    for path in pols:
        pol = pl.read_parquet(path, columns=["gate_id"])
        assert pol.join(lut, on="gate_id", how="semi").height == pol.height, f"{path.name} did not join fully"


def test_the_same_ray_keeps_its_gate_id_across_volumes(drifting_archive):
    """A gate must keep one identity across rotations or nothing can be compared."""
    sets = [
        set(pl.read_parquet(f, columns=["gate_id"])["gate_id"].to_list())
        for f in sorted((drifting_archive / RADAR).rglob("*_POL.parquet"))
    ]

    assert all(s == sets[0] for s in sets)


def test_the_whole_archive_reads_back_with_geometry(drifting_archive):
    """The loss was invisible because it only showed up in LUT joins."""
    rdf = RadDB(archive_dir=str(drifting_archive)).open(radars=RADAR)

    assert rdf.to_geopandas().shape[0] == rdf.data.height


def test_snap_volume_azimuths_moves_a_drifting_ray_onto_the_grid():
    """``sweeps`` and ``azimuths`` are parallel per-ray arrays; the grid is per sweep."""
    grid = np.arange(5, 3600, 10)  # 360 rays at x.5 degrees, in tenths
    azimuths = np.array([0.53, 1.47, 2.51])
    sweeps = np.ones(3, dtype=np.int64)

    snapped, worst = _snap_volume_azimuths(sweeps, azimuths, {1: grid}, RADAR)

    np.testing.assert_allclose(snapped, [0.5, 1.5, 2.5])
    assert worst == pytest.approx(0.03, abs=1e-9)


def test_snap_volume_azimuths_refuses_more_rays_than_the_grid():
    """A 720-ray volume against a 360-ray grid is a different scan strategy."""
    azimuths = np.linspace(0, 360, 720, endpoint=False)

    with pytest.raises(ValueError, match="different scan strategy"):
        _snap_volume_azimuths(np.ones(720, dtype=np.int64), azimuths, {1: np.arange(0, 3600, 10)}, RADAR)


def test_snap_volume_azimuths_accepts_fewer_rays_than_the_grid():
    """A rotation with holes: each surviving ray still snaps to its own grid point."""
    snapped, _ = _snap_volume_azimuths(
        np.ones(3, dtype=np.int64),
        np.array([0.53, 1.47, 2.51]),
        {1: np.arange(5, 3600, 10)},
        RADAR,
    )

    assert snapped.size == 3


def test_snap_volume_azimuths_refuses_an_unknown_sweep():
    """A sweep the LUT never saw has no grid to snap onto."""
    with pytest.raises(ValueError, match="no sweep"):
        _snap_volume_azimuths(np.array([9]), np.array([0.0]), {1: np.arange(0, 3600, 10)}, RADAR)


def test_snap_volume_azimuths_refuses_a_ray_beyond_the_tolerance():
    """Further than half a ray spacing is not antenna drift.

    Uniform grids cannot trigger this — the worst case is exactly half a spacing — so
    the guard only bites when a ray lands where the grid has no point at all.  Here the
    grid is a 1-degree rotation with 170..190 removed, and a ray is aimed into the gap.
    """
    gapped = np.concatenate([np.arange(0, 1700, 10), np.arange(1900, 3600, 10)])

    with pytest.raises(ValueError, match="not antenna drift"):
        _snap_volume_azimuths(np.array([1]), np.array([180.0]), {1: gapped}, RADAR)


def test_archiving_refuses_a_different_ray_count(tmp_path, make_datatree):
    """Supporting several geometries per radar is deliberately not done yet."""
    db = RadDB(archive_dir=str(tmp_path / "a"), crs=FMI_EPSG)
    db.archive(datatree=build_datatree(n_az=360, n_rng=20, n_sweeps=2), radar=RADAR)

    with pytest.raises(ValueError, match="different scan strategy"):
        db.archive(
            datatree=build_datatree(n_az=720, n_rng=20, n_sweeps=2, vol_time=pd.Timestamp("2024-08-01 13:00")),
            radar=RADAR,
        )


def test_archiving_accepts_a_volume_that_dropped_rays(tmp_path):
    """A volume short of a ray or two is a rotation with holes, not a new strategy."""
    db = RadDB(archive_dir=str(tmp_path / "a"), crs=FMI_EPSG)
    complete = build_datatree(n_az=360, n_rng=20, n_sweeps=2)
    db.archive(datatree=complete, radar=RADAR)

    drifted = retime(complete, pd.Timestamp("2024-08-01 18:00"), np.random.default_rng(21), ANTENNA_BIAS)
    holed = xr.DataTree.from_dict(
        {
            name: node.to_dataset().isel(azimuth=np.delete(np.arange(360), [5, 6, 200]))
            for name, node in drifted.children.items()
        },
    )

    result = db.archive(datatree=holed, radar=RADAR)

    assert (result["n_archived"], result["n_failed"]) == (1, 0)


def test_a_lut_built_from_a_holed_volume_still_holds_every_ray(tmp_path):
    """The LUT is written for the whole rotation, so a later complete volume joins."""
    complete = build_datatree(n_az=360, n_rng=20, n_sweeps=2)
    holed = xr.DataTree.from_dict(
        {
            name: node.to_dataset().isel(azimuth=np.delete(np.arange(360), [5, 6, 200]))
            for name, node in complete.children.items()
        },
    )
    db = RadDB(archive_dir=str(tmp_path / "a"), crs=FMI_EPSG)
    db.archive(datatree=holed, radar=RADAR)

    lut = db.get_lut(RADAR)
    assert lut.filter(pl.col("sweep") == 1)["azimuth"].n_unique() == 360

    result = db.archive(
        datatree=retime(complete, pd.Timestamp("2024-08-01 19:00"), np.random.default_rng(22), ANTENNA_BIAS),
        radar=RADAR,
    )
    assert (result["n_archived"], result["n_failed"]) == (1, 0)

    lut_ids = set(lut["gate_id"].to_list())
    assert all(g in lut_ids for g in db.open(radars=RADAR).data["gate_id"].to_list())


def test_a_batch_reports_a_refusal_instead_of_aborting(tmp_path):
    """One incompatible volume must not take the whole batch down."""
    db = RadDB(archive_dir=str(tmp_path / "a"), crs=FMI_EPSG)
    db.archive(datatree=build_datatree(n_az=360, n_rng=20, n_sweeps=2), radar=RADAR)

    good = retime(
        build_datatree(n_az=360, n_rng=20, n_sweeps=2),
        pd.Timestamp("2024-08-01 16:00"),
        np.random.default_rng(12),
        ANTENNA_BIAS,
    )
    bad = build_datatree(n_az=720, n_rng=20, n_sweeps=2, vol_time=pd.Timestamp("2024-08-01 17:00"))

    result = db.archive(datatree={"good": good, "bad": bad}, radar=RADAR)

    assert (result["n_archived"], result["n_failed"]) == (1, 1)


def test_building_without_a_grid_warns(caplog):
    """A pre-grid archive keeps working, but must say that gates may not join."""
    df = pl.DataFrame(
        {
            "sweep": [1, 1],
            "azimuth": [0.53, 1.53],
            "range": [1000.0, 1000.0],
            "DBZH": [10.0, 20.0],
            "time": [pd.Timestamp("2024-01-01")] * 2,
        },
    )

    with caplog.at_level("WARNING"):
        _build_polar_dataframe(df, RADAR, "DBZH", 0.0, ">", azimuth_grids=None)

    assert "no nominal azimuth grid" in caplog.text


# ---------------------------------------------------------------------------
# The read path
# ---------------------------------------------------------------------------


def test_parquet_to_dataframe(archive_dir):
    """The archive reads back as polars, one row per surviving gate."""
    df = parquet_to_dataframe(RADAR, archive_dir, *TIME_WINDOW)

    assert isinstance(df, pl.DataFrame)
    assert not df.is_empty()
    assert "gate_id" in df.columns


def test_parquet_to_dataframe_merges_the_lut(archive_dir):
    """``merge_lut=True`` attaches the static geometry, ``sweep`` included."""
    df = parquet_to_dataframe(RADAR, archive_dir, *TIME_WINDOW, merge_lut=True)

    assert {"sweep", "azimuth", "range"} <= set(df.columns)
    columns = list(df.columns)
    assert abs(columns.index("sweep") - columns.index("azimuth")) == 1, "sweep must sit beside azimuth"


def test_parquet_to_dataframe_selects_columns(archive_dir):
    """``columns=`` is pushed into the reader."""
    df = parquet_to_dataframe(RADAR, archive_dir, *TIME_WINDOW, columns=["gate_id", "DBZH"])

    assert set(df.columns) <= {"gate_id", "DBZH", "volume_time", "radar"}


def test_parquet_to_dataframe_honors_the_time_window(archive_dir_two_volumes):
    """Only volumes inside the window are read."""
    both = parquet_to_dataframe(RADAR, archive_dir_two_volumes, *TIME_WINDOW)
    first = parquet_to_dataframe(RADAR, archive_dir_two_volumes, "2024-08-01 11:59", "2024-08-01 12:01")

    assert 0 < first.height < both.height


def test_scan_polar_parquet(archive_dir):
    """The lazy entry point, for queries that should not materialize the archive."""
    lazy = scan_polar_parquet(RADAR, archive_dir, *TIME_WINDOW)

    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().height > 0


def test_scan_polar_parquet_returns_none_when_nothing_matches(archive_dir):
    """No files in range is ``None``, which callers check before collecting."""
    assert scan_polar_parquet(RADAR, archive_dir, "1999-01-01", "1999-12-31") is None


def test_two_volumes_join_whatever_order_their_moments_are_in(archive_dir_two_volumes):
    """``_gate_variables`` keeps a volume's own column order, so files disagree.

    The same radar read from two sources — a converted zarr and a raw ODIM, say —
    archives the same moments in a different order, and a plain vertical concat
    rejects that outright with ``schema names differ``.  A volume may also carry a
    moment another one does not: that column reads back null-filled, not as an error.
    """
    files = sorted(archive_dir_two_volumes.rglob("*POL.parquet"))
    assert len(files) == 2

    df = pl.read_parquet(files[1])
    reordered = df.select([*reversed(df.columns)]).with_columns(pl.lit(1.0).alias("EXTRA"))
    reordered.write_parquet(files[1])

    out = scan_polar_parquet(RADAR, archive_dir_two_volumes, *TIME_WINDOW).collect()

    assert out.height == pl.read_parquet(files[0]).height + df.height
    assert out["EXTRA"].null_count() == pl.read_parquet(files[0]).height


# ---------------------------------------------------------------------------
# Reconstruction back to a DataTree
# ---------------------------------------------------------------------------


def test_parquet_to_datatree(archive_dir):
    """A round trip: archived gates come back as sweeps carrying the label column."""
    dt = parquet_to_datatree(RADAR, archive_dir, *TIME_WINDOW, label_column="DBZH")

    from raddb.helper import list_sweep_names

    names = list_sweep_names(dt)
    assert names
    for name in names:
        assert "DBZH" in dt[name].to_dataset()


def test_parquet_to_datatree_handles_several_volumes(archive_dir_two_volumes):
    """Two volumes share every ``gate_id``; the duplicates used to crash the rebuild."""
    dt = parquet_to_datatree(RADAR, archive_dir_two_volumes, *TIME_WINDOW, label_column="DBZH")

    from raddb.helper import list_sweep_names

    assert list_sweep_names(dt)


def test_dataframe_to_datatree(archive_dir):
    """The same reconstruction, driven from a frame the caller already holds."""
    df = parquet_to_dataframe(RADAR, archive_dir, *TIME_WINDOW)

    dt = dataframe_to_datatree(df, RADAR, archive_dir, label_column="DBZH")

    from raddb.helper import list_sweep_names

    assert list_sweep_names(dt)


def test_reconstruct_sweep_dataset(archive_dir):
    """One sweep, reindexed onto its full azimuth x range grid."""
    from raddb.lut import load_radar_info, load_radar_lut

    lut = load_radar_lut(RADAR, archive_dir)
    df = parquet_to_dataframe(RADAR, archive_dir, *TIME_WINDOW)
    joined = df.join(lut, on="gate_id", how="inner")

    ds = reconstruct_sweep_dataset(joined, 1, lut, load_radar_info(RADAR, archive_dir), label_column="DBZH")

    assert isinstance(ds, xr.Dataset)
    assert {"azimuth", "range"} <= set(ds.dims)
    assert "DBZH" in ds


def test_reconstruct_datatree(archive_dir):
    """The whole volume, from a joined frame plus the two LUT files."""
    from raddb.lut import load_radar_lut

    lut = load_radar_lut(RADAR, archive_dir)
    df = parquet_to_dataframe(RADAR, archive_dir, *TIME_WINDOW)

    dt = reconstruct_datatree(
        df.join(lut, on="gate_id", how="inner"),
        lut_path=lut_file_path(RADAR, "lut", archive_dir),
        radar_info_path=lut_file_path(RADAR, "info", archive_dir),
        label_column="DBZH",
    )

    from raddb.helper import list_sweep_names

    assert list_sweep_names(dt) == ["sweep_1", "sweep_2"]


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def test_labels_to_dataframe():
    """External model output is turned into a joinable two-column frame."""
    df = labels_to_dataframe(np.array([1, 2, 3]), np.array([10, 20, 30], dtype=np.int64))

    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["gate_id", "hydrometeor_class"]
    assert df["gate_id"].to_list() == [10, 20, 30]


def test_labels_to_dataframe_carries_extra_columns():
    """``extra_columns=`` rides along, e.g. a per-gate confidence."""
    df = labels_to_dataframe(
        np.array([1, 2]),
        np.array([10, 20], dtype=np.int64),
        extra_columns={"confidence": np.array([0.9, 0.8])},
    )

    assert "confidence" in df.columns


def test_join_labels_with_lut(archive_dir):
    """Labels gain the static geometry their ``gate_id`` points at.

    The join is LUT-left, so every gate keeps a row and unlabeled ones carry a null —
    which is what lets a partial model output be reconstructed onto the full grid.
    """
    lut_path = lut_file_path(RADAR, "lut", archive_dir)
    lut = pl.read_parquet(lut_path, columns=["gate_id"])
    labels = labels_to_dataframe(np.ones(5, dtype=np.int64), lut["gate_id"].to_numpy()[:5])

    joined = join_labels_with_lut(labels, lut_path)

    assert joined.height == lut.height
    assert {"azimuth", "range", "sweep", "hydrometeor_class"} <= set(joined.columns)
    assert joined["hydrometeor_class"].null_count() == lut.height - 5


# ---------------------------------------------------------------------------
# add_feature_to_df / add_feature_to_dt
# ---------------------------------------------------------------------------


def test_add_feature_to_df():
    """A computed column is appended without touching the existing ones."""
    df = pl.DataFrame({"DBZH": [10.0, 20.0]})

    out = add_feature_to_df(df, "Z_lin", lambda d: 10 ** (np.asarray(d["DBZH"]) / 10.0))

    assert "Z_lin" in out.columns
    assert out["Z_lin"].to_numpy() == pytest.approx([10.0, 100.0])


def test_add_feature_to_df_returns_the_kind_it_was_given():
    """The same-kind-in-same-kind-out contract as ``filter_df``."""
    data = {"DBZH": [10.0, 20.0]}
    compute = lambda d: np.asarray(d["DBZH"]) * 2  # noqa: E731 - a one-line test double

    assert isinstance(add_feature_to_df(pl.DataFrame(data), "x", compute), pl.DataFrame)
    assert isinstance(add_feature_to_df(pd.DataFrame(data), "x", compute), pd.DataFrame)


def test_add_feature_to_dt(datatree):
    """The DataTree counterpart adds the variable to every sweep."""
    out = add_feature_to_dt(datatree, "Z_lin", lambda ds: 10 ** (ds["DBZH"] / 10.0))

    for name in ("sweep_1", "sweep_2"):
        assert "Z_lin" in out[name].to_dataset()


# ---------------------------------------------------------------------------
# The polars/pandas seam helpers
# ---------------------------------------------------------------------------


def test_to_polars_frame_and_back():
    """Both coercions are identity on the kind they target."""
    pl_df = pl.DataFrame({"a": [1, 2]})
    pd_df = pd.DataFrame({"a": [1, 2]})

    assert _to_polars_frame(pl_df) is pl_df
    assert isinstance(_to_polars_frame(pd_df), pl.DataFrame)
    assert isinstance(_to_pandas_frame(pl_df), pd.DataFrame)
    assert _to_pandas_frame(pd_df) is pd_df


def test_col_reads_from_either_kind():
    """Polars' ``Series.to_numpy()`` takes no ``dtype``, unlike pandas'."""
    for df in (pl.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 2]})):
        out = _col(df, "a", np.float64)
        assert out.dtype == np.float64
        assert out.tolist() == [1.0, 2.0]


def test_a_malformed_volume_is_reported_as_a_failure(lut_base):
    """A batch records the error rather than propagating it.

    A 530-volume run must survive one bad file; the record carries the reason so the
    caller can see what happened without re-running everything.
    """
    bad = xr.DataTree.from_dict({"sweep_1": xr.Dataset({"DBZH": (["azimuth", "range"], np.ones((5, 5)))})})

    records = archive_multiple_volumes({"bad_vol": bad}, radar=RADAR, base_output_path=lut_base, verbose=False)

    assert len(records) == 1
    assert records[0]["success"] is False
    assert records[0]["error"] is not None

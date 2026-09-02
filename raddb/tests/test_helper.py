"""Tests for :mod:`raddb.helper` — filters, radar-name normalization, StageTimer.

Two contracts carry most of the weight here.

**Radar names.** ``normalize_radar_name`` used to return the last character of a name,
which silently turned ``KTLX`` into ``X`` and let two NEXRAD sites overwrite each other's
archive.  It now raises instead of truncating, and the legacy ``ML*`` alias rule is
restricted to exactly three characters so a genuine four-character name is not eaten.

**Same kind in, same kind out.** ``filter_df`` accepts polars or pandas and returns what
it was given, so pandas callers predating the polars migration keep working.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from raddb.helper import (
    FILTER_LOGICS,
    RADAR_ALPHABET,
    RADAR_CODE_LEN,
    StageTimer,
    _vprint,
    check_dataframe,
    ensure_utc,
    filter_df,
    filter_dt,
    is_valid_radar_name,
    list_sweep_names,
    normalize_radar_name,
    read_parquet_files,
    resolve_filter_logic,
)
from raddb.tests.conftest import FMI_EPSG, RADAR

# ---------------------------------------------------------------------------
# list_sweep_names
# ---------------------------------------------------------------------------


def test_list_sweep_names(datatree):
    """Only ``sweep_N`` groups are returned, sorted and without the leading slash."""
    assert list_sweep_names(datatree) == ["sweep_1", "sweep_2"]


def test_list_sweep_names_ignores_other_groups():
    """A CfRadial2 tree carries ``radar_parameters`` and friends alongside the sweeps."""
    dt = xr.DataTree.from_dict(
        {
            "sweep_1": xr.Dataset({"DBZH": ("range", [1.0])}),
            "radar_parameters": xr.Dataset({"beamwidth": 1.0}),
            "georeferencing_correction": xr.Dataset({"dx": 0.0}),
        },
    )

    assert list_sweep_names(dt) == ["sweep_1"]


def test_list_sweep_names_on_an_empty_tree():
    """No sweeps is an empty list, not an error."""
    assert list_sweep_names(xr.DataTree()) == []


# ---------------------------------------------------------------------------
# ensure_utc
# ---------------------------------------------------------------------------


def test_ensure_utc():
    """Naive input is localised, aware input is converted, ``None`` passes through."""
    assert ensure_utc(None) is None
    assert ensure_utc("2024-08-01 12:00") == pd.Timestamp("2024-08-01 12:00", tz="UTC")
    assert ensure_utc(pd.Timestamp("2024-08-01 12:00")).tzinfo is not None


def test_ensure_utc_converts_rather_than_relabels():
    """A +02:00 timestamp becomes the same instant in UTC, not the same clock reading."""
    local = pd.Timestamp("2024-08-01 14:00", tz="Europe/Zurich")

    assert ensure_utc(local) == pd.Timestamp("2024-08-01 12:00", tz="UTC")


def test_ensure_utc_is_idempotent():
    """Running it twice must not shift the instant."""
    once = ensure_utc("2024-08-01 12:00")

    assert ensure_utc(once) == once


# ---------------------------------------------------------------------------
# read_parquet_files
# ---------------------------------------------------------------------------


def test_read_parquet_files(archive_dir_two_volumes, capsys):
    """Every matching parquet under the tree is concatenated into one frame."""
    df = read_parquet_files(archive_dir_two_volumes)

    assert isinstance(df, pl.DataFrame)
    assert "gate_id" in df.columns
    assert not df.is_empty()


def test_read_parquet_files_selects_columns(archive_dir):
    """``columns=`` is pushed into the parquet reader, not applied afterwards."""
    df = read_parquet_files(archive_dir, columns=["gate_id", "DBZH"], verbose=False)

    assert df.columns == ["gate_id", "DBZH"]


def test_read_parquet_files_with_no_matches(tmp_path, capsys):
    """No files yields an empty frame and says so."""
    df = read_parquet_files(tmp_path, verbose=True)

    assert df.is_empty()
    assert "No files found" in capsys.readouterr().out


def test_read_parquet_files_is_quiet_when_asked(tmp_path, capsys):
    """``verbose=False`` prints nothing at all."""
    read_parquet_files(tmp_path, verbose=False)

    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# check_dataframe
# ---------------------------------------------------------------------------


def test_check_dataframe(capsys):
    """A polars frame's shape, columns and null counts are printed."""
    check_dataframe(pl.DataFrame({"DBZH": [1.0, None], "gate_id": [1, 2]}))

    out = capsys.readouterr().out
    assert "Shape:" in out
    assert "(2, 2)" in out
    assert "DBZH" in out
    assert "Missing values:" in out


def test_check_dataframe_accepts_pandas(capsys):
    """The pandas branch takes a different path to the same summary."""
    check_dataframe(pd.DataFrame({"DBZH": [1.0, np.nan]}))

    assert "Missing values:" in capsys.readouterr().out


def test_check_dataframe_on_an_empty_frame(capsys):
    """No columns must not raise on the ``null_count().to_dicts()[0]`` indexing."""
    check_dataframe(pl.DataFrame())

    assert "Shape:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# normalize_radar_name / is_valid_radar_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A", "A"),
        ("a", "A"),
        (" L ", "L"),
        ("MLA", "A"),  # the legacy three-character alias
        ("mlw", "W"),
        ("KTLX", "KTLX"),  # NEXRAD survives whole
        ("koun", "KOUN"),
        ("000A", "A"),  # zero padding is not part of the name
        ("0A", "A"),
        ("0", "0"),  # ... but a radar may be named "0"
        ("ZZZZ", "ZZZZ"),
    ],
)
def test_normalize_radar_name(raw, expected):
    """The canonical form: upper-case, zero-stripped, ``ML*`` reduced."""
    assert normalize_radar_name(raw) == expected


def test_multi_letter_names_are_not_truncated():
    """The old bug: every name collapsed to its last character."""
    assert normalize_radar_name("KTLX") != "X"
    assert normalize_radar_name("KOUN") != "N"
    # Two sites sharing a final letter must stay distinct, or one would overwrite the
    # other's archive.
    assert normalize_radar_name("KTLX") != normalize_radar_name("KABX")


def test_the_ml_rule_only_applies_at_three_characters():
    """``MLAB`` is a real four-character name, not ``ML`` plus ``AB``."""
    assert normalize_radar_name("MLA") == "A"
    assert normalize_radar_name("MLAB") == "MLAB"


@pytest.mark.parametrize("bad", ["", "   ", "chlem", "ABCDE", "A-B", "vol.", "A B", "MLABC", "é"])
def test_normalize_radar_name_rejects_unusable_names(bad):
    """Raised, not truncated — five-character ODIM codes must be aliased explicitly."""
    assert not is_valid_radar_name(bad)
    with pytest.raises(ValueError, match="not usable"):
        normalize_radar_name(bad)


def test_normalize_radar_name_rejects_a_non_string():
    """A number is not a radar name, and the message says which type arrived."""
    assert not is_valid_radar_name(7)
    with pytest.raises(ValueError, match="must be a string"):
        normalize_radar_name(7)


def test_is_valid_radar_name():
    """The non-raising counterpart, for callers that want to skip rather than fail."""
    assert is_valid_radar_name("A")
    assert is_valid_radar_name("KTLX")
    assert not is_valid_radar_name("OVERLONG")
    assert not is_valid_radar_name(None)


def test_normalize_radar_name_is_idempotent():
    """A canonical name normalizes to itself, so archiving twice hits the same path."""
    for name in ("A", "KTLX", "0", "ZZZZ"):
        assert normalize_radar_name(normalize_radar_name(name)) == name


def test_the_alphabet_and_length_match_the_gate_id_layout():
    """Base-36 over four characters is what the ``gate_id`` radar field can hold."""
    assert RADAR_ALPHABET == "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert len(RADAR_ALPHABET) == 36
    assert RADAR_CODE_LEN == 4


# ---------------------------------------------------------------------------
# resolve_filter_logic / FILTER_LOGICS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("logic", "a", "b", "expected"),
    [
        ("==", 1, 1, True),
        ("!=", 1, 2, True),
        (">", 2, 1, True),
        (">=", 1, 1, True),
        ("<", 1, 2, True),
        ("<=", 1, 1, True),
    ],
)
def test_resolve_filter_logic(logic, a, b, expected):
    """All six operators resolve to the comparison they spell."""
    assert resolve_filter_logic(logic)(a, b) is expected


def test_resolve_filter_logic_rejects_an_unknown_operator():
    """The error lists the valid choices rather than just refusing."""
    with pytest.raises(ValueError, match="Unknown logic"):
        resolve_filter_logic("=~")


def test_filter_logics_registry_is_complete():
    """The registry is the single source of truth for the documented operator set."""
    assert set(FILTER_LOGICS) == {"==", "!=", ">", ">=", "<", "<="}


# ---------------------------------------------------------------------------
# filter_df
# ---------------------------------------------------------------------------


def test_filter_df():
    """Rows failing the comparison are dropped entirely."""
    df = pl.DataFrame({"DBZH": [-5.0, 0.0, 5.0, 10.0]})

    assert filter_df(df, threshold=0.0, logic=">")["DBZH"].to_list() == [5.0, 10.0]


def test_filter_df_returns_the_kind_it_was_given():
    """The polars migration must not break existing pandas callers."""
    data = {"DBZH": [-5.0, 5.0]}

    assert isinstance(filter_df(pl.DataFrame(data)), pl.DataFrame)
    assert isinstance(filter_df(pd.DataFrame(data)), pd.DataFrame)


def test_filter_df_resets_the_pandas_index():
    """A dropped row must not leave a gap in the index for downstream ``iloc``."""
    out = filter_df(pd.DataFrame({"DBZH": [-5.0, 5.0, 10.0]}), threshold=0.0)

    assert out.index.tolist() == [0, 1]


@pytest.mark.parametrize("logic", ["==", "!=", ">", ">=", "<", "<="])
def test_filter_df_honors_every_operator(logic):
    """Each operator selects what the plain numpy comparison would."""
    values = np.array([-5.0, 0.0, 5.0])
    out = filter_df(pl.DataFrame({"DBZH": values}), threshold=0.0, logic=logic)

    assert out["DBZH"].to_numpy().tolist() == values[FILTER_LOGICS[logic](values, 0.0)].tolist()


def test_filter_df_rejects_a_missing_column():
    """A typo in the variable name must not silently keep every row."""
    with pytest.raises(KeyError, match="ZDR"):
        filter_df(pl.DataFrame({"DBZH": [1.0]}), feature="ZDR")


def test_filter_df_rejects_an_unknown_logic():
    """Validation happens before the column lookup, so both errors stay distinct."""
    with pytest.raises(ValueError, match="Unknown logic"):
        filter_df(pl.DataFrame({"DBZH": [1.0]}), logic="=~")


def test_filter_df_can_keep_nothing():
    """An empty result is a valid answer — a clear-air volume produces one."""
    assert filter_df(pl.DataFrame({"DBZH": [-5.0, -1.0]}), threshold=0.0).is_empty()


# ---------------------------------------------------------------------------
# filter_dt
# ---------------------------------------------------------------------------


def test_filter_dt(datatree):
    """Non-matching gates become NaN; the tree keeps its shape."""
    out = filter_dt(datatree, feature="DBZH", threshold=15.0, logic=">")

    ds_in = datatree["sweep_1"].to_dataset()
    ds_out = out["sweep_1"].to_dataset()
    assert ds_out["DBZH"].shape == ds_in["DBZH"].shape
    assert np.isnan(ds_out["DBZH"].values).any()
    assert np.nanmin(ds_out["DBZH"].values) > 15.0


def test_filter_dt_leaves_matching_gates_untouched(datatree):
    """A legitimate zero must survive; masking is not thresholding twice."""
    out = filter_dt(datatree, feature="DBZH", threshold=0.0, logic=">")

    np.testing.assert_allclose(
        out["sweep_1"].to_dataset()["DBZH"].values,
        datatree["sweep_1"].to_dataset()["DBZH"].values,
    )


def test_filter_dt_masks_every_variable_sharing_the_mask_dims(datatree):
    """The point of a DataTree filter: ZDR is masked wherever DBZH failed."""
    out = filter_dt(datatree, feature="DBZH", threshold=15.0, logic=">")

    ds = out["sweep_1"].to_dataset()
    np.testing.assert_array_equal(np.isnan(ds["DBZH"].values), np.isnan(ds["ZDR"].values))


def test_filter_dt_leaves_disjoint_variables_alone():
    """Gate-edge geometry lives on other dims; broadcasting the mask onto it explodes."""
    ds = xr.Dataset(
        {
            "DBZH": (["azimuth", "range"], np.array([[1.0, 20.0]])),
            "x_edges": (["azimuth_edge"], np.array([0.0, 1.0])),
        },
        coords={"azimuth": [0.0], "range": [1000.0, 2000.0], "azimuth_edge": [0.0, 1.0]},
    )
    dt = xr.DataTree.from_dict({"sweep_1": ds})

    out = filter_dt(dt, feature="DBZH", threshold=15.0, logic=">")

    np.testing.assert_array_equal(out["sweep_1"].to_dataset()["x_edges"].values, [0.0, 1.0])


def test_filter_dt_skips_a_sweep_without_the_variable():
    """A sweep missing DBZH is passed through rather than dropped."""
    dt = xr.DataTree.from_dict({"sweep_1": xr.Dataset({"ZDR": ("range", [1.0, 2.0])})})

    out = filter_dt(dt, feature="DBZH", threshold=0.0)

    np.testing.assert_array_equal(out["sweep_1"].to_dataset()["ZDR"].values, [1.0, 2.0])


def test_filter_dt_rejects_an_unknown_logic(datatree):
    """Operator validation happens before any sweep is touched."""
    with pytest.raises(ValueError, match="Unknown logic"):
        filter_dt(datatree, logic="=~")


# ---------------------------------------------------------------------------
# StageTimer
# ---------------------------------------------------------------------------


def test_StageTimer():
    """A fresh timer holds no records."""
    assert StageTimer().records == []


def test_StageTimer_init():
    """Each instance gets its own list — a shared class attribute would pool runs."""
    a, b = StageTimer(), StageTimer()
    a.record("stage", 1.0)

    assert b.records == []


def test_StageTimer_time_stage():
    """The context manager records stage, volume, sweep and a positive duration."""
    timer = StageTimer()

    with timer.time_stage("build", volume="vol_001", sweep=2):
        pass

    (rec,) = timer.records
    assert rec["stage"] == "build"
    assert rec["volume"] == "vol_001"
    assert rec["sweep"] == 2
    assert rec["duration"] >= 0.0


def test_time_stage_records_even_when_the_body_raises():
    """Timing lives in a ``finally``, so a failed stage still shows up in the profile."""
    timer = StageTimer()

    with pytest.raises(RuntimeError), timer.time_stage("boom"):
        raise RuntimeError("stage failed")

    assert [r["stage"] for r in timer.records] == ["boom"]


def test_StageTimer_record():
    """A pre-measured entry has the same shape as a timed one."""
    timer = StageTimer()

    timer.record("io", 1.5, volume="vol_002", sweep=1, t_start=100.0)

    assert timer.records == [{"volume": "vol_002", "sweep": 1, "stage": "io", "t_start": 100.0, "duration": 1.5}]


def test_StageTimer_to_dataframe():
    """Records become a pandas frame — this is one of the three pandas seams."""
    timer = StageTimer()
    timer.record("io", 1.0)
    timer.record("build", 2.0)

    df = timer.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df["stage"].tolist() == ["io", "build"]


def test_to_dataframe_on_an_empty_timer_keeps_the_schema():
    """Downstream ``groupby("stage")`` needs the columns even with no rows."""
    df = StageTimer().to_dataframe()

    assert df.empty
    assert list(df.columns) == ["volume", "sweep", "stage", "duration"]


def test_StageTimer_summary():
    """Aggregated per stage and sorted by total time, slowest first."""
    timer = StageTimer()
    timer.record("fast", 0.1)
    timer.record("slow", 5.0)
    timer.record("slow", 5.0)

    summary = timer.summary()

    assert summary.index.tolist() == ["slow", "fast"]
    assert summary.loc["slow", "sum"] == pytest.approx(10.0)
    assert summary.loc["slow", "count"] == 2
    assert summary.loc["fast", "mean"] == pytest.approx(0.1)


def test_summary_on_an_empty_timer():
    """No records means an empty frame, which ``print_summary`` then reports."""
    assert StageTimer().summary().empty


def test_StageTimer_print_summary(capsys):
    """The table names each stage and its share of the total."""
    timer = StageTimer()
    timer.record("archive_volume", 3.0)
    timer.record("build_lut", 1.0)

    timer.print_summary()

    out = capsys.readouterr().out
    assert "PIPELINE PROFILING SUMMARY" in out
    assert "archive_volume" in out
    assert "TOTAL" in out
    assert "75.0%" in out


def test_print_summary_on_an_empty_timer(capsys):
    """A run that recorded nothing says so instead of printing an empty table."""
    StageTimer().print_summary()

    assert "No timing data recorded" in capsys.readouterr().out


def test_timer_accumulates_across_a_batch(tmp_path, make_datatree):
    """Records pool across volumes, which is what makes the profile useful."""
    from raddb.io_core import archive_multiple_volumes
    from raddb.lut import generate_lut_from_datatree

    timer = StageTimer()
    volumes = {f"vol_{i:03d}": make_datatree(vol_time=pd.Timestamp(f"2024-08-01 19:0{i}:00")) for i in range(3)}
    generate_lut_from_datatree(
        volumes["vol_000"],
        radar=RADAR,
        output_base_path=str(tmp_path),
        projection_epsg=FMI_EPSG,
    )

    archive_multiple_volumes(volumes, radar=RADAR, base_output_path=str(tmp_path), timer=timer, verbose=False)

    df = timer.to_dataframe()
    assert len(df) >= 3
    assert "archive_volume" in df["stage"].to_numpy()


# ---------------------------------------------------------------------------
# _vprint
# ---------------------------------------------------------------------------


def test_vprint_is_silent_by_default(capsys):
    """Progress output is opt-in; library code must not print unasked."""
    _vprint("hello")

    assert capsys.readouterr().out == ""


def test_vprint_timestamps_its_output(capsys):
    """A millisecond timestamp is what makes the messages useful in a long batch."""
    _vprint("hello", verbose=True)

    out = capsys.readouterr().out
    assert "hello" in out
    assert out.startswith("[")
    assert out.count(":") == 2


def test_read_parquet_files_accepts_a_path_object(archive_dir):
    """``base_path`` is used through ``Path``, so both spellings work."""
    assert not read_parquet_files(Path(archive_dir), verbose=False).is_empty()

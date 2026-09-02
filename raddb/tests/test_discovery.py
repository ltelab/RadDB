"""Tests for :mod:`raddb.discovery` — DataTree file discovery and filename-time parsing.

Both sides of the archive live here: finding DataTree inputs on disk, and finding
``*_POL.parquet`` outputs in a time range.  Everything is pure filesystem plus pandas, so
these tests touch no radar data at all — empty files with the right *names* are enough.

The one behavior worth stating up front: a ``.zarr`` store is a **directory**, and it is
matched as a leaf.  Descending into one would return its internal chunk files as if they
were volumes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raddb.discovery import (
    _find_polar_files_in_range,
    _parse_datatree_file_time,
    _parse_pol_time,
    find_datatree_files,
)

# ---------------------------------------------------------------------------
# find_datatree_files — the module's only public callable
# ---------------------------------------------------------------------------


def test_find_datatree_files(tmp_path):
    """NetCDF files and Zarr stores are found; a Zarr store is never descended into."""
    (tmp_path / "vol_20240101_000000.nc").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "vol_20240101_000500.nc").touch()
    store = tmp_path / "vol_20240101_001000.zarr"
    store.mkdir()
    (store / "inner_20240101_002000.nc").touch()  # must NOT be matched
    (tmp_path / "notes.txt").touch()

    found = find_datatree_files(tmp_path)

    assert [p.name for p in found] == [
        "vol_20240101_000000.nc",
        "vol_20240101_000500.nc",
        "vol_20240101_001000.zarr",
    ]


def test_recursive_false_stays_in_the_top_directory(tmp_path):
    """``recursive=False`` ignores subdirectories entirely."""
    (tmp_path / "vol_20240101_000000.nc").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "vol_20240101_000500.nc").touch()

    assert [p.name for p in find_datatree_files(tmp_path, recursive=False)] == ["vol_20240101_000000.nc"]


def test_results_are_sorted_by_filename_timestamp(tmp_path):
    """Directory order is arbitrary; the returned order is chronological."""
    for name in ("vol_20240101_020000.nc", "vol_20240101_000000.nc", "vol_20240101_010000.nc"):
        (tmp_path / name).touch()

    found = find_datatree_files(tmp_path)

    assert [p.name for p in found] == [
        "vol_20240101_000000.nc",
        "vol_20240101_010000.nc",
        "vol_20240101_020000.nc",
    ]


def test_unparsable_names_sort_last(tmp_path):
    """A file with no timestamp is kept but pushed to the end, never interleaved."""
    (tmp_path / "aaa_no_timestamp.nc").touch()
    (tmp_path / "vol_20240101_000000.nc").touch()

    assert [p.name for p in find_datatree_files(tmp_path)][-1] == "aaa_no_timestamp.nc"


def test_time_range_filters_and_keeps_unparsable_names(tmp_path):
    """Out-of-range files drop out; an unparsable name survives by default."""
    for name in (
        "vol_20240101_000000.nc",
        "vol_20240101_010000.nc",
        "vol_20240101_020000.nc",
        "no_timestamp_here.nc",
    ):
        (tmp_path / name).touch()

    found = find_datatree_files(tmp_path, start_time="2024-01-01 00:30", end_time="2024-01-01 01:30")

    assert [p.name for p in found] == ["vol_20240101_010000.nc", "no_timestamp_here.nc"]


def test_strict_time_drops_unparsable_names(tmp_path):
    """``strict_time=True`` refuses to guess: no timestamp, no file."""
    (tmp_path / "vol_20240101_010000.nc").touch()
    (tmp_path / "no_timestamp_here.nc").touch()

    found = find_datatree_files(
        tmp_path,
        start_time="2024-01-01 00:30",
        end_time="2024-01-01 01:30",
        strict_time=True,
    )

    assert [p.name for p in found] == ["vol_20240101_010000.nc"]


def test_strict_time_is_ignored_without_a_range(tmp_path):
    """With no range there is nothing to be strict about — the file is kept."""
    (tmp_path / "no_timestamp_here.nc").touch()

    assert len(find_datatree_files(tmp_path, strict_time=True)) == 1


def test_extensions_are_matched_case_insensitively(tmp_path):
    """Uppercase suffixes appear on data written on case-preserving filesystems."""
    (tmp_path / "vol_20240101_000000.NC").touch()

    assert [p.name for p in find_datatree_files(tmp_path)] == ["vol_20240101_000000.NC"]


def test_extensions_can_be_narrowed(tmp_path):
    """The ``extensions`` argument is a whitelist, not an addition."""
    (tmp_path / "vol_20240101_000000.nc").touch()
    store = tmp_path / "vol_20240101_001000.zarr"
    store.mkdir()

    assert [p.name for p in find_datatree_files(tmp_path, extensions=(".zarr",))] == ["vol_20240101_001000.zarr"]


def test_an_empty_directory_returns_an_empty_list(tmp_path):
    """No matches is not an error."""
    assert find_datatree_files(tmp_path) == []


def test_a_missing_directory_raises(tmp_path):
    """A typo in the input path must fail loudly, not silently archive nothing."""
    with pytest.raises(FileNotFoundError):
        find_datatree_files(tmp_path / "nope")


def test_a_file_passed_as_the_directory_raises(tmp_path):
    """``directory`` must be a directory."""
    f = tmp_path / "vol_20240101_000000.nc"
    f.touch()
    with pytest.raises(FileNotFoundError):
        find_datatree_files(f)


# ---------------------------------------------------------------------------
# _parse_datatree_file_time — the stem patterns behind the time filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("vol_20240101_000000.nc", "2024-01-01 00:00:00"),
        ("A_20240715T235959.nc", "2024-07-15 23:59:59"),
        ("radar-20240101-1230.zarr", "2024-01-01 12:30:00"),
        ("x_202401011230.nc", "2024-01-01 12:30:00"),
        ("KTLX_20130520_195111.zarr", "2013-05-20 19:51:11"),
        ("no_time.nc", None),
    ],
)
def test_parse_datatree_file_time_patterns(name, expected):
    """The three stem patterns, tried in order, and the give-up case."""
    ts = _parse_datatree_file_time(Path(name))

    assert ts is None if expected is None else ts == pd.Timestamp(expected, tz="UTC")


def test_parse_datatree_file_time_is_always_utc():
    """Downstream comparisons assume tz-aware UTC; a naive timestamp would raise."""
    assert _parse_datatree_file_time("vol_20240101_000000.nc").tzinfo is not None


def test_parse_datatree_file_time_rejects_an_impossible_date():
    """A digit run that matches the pattern but is not a date yields ``None``."""
    assert _parse_datatree_file_time("vol_20241332_000000.nc") is None


# ---------------------------------------------------------------------------
# _parse_pol_time / _find_polar_files_in_range — the archive side
# ---------------------------------------------------------------------------


def test_parse_pol_time_reads_the_archive_layout():
    """``{radar}_{YYYYMMDD}_{HHMMSS}_POL.parquet`` is the only accepted shape."""
    assert _parse_pol_time("L_20240826_025000_POL.parquet") == pd.Timestamp("2024-08-26 02:50:00", tz="UTC")
    assert _parse_pol_time("KTLX_20130520_195111_POL.parquet") == pd.Timestamp("2013-05-20 19:51:11", tz="UTC")


@pytest.mark.parametrize("name", ["nope.parquet", "L_20240826_POL.parquet", "L_notadate_025000_POL.parquet"])
def test_parse_pol_time_returns_none_off_layout(name):
    """Anything else is skipped rather than guessed at."""
    assert _parse_pol_time(name) is None


def test_find_polar_files_in_range(tmp_path):
    """POL files are collected recursively, filtered by time and sorted."""
    day = tmp_path / "2024" / "08" / "26"
    day.mkdir(parents=True)
    for hhmmss in ("030000", "010000", "020000"):
        (day / f"L_20240826_{hhmmss}_POL.parquet").touch()
    (day / "not_a_pol_file.parquet").touch()

    found = _find_polar_files_in_range(tmp_path, "2024-08-26 01:30", "2024-08-26 03:30")

    assert [p.name for p in found] == ["L_20240826_020000_POL.parquet", "L_20240826_030000_POL.parquet"]


def test_find_polar_files_in_range_without_bounds(tmp_path):
    """No range means everything, still time-sorted."""
    day = tmp_path / "2024" / "08" / "26"
    day.mkdir(parents=True)
    for hhmmss in ("030000", "010000"):
        (day / f"L_20240826_{hhmmss}_POL.parquet").touch()

    assert [p.name for p in _find_polar_files_in_range(tmp_path)] == [
        "L_20240826_010000_POL.parquet",
        "L_20240826_030000_POL.parquet",
    ]


def test_find_polar_files_in_range_on_an_empty_tree(tmp_path):
    """A radar directory with no volumes yields an empty list, not an error."""
    assert _find_polar_files_in_range(tmp_path) == []

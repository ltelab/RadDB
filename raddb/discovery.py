"""File discovery for RadDB, on both sides of the archive.

- **DataTree file discovery** (input side): locate DataTree files
  (NetCDF / Zarr) on disk, optionally filtered by a filename timestamp,
  ready to load with :func:`raddb.io_core.open_any_datatree` and archive
  with :meth:`raddb.RadDB.archive_from_datatrees`.
- **Archived POL search** (output side): locate ``*_POL.parquet`` files in a
  time range inside an existing RadDB archive.

Everything here is pure filesystem + pandas — no pyart dependency — so
discovery works in any environment.  Scanning a national archive's own raw
format belongs in the package that reads it, not here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from raddb.helper import ensure_utc

# ============================================================================
# DataTree file discovery  (input side)
# ============================================================================

# Filename-stem timestamp patterns, tried in order.
_DT_TIME_PATTERNS = [
    (r"(\d{8})[T_-]?(\d{6})", "%Y%m%d%H%M%S"),  # YYYYMMDD[T_-]HHMMSS
    (r"(\d{8})[T_-]?(\d{4})", "%Y%m%d%H%M"),  # YYYYMMDD[T_-]HHMM
    (r"(\d{12})", "%Y%m%d%H%M"),  # YYYYMMDDHHMM
]


def _parse_datatree_file_time(path: str | Path) -> pd.Timestamp | None:
    """Best-effort UTC timestamp from a DataTree filename stem.

    Tries the patterns in :data:`_DT_TIME_PATTERNS` against the stem
    (e.g. ``vol_20240101_000000.nc`` or ``A_202401010000.zarr``).
    Returns ``None`` when no pattern yields a valid timestamp.
    """
    stem = Path(path).stem
    for pattern, fmt in _DT_TIME_PATTERNS:
        m = re.search(pattern, stem)
        if not m:
            continue
        try:
            ts = pd.to_datetime("".join(m.groups()), format=fmt)
        except Exception:
            continue
        return ensure_utc(ts)
    return None


def find_datatree_files(
    directory: str | Path,
    recursive: bool = True,
    extensions: tuple[str, ...] = (".nc", ".nc4", ".cdf", ".zarr"),
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    strict_time: bool = False,
) -> list[Path]:
    """Find DataTree files (NetCDF files / Zarr stores) under *directory*.

    Zarr stores are **directories** — they are matched as leaves and never
    descended into.  Results are sorted by (filename timestamp, path); files
    whose stem has no parseable timestamp sort last.

    Parameters
    ----------
    directory : str or Path
        Directory to scan.
    recursive : bool
        Recurse into subdirectories (default ``True``).
    extensions : tuple of str
        File/store suffixes to match (case-insensitive).
    start_time, end_time : str or Timestamp, optional
        Keep only files whose filename timestamp (see
        :func:`_parse_datatree_file_time`) falls in this range.
    strict_time : bool
        When a time range is given, drop files whose timestamp cannot be
        parsed from the filename (default ``False`` — they are kept).

    Returns
    -------
    list of Path
        Matching files/stores, time-sorted.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    exts = {e.lower() for e in extensions}
    matches: list[Path] = []
    stack = [root]
    while stack:
        d = stack.pop()
        for entry in sorted(d.iterdir()):
            if entry.suffix.lower() in exts:
                matches.append(entry)  # .zarr dirs are leaves: no descent
            elif entry.is_dir() and recursive:
                stack.append(entry)

    start_dt = ensure_utc(start_time) if start_time else None
    end_dt = ensure_utc(end_time) if end_time else None
    has_range = start_dt is not None or end_dt is not None

    kept: list[tuple[pd.Timestamp | None, Path]] = []
    for p in matches:
        ts = _parse_datatree_file_time(p)
        if ts is None:
            if has_range and strict_time:
                continue
        else:
            if start_dt and ts < start_dt:
                continue
            if end_dt and ts > end_dt:
                continue
        kept.append((ts, p))

    kept.sort(key=lambda x: (x[0] is None, x[0] or pd.Timestamp(0, tz="UTC"), str(x[1])))
    return [p for _, p in kept]


# ============================================================================
# Archived POL parquet search  (output side)
# ============================================================================


def _parse_pol_time(path: str | Path) -> pd.Timestamp | None:
    """UTC volume timestamp from a ``{radar}_{YYYYMMDD}_{HHMMSS}_POL.parquet`` name.

    Returns ``None`` when the stem does not match the archive POL layout.
    """
    stem = Path(path).stem.replace("_POL", "")
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        ts = pd.to_datetime(parts[-2] + "_" + parts[-1], format="%Y%m%d_%H%M%S")
    except Exception:
        return None
    return ensure_utc(ts)


def _find_polar_files_in_range(
    radar_path: Path,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> list[Path]:
    """Return POLAR parquet files within the given time range, sorted by timestamp."""
    polar_files = sorted(radar_path.rglob("*_POL.parquet"))
    if not polar_files:
        return []

    start_dt = ensure_utc(start_time) if start_time else None
    end_dt = ensure_utc(end_time) if end_time else None

    valid = []
    for f in polar_files:
        ts = _parse_pol_time(f)  # {radar}_{YYYYMMDD}_{HHMMSS}_POL.parquet
        if ts is None:
            continue
        if start_dt and ts < start_dt:
            continue
        if end_dt and ts > end_dt:
            continue
        valid.append((ts, f))

    valid.sort(key=lambda x: x[0])
    return [f for _, f in valid]

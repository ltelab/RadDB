"""High-level interface for RadDB — a generic radar data archiving library.

``RadDB`` is a single, dual-role class:

* **Archive-bound** — ``db = RadDB(archive_dir, crs=2056)``.  Use it to
  :meth:`~RadDB.archive` DataTree volumes and to :meth:`~RadDB.open` archived
  data.  ``archive_dir`` / ``crs`` are the shared defaults for every task.
* **Data-carrying** — the object returned by :meth:`~RadDB.open` (and by
  ``filter`` / ``crop_*`` / ``extract_cross_section``).  It holds the loaded
  data as a **polars** DataFrame (``rdf.data``) and exposes the query,
  conversion, area-of-interest, cross-section and plotting methods.  These
  return a **new** ``RadDB`` (fluent ``open → filter → crop → plot``).

Note: network-specific constants such as a list of radar identifiers
(e.g. the FMI sites ``FKUO``, ``FANJ``, ``FKOR``) belong in the user script or
in the network-specific ingestion package, not here.
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import shapely
import xarray as xr

from raddb.aoi import (
    _apply_gate_ids,
    _cross_section_gates,
    _load_aoi_polygon,
    _lut_centroids,
    _lut_cs_table,
    _radars_from_gate_ids,
    _reproject_to_aoi,
    _resolve_aoi_centroids,
    aoi_epsg_for,
)
from raddb.discovery import (
    _find_polar_files_in_range,
    _parse_datatree_file_time,
    _parse_pol_time,
    find_datatree_files,
)
from raddb.helper import (
    RADAR_CODE_LEN,
    ensure_utc,
    is_valid_radar_name,
    normalize_radar_name,
)
from raddb.io_core import (
    archive_multiple_volumes,
    archive_volume,
    archive_volumes_multi_radar,
    dataframe_to_datatree,
    open_any_datatree,
    scan_polar_parquet,
)
from raddb.lut import (
    GATE_ID_RADAR_BASE,
    add_lut_projection,
    cartesian_to_geographic,
    encode_radar_code,
    gate_corner_table,
    generate_lut_from_datatree,
    load_plane_nodes,
    load_radar_info,
    load_radar_lut,
)

logger = logging.getLogger(__name__)


# ================================================================
# Private helpers for end-to-end archiving
# ================================================================


def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (day_start, day_end) pairs covering [start, end] inclusively."""
    day = start.normalize()
    last = end.normalize()
    while day <= last:
        day_start = max(day, start)
        day_end = min(day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), end)
        yield day_start, day_end
        day = day + pd.Timedelta(days=1)


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r") as f:
        return {line.strip() for line in f if line.strip()}


def _append_checkpoint(path: Path, entry: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(entry + "\n")


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# ================================================================
# Private helpers for the object model (filters, time, geometry)
# ================================================================

# Shapely-geometry columns are stored WKB-encoded (Binary) inside the polars
# backend and decoded back to shapely on ``to_pandas``.
_GEOM_COLS = ("cs_polygon",)


def _filter_expr(var: str, logic: str, threshold) -> pl.Expr:
    """Build a polars boolean expression ``var <logic> threshold``."""
    col = pl.col(var)
    ops = {
        "==": col == threshold,
        "!=": col != threshold,
        ">": col > threshold,
        ">=": col >= threshold,
        "<": col < threshold,
        "<=": col <= threshold,
    }
    if logic not in ops:
        raise ValueError(
            f"Unknown filter logic {logic!r}; use one of {sorted(ops)}.",
        )
    return ops[logic]


# ---------------------------------------------------------------------------
# .sel() support — xarray-style label selection
# ---------------------------------------------------------------------------

#: Convenience aliases accepted by :meth:`RadDB.sel`.  Resolved only *after* the
#: literal name fails to match a data or LUT column, so a real column always wins.
_SEL_ALIASES: dict[str, str] = {
    "radars": "radar",
    "lat": "latitude",
    "lats": "latitude",
    "lon": "longitude",
    "long": "longitude",
    "lons": "longitude",
    "alt": "altitude",
    "altitudes": "altitude",
    "sweeps": "sweep",
    "azimuths": "azimuth",
    "ranges": "range",
    "elevation": "elevation_angle",
    "elevations": "elevation_angle",
    "times": "time",
}

#: Columns that carry a timestamp — selection on these accepts partial strings.
_TIME_COLUMNS = frozenset({"time", "volume_time"})


def _is_time_dtype(dtype) -> bool:
    return dtype in (pl.Datetime, pl.Date) or isinstance(dtype, (pl.Datetime, pl.Date))


def _time_bound(value, dtype, *, upper: bool):
    """Coerce ``value`` to a timestamp comparable with a column of ``dtype``.

    A *partial* string expands to the edge of the period it names, matching
    pandas/xarray partial-string indexing: ``"2022-01"`` becomes
    ``2022-01-01 00:00:00`` as a lower bound and ``2022-01-31 23:59:59.999…``
    as an upper bound.  The result's tz-awareness is matched to the column.
    """
    if isinstance(value, (datetime.datetime, datetime.date, np.datetime64, pd.Timestamp)):
        ts = pd.Timestamp(value)
    else:
        s = str(value).strip()
        if upper:
            try:
                ts = pd.Period(s).end_time
            except Exception:
                ts = pd.Timestamp(s)
        else:
            try:
                ts = pd.Period(s).start_time
            except Exception:
                ts = pd.Timestamp(s)

    tz = getattr(dtype, "time_zone", None)
    if tz:
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert(tz)
    elif ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _sel_expr(name: str, value, dtype) -> pl.Expr:
    """Build the boolean expression selecting ``value`` on column ``name``.

    ``value`` may be a ``slice`` (inclusive on both ends, as in xarray), a
    list/tuple/set (membership), or a scalar (equality — or the enclosing period
    for a timestamp column, so ``time="2024-08-26 02:46:08"`` still matches a
    row stored as ``02:46:08.050``).
    """
    col = pl.col(name)
    is_time = _is_time_dtype(dtype)

    def bound(v, upper):
        return _time_bound(v, dtype, upper=upper) if is_time else v

    if isinstance(value, slice):
        if value.step is not None:
            raise ValueError(
                f"sel({name}=...): a step is not supported (got step={value.step!r}); "
                "use a plain slice(start, stop).",
            )
        parts = []
        if value.start is not None:
            parts.append(col >= bound(value.start, False))
        if value.stop is not None:
            parts.append(col <= bound(value.stop, True))
        if not parts:
            return pl.lit(True)
        expr = parts[0]
        for p in parts[1:]:
            expr = expr & p
        return expr

    if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pl.Series)):
        vals = list(value.to_list() if isinstance(value, pl.Series) else list(value))
        if is_time:
            # a list of timestamps/partial strings -> union of their periods
            expr = None
            for v in vals:
                one = (col >= bound(v, False)) & (col <= bound(v, True))
                expr = one if expr is None else (expr | one)
            return pl.lit(False) if expr is None else expr
        return col.is_in(vals)

    if is_time:
        return (col >= bound(value, False)) & (col <= bound(value, True))
    return col == value


def _ccw_polygons(polys: np.ndarray) -> np.ndarray:
    """Force counter-clockwise exterior rings, as GeoParquet/GeoArrow prefer.

    The gate corner order is deterministically clockwise (inherited from the
    reference prototype), so serialized output needs flipping.
    """
    if hasattr(shapely, "orient_polygons"):  # shapely >= 2.1
        return shapely.orient_polygons(polys)
    ccw = shapely.is_ccw(shapely.get_exterior_ring(polys))
    return np.where(ccw, polys, shapely.reverse(polys))


_FILTER_KEYS = ("var", "logic", "threshold")


def _resolve_filters(filters) -> list[tuple[str, str, float]]:
    """Normalize a filter dict / list-of-dicts to ``[(var, logic, threshold), ...]``.

    Unknown keys are rejected rather than ignored: ``threshold`` defaults to 0,
    so a misspelled one (``{"var": "DBZH", "logic": ">", "value": 45}``) would
    otherwise silently become ``DBZH > 0`` — a filter that keeps everything and
    looks like it ran.
    """
    if filters is None:
        return []
    if isinstance(filters, dict):
        filters = [filters]
    specs = []
    for f in filters:
        extra = [k for k in f if k not in _FILTER_KEYS]
        if extra:
            raise KeyError(
                f"unknown filter key(s) {sorted(extra)} in {f!r}; "
                f"a filter is {{{', '.join(repr(k) for k in _FILTER_KEYS)}}}.",
            )
        specs.append((f["var"], f.get("logic", ">"), f.get("threshold", 0.0)))
    return specs


def _normalize_time_period(time_period):
    """Return ``(start, end)`` datetimes from str | datetime | (start, end)."""

    def _u(x):
        return ensure_utc(x) if x is not None else None

    if time_period is None:
        return None, None
    if isinstance(time_period, (tuple, list)):
        if len(time_period) == 1:
            return _u(time_period[0]), None
        return _u(time_period[0]), _u(time_period[1])
    return _u(time_period), None


def _encode_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """WKB-encode shapely-geometry columns so the frame can enter polars."""
    for c in _GEOM_COLS:
        if c in df.columns and len(df):
            sample = df[c].dropna()
            if len(sample) and hasattr(sample.iloc[0], "geom_type"):
                df = df.copy()
                df[c] = [None if g is None else shapely.to_wkb(g) for g in df[c]]
    return df


def _decode_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Decode WKB-encoded geometry columns back to shapely objects."""
    for c in _GEOM_COLS:
        if c in df.columns and len(df):
            sample = df[c].dropna()
            if len(sample) and isinstance(sample.iloc[0], (bytes, bytearray)):
                df[c] = [None if b is None else shapely.from_wkb(b) for b in df[c]]
    return df


def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Pandas -> polars, WKB-encoding any shapely-geometry columns first."""
    return pl.from_pandas(_encode_geometry(df))


def _radar_from_filename(path) -> str:
    """Infer the radar id from a saved DataTree filename (``<RADAR>_...``)."""
    return Path(path).stem.split("_")[0]


def _format_size(n_bytes: float) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:,.1f} {unit}" if unit != "B" else f"{n_bytes:,.0f} B"
        n_bytes /= 1024
    return f"{n_bytes:,.1f} TB"


def _path_size(path: Path) -> int:
    """Size of a file, or the total size of a directory (e.g. a ``.zarr`` store)."""
    # ponytail: stats every file; fine for archives up to ~1e5 volumes, add a
    # `size=False` switch if scanning ever gets slow.
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _time_span(times: list) -> str:
    """``'first .. last'`` for a list of timestamps (``'unknown'`` when empty)."""
    known = sorted(t for t in times if t is not None)
    if not known:
        return "unknown (no timestamp in the filenames)"
    if len(known) == 1 or known[0] == known[-1]:
        return f"{known[0]:%Y-%m-%d %H:%M:%S}"
    return f"{known[0]:%Y-%m-%d %H:%M:%S} .. {known[-1]:%Y-%m-%d %H:%M:%S}"


def _print_daily_breakdown(times: list, indent: str = "      ") -> None:
    """One line per day: number of volumes and the first/last time of that day."""
    per_day = defaultdict(list)
    for t in times:
        if t is not None:
            per_day[t.date()].append(t)
    for day in sorted(per_day):
        ts = sorted(per_day[day])
        print(f"{indent}{day}  {len(ts):>5} volume(s)  {ts[0]:%H:%M:%S} .. {ts[-1]:%H:%M:%S}")
    n_undated = sum(1 for t in times if t is None)
    if n_undated:
        print(f"{indent}(no parseable timestamp)  {n_undated:>5} file(s)")


def _list_archive_radars(archive_dir: Path) -> list[str]:
    """Radar subdirectories that hold archived data.

    A directory counts as a radar when its name is a usable radar name *and* it
    holds a ``LUT/`` subdirectory — the name test alone would also match
    scratch directories, and RadDB cannot read a radar without its LUT anyway.
    """
    if not archive_dir.exists():
        return []
    return sorted(
        p.name for p in archive_dir.iterdir() if p.is_dir() and is_valid_radar_name(p.name) and (p / "LUT").is_dir()
    )


class RadDB:
    """Generic radar-data archiving and analysis interface (see module docstring).

    Examples
    --------
    >>> db = RadDB(archive_dir="/data/raddb", crs=2056)
    >>> db.archive(datatree_dir="/data/FMI_datatree")  # or datatree=dt
    >>> rdf = db.open(time_period=("2024-08-26", "2024-08-27"))
    >>> rdf.filter({"var": "DBZH", "logic": ">", "threshold": 20}).crop_by_bbox(extent=rdf.extent()).plot_ppi(
    ...     variable="DBZH", save="ppi.png"
    ... )
    """

    # ================================================================
    # Construction & state
    # ================================================================

    def __init__(
        self,
        archive_dir: str | None = None,
        crs: int | str | None = None,
        network: str = "",
        *,
        _data: pl.DataFrame | None = None,
        _meta: dict | None = None,
    ):
        """Create an archive-bound RadDB.

        Parameters
        ----------
        archive_dir : str, optional
            Base directory of the RadDB archive (LUT + POL parquet under
            ``{archive_dir}/{radar}/``).  Shared default for every task.
        crs : int, str or CRS-coercible, optional
            Projection for the LUT (e.g. ``2056`` for CH1903+/LV95).  Used both
            when generating LUTs during :meth:`archive` and to resolve projected
            coordinates (:meth:`extent`, :meth:`to_geopandas`).
        network : str, optional
            Network label stored in generated LUT metadata.

        The private ``_data`` / ``_meta`` arguments carry the polars backend of
        a data-carrying instance and are not meant to be passed directly; use
        :meth:`open`.
        """
        self.archive_dir = Path(archive_dir) if archive_dir is not None else None
        self._crs = crs
        self.network = network
        self._data = _data
        self._meta = dict(_meta) if _meta else {}

    def _derive(self, data: pl.DataFrame, *, archive_dir=None, crs=None, **meta) -> RadDB:
        """Build a new data-carrying ``RadDB`` sharing this one's configuration."""
        return RadDB(
            archive_dir=(
                str(archive_dir) if archive_dir is not None else (str(self.archive_dir) if self.archive_dir else None)
            ),
            crs=crs if crs is not None else self._crs,
            network=self.network,
            _data=data,
            _meta={**self._meta, **meta},
        )

    def _require_archive_dir(self) -> Path:
        if self.archive_dir is None:
            raise ValueError(
                "This RadDB has no archive_dir; pass it to RadDB(archive_dir=...) " "or to the method call.",
            )
        return self.archive_dir

    def _require_data(self) -> pl.DataFrame:
        if self._data is None:
            raise ValueError(
                "This RadDB carries no data (it is archive-bound). Load data " "first with db.open(...).",
            )
        return self._data

    @property
    def data(self) -> pl.DataFrame:
        """The loaded data as a polars DataFrame."""
        return self._require_data()

    def __len__(self) -> int:
        """Number of gates held; ``0`` when archive-bound with no data loaded."""
        return 0 if self._data is None else self._data.height

    def __repr__(self) -> str:
        """One-line summary when archive-bound, a multi-line one when data-carrying."""
        if self._data is None:
            return f"RadDB(archive_dir={self.archive_dir!s}, crs={self._crs!r}) " f"[archive-bound, no data loaded]"
        lines = [f"RadDB [{len(self):,} gates]"]
        with contextlib.suppress(Exception):
            lines.append(f"  radars     : {self.radars()}")
        try:
            t0, t1 = self.start_time(), self.end_time()
            lines.append(f"  time range : {t0} .. {t1}")
        except Exception:
            pass
        schema = self._data.schema
        cols = ", ".join(f"{n}:{t!s}" for n, t in list(schema.items())[:12])
        more = "" if len(schema) <= 12 else f" (+{len(schema) - 12} more)"
        lines.append(f"  columns    : {cols}{more}")
        lines.append(f"  archive_dir: {self.archive_dir}")
        return "\n".join(lines)

    __str__ = __repr__

    # ================================================================
    # ARCHIVE
    # ================================================================

    def archive(
        self,
        datatree_dir: str | None = None,
        datatree=None,
        archive_dir: str | None = None,
        crs: int | str | None = None,
        radar: str | list[str] | None = None,
        filter: dict | None = None,
        time_period=None,
        variables: list[str] | None = None,
    ) -> dict:
        """Archive DataTree volumes to the RadDB Parquet store.

        Exactly one input source must be given:

        * ``datatree`` — an in-memory ``xr.DataTree``, a list of them, a
          ``{label: DataTree}`` dict (one radar), or a ``{radar: [DataTree, ...]}``
          dict (several radars).
        * ``datatree_dir`` — a directory of **saved** DataTree files
          (``.zarr`` / ``.nc``); volumes are grouped by radar from the filename.

        Passing both raises ``ValueError``.

        Parameters
        ----------
        datatree_dir : str, optional
            Directory of saved DataTree files to archive.
        datatree : xarray.DataTree, list, or dict, optional
            In-memory volume(s) to archive.
        archive_dir, crs : optional
            Override the instance defaults for this call.
        radar : str or list of str, optional
            Restrict / assign the radar(s).  For ``datatree_dir`` with
            ``radar=None`` (default) the radar is inferred per file from the
            filename and **all** radars are archived.  For a single in-memory
            ``DataTree`` / list, ``radar`` is required.
        filter : dict, optional
            Gate filter ``{"var", "logic", "threshold"}`` defining which gates
            to keep.  Default: ``{"var": "DBZH", "logic": ">", "threshold": 0}``
            (drops no-echo gates).
        time_period : str, datetime or (start, end), optional
            For ``datatree_dir``: keep only files whose filename timestamp falls
            in the period.
        variables : list of str, optional
            Which moments to archive.  ``None`` (default) archives every **per-gate**
            variable each volume carries — anything measured on both the azimuth and
            the range dimension — so nothing a network records is silently dropped.
            The per-ray and scalar metadata a sweep also holds (``sweep_mode``,
            ``prt_mode``, ``nyquist_velocity``, ...) is never archived: it describes
            the scan, not the weather.  Pass a list to keep an archive lean, e.g.
            ``variables=["DBZH", "ZDR", "RHOHV"]``.

        Returns
        -------
        dict
            ``{"n_archived": int, "n_failed": int, "n_skipped": int,
            "radars": [...]}``.  The three counts sum to the number of volumes
            attempted: ``n_skipped`` covers volumes that held nothing to
            archive (every gate failed the filter, or an all-``NaT`` time),
            which is neither a success nor a failure.
        """
        archive_dir = Path(archive_dir) if archive_dir is not None else self.archive_dir
        if archive_dir is None:
            raise ValueError("archive_dir must be given (via RadDB(...) or this call).")
        crs = crs if crs is not None else self._crs
        if crs is None:
            # Checked up front: archive() reports per-volume failures rather than
            # raising, so without this a missing CRS would read as "0 archived"
            # instead of saying what is actually wrong.
            raise ValueError(
                "archiving requires a CRS — the LUT stores projected gate "
                "coordinates, and crops and cross-sections are computed in them. "
                "There is no default because a wrong one is silently wrong: "
                "EPSG:2056 outside Switzerland mis-measures distance by ~20%. "
                "Pass RadDB(crs=<epsg>) or archive(crs=<epsg>), choosing one valid "
                "at your radar's site (raddb.lut.suggest_crs(lon, lat) gives the "
                "UTM zone).",
            )

        if (datatree is None) == (datatree_dir is None):
            raise ValueError(
                "Pass exactly one of `datatree` (in-memory) or `datatree_dir` " "(saved files).",
            )

        if filter is None:
            feat, logic, thr = "DBZH", ">", 0.0
        else:
            feat = filter.get("var", "DBZH")
            logic = filter.get("logic", ">")
            thr = filter.get("threshold", 0.0)

        t0 = time.time()
        if datatree is not None:
            radars_done, n_ok, n_fail, n_skip = self._archive_in_memory(
                datatree,
                radar,
                archive_dir,
                crs,
                feat,
                logic,
                thr,
                variables,
            )
        else:
            radars_done, n_ok, n_fail, n_skip = self._archive_from_disk(
                datatree_dir,
                radar,
                archive_dir,
                crs,
                feat,
                logic,
                thr,
                time_period,
                variables,
            )

        print("=" * 70)
        print("RadDB archive")
        print(f"  archive_dir : {archive_dir}")
        print(f"  crs         : {crs}")
        print(f"  radars      : {radars_done}")
        print(f"  filter      : keep {feat} {logic} {thr}")
        print(
            f"  volumes     : {n_ok} archived, {n_fail} failed"
            + (f", {n_skip} skipped (nothing to archive)" if n_skip else ""),
        )
        print(f"  elapsed     : {_format_elapsed_time(time.time() - t0)}")
        print("=" * 70)
        return {
            "n_archived": n_ok,
            "n_failed": n_fail,
            "n_skipped": n_skip,
            "radars": radars_done,
        }

    def _ensure_lut(self, radar: str, sample_dt, archive_dir: Path, crs) -> None:
        """Generate the per-radar LUT from a sample volume if it does not exist."""
        lut_path = archive_dir / radar / "LUT" / f"{radar}_LUT.parquet"
        if lut_path.exists():
            return
        try:
            generate_lut_from_datatree(
                dt=sample_dt,
                radar=radar,
                output_base_path=str(archive_dir),
                network=self.network,
                projection_epsg=crs if isinstance(crs, int) else None,
                projection_crs=None if isinstance(crs, int) else crs,
            )
        except ValueError:
            # A rejected CRS is fatal, not a per-volume hiccup: continuing would
            # write POL files against a LUT that does not exist or is wrong, and
            # report "archived" for data no crop or section could ever use.
            raise
        except Exception as e:
            print(f"  [{radar}] LUT generation failed: {e}")

    def _archive_in_memory(self, datatree, radar, archive_dir, crs, feat, logic, thr, variables=None):
        archive_dir = Path(archive_dir)
        # {radar: [DataTree, ...]} -- multi-radar
        if isinstance(datatree, dict) and datatree and all(not isinstance(v, xr.DataTree) for v in datatree.values()):
            for r, vols in datatree.items():
                rn = normalize_radar_name(r)
                first = next(iter(vols.values())) if isinstance(vols, dict) else vols[0]
                self._ensure_lut(rn, first, archive_dir, crs)
            results = archive_volumes_multi_radar(
                volumes_by_radar=datatree,
                base_output_path=str(archive_dir),
                filter_feature=feat,
                filter_threshold=thr,
                filter_logic=logic,
                verbose=False,
                variables=variables,
            )
            # Count outcomes, not attempts: archive_multiple_volumes reports a
            # failed volume as a record with success=False, and calling every
            # record an archive is how a broken volume gets announced as stored.
            flat = [r for v in results.values() for r in v]
            n_ok = sum(1 for r in flat if r.get("success"))
            n_skip = sum(1 for r in flat if r.get("skipped"))
            return list(results.keys()), n_ok, len(flat) - n_ok - n_skip, n_skip

        if radar is None or not isinstance(radar, str):
            raise ValueError(
                "For in-memory archiving, pass a single radar letter, e.g. " "archive(datatree=dt, radar='A').",
            )
        r = normalize_radar_name(radar)
        if isinstance(datatree, xr.DataTree):
            self._ensure_lut(r, datatree, archive_dir, crs)
            path = archive_volume(
                dt=datatree,
                radar=r,
                base_output_path=str(archive_dir),
                filter_feature=feat,
                filter_threshold=thr,
                filter_logic=logic,
                variables=variables,
            )
            # A None path means the volume held nothing to archive; counting it
            # as archived is how an empty volume gets reported as stored.
            return ([r], 0, 0, 1) if path is None else ([r], 1, 0, 0)
        # list or {label: DataTree}
        first = next(iter(datatree.values())) if isinstance(datatree, dict) else datatree[0]
        self._ensure_lut(r, first, archive_dir, crs)
        results = archive_multiple_volumes(
            volumes=datatree,
            radar=r,
            base_output_path=str(archive_dir),
            filter_feature=feat,
            filter_threshold=thr,
            filter_logic=logic,
            verbose=False,
            variables=variables,
        )
        n_ok = sum(1 for res in results if res.get("success"))
        n_skip = sum(1 for res in results if res.get("skipped"))
        return [r], n_ok, len(results) - n_ok - n_skip, n_skip

    def _archive_from_disk(self, datatree_dir, radar, archive_dir, crs, feat, logic, thr, time_period, variables=None):
        archive_dir = Path(archive_dir)
        start, end = _normalize_time_period(time_period)
        files = find_datatree_files(
            Path(datatree_dir),
            recursive=True,
            start_time=start,
            end_time=end,
        )
        if isinstance(radar, str):
            # A single radar name: archive every file as that radar (the
            # filename need not encode the radar, e.g. generic ``vol_*`` files).
            by_radar: dict[str, list] = {normalize_radar_name(radar): list(files)}
        else:
            # Infer the radar per file from its filename (``<RADAR>_...``).
            # Group on the canonical name so that case and the legacy ``ML*``
            # spelling collapse together; an unusable prefix is kept verbatim
            # so the per-radar loop can report and skip it.
            by_radar = defaultdict(list)
            for f in files:
                prefix = _radar_from_filename(f)
                key = normalize_radar_name(prefix) if is_valid_radar_name(prefix) else prefix
                by_radar[key].append(f)
            if radar is not None:  # a list of radars -> keep only those
                wanted = {normalize_radar_name(x) for x in radar}
                by_radar = {r: fs for r, fs in by_radar.items() if r in wanted}

        radars_done, total_ok, total_fail, total_skip = [], 0, 0, 0
        for r, rfiles in sorted(by_radar.items()):
            n_ok, n_fail, n_skip = self._archive_files_one_radar(
                r,
                sorted(rfiles),
                archive_dir,
                crs,
                feat,
                logic,
                thr,
                variables,
            )
            radars_done.append(r)
            total_ok += n_ok
            total_fail += n_fail
            total_skip += n_skip
        return radars_done, total_ok, total_fail, total_skip

    def _archive_files_one_radar(self, radar, files, archive_dir, crs, feat, logic, thr, variables=None):
        """Archive every saved DataTree file for one radar (LUT autogen, resume)."""
        try:
            radar = normalize_radar_name(radar)
        except ValueError as exc:
            print(f"  [skip] {exc} Skipping {len(files)} file(s).")
            return (0, 0, len(files))
        archive_dir.mkdir(parents=True, exist_ok=True)
        ckpt = archive_dir / f"_archive_checkpoint_datatrees_{radar}.txt"
        seen = _load_checkpoint(ckpt)

        preopened: dict = {}
        lut_path = archive_dir / radar / "LUT" / f"{radar}_LUT.parquet"
        if not lut_path.exists() and files:
            try:
                dt0 = open_any_datatree(files[0])
            except Exception as e:
                print(f"  [{radar}] LUT generation failed: {e}")
            else:
                preopened[files[0]] = dt0
                # Not wrapped: _ensure_lut re-raises a rejected CRS deliberately, and
                # catching it here would write POL files against a LUT that was never
                # generated and still report them as archived.
                self._ensure_lut(radar, dt0, archive_dir, crs)

        n_ok = n_fail = n_skip = 0
        for f in files:
            stem = Path(f).stem
            key = f"{radar}:{stem}"
            if key in seen:
                preopened.pop(f, None)
                continue
            try:
                dt = preopened.pop(f, None)
                if dt is None:
                    dt = open_any_datatree(f)
                path = archive_volume(
                    dt=dt,
                    radar=radar,
                    base_output_path=str(archive_dir),
                    filter_feature=feat,
                    filter_threshold=thr,
                    filter_logic=logic,
                    volume=stem,
                    variables=variables,
                )
                # Checkpoint either way: a volume with nothing to archive is
                # settled, and re-reading it on resume would only skip again.
                _append_checkpoint(ckpt, key)
                seen.add(key)
                if path is None:
                    n_skip += 1
                    print(f"  [{radar}] SKIP {stem}: no gates to archive")
                else:
                    n_ok += 1
                del dt
            except Exception as e:
                n_fail += 1
                print(f"  [{radar}] FAIL {stem}: {e}")
        return (n_ok, n_fail, n_skip)

    # ---- LUT read accessors (archive-bound) ----

    def get_lut(self, radar: str) -> pl.DataFrame:
        """Load the LUT (static gate geometry) for a radar, as polars."""
        return load_radar_lut(normalize_radar_name(radar), self._require_archive_dir())

    def get_radar_info(self, radar: str) -> dict:
        """Load radar metadata (location, sweep geometry)."""
        return load_radar_info(normalize_radar_name(radar), self._require_archive_dir())

    def add_lut_projection(self, radar: str, epsg: int | None = None, crs=None) -> pl.DataFrame:
        """Return the radar LUT enriched with projected ``x_{epsg}`` / ``y_{epsg}`` columns."""
        lut_df = load_radar_lut(normalize_radar_name(radar), self._require_archive_dir())
        return add_lut_projection(lut_df, epsg=epsg, crs=crs)

    def get_h_plane(
        self,
        radar: str,
        sweep: int | None = None,
        per_gate: bool = False,
    ) -> pl.DataFrame:
        """Horizontal-face geometry of each gate — the precise PPI footprint.

        ``per_gate=False`` (default) returns the compact **node lattice** as
        stored: ``sweep, az_idx, rng_idx, x, y`` (+ ``x_<epsg>, y_<epsg>`` when
        the LUT was generated with a projection).  Neighboring gates share
        nodes, which is why the file is ~4x smaller than per-gate corners.

        ``per_gate=True`` expands it to **4 corners per gate**, keyed by
        ``gate_id``: ``x_1..x_4``, ``y_1..y_4`` in ring order.  Feed straight to
        ``matplotlib.collections.PolyCollection`` or ``shapely.polygons``.

        Note the first range bin is degenerate: its inner edge falls at the radar
        (range_start ~= dR), so its footprint is a triangle rather than a
        trapezoid.
        """
        radar = normalize_radar_name(radar)
        base = self._require_archive_dir()
        if per_gate:
            return gate_corner_table(radar, base, kind="h_plane", sweep=sweep)
        return load_plane_nodes(radar, base, "h_plane", sweep=sweep)

    def get_v_plane(
        self,
        radar: str,
        sweep: int | None = None,
        azimuth: float | None = None,
        per_gate: bool = False,
    ) -> pl.DataFrame:
        """Vertical-face geometry of each gate — the precise RHI footprint.

        Coordinates are ``(d, z)``: ``d`` is the ground distance from the radar
        [m] and altitude is given **both ways** — ``z_asl`` (absolute, m above sea
        level) and ``z_rel`` (relative to the radar).  They differ by the site
        altitude in ``{radar}_info.yaml``.

        ``per_gate=True`` returns 4 corners per gate ordered
        (near-bottom, far-bottom, far-top, near-top): ``d_1..d_4``,
        ``z_asl_1..z_asl_4``, ``z_rel_1..z_rel_4``.

        ``azimuth`` keeps only the ray nearest that azimuth (requires
        ``per_gate=True``), which is exactly the slice an RHI plots.
        """
        radar = normalize_radar_name(radar)
        base = self._require_archive_dir()
        if not per_gate:
            if azimuth is not None:
                raise ValueError("azimuth= selection requires per_gate=True.")
            return load_plane_nodes(radar, base, "v_plane", sweep=sweep)

        tbl = gate_corner_table(radar, base, kind="v_plane", sweep=sweep)
        if azimuth is None:
            return tbl
        # Pick the nearest stored ray, comparing on the circle.
        lut = load_radar_lut(radar, base).select(["gate_id", "azimuth"])
        az = lut["azimuth"].to_numpy()
        target = float(azimuth) % 360.0
        diff = np.abs((az - target + 180.0) % 360.0 - 180.0)
        nearest = float(az[np.argmin(diff)])
        keep = lut.filter(pl.col("azimuth") == nearest).select("gate_id")
        return tbl.join(keep, on="gate_id", how="semi")

    def get_corners(
        self,
        radar: str,
        sweep: int | None = None,
        per_gate: bool = False,
    ) -> pl.DataFrame:
        """Full 3-D gate corners — 8 per gate, for volume reconstruction.

        ``per_gate=True`` returns ``x_1..x_8``, ``y_1..y_8``, ``z_rel_1..z_rel_8``
        where **1-4 are the near face** (towards the radar) and **5-8 the far
        face**; within a face the order is (az-, el-), (az+, el-), (az+, el+),
        (az-, el+).

        Because the beam's angular extent grows with range, the far face is
        strictly larger than the near face — a useful invariant to assert.  The
        one exception is the first range bin, whose near face collapses onto the
        radar itself.

        Add the site altitude from :meth:`get_radar_info` to ``z_rel`` for meters
        above sea level.
        """
        radar = normalize_radar_name(radar)
        base = self._require_archive_dir()
        if per_gate:
            return gate_corner_table(radar, base, kind="corners", sweep=sweep)
        return load_plane_nodes(radar, base, "corners", sweep=sweep)

    def export_h_plane_geoparquet(
        self,
        radar: str,
        path: str | Path,
        sweep: int | None = None,
        epsg: int | None = None,
    ) -> str:
        """Write the horizontal gate footprints as **GeoParquet** (CRS embedded).

        The archive keeps plain parquet — small and uniform.  This is the opt-in
        interoperability path: the output opens directly in QGIS or
        ``geopandas.read_parquet``.

        ``epsg`` selects the output CRS: the LUT's projected columns when they
        exist and match, otherwise WGS-84 (4326) derived from the radar-relative
        ``x``/``y``.  Exterior rings are normalized counter-clockwise, as the
        GeoParquet spec prefers.
        """
        import geopandas as gpd

        radar = normalize_radar_name(radar)
        base = self._require_archive_dir()
        tbl = gate_corner_table(radar, base, kind="h_plane", sweep=sweep)

        if epsg is None:
            epsg = int(self._crs) if isinstance(self._crs, int) else None

        xs = [f"x_{epsg}_{k}" for k in range(1, 5)] if epsg else []
        if xs and all(c in tbl.columns for c in xs):
            xcols = xs
            ycols = [f"y_{epsg}_{k}" for k in range(1, 5)]
            out_crs = f"EPSG:{epsg}"
        else:
            # Fall back to WGS-84 from the radar-relative meters.
            info = load_radar_info(radar, base)
            ring = np.stack(
                [np.stack([tbl[f"x_{k}"].to_numpy(), tbl[f"y_{k}"].to_numpy()], axis=1) for k in range(1, 5)],
                axis=1,
            )
            lat, lon, _ = cartesian_to_geographic(
                ring[:, :, 0],
                ring[:, :, 1],
                np.zeros(ring.shape[:2]),
                info["latitude"],
                info["longitude"],
                info["altitude"],
            )
            ring = np.concatenate([np.stack([lon, lat], axis=2), np.stack([lon[:, :1], lat[:, :1]], axis=2)], axis=1)
            gdf = gpd.GeoDataFrame(
                {"gate_id": tbl["gate_id"].to_numpy(), "sweep": tbl["sweep"].to_numpy()},
                geometry=_ccw_polygons(shapely.polygons(ring)),
                crs="EPSG:4326",
            )
            gdf.to_parquet(path)
            return str(path)

        ring = np.stack(
            [np.stack([tbl[xc].to_numpy(), tbl[yc].to_numpy()], axis=1) for xc, yc in zip(xcols, ycols, strict=False)],
            axis=1,
        )
        ring = np.concatenate([ring, ring[:, :1, :]], axis=1)  # close the ring
        gdf = gpd.GeoDataFrame(
            {"gate_id": tbl["gate_id"].to_numpy(), "sweep": tbl["sweep"].to_numpy()},
            geometry=_ccw_polygons(shapely.polygons(ring)),
            crs=out_crs,
        )
        gdf.to_parquet(path)
        return str(path)

    def list_radars(self) -> list[str]:
        """List radar identifiers that have data in the archive."""
        return _list_archive_radars(self._require_archive_dir())

    # ---- what is on disk? ----

    def inventory(
        self,
        datatree_dir: str | None = None,
        detailed: bool = False,
        archive_dir: str | None = None,
    ) -> None:
        """Print what data is available on disk — which radars, which time periods.

        Answers "what can I analyze?" before :meth:`open` (archive side) or
        :meth:`archive` (input side).  Prints only; nothing is loaded into memory.

        Parameters
        ----------
        datatree_dir : str, optional
            Scan a directory of **saved DataTree files** (``.zarr`` / ``.nc``) —
            i.e. data not archived yet.  Radar and time come from the filename
            (``<RADAR>_<YYYYMMDD>_<HHMMSS>``).  When omitted, the **RadDB
            archive** is scanned instead.
        detailed : bool
            Add a per-radar, per-day breakdown; for the archive also the LUT size,
            the sweep count and the moment columns stored per volume.
        archive_dir : str, optional
            Override the archive location for this call.

        Examples
        --------
        >>> db.inventory()  # what is archived
        >>> db.inventory(detailed=True)  # ... day by day
        >>> db.inventory(datatree_dir="/data/FMI_datatree")  # what could be archived
        """
        if datatree_dir is not None:
            self._inventory_datatrees(Path(datatree_dir), detailed)
        else:
            base = Path(archive_dir) if archive_dir is not None else self._require_archive_dir()
            self._inventory_archive(base, detailed)

    def _inventory_datatrees(self, directory: Path, detailed: bool) -> None:
        """Inventory of saved DataTree files (the input side of the archive)."""
        files = find_datatree_files(directory, recursive=True)
        print("=" * 78)
        print("RadDB inventory — DataTree files on disk (not archived yet)")
        print(f"  directory : {directory}")
        if not files:
            print("  no .zarr / .nc / .nc4 / .cdf files found")
            print("=" * 78)
            return

        by_radar: dict[str, list] = defaultdict(list)
        for f in files:
            by_radar[_radar_from_filename(f)].append(f)

        times_all = [_parse_datatree_file_time(f) for f in files]
        print(f"  files     : {len(files)}")
        print(f"  radars    : {', '.join(sorted(by_radar))}  (from the filename prefix)")
        print(f"  time range: {_time_span(times_all)}")
        print("-" * 78)
        print(f"  {'radar':<7}{'files':>8}  {'time range':<45}{'size':>12}")
        for r in sorted(by_radar):
            rfiles = by_radar[r]
            times = [_parse_datatree_file_time(f) for f in rfiles]
            size = sum(_path_size(f) for f in rfiles)
            print(f"  {r:<7}{len(rfiles):>8}  {_time_span(times):<45}{_format_size(size):>12}")
            if detailed:
                _print_daily_breakdown(times)
                if not is_valid_radar_name(r):
                    print(
                        f"      [!] {r!r} is not a usable radar name (1-{RADAR_CODE_LEN} "
                        f"characters from [0-9A-Z]) — archive() would skip it unless you "
                        f"pass radar='<name>'",
                    )
        print("-" * 78)
        print(f"  archive with: db.archive(datatree_dir={str(directory)!r})")
        print("=" * 78)

    def _inventory_archive(self, base: Path, detailed: bool) -> None:
        """Inventory of an existing RadDB archive (the output side)."""
        print("=" * 78)
        print("RadDB inventory — archived data")
        print(f"  archive_dir : {base}")
        radars = _list_archive_radars(base)
        if not radars:
            print("  no radar directories found — nothing archived here yet")
            print("=" * 78)
            return

        per_radar = {}
        for r in radars:
            pol = _find_polar_files_in_range(base / r)
            per_radar[r] = (pol, [_parse_pol_time(f) for f in pol])

        n_vol = sum(len(v[0]) for v in per_radar.values())
        all_times = [t for v in per_radar.values() for t in v[1]]
        print(f"  radars      : {', '.join(radars)}")
        print(f"  volumes     : {n_vol}")
        print(f"  time range  : {_time_span(all_times)}")
        print("-" * 78)
        print(f"  {'radar':<7}{'volumes':>8}  {'time range':<45}{'size':>12}")
        for r in radars:
            pol, times = per_radar[r]
            size = sum(_path_size(f) for f in pol)
            print(f"  {r:<7}{len(pol):>8}  {_time_span(times):<45}{_format_size(size):>12}")
            if not detailed:
                continue
            lut_path = base / r / "LUT" / f"{r}_LUT.parquet"
            if lut_path.exists():
                try:
                    info = load_radar_info(r, base)
                    print(
                        f"      LUT: {_format_size(_path_size(lut_path))}, "
                        f"{len(info.get('sweeps', {}))} sweeps, site "
                        f"({info.get('latitude'):.4f}, {info.get('longitude'):.4f}) "
                        f"at {info.get('altitude'):.0f} m",
                    )
                except Exception as e:
                    print(f"      LUT: present, metadata unreadable ({e})")
            else:
                print("      LUT: MISSING — open() will have no geometry for this radar")
            if pol:
                try:
                    cols = pl.read_parquet_schema(pol[0]).keys()
                    print(f"      columns: {', '.join(c for c in cols if c != 'gate_id')}")
                except Exception as e:
                    print(f"      columns: unreadable ({e})")
            _print_daily_breakdown(times)
        print("-" * 78)
        print("  load with   : db.open(radars=..., time_period=(start, end))")
        print("=" * 78)

    # ================================================================
    # OPEN & FILTER
    # ================================================================

    def open(
        self,
        time_period=None,
        radars: str | list[str] | None = None,
        columns: list[str] | None = None,
        filters=None,
        archive_dir: str | None = None,
    ) -> RadDB:
        """Load archived data into a data-carrying ``RadDB``.

        Parameters
        ----------
        time_period : str, datetime or (start, end), optional
            Time range to load.  ``None`` loads everything available.
        radars : str or list of str, optional
            Radar(s) to load.  ``None`` (default) loads all radars in the archive.
        columns : list of str, optional
            Subset of moment columns to load (``gate_id`` is always included).
        filters : dict or list of dict, optional
            Gate filter(s) ``{"var", "logic", "threshold"}`` applied after load
            (a list is combined with AND).
        archive_dir : str, optional
            Override the archive location for this call.

        Returns
        -------
        RadDB
            A data-carrying instance (``rdf``).
        """
        archive_dir = Path(archive_dir) if archive_dir is not None else self._require_archive_dir()
        start, end = _normalize_time_period(time_period)

        if radars is None:
            radars = _list_archive_radars(archive_dir)
        elif isinstance(radars, str):
            radars = [radars]

        if columns is not None and "gate_id" not in columns:
            columns = ["gate_id", *columns]

        scans = []
        for r in radars:
            lf = scan_polar_parquet(
                radar=normalize_radar_name(r),
                base_path=archive_dir,
                start_time=start,
                end_time=end,
                columns=columns,
            )
            if lf is not None:
                scans.append(lf)

        # Filters are applied to the plan, so polars only materializes the rows
        # that survive them.
        if scans:
            # Diagonal: two radars in one archive need not record the same moments,
            # and the same moment set can be listed in a different column order —
            # see the concat in `io_core._scan_polar_files`.
            plan = pl.concat(scans, how="diagonal_relaxed")
            for var, logic, thr in _resolve_filters(filters):
                plan = plan.filter(_filter_expr(var, logic, thr))
            data = plan.collect()
        else:
            data = pl.DataFrame()
        return self._derive(data, archive_dir=archive_dir)

    def filter(self, filters) -> RadDB:
        """Keep only gates satisfying ``filters`` (row removal); returns a new RadDB.

        ``filters`` is a dict ``{"var", "logic", "threshold"}`` or a list of such
        dicts (combined with AND).  Vertical subsetting works the same way on an
        ``altitude`` column, e.g.
        ``rdf.filter([{"var": "altitude", "logic": ">", "threshold": 2000},
        {"var": "altitude", "logic": "<", "threshold": 4000}])``.

        Filtering on a static column (``altitude``, ``sweep``, ``latitude`` …)
        joins it from the LUT just long enough to evaluate the predicate and
        drops it again, so the returned frame still carries dynamic values only.
        """
        data = self._require_data()
        specs = _resolve_filters(filters)

        borrowed = []
        missing = [v for v, _, _ in specs if v not in data.columns]
        if missing:
            geo = self._gate_geometry()
            borrowed = [c for c in dict.fromkeys(missing) if c in geo.columns]
            unknown = [c for c in missing if c not in geo.columns]
            if unknown:
                raise KeyError(
                    f"cannot filter on {unknown}: not a data column "
                    f"{sorted(data.columns)} nor a LUT column {sorted(geo.columns)}.",
                )
            data = data.join(geo.select("gate_id", *borrowed), on="gate_id", how="left")

        for var, logic, thr in specs:
            data = data.filter(_filter_expr(var, logic, thr))
        if borrowed:
            data = data.drop(borrowed)
        return self._derive(data)

    def _lut_paths(self) -> dict[str, Path]:
        """``{radar: LUT parquet path}`` for the radars present in the data."""
        archive_dir = self._require_archive_dir()
        out = {}
        for r in self.radars():
            rr = normalize_radar_name(r)
            p = Path(archive_dir) / rr / "LUT" / f"{rr}_LUT.parquet"
            if p.exists():
                out[rr] = p
        return out

    def _lut_column_names(self) -> list[str]:
        """LUT column names, read from the parquet **schema** (no data loaded)."""
        for p in self._lut_paths().values():
            return list(pl.scan_parquet(p).collect_schema().names())
        return []

    def _borrow_lut_columns(self, cols: list[str]) -> pl.DataFrame:
        """Load ``cols`` from the LUT for the gates present, keyed by ``gate_id``.

        The general form of :meth:`_gate_geometry` (which exposes only
        lon/lat/alt/sweep): this can borrow **any** LUT column — ``range``,
        ``azimuth``, ``elevation_angle``, ``x``/``y``/``z`` … .  Column
        projection is pushed into the parquet reader, and the result is
        restricted to the gates currently in ``.data``, so the geometry stays
        synchronized with the (possibly already filtered) values.
        """
        paths = self._lut_paths()
        if not paths:
            raise ValueError(
                "no LUT found for the radars in this data; cannot select on " f"static columns {cols}.",
            )
        present = self._require_data().select("gate_id").unique()
        parts = []
        for p in paths.values():
            names = pl.scan_parquet(p).collect_schema().names()
            keep = ["gate_id", *[c for c in cols if c in names and c != "gate_id"]]
            parts.append(
                pl.scan_parquet(p).select(keep).join(present.lazy(), on="gate_id", how="semi"),
            )
        return (
            # Diagonal: a radar may lack a LUT column another one has (a projection
            # only some were given), which a vertical concat rejects outright.
            pl.concat(parts, how="diagonal_relaxed")
            .collect()
            .unique(
                subset="gate_id",
                maintain_order=True,
            )
        )

    def sel(self, **indexers) -> RadDB:
        """Select gates by label, xarray-style; returns a **new** ``RadDB``.

        Each keyword names a column and gives what to keep:

        * ``slice(start, stop)`` — a range, **inclusive of both ends** (as in
          ``xarray.Dataset.sel``, not like Python list slicing).  Either end may
          be ``None`` to leave it open.  A ``step`` is rejected.
        * a list / tuple / set — membership, e.g. ``radars=["A", "L"]``.
        * a scalar — equality.  For a timestamp column a scalar (or a *partial*
          string) selects the whole period it names, so
          ``time="2024-08-26 02:46:08"`` still matches a row stored as
          ``02:46:08.050``, and ``time="2024-08"`` selects that month.

        Dynamic columns (``DBZH``, ``ZDR``, ``time`` …) are matched directly.
        **Static/geometry columns come from the LUT** (``latitude``,
        ``longitude``, ``altitude``, ``range``, ``azimuth``, ``sweep``,
        ``elevation_angle``, ``x``/``y``/``z``, ``x_<epsg>``/``y_<epsg>``): they
        are borrowed from the LUT only for as long as the predicate needs them
        and then dropped, so the returned object still carries **dynamic values
        only** — the LUT is never concatenated onto the data.  Because the LUT is
        always re-derived for the gates currently present, it stays in sync
        automatically after any number of chained selections.

        Aliases are accepted when they do not collide with a real column:
        ``radars``→``radar``, ``lat``→``latitude``, ``lon``→``longitude``,
        ``alt``→``altitude``, ``ranges``→``range``, ``sweeps``→``sweep``,
        ``elevation``→``elevation_angle``.

        All keywords are combined with **AND**.  The original object is never
        modified.

        Examples
        --------
        >>> rdf.sel(time=slice("2021-02", "2022-03"), DBZH=slice(0, 10))
        >>> rdf.sel(time="2022-01-03 14:00:00")
        >>> rdf.sel(radars=["A", "L"])
        >>> rdf.sel(range=slice(10_000, 50_000))
        >>> rdf.sel(lon=slice(8.0, 9.0), lat=slice(46.0, 47.0))

        Raises
        ------
        KeyError
            If a keyword matches neither a data column nor a LUT column.
        """
        if not indexers:
            return self._derive(self._require_data())

        data = self._require_data()
        lut_names = None  # loaded lazily, only if a static column is requested

        resolved: list[tuple[str, object]] = []  # (column, value)
        static: list[str] = []
        radar_from_gate_id = None

        for key, value in indexers.items():
            name = key
            if name not in data.columns:
                if lut_names is None:
                    lut_names = self._lut_column_names()
                if name not in lut_names:
                    alias = _SEL_ALIASES.get(name)
                    if alias and (alias in data.columns or alias in lut_names):
                        name = alias
                    elif alias == "radar" or name in ("radar", "radars"):
                        # no radar column stored -> select via the gate_id prefix
                        radar_from_gate_id = value
                        continue
                    elif name in _TIME_COLUMNS or _SEL_ALIASES.get(name) in _TIME_COLUMNS:
                        try:
                            name = self._time_column()
                        except KeyError:
                            raise KeyError(
                                f"sel({key}=...): data has no time column.",
                            ) from None
                    else:
                        raise KeyError(
                            f"sel({key}=...): {name!r} is neither a data column "
                            f"{sorted(data.columns)} nor a LUT column {sorted(lut_names)}.",
                        )
            if name not in data.columns:
                static.append(name)
            resolved.append((name, value))

        # Borrow the static columns just long enough to evaluate their predicates.
        borrowed: list[str] = []
        if static:
            lut_tbl = self._borrow_lut_columns(sorted(set(static)))
            borrowed = [c for c in dict.fromkeys(static) if c in lut_tbl.columns]
            still_missing = [c for c in static if c not in borrowed]
            if still_missing:
                raise KeyError(
                    f"sel(): LUT has no column(s) {still_missing}; " f"available: {sorted(lut_tbl.columns)}.",
                )
            data = data.join(
                lut_tbl.select(["gate_id", *borrowed]),
                on="gate_id",
                how="left",
                maintain_order="left",
            )

        for name, value in resolved:
            data = data.filter(_sel_expr(name, value, data.schema[name]))

        if radar_from_gate_id is not None:
            wanted = [radar_from_gate_id] if isinstance(radar_from_gate_id, str) else list(radar_from_gate_id)
            codes = [encode_radar_code(r) for r in wanted]
            data = data.filter(
                (pl.col("gate_id") // GATE_ID_RADAR_BASE).is_in(codes),
            )

        if borrowed:
            data = data.drop(borrowed)
        return self._derive(data)

    def add_feature(self, name: str, compute_fn) -> RadDB:
        """Add a computed column ``name`` and return a new RadDB.

        ``compute_fn`` receives the polars DataFrame and returns a polars
        ``Expr`` or a column-length array/Series, e.g.
        ``rdf.add_feature("ZDR_lin", lambda df: 10 ** (df["ZDR"] / 10))``.
        """
        data = self._require_data()
        res = compute_fn(data)
        if isinstance(res, pl.Expr):
            data = data.with_columns(res.alias(name))
        else:
            data = data.with_columns(pl.Series(name, res))
        return self._derive(data)

    # ---- converters ----

    def to_pandas(self, with_geometry: bool = False, with_polar_coords: bool = False) -> pd.DataFrame:
        """Return the data as a pandas DataFrame.

        Parameters
        ----------
        with_geometry : bool
            Merge the per-gate LUT geometry on ``gate_id`` —
            ``latitude``/``longitude``/``altitude`` and, if ``crs`` is set,
            ``x_{epsg}``/``y_{epsg}``.
        with_polar_coords : bool
            Also merge the polar coordinates the geometry was derived from:
            ``range``, ``azimuth``, ``elevation_angle``.  Off by default because
            they duplicate information already in the Cartesian columns; turn them
            on to work in polar space.  Implies ``with_geometry``.
        """
        data = self._require_data()
        if with_geometry or with_polar_coords:
            data = data.join(
                self._gate_geometry(with_polar_coords=with_polar_coords),
                on="gate_id",
                how="left",
                suffix="_lut",
            )
        return _decode_geometry(data.to_pandas())

    def to_geopandas(self, with_polar_coords: bool = False):
        """Return a GeoDataFrame with a per-gate point geometry and CRS.

        ``with_polar_coords=True`` adds ``range`` / ``azimuth`` / ``elevation_angle``
        (see :meth:`to_pandas`).
        """
        import geopandas as gpd

        df = self.to_pandas(with_geometry=True, with_polar_coords=with_polar_coords)
        epsg = int(self._crs) if isinstance(self._crs, int) else None
        xcol, ycol = (f"x_{epsg}", f"y_{epsg}") if epsg else (None, None)
        if xcol and xcol in df.columns:
            geom = gpd.points_from_xy(df[xcol], df[ycol])
            crs = self._crs
        else:
            geom = gpd.points_from_xy(df["longitude"], df["latitude"])
            crs = 4326
        return gpd.GeoDataFrame(df, geometry=geom, crs=crs)

    def to_geoarrow(
        self,
        geometry: str = "point",
        columns: list[str] | None = None,
        max_rows: int | None = 5_000_000,
    ):
        """Return the data as a GeoArrow-tagged ``pyarrow.Table``.

        This is the export path for GPU map renderers (deck.gl / lonboard) and
        for GeoParquet.  The dynamic columns come straight from the polars frame
        with no copy; the LUT geometry is joined **here and only here**.

        Parameters
        ----------
        geometry : {"point", "polygon"}
            ``"point"`` gives one gate centroid per row (~24 B/gate).
            ``"polygon"`` gives the exact wedge — the gate's four corners taken
            from the LUT's per-sweep edge arrays (~100 B/gate).
        columns : list of str, optional
            Dynamic columns to export.  ``None`` (default) exports all of them.
            ``gate_id`` is always included.
        max_rows : int, optional
            Guardrail: raise if the frame is larger, since renderers top out
            around a few million features.  Pass ``None`` to disable.

        Returns
        -------
        pyarrow.Table
            The dynamic columns plus a ``geometry`` column carrying the GeoArrow
            extension metadata.

        Examples
        --------
        >>> import lonboard
        >>> lonboard.viz(rdf.crop_by_bbox(bounds).to_geoarrow(geometry="polygon"))
        """
        import pyarrow as pa

        from raddb.lut import gate_polygons_geoarrow, geoarrow_field

        if geometry not in ("point", "polygon"):
            raise ValueError(f"geometry must be 'point' or 'polygon'; got {geometry!r}.")

        data = self._require_data()
        if max_rows is not None and len(data) > max_rows:
            raise ValueError(
                f"to_geoarrow() would build {len(data):,} features, over the "
                f"max_rows={max_rows:,} guardrail. Narrow the selection first "
                "(crop_by_bbox / crop_by_polygon / crop_around_point / filter), "
                "or pass max_rows=None to override.",
            )
        if columns is not None:
            data = data.select(dict.fromkeys(["gate_id", *columns]))

        table = data.to_arrow()
        gate_ids = data["gate_id"].to_numpy()

        if geometry == "polygon":
            radars = self.radars()
            if len(radars) != 1:
                raise ValueError(
                    f"polygon geometry needs a single radar; data spans {radars}. "
                    "Select one with open(radars=...) or filter first.",
                )
            geom = gate_polygons_geoarrow(
                normalize_radar_name(radars[0]),
                self._require_archive_dir(),
                gate_ids,
            )
            field = geoarrow_field("geometry", geom.type, "polygon", "EPSG:4326")
        else:
            geo = (
                pl.DataFrame({"gate_id": gate_ids, "_ord": np.arange(len(gate_ids))})
                .join(self._gate_geometry(), on="gate_id", how="left")
                .sort("_ord")
            )
            xy = np.column_stack([geo["longitude"].to_numpy(), geo["latitude"].to_numpy()])
            geom = pa.FixedSizeListArray.from_arrays(
                pa.array(xy.reshape(-1), type=pa.float64()),
                2,
            )
            field = geoarrow_field("geometry", geom.type, "point", "EPSG:4326")

        return table.append_column(field, geom)

    def to_datatree(self, radar: str | None = None, timestep=None, label_column: str = "DBZH") -> xr.DataTree:
        """Reconstruct a single-volume ``xr.DataTree`` from the loaded data.

        Recovers geometry from the LUT and NaN-fills gates absent from the data.
        Selects one radar and one volume:

        - ``radar`` — inferred when the data covers exactly one radar, else required.
        - ``timestep`` — nearest ``volume_time``; required only when several volumes.
        """
        # Radar/volume selection runs on the polars frame, so only the single
        # selected volume crosses into pandas (inside dataframe_to_datatree),
        # not the whole loaded dataset.
        data = self._require_data()
        if "gate_id" not in data.columns:
            raise KeyError("data has no 'gate_id' column; cannot reconstruct a DataTree.")

        present = _radars_from_gate_ids(data["gate_id"].to_numpy())
        if radar is None:
            if len(present) != 1:
                raise ValueError(f"data spans radars {present}; pass radar= to pick one.")
            radar = present[0]
        radar = normalize_radar_name(radar)

        if "radar" in data.columns:
            df_r = data.filter(pl.col("radar") == radar)
        else:
            df_r = data.filter(
                (pl.col("gate_id") // GATE_ID_RADAR_BASE) == encode_radar_code(radar),
            )
        if df_r.is_empty():
            raise ValueError(f"No rows for radar {radar!r} in data.")

        if "volume_time" in df_r.columns and df_r["volume_time"].is_not_null().any():
            vols = df_r["volume_time"].drop_nulls().unique().sort().to_list()
            if timestep is None:
                if len(vols) != 1:
                    raise ValueError(f"data holds {len(vols)} volumes; pass timestep= to pick one.")
                chosen = vols[0]
            else:
                ts = pd.to_datetime(timestep)
                # Match the tz-awareness of the stored volume_time before comparing.
                aware = getattr(vols[0], "tzinfo", None) is not None
                if aware and ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                elif not aware and ts.tzinfo is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                chosen = min(vols, key=lambda v: abs(pd.Timestamp(v) - ts))
            df_vol = df_r.filter(pl.col("volume_time") == chosen)
        else:
            df_vol = df_r

        return dataframe_to_datatree(
            df=df_vol,
            radar=radar,
            base_path=str(self._require_archive_dir()),
            label_column=label_column,
        )

    # ---- accessors (return values; repr prints the summary) ----

    def head(self, n: int = 5) -> pl.DataFrame:
        """First ``n`` rows (polars)."""
        return self._require_data().head(n)

    def tail(self, n: int = 5) -> pl.DataFrame:
        """Last ``n`` rows (polars)."""
        return self._require_data().tail(n)

    def columns(self) -> list[str]:
        """Column names of the loaded data."""
        return list(self._require_data().columns)

    def radars(self) -> list[str]:
        """Radar identifiers present in the loaded data."""
        data = self._require_data()
        if "radar" in data.columns:
            return sorted(data["radar"].unique().to_list())
        return list(_radars_from_gate_ids(data["gate_id"].to_pandas()))

    def _time_column(self) -> str:
        data = self._require_data()
        for c in ("volume_time", "time"):
            if c in data.columns:
                return c
        raise KeyError("data has no 'volume_time' or 'time' column.")

    def start_time(self) -> datetime.datetime:
        """Earliest volume/ray time in the loaded data."""
        return self._require_data()[self._time_column()].min()

    def end_time(self) -> datetime.datetime:
        """Latest volume/ray time in the loaded data."""
        return self._require_data()[self._time_column()].max()

    #: LUT columns the ``to_*`` converters attach as gate geometry.
    _GEOMETRY_COLS = ("gate_id", "latitude", "longitude", "altitude", "sweep")

    #: Polar coordinates, attached only on request — they are what the gate
    #: geometry was *derived from*, so they are redundant with the Cartesian
    #: columns unless you are working in polar (antenna) space.
    _POLAR_COLS = ("range", "azimuth", "elevation_angle")

    def _gate_geometry(self, with_polar_coords: bool = False) -> pl.DataFrame:
        """LUT geometry (lon/lat/alt [+ projected x/y]) for the gates present.

        Returned as its **own** table — the LUT is never carried alongside the
        dynamic values; the ``to_*`` converters join it on demand.
        """
        archive_dir = self._require_archive_dir()
        epsg = int(self._crs) if isinstance(self._crs, int) else None
        wanted = list(self._GEOMETRY_COLS) + (list(self._POLAR_COLS) if with_polar_coords else [])
        radars = self.radars()
        if not radars:  # empty selection (e.g. a crop that matched nothing)
            cols = list(wanted)
            if epsg:
                cols += [f"x_{epsg}", f"y_{epsg}"]
            return pl.DataFrame(schema={c: (pl.Int64 if c == "gate_id" else pl.Float64) for c in cols})
        parts = []
        for r in radars:
            lut = load_radar_lut(normalize_radar_name(r), archive_dir)
            keep = [c for c in wanted if c in lut.columns]
            if epsg and f"x_{epsg}" in lut.columns:
                keep += [f"x_{epsg}", f"y_{epsg}"]
            parts.append(lut.select(keep))
        geo = pl.concat(parts, how="diagonal").unique(subset="gate_id", maintain_order=True)
        present = self._require_data().select("gate_id").unique()
        return geo.join(present, on="gate_id", how="semi")

    def extent(self) -> list[float]:
        """Projected bounding box ``[xmin, xmax, ymin, ymax]`` (in ``crs``)."""
        epsg = int(self._crs) if isinstance(self._crs, int) else None
        if epsg is None:
            raise ValueError("extent() needs a projected crs; set RadDB(crs=<epsg>).")
        geo = self._gate_geometry()
        xcol, ycol = f"x_{epsg}", f"y_{epsg}"
        if xcol not in geo.columns:
            raise ValueError(
                f"LUT has no {xcol} column; archive with crs={epsg} to store projected coords.",
            )
        return [float(geo[xcol].min()), float(geo[xcol].max()), float(geo[ycol].min()), float(geo[ycol].max())]

    def geographic_extent(self) -> list[float]:
        """Geographic bounding box ``[lon_min, lon_max, lat_min, lat_max]``."""
        geo = self._gate_geometry()
        return [
            float(geo["longitude"].min()),
            float(geo["longitude"].max()),
            float(geo["latitude"].min()),
            float(geo["latitude"].max()),
        ]

    def crs(self):
        """Projected CRS (x, y) as a ``pyproj.CRS``.

        Falls back to the CRS the archive was written with, so reading never
        requires restating it — that is recorded precisely so it need not be
        guessed or repeated.
        """
        import pyproj

        spec = self._crs
        if spec is None and self.archive_dir is not None:
            from raddb.aoi import aoi_epsg

            for radar in self.radars() if self.data is not None else []:
                try:
                    spec = aoi_epsg(self.archive_dir, radar)
                    break
                except (ValueError, FileNotFoundError):
                    continue
        if spec is None:
            raise ValueError(
                "no projected CRS: this object has none and no archive records " "one. Pass RadDB(crs=<epsg>).",
            )
        return pyproj.CRS.from_user_input(spec)

    def geographic_crs(self):
        """Geographic CRS (lat, lon) as a ``pyproj.CRS`` (EPSG:4326)."""
        import pyproj

        return pyproj.CRS.from_epsg(4326)

    # ================================================================
    # EXTRACT AREA OF INTEREST
    # ================================================================

    def crop_by_bbox(
        self,
        bounds=None,
        extent=None,
        crs: int | str | None = None,
        quicklook: bool = False,
        aoi_crs=None,
    ) -> RadDB:
        """Crop to a rectangle; returns a new RadDB.

        Give **exactly one** of ``bounds=(xmin, ymin, xmax, ymax)`` or
        ``extent=[xmin, xmax, ymin, ymax]`` (the latter matches :meth:`extent`),
        in ``crs`` (default EPSG:2056).  A single horizontal footprint selects
        the whole vertical column above it.
        """
        if (bounds is None) == (extent is None):
            raise ValueError("Pass exactly one of `bounds` or `extent`.")
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
        else:
            xmin, ymin, xmax, ymax = bounds
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("expected xmin < xmax and ymin < ymax.")
        epsg = self._aoi_epsg(aoi_crs)
        geom = _reproject_to_aoi(shapely.box(xmin, ymin, xmax, ymax), crs, epsg)
        return self._derive(self._crop_to_aoi(self._require_data(), geom, quicklook, epsg))

    def crop_by_polygon(self, polygon, crs: int | str | None = None, quicklook: bool = False, aoi_crs=None) -> RadDB:
        """Crop to an arbitrary polygon; returns a new RadDB.

        ``polygon`` is a shapely geometry, a GeoDataFrame/GeoSeries, or a
        ``.shp``/``.geojson`` path.  ``crs=None`` auto-detects.
        """
        epsg = self._aoi_epsg(aoi_crs)
        geom = _load_aoi_polygon(polygon, crs, epsg)
        return self._derive(self._crop_to_aoi(self._require_data(), geom, quicklook, epsg))

    def crop_around_point(
        self,
        point,
        distance: float,
        crs: int | str | None = None,
        quicklook: bool = False,
        aoi_crs=None,
    ) -> RadDB:
        """Crop to a circle of radius ``distance`` (meters) around ``point``.

        Returns a new RadDB.  ``point`` is ``(x, y)`` or a shapely Point in ``crs``.
        """
        if distance <= 0:
            raise ValueError(f"distance must be positive (meters); got {distance!r}.")
        if hasattr(point, "geom_type"):
            if point.geom_type != "Point":
                raise TypeError(f"point geometry must be a Point; got {point.geom_type}.")
            pt = point
        elif isinstance(point, (tuple, list)) and len(point) == 2:
            pt = shapely.Point(float(point[0]), float(point[1]))
        else:
            raise TypeError(f"point must be (x, y) or a shapely Point; got {type(point).__name__}.")
        epsg = self._aoi_epsg(aoi_crs)
        geom = _reproject_to_aoi(pt, crs, epsg).buffer(distance)
        return self._derive(self._crop_to_aoi(self._require_data(), geom, quicklook, epsg))

    def _aoi_epsg(self, override=None) -> int:
        """The CRS this object's AOI operations run in.

        The archive's own, from ``info.yaml`` — never a built-in default, so a
        crop radius always means meters in a frame that is valid where the radar
        actually is.  ``aoi_crs=`` names a different one explicitly and is
        validated against every site before use.
        """
        data = self._require_data()
        radars = _radars_from_gate_ids(data["gate_id"].to_numpy())
        return aoi_epsg_for(self._require_archive_dir(), radars, override=override)

    def _crop_to_aoi(self, data: pl.DataFrame, geom, quicklook: bool = False, epsg: int | None = None) -> pl.DataFrame:
        """Intersect ``geom`` (in the AOI CRS) with LUT centroids and keep matching rows.

        This **selects** rows, it never widens them: the LUT geometry stays in its
        own table, reachable through the ``to_*`` converters.  The intersection
        itself runs against the static LUT centroids, so it costs the same no
        matter how many volumes ``data`` spans.
        """
        if "gate_id" not in data.columns:
            raise KeyError("data has no 'gate_id' column; cannot crop by AOI.")
        radars = _radars_from_gate_ids(data["gate_id"].to_numpy())
        centroids = _lut_centroids(self._require_archive_dir(), radars, epsg=epsg)
        aoi_cen = _resolve_aoi_centroids(centroids, geom)
        data_aoi = _apply_gate_ids(data, aoi_cen["gate_id"].to_numpy())

        if quicklook:
            from raddb.viz.plot import plot_aoi_quicklook

            # ponytail: the quicklook is the one consumer that needs coordinates,
            # so join them for the plot only — never into the returned frame.
            # _lut_centroids returns the archive's projected pair as plain x/y,
            # whatever EPSG that is — the quicklook draws in the AOI frame.
            selected = data_aoi.join(
                aoi_cen.select("gate_id", "x", "y"),
                on="gate_id",
                how="left",
            )
            plot_aoi_quicklook(geom, selected=selected, radars=radars, base_path=self.archive_dir, epsg=epsg)
        return data_aoi

    def interactive_crop(self, **kwargs):
        """Draw an AOI on an interactive ipyleaflet map and crop (Jupyter only).

        Returns an ``AOISelector``; after drawing a shape and clicking *Apply
        crop*, the cropped result (a new ``RadDB``) is on ``selector.result``.
        """
        self._require_data()
        from raddb.viz.interactive import AOISelector

        return AOISelector(self, **kwargs).display()

    # ================================================================
    # EXTRACT CROSS-SECTION
    # ================================================================

    def extract_cross_section(
        self,
        p1,
        p2,
        crs: int | str | None = None,
        beamwidth_deg: float = 1.0,
        quicklook: bool = False,
        aoi_crs=None,
    ) -> RadDB:
        """Extract a vertical cross-section along the line ``p1 -> p2``; returns a new RadDB.

        The line need not pass through a radar.  Each selected gate gets a polygon
        in the (distance-along-line, altitude) plane (``cs_polygon``) plus its
        center ``d_center``/``z_center``; visualize with
        :meth:`plot_vcs`.  ``p1``/``p2`` are ``(x, y)`` or shapely Points
        in ``crs``; distance is measured from ``p1``.
        """
        data = self._require_data()
        if "gate_id" not in data.columns:
            raise KeyError("data has no 'gate_id' column; cannot extract a cross-section.")

        epsg = self._aoi_epsg(aoi_crs)
        pts = []
        for name, p in (("p1", p1), ("p2", p2)):
            if hasattr(p, "geom_type"):
                if p.geom_type != "Point":
                    raise TypeError(f"{name} must be a Point; got {p.geom_type}.")
                pt = p
            elif isinstance(p, (tuple, list)) and len(p) == 2:
                pt = shapely.Point(float(p[0]), float(p[1]))
            else:
                raise TypeError(f"{name} must be (x, y) or a shapely Point.")
            pt = _reproject_to_aoi(pt, crs, epsg)
            pts.append((pt.x, pt.y))
        (x1, y1), (x2, y2) = pts

        radars = _radars_from_gate_ids(data["gate_id"].to_numpy())
        base = self._require_archive_dir()
        cs_t = _lut_cs_table(base, radars, beamwidth_deg=beamwidth_deg, epsg=epsg)
        cs_geom = _cross_section_gates(
            cs_t,
            (x1, y1),
            (x2, y2),
            beamwidth_deg=beamwidth_deg,
            base_path=base,
            epsg=epsg,
        )

        gate_ids = cs_geom["gate_id"].to_numpy(dtype=np.int64) if len(cs_geom) else np.empty(0, dtype=np.int64)
        data_cs = _apply_gate_ids(data, gate_ids)

        # Unlike an AOI crop these columns are *not* LUT data: they are the
        # per-gate geometry of this particular section line, computed here and
        # available nowhere else, so they travel with the rows.
        # The geometry table computes in the projected frame under plain x/y.
        # Publish it as x_<epsg>/y_<epsg>, and give x/y back to the LUT's
        # radar-relative meters, so both meanings are unambiguous downstream.
        cs_geom = cs_geom.rename(
            columns={
                "x": f"x_{epsg}",
                "y": f"y_{epsg}",
                "x_rel": "x",
                "y_rel": "y",
            },
        )
        geom_cols = [
            c
            for c in (
                "radar",
                "sweep",
                "azimuth",
                "range",
                "elevation_angle",
                "x",
                "y",
                f"x_{epsg}",
                f"y_{epsg}",
                "altitude",
                "d_center",
                "z_center",
                "cs_polygon",
            )
            if c in cs_geom.columns and (c == "cs_polygon" or c not in data_cs.columns)
        ]
        if not data_cs.is_empty() and len(cs_geom):
            data_cs = data_cs.join(
                _to_polars(cs_geom[["gate_id", *geom_cols]]),
                on="gate_id",
                how="left",
            )

        if quicklook:
            from raddb.viz.plot import plot_aoi_quicklook

            plot_aoi_quicklook(
                shapely.LineString([(x1, y1), (x2, y2)]),
                selected=data_cs,
                radars=radars,
                base_path=self.archive_dir,
                # The section was resolved in `epsg`; without it the quicklook
                # falls back to LV95 and frames a non-Swiss archive over
                # Switzerland.
                epsg=epsg,
            )
        return self._derive(data_cs)

    # ================================================================
    # PLOT
    # ================================================================

    @staticmethod
    def _save_fig(ret, save, kwargs):
        if not save:
            return
        import matplotlib.pyplot as plt

        fig = getattr(ret, "figure", None) or getattr(getattr(ret, "axes", None), "figure", None) or plt.gcf()
        fig.savefig(save, bbox_inches="tight", dpi=kwargs.get("dpi", 150))

    def plot_ppi(self, sweep: int | str = 1, variable: str = "DBZH", radar: str | None = None, timestep=None, **kwargs):
        """Plot a PPI of one sweep — one plot, one figure.

        Gate footprints come from the ``h_plane`` lattice, so a filtered, ``sel``-ed
        or cropped RadDB draws exactly the gates it still holds.  Pass ``ax=`` to
        compose several of these into a multi-panel figure.

        See :func:`raddb.viz.plot.plot_ppi` for the full parameter list
        (``coords``, ``context``, ``save``, ...).
        """
        from raddb.viz.plot import plot_ppi as _plot_ppi

        return _plot_ppi(self, sweep=sweep, variable=variable, radar=radar, timestep=timestep, **kwargs)

    def plot_rhi(self, azimuth: float = 0.0, variable: str = "DBZH", radar: str | None = None, timestep=None, **kwargs):
        """Plot an RHI along one azimuth, stacking every sweep.

        Gate faces come from the ``v_plane`` lattice in the
        ``(ground distance, altitude)`` plane.  See
        :func:`raddb.viz.plot.plot_rhi` for the full parameter list.
        """
        from raddb.viz.plot import plot_rhi as _plot_rhi

        return _plot_rhi(self, azimuth=azimuth, variable=variable, radar=radar, timestep=timestep, **kwargs)

    def plot_cappi(self, altitude: float, variable: str = "DBZH", radar: str | None = None, timestep=None, **kwargs):
        """Plot a CAPPI — a horizontal slice at constant ``altitude`` [m].

        Where :meth:`plot_ppi` fixes the sweep, this fixes the altitude and pulls
        from whichever elevation angles sample it.  See
        :func:`raddb.viz.plot.plot_cappi` for how the slice geometry is built and
        for ``overlap`` / ``fill_lowest``.
        """
        from raddb.viz.plot import plot_cappi as _plot_cappi

        return _plot_cappi(self, altitude=altitude, variable=variable, radar=radar, timestep=timestep, **kwargs)

    def plot_vcs(self, line=None, variable: str = "DBZH", radar: str | None = None, timestep=None, **kwargs):
        """Plot a vertical cross-section along an arbitrary line.

        Either pass ``line=`` — ``(p1, p2)``, a shapely ``LineString``, or a
        ``.shp``/``.geojson`` path — or call this on an
        :meth:`extract_cross_section` result, which already carries the section.
        Giving both is an error, as is giving neither.  A DataTree cannot be used:
        archive it first.  See :func:`raddb.viz.plot.plot_vcs`.
        """
        from raddb.viz.plot import plot_vcs as _plot_vcs

        return _plot_vcs(self, line=line, variable=variable, radar=radar, timestep=timestep, **kwargs)

    def plot_cross_section(self, variable: str = "DBZH", radar: str | None = None, timestep=None, **kwargs):
        """Deprecated alias of :meth:`plot_vcs`."""
        import warnings

        warnings.warn(
            "RadDB.plot_cross_section is deprecated; use plot_vcs() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.plot_vcs(variable=variable, radar=radar, timestep=timestep, **kwargs)

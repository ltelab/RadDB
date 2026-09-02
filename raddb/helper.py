"""Shared utilities and configuration for RadDB."""

import contextlib as _contextlib
import datetime as _dt
import re
import time as _time
from pathlib import Path

import pandas as pd
import polars as pl
import xarray as xr


# --- DataTree Helpers ---
def list_sweep_names(dt: xr.DataTree) -> list[str]:
    """Return sorted sweep group names from a DataTree."""
    pat = re.compile(r"^sweep_\d+$")
    return sorted(s.lstrip("/") for s in dt.groups if pat.match(s.lstrip("/")))


# --- Parquet Helpers ---
def ensure_utc(dt_input):
    """Ensure a datetime-like input is timezone-aware and in UTC."""
    if dt_input is None:
        return None

    ts = pd.to_datetime(dt_input)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def read_parquet_files(
    base_path: str | Path,
    pattern: str = "**/*POL.parquet",
    columns: list[str] | None = None,
    verbose: bool = True,
) -> "pl.DataFrame":
    """Read matching parquet files into a single **polars** DataFrame."""
    files = sorted(Path(base_path).rglob(pattern))
    if not files:
        if verbose:
            print(f"[RadDB] No files found matching '{pattern}' in {base_path}")
        return pl.DataFrame()
    if verbose:
        print(f"[RadDB] Found {len(files)} file(s) — loading...")
    return pl.concat(
        [pl.read_parquet(f, columns=columns) for f in files],
        how="vertical_relaxed",
    )


def check_dataframe(df: "pl.DataFrame | pd.DataFrame") -> None:
    """Print a quick structural summary; accepts polars or pandas."""
    print("-" * 50)
    print(f"Shape:   {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 50)
    if isinstance(df, pl.DataFrame):
        nulls = df.null_count().to_dicts()[0] if len(df.columns) else {}
        print("Missing values:")
        for k, v in nulls.items():
            print(f"{k}    {v}")
    else:
        print(f"Missing values:\n{df.isna().sum()}")
    print("-" * 50)
    print(df.head())
    print("-" * 50)


# --- Radar Name Normalization ---

#: Characters a radar name may be built from, in the order that gives each its
#: numeric value in the base-36 ``gate_id`` radar code (see
#: :func:`raddb.lut.encode_radar_code`).
RADAR_ALPHABET: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Maximum radar-name length.  Four base-36 characters is what the ``gate_id``
#: layout can hold: ``36**4 = 1_679_616`` codes against the ``9_223_371`` slots
#: int64 leaves above the ``10**12`` gate field, whereas ``36**5`` would need
#: 60 million.  It fits both single-letter identifiers and four-letter ones
#: (NEXRAD ``KTLX``, FMI ``FKUO``); five-character ODIM NOD codes such as
#: ``fikuo`` must be aliased to four and kept in the info YAML's ``network``.
RADAR_CODE_LEN: int = 4

_RADAR_NAME_RE = re.compile(rf"^[{RADAR_ALPHABET}]{{1,{RADAR_CODE_LEN}}}$")


def normalize_radar_name(radar: str) -> str:
    """
    Normalize a radar name to its canonical archive form.

    The canonical form is upper-case, 1 to :data:`RADAR_CODE_LEN` characters
    drawn from :data:`RADAR_ALPHABET`, with leading zeros stripped (they are
    padding in the ``gate_id`` radar code, not part of the name, so ``"0A"``
    and ``"A"`` are the same radar).

    A three-character ``ML*`` name is a legacy alias for a single-letter radar
    and is reduced to that letter (``"MLA"`` -> ``"A"``).  Every other name is
    kept **whole** — ``"KTLX"`` stays ``"KTLX"``.

    Parameters
    ----------
    radar : str
        Radar name, e.g. ``"A"``, ``"MLA"`` or ``"KTLX"``.

    Returns
    -------
    str
        Canonical radar name.

    Raises
    ------
    ValueError
        If the name is empty, too long, or uses characters outside
        :data:`RADAR_ALPHABET`.  It is raised rather than silently truncating:
        two sites reduced to the same letter would overwrite each other's
        archive.

    Examples
    --------
    >>> normalize_radar_name("MLA")
    'A'
    >>> normalize_radar_name("A")
    'A'
    >>> normalize_radar_name("KTLX")
    'KTLX'
    """
    if not isinstance(radar, str):
        raise ValueError(f"radar name must be a string, got {type(radar).__name__}.")

    name = radar.upper().strip()

    # Legacy "ML<letter>" alias.  Restricted to exactly three characters so that
    # a genuine four-character name beginning with "ML" is not mistaken for one.
    if len(name) == 3 and name.startswith("ML"):
        name = name[-1]

    # Leading zeros are gate_id padding ("000A"), never part of the name.
    stripped = name.lstrip("0")
    if stripped:
        name = stripped

    if not _RADAR_NAME_RE.match(name):
        raise ValueError(
            f"radar name {radar!r} is not usable: a radar name must be 1 to "
            f"{RADAR_CODE_LEN} characters from [0-9A-Z] (e.g. 'A', 'KTLX'). "
            f"Longer identifiers must be aliased to {RADAR_CODE_LEN} characters.",
        )
    return name


def is_valid_radar_name(radar) -> bool:
    """``True`` when :func:`normalize_radar_name` would accept *radar*."""
    try:
        normalize_radar_name(radar)
    except ValueError:
        return False
    return True


# --- Filter Logic Registry ---
FILTER_LOGICS: dict[str, callable] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}


def resolve_filter_logic(logic: str):
    """Return the comparison function for *logic*.

    Raises
    ------
    ValueError
        If ``logic`` is not one of the supported operators.
    """
    fn = FILTER_LOGICS.get(logic)
    if fn is None:
        raise ValueError(
            f"Unknown logic '{logic}'. Choose from: {list(FILTER_LOGICS)}",
        )
    return fn


def filter_df(
    df: "pl.DataFrame | pd.DataFrame",
    feature: str = "DBZH",
    threshold: float = 0.0,
    logic: str = ">",
) -> "pl.DataFrame | pd.DataFrame":
    """Filter a DataFrame, keeping rows where ``feature [logic] threshold``.

    Accepts polars or pandas and returns the **same kind**, so an existing
    pandas caller keeps getting pandas back.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Input DataFrame.
    feature : str
        Column name to filter on (default ``"DBZH"``).
    threshold : float
        Comparison value.
    logic : str
        Comparison operator: ``'>'``, ``'>='``, ``'<'``, ``'<='``,
        ``'=='``, ``'!='``.  Default ``'>'``.

    Returns
    -------
    pl.DataFrame or pd.DataFrame
        Filtered DataFrame with non-matching rows dropped (pandas index reset).

    Raises
    ------
    KeyError
        If ``feature`` is not a column of ``df``.
    ValueError
        If ``logic`` is not one of the supported operators.
    """
    fn = resolve_filter_logic(logic)
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame columns.")
    mask = fn(df[feature].to_numpy(), threshold)
    if isinstance(df, pl.DataFrame):
        return df.filter(pl.Series(mask))
    return df[mask].reset_index(drop=True)


def filter_dt(
    dt: xr.DataTree,
    feature: str = "DBZH",
    threshold: float = 0.0,
    logic: str = ">",
) -> xr.DataTree:
    """Filter a DataTree, masking gates where ``feature [logic] threshold`` is False.

    Gates that do **not** satisfy the condition are set to NaN across all
    data variables in each sweep.  Gates that *do* satisfy the condition
    keep their original values unchanged — including legitimate zero values.

    .. note::
        This operates on the multidimensional DataTree structure via
        ``xr.Dataset.where()``.  For tabular (row-level) filtering use
        :func:`filter_df` instead, which drops non-matching rows entirely.

    Parameters
    ----------
    dt : xarry.DataTree
        Input DataTree with ``sweep_N`` groups.
    feature : str
        Variable name to use as the filter criterion (default ``"DBZH"``).
    threshold : float
        Comparison value.
    logic : str
        Comparison operator: ``'>'``, ``'>='``, ``'<'``, ``'<='``,
        ``'=='``, ``'!='``.  Default ``'>'``.

    Returns
    -------
    xr.DataTree
        New DataTree where non-matching gates have NaN for all variables.
        Matching gates are left entirely unchanged (zeros remain zeros).

    Raises
    ------
    ValueError
        If ``logic`` is not a supported operator.
    """
    fn = resolve_filter_logic(logic)

    sweep_names = list_sweep_names(dt)
    dict_ds = {}

    for sweep_name in sweep_names:
        ds = dt[sweep_name].to_dataset()
        if feature in ds:
            keep_mask = fn(ds[feature], threshold)
            # Mask only variables that share the mask's dimensions.
            # Variables on other dims (e.g. gate-edge arrays x_edges/y_edges
            # on azimuth_edge/range_edge) are static geometry — masking them
            # is meaningless and broadcasting the mask onto them explodes
            # memory (dims are disjoint).
            mask_dims = set(keep_mask.dims)
            ds = ds.assign(
                {name: var.where(keep_mask) for name, var in ds.data_vars.items() if mask_dims & set(var.dims)},
            )
        dict_ds[sweep_name] = ds

    return xr.DataTree.from_dict(dict_ds)


# ============================================================
# --- Profiling Utilities ---
# ============================================================


def _vprint(msg: str, verbose: bool = False) -> None:
    """Print a timestamped progress message to stdout when verbose is True."""
    if verbose:
        ts = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}]  {msg}", flush=True)


class StageTimer:
    """Accumulates per-stage timing records for pipeline profiling.

    Example
    -------
    >>> timer = StageTimer()
    >>> with timer.time_stage("my_stage", volume="vol_001", sweep=2):
    ...     do_work()
    ...
    >>> timer.print_summary()
    """

    def __init__(self):
        self.records: list[dict] = []

    @_contextlib.contextmanager
    def time_stage(self, stage: str, volume: str | None = None, sweep: int | None = None):
        """Context manager — records elapsed wall-clock time for *stage*."""
        t0 = _time.perf_counter()
        try:
            yield
        finally:
            self.records.append(
                {
                    "volume": volume,
                    "sweep": sweep,
                    "stage": stage,
                    "t_start": t0,
                    "duration": _time.perf_counter() - t0,
                },
            )

    def record(
        self,
        stage: str,
        duration: float,
        volume: str | None = None,
        sweep: int | None = None,
        t_start: float | None = None,
    ):
        """Manually append a pre-measured timing entry."""
        self.records.append(
            {"volume": volume, "sweep": sweep, "stage": stage, "t_start": t_start, "duration": duration},
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return all records as a DataFrame with columns [volume, sweep, stage, duration]."""
        if not self.records:
            return pd.DataFrame(columns=["volume", "sweep", "stage", "duration"])
        return pd.DataFrame(self.records)

    def summary(self) -> pd.DataFrame:
        """Aggregate by stage — returns (sum, mean, min, max, count) sorted by total time."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        return (
            df.groupby("stage")["duration"]
            .agg(["sum", "mean", "min", "max", "count"])
            .sort_values("sum", ascending=False)
        )

    def print_summary(self):
        """Print a formatted profiling table to stdout."""
        summary = self.summary()
        if summary.empty:
            print("[RadDB profiling] No timing data recorded.")
            return
        total = summary["sum"].sum()
        print("\n" + "=" * 68)
        print("  PIPELINE PROFILING SUMMARY")
        print("=" * 68)
        print(f"  {'Stage':<34} {'Total':>7}  {'Mean':>6}  {'N':>5}  {'%':>5}")
        print("-" * 68)
        for stage, row in summary.iterrows():
            pct = 100.0 * row["sum"] / total if total > 0 else 0.0
            print(
                f"  {stage:<34} {row['sum']:>6.2f}s  {row['mean']:>5.2f}s" f"  {int(row['count']):>5}  {pct:>4.1f}%",
            )
        print("-" * 68)
        print(f"  {'TOTAL':<34} {total:>6.2f}s")
        print("=" * 68)

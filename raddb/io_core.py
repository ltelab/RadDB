"""Core I/O conversion functions for radar data.

This module provides generic conversions between xarray DataTree,
polars DataFrame, and Parquet files.  pandas survives only at the xarray
seam, where DataTree reconstruction needs ``reindex(MultiIndex)`` and
``to_xarray()``.  It does **not** depend on pyart, and it makes no assumption
about which network a volume came from — reading a national archive's own raw
format belongs in a separate package.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time as _time
from contextlib import nullcontext as _nullctx
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
import yaml

from raddb.discovery import _find_polar_files_in_range, _parse_pol_time
from raddb.helper import (
    StageTimer,
    _vprint,
    list_sweep_names,
    normalize_radar_name,
    resolve_filter_logic,
)
from raddb.lut import (
    AZIMUTH_SCALE,
    _parse_corners_npz,
    azimuth_grid_tolerance,
    encode_gate_ids,
    get_full_sweep_index,
    load_azimuth_grids,
    snap_azimuths_to_grid,
)

logger = logging.getLogger(__name__)

# --- Constants --- #
LUT_COLUMNS = [
    "gate_id",
    "sweep",
    "azimuth",
    "range",
    "elevation_angle",
    "latitude",
    "longitude",
    "altitude",
    "x",
    "y",
    "z",
]

#: Per-gate geometry, which lives in the LUT and is joined back on ``gate_id`` — never
#: duplicated into a POL parquet.  Used to pick the moments out of a *flattened* volume,
#: where the per-ray/scalar metadata a DataTree carries has already been broadcast away
#: and only the column names distinguish geometry from data.  Projected ``x_<epsg>`` /
#: ``y_<epsg>`` pairs are matched separately by :func:`_projection_columns`.
_GEOMETRY_COLUMNS: frozenset = frozenset(set(LUT_COLUMNS) - {"gate_id"} | {"sweep", "elevation"})


def _projection_columns(df: pl.DataFrame | pd.DataFrame) -> list[str]:
    """Columns added by :func:`raddb.lut.add_lut_projection` (e.g. x_2056 / y_2056)."""
    return [c for c in df.columns if re.match(r"^[xy]_\w+$", c)]


def _col(df: pl.DataFrame | pd.DataFrame, name: str, dtype=None) -> np.ndarray:
    """Column ``name`` of ``df`` as a numpy array, for polars **or** pandas.

    ``polars.Series.to_numpy`` takes no ``dtype`` argument (pandas' does), so the
    cast is applied afterwards.  Used by the write path, which is numpy-based
    internally and therefore backend-agnostic.
    """
    arr = df[name].to_numpy()
    return arr if dtype is None else arr.astype(dtype)


def _to_polars_frame(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """Coerce a pandas frame to polars; pass polars frames straight through."""
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


def _to_pandas_frame(df: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    """Coerce a polars frame to pandas; pass pandas frames straight through.

    Used only at the **xarray seam**: DataTree reconstruction needs
    ``set_index().reindex(MultiIndex)`` and ``to_xarray()``, which have no
    polars equivalent.  Everywhere else the backend stays polars.
    """
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


# ============================================================================
# DataTree file loading  (NetCDF / Zarr)
# ============================================================================


def open_any_datatree(
    path: str | Path,
    engine: str | None = None,
    **open_kwargs,
) -> xr.DataTree:
    """Open a DataTree from disk — NetCDF file or Zarr store.

    Engine resolution: an explicit ``engine`` wins; a ``*.zarr`` suffix or a
    directory containing ``.zgroup`` / ``zarr.json`` selects ``"zarr"``;
    otherwise xarray auto-detects the NetCDF backend (netCDF4 / h5netcdf).

    Parameters
    ----------
    path : str or Path
        Path to a NetCDF file or Zarr store.
    engine : str, optional
        xarray backend override (e.g. ``"h5netcdf"``, ``"zarr"``).
    **open_kwargs
        Forwarded to :func:`xarray.open_datatree`.

    Returns
    -------
    xr.DataTree
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DataTree file/store not found: {p}")

    if engine is None and (
        p.suffix.lower() == ".zarr" or (p.is_dir() and ((p / ".zgroup").exists() or (p / "zarr.json").exists()))
    ):
        engine = "zarr"

    try:
        return xr.open_datatree(p, engine=engine, **open_kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            f"Opening {p.name} requires an xarray backend that is not "
            "installed (netCDF4/h5netcdf for NetCDF, zarr for Zarr stores). "
            "Install with: pip install raddb[io]",
        ) from exc


# ============================================================================
# DataTree -> DataFrame / Parquet
# ============================================================================


def datatree_to_dataset(dt: xr.DataTree, sweep: str | int) -> xr.Dataset:
    """Extract a single sweep Dataset from a DataTree."""
    sweep_name = f"sweep_{sweep}" if isinstance(sweep, int) else sweep
    return dt[sweep_name].to_dataset()


def datatree_to_dataframe(
    dt: xr.DataTree,
    max_workers: int = 1,
) -> pl.DataFrame:
    """Flatten a DataTree into a single **polars** DataFrame.

    Each sweep is converted independently and concatenated, with a ``sweep``
    column indicating the source sweep number.

    ``xarray.Dataset.to_dataframe`` only emits pandas, so each sweep is
    flattened through pandas and the whole volume is handed to polars in a
    single conversion at the end — the xarray seam is the one place pandas is
    unavoidable.
    """
    names = list_sweep_names(dt)

    def _flatten(name):
        ds = dt[name].to_dataset()
        # A dimension coordinate can exist without an index — raw NEXRAD Level II
        # sweeps arrive that way for ``range``.  ``to_dataframe`` then indexes that
        # dimension by position and emits the real values as a *column* of the same
        # name, which ``reset_index`` cannot insert.  Re-assigning rebuilds the index.
        unindexed = [d for d in ds.sizes if d in ds.coords and d not in ds.xindexes]
        if unindexed:
            ds = ds.assign_coords({d: ds[d].to_numpy() for d in unindexed})
        df = ds.to_dataframe().reset_index()
        df["sweep"] = int(name.split("_")[-1])
        return df

    if max_workers <= 1:
        list_df = [_flatten(name) for name in names]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as ex:
            list_df = list(ex.map(_flatten, names))

    return pl.from_pandas(pd.concat(list_df, ignore_index=True))


def _save_polar_parquet(
    df_polar: pl.DataFrame | pd.DataFrame,
    radar: str,
    base_path: str,
) -> str | None:
    """Save a POLAR DataFrame to the standard directory layout.

    Accepts polars (the native write-path format) or pandas.

    Returns ``None`` — writing nothing — when the volume carries no usable
    timestamp to build a path from.  That happens for a clear-air volume whose
    every gate fails the filter (``DBZH`` all-null), and for one whose ``time``
    is all-``NaT``.  Both used to crash here rather than being skipped:
    ``.min()`` on an empty column is ``None``, ``pd.to_datetime(None)`` is
    ``NaT``, and ``pd.NaT.month`` is *nan* (a float), so the ``:02d`` below
    raised ``Unknown format code 'd' for object of type 'float'`` — an error
    naming neither the volume nor the cause.
    """
    df_polar = _to_polars_frame(df_polar)
    if df_polar.is_empty():
        logger.info(
            "radar %s: no gates satisfied the filter; no POL file written.",
            radar,
        )
        return None

    vol_time = pd.to_datetime(df_polar["time"].min())
    if pd.isna(vol_time):
        logger.warning(
            "radar %s: volume time is NaT for all %d surviving gates; " "no POL file written.",
            radar,
            len(df_polar),
        )
        return None

    save_dir = Path(base_path) / radar / str(vol_time.year) / f"{vol_time.month:02d}" / f"{vol_time.day:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = vol_time.strftime("%Y%m%d_%H%M%S")
    pp = save_dir / f"{radar}_{ts}_POL.parquet"
    df_polar.write_parquet(pp)
    return str(pp)


def _gate_variables(dt: xr.DataTree) -> list[str]:
    """Names of the **per-gate** data variables of a volume, in first-seen order.

    A moment is measured per gate, so it is carried on both the azimuth and the range
    dimension.  Everything else a sweep holds — ``sweep_mode``, ``prt_mode``,
    ``follow_mode``, ``sweep_number``, ``sweep_fixed_angle``, ``nyquist_velocity`` — is
    per-ray or scalar metadata that describes the scan rather than the weather, and the
    first three are strings that cannot go in a numeric column at all.

    This is what replaces a hardcoded list of moment names: a network is free to record
    whatever it measures, and all of it is archived.  Checked against real volumes, it
    selects 19 of 25 variables on an FMI volume and 6 of 11 on a NEXRAD one, leaving exactly
    the scalar metadata behind.
    """
    seen: dict[str, None] = {}
    for name in list_sweep_names(dt):
        ds = dt[name].to_dataset()
        for var, arr in ds.data_vars.items():
            if {"azimuth", "range"} <= set(arr.dims):
                seen.setdefault(str(var), None)
    return list(seen)


def _resolve_polar_columns(
    df: pl.DataFrame | pd.DataFrame,
    variables: list[str] | None,
) -> list[str]:
    """Moment columns to write for a flattened volume, ``gate_id`` and ``time`` aside.

    ``variables`` — typically :func:`_gate_variables` of the source tree — is honoured
    in the order given, minus anything the frame does not carry.  ``None`` falls back to
    "every column that is not geometry", which is what the frame-in entry points
    (:func:`labels_to_dataframe`, a hand-built DataFrame) have to use: once a volume is
    flattened, the per-ray metadata has been broadcast to one value per row and only the
    column name still says what a column is.
    """
    if variables is not None:
        return [c for c in variables if c in df.columns and c not in ("gate_id", "time")]
    skip = _GEOMETRY_COLUMNS | {"gate_id", "time"} | set(_projection_columns(df))
    return [c for c in df.columns if c not in skip]


def _snap_volume_azimuths(sweeps, azimuths, grids, radar):
    """Move each ray onto its sweep's canonical azimuth, in place of the measured one.

    The antenna reports where it actually pointed, which drifts by a few
    hundredths of a degree between volumes; ``gate_id`` resolves 0.1°, so an
    unsnapped ray lands in a neighboring bin and its gates match no LUT row.
    See :func:`raddb.lut.nominal_azimuth_grid`.

    Returns
    -------
    (azimuths, worst_move_deg)

    Raises
    ------
    ValueError
        If a sweep is missing from the LUT, holds a different number of rays, or
        a ray sits further than half a ray spacing from the grid — all of which
        mean this volume was scanned with a different strategy than the LUT was
        built for, and no snapping can reconcile them.
    """
    out = np.array(azimuths, dtype=np.float64, copy=True)
    worst = 0.0
    for sweep in np.unique(sweeps):
        grid = grids.get(int(sweep))
        if grid is None:
            raise ValueError(
                f"radar {radar!r}: the LUT has no sweep {int(sweep)}, but this "
                f"volume does — it uses a different scan strategy.",
            )
        sel = sweeps == sweep
        n_rays = np.unique(out[sel]).size
        # Fewer rays than the grid is a rotation with holes — a volume that
        # dropped a ray or two, which every network does — and each surviving ray
        # still snaps to its own grid point, so it archives correctly.  More rays
        # than the grid cannot: they have nowhere to go.
        if n_rays > len(grid):
            raise ValueError(
                f"radar {radar!r} sweep {int(sweep)}: volume has {n_rays} rays, "
                f"the LUT was built for {len(grid)} — a different scan strategy. "
                f"Archive it under its own radar name, or rebuild the LUT.",
            )
        snapped, dist = snap_azimuths_to_grid(out[sel], grid)
        tol = azimuth_grid_tolerance(grid)
        if dist.size and dist.max() > tol:
            raise ValueError(
                f"radar {radar!r} sweep {int(sweep)}: a ray sits "
                f"{dist.max() / AZIMUTH_SCALE:.3f}° from the nearest LUT azimuth, "
                f"beyond the half-spacing tolerance of {tol / AZIMUTH_SCALE:.3f}° "
                f"— this is not antenna drift.",
            )
        out[sel] = snapped.astype(np.float64) / AZIMUTH_SCALE
        worst = max(worst, float(dist.max()) if dist.size else 0.0)
    return out, worst / AZIMUTH_SCALE


def _build_polar_dataframe(
    df: pl.DataFrame | pd.DataFrame,
    radar: str,
    filter_feature: str,
    filter_threshold: float,
    filter_logic: str,
    azimuth_grids: dict | None = None,
    variables: list[str] | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Filter a flattened volume DataFrame and attach gate_ids.

    Rows that do not satisfy ``filter_feature [filter_logic] filter_threshold``
    are dropped (row removal — zeros in surviving gates stay zeros).

    ``variables`` names the moment columns to keep; ``None`` keeps every non-geometry
    column.  See :func:`_resolve_polar_columns`.

    This is the single shared core of :func:`datatree_to_parquet` and
    :func:`archive_volume`.

    ``azimuth_grids`` maps sweep -> canonical azimuths (tenths of a degree); when
    given, every ray is snapped onto it before its ``gate_id`` is built, so the
    volume joins its LUT exactly.  ``None`` — an archive predating the grid —
    keeps the measured azimuths, i.e. the previous behavior.

    Returns
    -------
    (df_polar, mask) : the polar DataFrame (gate_id + polar columns) and the
    boolean row mask, which callers reuse to align other per-row arrays.
    """
    fn = resolve_filter_logic(filter_logic)

    if filter_feature in df.columns:
        mask = fn(_col(df, filter_feature), filter_threshold)
    else:
        logger.warning(
            "filter_feature '%s' not found in DataFrame; keeping all gates.",
            filter_feature,
        )
        mask = np.ones(len(df), dtype=bool)

    sweeps_all = _col(df, "sweep", np.int64)
    azimuths_all = _col(df, "azimuth", np.float64)
    if azimuth_grids:
        # Snapped before the filter, so the scan-strategy check counts the
        # volume's rays rather than only those that survived the filter.
        azimuths_all, worst = _snap_volume_azimuths(
            sweeps_all,
            azimuths_all,
            azimuth_grids,
            radar,
        )
        logger.debug("radar %s: rays snapped, max move %.3f deg.", radar, worst)
    else:
        # The old, silent failure mode: without a grid the measured azimuths go
        # straight into gate_id, and every ray whose 0.1° bin drifted since the
        # LUT was built produces gates that match no LUT row and vanish from
        # every join.  Say so rather than losing 6-35% of the volume quietly.
        logger.warning(
            "radar %s: the LUT records no nominal azimuth grid, so measured "
            "azimuths are used as-is and some gates may not join it. Regenerate "
            "the LUT to fix this.",
            radar,
        )

    gate_ids = encode_gate_ids(
        radar,
        sweeps_all[mask],
        azimuths_all[mask],
        _col(df, "range")[mask],
    )

    polar_cols = _resolve_polar_columns(df, variables)
    if "time" in df.columns:
        polar_cols = ["time", *polar_cols]
    df_polar = pl.DataFrame(
        {"gate_id": gate_ids, **{c: _col(df, c)[mask] for c in polar_cols}},
    )
    return df_polar, mask


def archive_volume(
    dt: xr.DataTree,
    radar: str,
    base_output_path: str,
    filter_feature: str = "DBZH",
    filter_threshold: float = 0.0,
    filter_logic: str = ">",
    timer=None,
    volume: str | None = None,
    variables: list[str] | None = None,
) -> str | None:
    """Archive a single DataTree volume to Parquet format.

    Converts the DataTree to a DataFrame, generates a ``gate_id`` for each
    gate (linking back to the LUT), drops gates that do not satisfy
    ``filter_feature [filter_logic] filter_threshold``, and saves the
    result as a POL parquet file.

    Non-matching gates are **dropped** (row removal), so zero values in
    surviving gates are never converted to NaN.  NaN values in the
    reconstruction come only from gates absent in the parquet (i.e. gates
    that were filtered out or had no data in the original DataTree).

    Parameters
    ----------
    dt : xarry.DataTree
        Processed volume (any radar network).
    radar : str
        Radar identifier, e.g. ``"A"`` or ``"KTLX"``.
    base_output_path : str
        Base output directory for parquet files.
    filter_feature : str
        Column to use for gate filtering (default ``"DBZH"``).
    filter_threshold : float
        Threshold value for the filter (default ``0.0``).
    filter_logic : str
        Comparison operator (default ``">"``).
    timer : StageTimer, optional
        Profiling timer.
    volume : str, optional
        Volume label for timer records.
    variables : list of str, optional
        Moments to archive.  ``None`` (default) archives every **per-gate** variable the
        volume carries — see :func:`_gate_variables` — so nothing a network records is
        silently dropped.  Pass a list to keep an archive lean.

    Returns
    -------
    str or None
        Path to the saved POL parquet file, or ``None`` when the volume held
        nothing to archive — every gate failed the filter, or its ``time`` was
        all-``NaT``.  That is a *skip*, not a failure: callers should count it
        separately rather than treating it as either stored or broken.
    """
    radar = normalize_radar_name(radar)
    resolve_filter_logic(filter_logic)  # fail fast before flattening

    with timer.time_stage("datatree_to_df", volume=volume) if timer else _nullctx():
        df = datatree_to_dataframe(dt)

    with timer.time_stage("generate_gate_ids", volume=volume) if timer else _nullctx():
        df_polar, _mask = _build_polar_dataframe(
            df,
            radar,
            filter_feature,
            filter_threshold,
            filter_logic,
            azimuth_grids=load_azimuth_grids(radar, base_output_path),
            variables=_gate_variables(dt) if variables is None else variables,
        )

    with timer.time_stage("save_parquet", volume=volume) if timer else _nullctx():
        return _save_polar_parquet(_finalize_polar_dtypes(df_polar), radar, base_output_path)


def archive_multiple_volumes(
    volumes: list[xr.DataTree] | dict[str, xr.DataTree],
    radar: str,
    base_output_path: str,
    filter_feature: str = "DBZH",
    filter_threshold: float = 0.0,
    filter_logic: str = ">",
    verbose: bool = True,
    timer: StageTimer | None = None,
    variables: list[str] | None = None,
) -> list[dict]:
    """Archive multiple DataTree volumes sequentially.

    Parameters
    ----------
    volumes : list or dict
        If a list, each element is a DataTree.
        If a dict, keys are volume labels and values are DataTrees.
    radar : str
        Radar identifier.
    base_output_path : str
        Base output directory.
    filter_feature : str
        Column to use for gate filtering (default ``"DBZH"``).
    filter_threshold : float
        Threshold value (default ``0.0``).
    filter_logic : str
        Comparison operator (default ``">"``).
    verbose : bool
        Print progress.
    timer : StageTimer, optional
        Profiling timer.
    variables : list of str, optional
        Moments to archive.  ``None`` (default) archives every **per-gate** variable each
        volume carries — see :func:`_gate_variables`.

    Returns
    -------
    list of dict
        Results with keys: label, success, skipped, error, polar_path, n_gates.
        ``skipped`` marks a volume that held nothing to archive (every gate
        failed the filter, or an all-``NaT`` time); it has ``success=False``
        and ``error=None``, so the three states stay distinguishable.
    """
    radar = normalize_radar_name(radar)

    if isinstance(volumes, dict):
        items = list(volumes.items())
    else:
        items = [(f"volume_{i}", dt) for i, dt in enumerate(volumes)]

    results = []
    pipeline_t0 = _time.perf_counter()

    for i, (label, dt) in enumerate(items, 1):
        _vprint(f"\n>>  Volume {i}/{len(items)}: {label}", verbose)

        result = {
            "label": label,
            "radar": radar,
            "success": False,
            "skipped": False,
            "error": None,
            "n_gates": 0,
        }

        vol_t0 = _time.perf_counter()
        try:
            with timer.time_stage("archive_volume", volume=label) if timer else _nullctx():
                polar_path = archive_volume(
                    dt,
                    radar=radar,
                    base_output_path=base_output_path,
                    filter_feature=filter_feature,
                    filter_threshold=filter_threshold,
                    filter_logic=filter_logic,
                    timer=timer,
                    volume=label,
                    variables=variables,
                )

            vol_elapsed = _time.perf_counter() - vol_t0
            if polar_path is None:
                # Nothing to archive — not an error, so it must not be counted
                # as one; see archive_volume's return contract.
                result["skipped"] = True
                result["polar_path"] = None
                _vprint(
                    f"SKIP  Volume {i}/{len(items)} held no gates to archive " f"({vol_elapsed:.1f}s)",
                    verbose,
                )
            else:
                result["success"] = True
                result["polar_path"] = polar_path

                df_polar = pd.read_parquet(polar_path)
                result["n_gates"] = len(df_polar)

                _vprint(
                    f"OK  Volume {i}/{len(items)} done in " f"{vol_elapsed:.1f}s -- {result['n_gates']:,} gates saved",
                    verbose,
                )

        except Exception as e:
            vol_elapsed = _time.perf_counter() - vol_t0
            result["error"] = str(e)
            _vprint(
                f"FAIL  Volume {i}/{len(items)} FAILED in " f"{vol_elapsed:.1f}s: {e}",
                verbose,
            )
            logger.error(f"[{i}/{len(items)}] {label} - FAIL: {e}")

        results.append(result)

    total_elapsed = _time.perf_counter() - pipeline_t0
    n_ok = sum(1 for r in results if r["success"])
    n_skip = sum(1 for r in results if r["skipped"])
    _vprint(
        f"\nArchiving complete: {n_ok}/{len(results)} volumes "
        f"in {total_elapsed:.1f}s" + (f" ({n_skip} skipped, nothing to archive)" if n_skip else ""),
        verbose,
    )
    return results


def archive_volumes_multi_radar(
    volumes_by_radar: dict[str, list[xr.DataTree] | dict[str, xr.DataTree]],
    base_output_path: str,
    filter_feature: str = "DBZH",
    filter_threshold: float = 0.0,
    filter_logic: str = ">",
    verbose: bool = True,
    timer: StageTimer | None = None,
    variables: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Archive volumes for multiple radars sequentially.

    Parameters
    ----------
    volumes_by_radar : dict
        Keys are radar names, values are lists or dicts of DataTrees.
        Example: ``{"A": [dt1, dt2], "D": [dt3, dt4]}``
    base_output_path : str
        Base output directory.
    filter_feature : str
        Column to use for gate filtering (default ``"DBZH"``).
    filter_threshold : float
        Threshold value (default ``0.0``).
    filter_logic : str
        Comparison operator (default ``">"``).
    verbose : bool
        Print progress.
    timer : StageTimer, optional
        Profiling timer.
    variables : list of str, optional
        Moments to archive.  ``None`` (default) archives every **per-gate** variable each
        volume carries — see :func:`_gate_variables`.

    Returns
    -------
    dict
        Keys are radar names, values are lists of result dicts.
    """
    all_results = {}

    for radar, volumes in volumes_by_radar.items():
        radar = normalize_radar_name(radar)
        _vprint(
            f"\n{'='*60}\nArchiving radar {radar}\n{'='*60}",
            verbose,
        )

        results = archive_multiple_volumes(
            volumes,
            radar=radar,
            base_output_path=base_output_path,
            filter_feature=filter_feature,
            filter_threshold=filter_threshold,
            filter_logic=filter_logic,
            verbose=verbose,
            timer=timer,
            variables=variables,
        )

        all_results[radar] = results

    return all_results


def _finalize_polar_dtypes(df_polar: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """Narrow every float column to float32.

    Decided by dtype rather than by column name, so a moment this package has never
    heard of is stored as compactly as ``DBZH``.  float32 gives 7 significant digits,
    which is more than any radar moment carries; ``gate_id`` (int64) and ``time``
    (datetime) are untouched.
    """
    df_polar = _to_polars_frame(df_polar)
    floats = [c for c, dtype in zip(df_polar.columns, df_polar.dtypes, strict=False) if dtype == pl.Float64]
    if floats:
        df_polar = df_polar.with_columns([pl.col(c).cast(pl.Float32) for c in floats])
    return df_polar


def datatree_to_parquet(
    dt: xr.DataTree,
    radar: str,
    base_output_path: str,
    filter_feature: str = "DBZH",
    filter_threshold: float = 0.0,
    filter_logic: str = ">",
    max_workers: int = 1,
    variables: list[str] | None = None,
) -> str:
    """Convert a DataTree to a filtered POL parquet file.

    Gates that do not satisfy ``filter_feature [filter_logic] filter_threshold``
    are dropped (row removal) before saving.  Zero values in surviving gates
    are preserved as-is.

    ``variables`` names the moments to write; ``None`` writes every per-gate variable
    the tree carries (:func:`_gate_variables`).
    """
    resolve_filter_logic(filter_logic)  # fail fast before flattening

    df = datatree_to_dataframe(dt, max_workers)
    df_polar, _mask = _build_polar_dataframe(
        df,
        radar,
        filter_feature,
        filter_threshold,
        filter_logic,
        azimuth_grids=load_azimuth_grids(radar, base_output_path),
        variables=_gate_variables(dt) if variables is None else variables,
    )
    return _save_polar_parquet(_finalize_polar_dtypes(df_polar), radar, base_output_path)


# ============================================================================
# Parquet -> DataFrame / DataTree  (reading archived data)
# ============================================================================


def parquet_to_dataframe(
    radar: str,
    base_path: str | Path,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    columns: list[str] | None = None,
    merge_lut: bool = False,
) -> pl.DataFrame:
    """Load archived POLAR parquet files as a single **polars** DataFrame.

    Parameters
    ----------
    radar : str
        Radar identifier, e.g. ``"A"`` or ``"KTLX"``.
    base_path : str or Path
        RadDB base directory.
    start_time, end_time : optional
        Filter by volume timestamp.
    columns : list of str, optional
        Columns to load from parquet files.
    merge_lut : bool
        If True, merge with the LUT to add spatial coordinates
        (azimuth, range, latitude, longitude, altitude, x, y, z).

    Returns
    -------
    pl.DataFrame
        Includes a ``volume_time`` column (the volume timestamp parsed from each
        source filename) so a multi-volume frame can be split back into single
        volumes — used by :func:`dataframe_to_datatree` / PPI plotting.
    """
    radar_path = Path(base_path) / radar
    if not radar_path.exists():
        logger.warning(f"Radar directory not found: {radar_path}")
        return pl.DataFrame()

    polar_files = _find_polar_files_in_range(radar_path, start_time, end_time)

    if not polar_files:
        logger.warning(
            f"No POLAR data found for radar {radar} " f"between {start_time} and {end_time}",
        )
        return pl.DataFrame()

    if columns is not None:
        # ``volume_time`` / ``radar`` are derived below (and in RadDB.open), not
        # stored in the POL files; asking pyarrow for them raises and would drop
        # every volume.
        columns = [c for c in columns if c not in _NON_GATE_METADATA_COLS]

    dfs = []
    for f in polar_files:
        try:
            df = pl.read_parquet(f, columns=columns)
            # Tag each row with its volume timestamp (from the filename) so a
            # multi-volume DataFrame can later be split back into single volumes
            # (per-gate `time` spans the whole ~5 min scan and cannot separate
            # back-to-back volumes reliably).
            vt = _parse_pol_time(f)
            df = df.with_columns(
                pl.lit(vt.tz_localize(None) if vt is not None else None).cast(pl.Datetime("ns")).alias("volume_time"),
            )
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
            continue

    if not dfs:
        return pl.DataFrame()

    df_all = pl.concat(dfs, how="vertical_relaxed")

    if merge_lut:
        lut_path = radar_path / "LUT" / f"{radar}_LUT.parquet"
        if lut_path.exists():
            lut_df = pl.read_parquet(lut_path)
            lut_cols = [
                "gate_id",
                "sweep",
                "azimuth",
                "range",
                "elevation_angle",
                "latitude",
                "longitude",
                "altitude",
                "x",
                "y",
                "z",
            ]
            # Include any projected coordinate columns added by add_lut_projection
            # (e.g. x_2056, y_2056 for Swiss LV95 / EPSG:2056)
            lut_cols += _projection_columns(lut_df)
            lut_cols = [c for c in lut_cols if c in lut_df.columns]
            # maintain_order="left" reproduces pandas' left-merge row order.
            df_all = df_all.join(
                lut_df.select(lut_cols),
                on="gate_id",
                how="left",
                maintain_order="left",
            )
        else:
            logger.warning(
                f"LUT not found at {lut_path}. " "Returning data without spatial coordinates.",
            )

    return df_all


def scan_polar_parquet(
    radar: str,
    base_path: str | Path,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    columns: list[str] | None = None,
) -> pl.LazyFrame | None:
    """Scan archived POLAR parquet files as a single polars LazyFrame.

    The polars counterpart of :func:`parquet_to_dataframe`, used by
    :meth:`raddb.RadDB.open`.  Scanning (rather than reading) lets polars push the
    column projection into the parquet reader, and avoids the pandas
    intermediate frames and the ``pd.concat`` copy of the full result.

    Unlike :func:`parquet_to_dataframe` this never merges the LUT: the static
    geometry stays in its own table and is joined only by the ``to_*``
    converters.

    Parameters
    ----------
    radar : str
        Radar identifier, e.g. ``"A"`` or ``"KTLX"``.
    base_path : str or Path
        RadDB base directory.
    start_time, end_time : optional
        Filter by volume timestamp.
    columns : list of str, optional
        Columns to project.  ``volume_time`` / ``radar`` are added here rather
        than read, so they are dropped from the parquet projection.

    Returns
    -------
    pl.LazyFrame or None
        ``None`` when no volume matches — callers decide what an empty result
        means.  The frame carries a ``volume_time`` and a ``radar`` column.
    """
    radar_path = Path(base_path) / radar
    if not radar_path.exists():
        logger.warning(f"Radar directory not found: {radar_path}")
        return None

    polar_files = _find_polar_files_in_range(radar_path, start_time, end_time)
    if not polar_files:
        logger.warning(
            f"No POLAR data found for radar {radar} " f"between {start_time} and {end_time}",
        )
        return None

    if columns is not None:
        columns = [c for c in columns if c not in _NON_GATE_METADATA_COLS]

    scans = []
    for f in polar_files:
        try:
            lf = pl.scan_parquet(f)
            if columns is not None:
                lf = lf.select(columns)
            # Tag each row with its volume timestamp (from the filename) so a
            # multi-volume frame can later be split back into single volumes
            # (per-gate `time` spans the whole ~5 min scan and cannot separate
            # back-to-back volumes reliably).  The dtype is pinned so files with
            # an unparsable name still concatenate with the rest.  Microsecond
            # resolution matches what :func:`parquet_to_dataframe` produces.
            ts = _parse_pol_time(f)
            scans.append(
                lf.with_columns(
                    pl.lit(ts.to_pydatetime() if ts is not None else None, dtype=pl.Datetime("us", "UTC")).alias(
                        "volume_time",
                    ),
                    pl.lit(radar).alias("radar"),
                ),
            )
        except Exception as e:
            logger.warning(f"Error scanning {f}: {e}")
            continue

    if not scans:
        return None
    # Diagonal, not vertical: :func:`_gate_variables` keeps a volume's moments in the
    # order its own file lists them, so two volumes of the same radar read from
    # different sources (a converted zarr and a raw ODIM, say) archive the same moments
    # in a different column order, and a plain vertical concat rejects that with
    # "schema names differ".  A radar can also gain or lose a moment over the life of an
    # archive.  Diagonal takes the union by name and null-fills what a file lacks.
    return pl.concat(scans, how="diagonal_relaxed")


def parquet_to_datatree(
    radar: str,
    base_path: str | Path,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    label_column: str = "DBZH",
    max_workers: int = 1,
) -> xr.DataTree:
    """Load archived POLAR parquet files and reconstruct a DataTree.

    Loads all volumes in the given time range, joins with the LUT to
    recover azimuth/range coordinates, and reconstructs an xarray DataTree.

    Parameters
    ----------
    radar : str
        Radar identifier, e.g. ``"A"`` or ``"KTLX"``.
    base_path : str or Path
        RadDB base directory.
    start_time, end_time : optional
        Filter by volume timestamp.
    label_column : str
        Column to use for reconstruction (default ``"DBZH"``).
    max_workers : int
        Parallel workers for sweep reconstruction.

    Returns
    -------
    xr.DataTree

    Raises
    ------
    FileNotFoundError
        If the LUT or radar info files are missing.
    ValueError
        If no data is found.
    """
    base = Path(base_path)
    radar_path = base / radar

    lut_path = radar_path / "LUT" / f"{radar}_LUT.parquet"
    info_path = radar_path / "LUT" / f"{radar}_info.yaml"

    if not lut_path.exists():
        raise FileNotFoundError(
            f"LUT not found at {lut_path}. Run generate_lut() first.",
        )
    if not info_path.exists():
        raise FileNotFoundError(f"Radar info not found at {info_path}.")

    polar_files = _find_polar_files_in_range(radar_path, start_time, end_time)
    if not polar_files:
        raise ValueError(
            f"No POLAR data found for {radar} " f"between {start_time} and {end_time}",
        )

    # Load all POLAR files in range
    dfs = []
    for f in polar_files:
        try:
            dfs.append(pd.read_parquet(f, engine="pyarrow"))
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
            continue

    if not dfs:
        raise ValueError(f"All POLAR files for {radar} failed to load.")

    df_polar = pd.concat(dfs, ignore_index=True)

    return dataframe_to_datatree(
        df=df_polar,
        radar=radar,
        base_path=base_path,
        label_column=label_column,
        max_workers=max_workers,
    )


# Columns the LUT owns; a DataFrame's own copies of these are replaced by the
# LUT's on reconstruction so geometry is always authoritative and never collides.
_LUT_GEOMETRY_COLS = (
    "sweep",
    "azimuth",
    "range",
    "latitude",
    "longitude",
    "altitude",
    "x",
    "y",
    "z",
)
# Pure per-volume metadata that must not become gridded data_vars.
_NON_GATE_METADATA_COLS = ("radar", "volume_time")


def dataframe_to_datatree(
    df: pl.DataFrame | pd.DataFrame,
    radar: str,
    base_path: str | Path,
    label_column: str = "DBZH",
    max_workers: int = 1,
) -> xr.DataTree:
    """Reconstruct a DataTree from an in-memory per-gate DataFrame.

    The df→DataTree core shared by :func:`parquet_to_datatree` and by DataFrame
    plotting: it joins ``df`` with the radar LUT on ``gate_id`` to recover
    geometry (sweep/azimuth/range + lat/lon/alt/x/y/z and any projection cols),
    fills the full ``(azimuth x range)`` grid, and NaN-fills gates absent from
    ``df`` — so a **cropped/filtered** DataFrame reconstructs to a DataTree that
    carries the correct geometry but only the rows present in ``df``, with
    **the DataFrame's own values** (honoring crops or added feature columns).

    ``df`` must be a **single radar and a single volume** already (see
    :meth:`raddb.RadDB.datatree_from_df` for the radar/volume selection helper).
    Any of the LUT's geometry columns already on ``df`` are dropped and taken
    from the LUT instead, so a ``crop_bbox`` frame (which carries
    ``sweep``/``x_2056``/``y_2056``/``z``/``altitude``) reconstructs cleanly.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-gate rows with a ``gate_id`` column (+ measurement columns).
    radar : str
        Radar identifier whose LUT to join against.
    base_path : str or Path
        RadDB archive base directory.
    label_column : str
        Feature used for reconstruction (default ``"DBZH"``).
    max_workers : int
        Parallel workers for sweep reconstruction.

    Returns
    -------
    xr.DataTree

    Raises
    ------
    FileNotFoundError
        If the LUT or radar info files are missing.
    ValueError
        If ``df`` is empty or no gates match the LUT.
    """
    base = Path(base_path)
    radar_path = base / radar
    lut_path = radar_path / "LUT" / f"{radar}_LUT.parquet"
    info_path = radar_path / "LUT" / f"{radar}_info.yaml"
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT not found at {lut_path}. Run generate_lut() first.")
    if not info_path.exists():
        raise FileNotFoundError(f"Radar info not found at {info_path}.")

    # xarray seam: reconstruction below needs pandas indexing/reindexing.
    df = _to_pandas_frame(df)
    if df.empty:
        raise ValueError("dataframe_to_datatree: input DataFrame is empty.")

    lut_df = pd.read_parquet(lut_path, engine="pyarrow")
    join_cols = ["gate_id", *(_LUT_GEOMETRY_COLS), *_projection_columns(lut_df)]
    join_cols = [c for c in join_cols if c in lut_df.columns]

    # Drop the df's own copies of LUT/metadata columns so geometry comes solely
    # from the LUT (authoritative, no _x/_y merge collisions) and constant
    # metadata doesn't turn into gridded variables.
    drop = [
        c
        for c in (*_LUT_GEOMETRY_COLS, *_projection_columns(lut_df), *_NON_GATE_METADATA_COLS)
        if c != "gate_id" and c in df.columns
    ]
    df_meas = df.drop(columns=drop)

    df_joined = df_meas.merge(lut_df[join_cols], on="gate_id", how="left")
    df_joined = df_joined.dropna(subset=["azimuth", "range"])

    # If several volumes slipped through, keep the latest obs per gate so each
    # (sweep, azimuth, range) cell is unique before gridding.
    if "time" in df_joined.columns:
        df_joined = df_joined.sort_values("time")
    df_joined = df_joined.drop_duplicates(subset=["sweep", "azimuth", "range"], keep="last")

    if df_joined.empty:
        raise ValueError("No matching gates found between the DataFrame and the LUT.")

    return reconstruct_datatree(
        df_joined=df_joined,
        lut_path=lut_path,
        radar_info_path=info_path,
        label_column=label_column,
        max_workers=max_workers,
    )


# ============================================================================
# Reconstruction (Parquet + LUT -> DataTree)
# ============================================================================


def labels_to_dataframe(
    labels: np.ndarray,
    gate_ids,
    extra_columns: dict | None = None,
) -> pl.DataFrame:
    """Create a **polars** DataFrame from prediction labels and gate IDs."""
    data = {"gate_id": np.asarray(gate_ids), "hydrometeor_class": np.asarray(labels)}
    if extra_columns:
        data.update(extra_columns)
    return pl.DataFrame(data)


def join_labels_with_lut(
    df_labels: pl.DataFrame | pd.DataFrame,
    lut_path: str | Path,
) -> pl.DataFrame:
    """Join label data with the LUT to recover spatial coordinates.

    Accepts a polars or pandas label frame; always returns polars.
    """
    df_labels = _to_polars_frame(df_labels)
    df_lut = pl.read_parquet(str(lut_path))
    cols = [c for c in df_labels.columns if c != "gate_id"]
    # maintain_order="left" reproduces pandas' left-merge row order.
    return df_lut.join(
        df_labels.select(["gate_id", *cols]),
        on="gate_id",
        how="left",
        maintain_order="left",
    )


_PER_GATE_COORDS = ("latitude", "longitude", "altitude", "x", "y", "z")


def _get_sweep_coords(sweep, radar_info):
    coords = {
        "site_latitude": radar_info["latitude"],
        "site_longitude": radar_info["longitude"],
        "site_altitude": radar_info["altitude"],
        "sweep_number": sweep,
    }
    meta = radar_info.get("sweeps", {}).get(sweep, {})
    if "elevation" in meta:
        coords["elevation_angle"] = meta["elevation"]
    return coords


def reconstruct_sweep_dataset(
    df_joined: pl.DataFrame | pd.DataFrame,
    sweep: int,
    lut_df: pl.DataFrame | pd.DataFrame,
    radar_info: dict,
    label_column: str = "hydrometeor_class",  # noqa: ARG001  kept for signature parity with the callers below
    sweep_corners: dict | None = None,
) -> xr.Dataset:
    """Reconstruct a single sweep Dataset from joined data.

    Data variables come from the filtered POL parquet (NaN where gates were
    dropped). Per-gate spatial coords (lat/lon/alt/x/y/z and any x_<epsg>/
    y_<epsg>) come from the LUT directly so every gate has a valid geometry
    regardless of filtering.

    If ``sweep_corners`` is provided, per-gate edge arrays
    (``x_edges``, ``y_edges``, ``z_edges``, ``lon_edges``, ``lat_edges``) of
    shape ``(n_az+1, n_range+1)`` are attached as data_vars for pcolormesh
    rendering with ``shading="flat"``.
    """
    # xarray seam: MultiIndex reindexing + to_xarray() are pandas-only.
    df_joined = _to_pandas_frame(df_joined)
    lut_df = _to_pandas_frame(lut_df)

    df_sweep = df_joined[df_joined["sweep"] == sweep].copy()

    # Identify spatial columns that should come from the LUT (always populated)
    # and non-spatial columns that should come from the polar data (may be NaN).
    spatial_cols = [c for c in _PER_GATE_COORDS if c in lut_df.columns]
    spatial_cols += _projection_columns(lut_df)

    non_spatial = [
        c for c in df_sweep.columns if c not in ("gate_id", "sweep", "azimuth", "range") and c not in spatial_cols
    ]

    idx = get_full_sweep_index(lut_df, sweep)

    # Data variables (may have NaN for filtered-out gates)
    df_reidx = df_sweep.set_index(["azimuth", "range"])[non_spatial].reindex(idx)

    # Spatial coords from the LUT, aligned to the same (azimuth, range) index
    lut_sweep = lut_df[lut_df["sweep"] == sweep].set_index(["azimuth", "range"])
    lut_spatial = lut_sweep[spatial_cols].reindex(idx)
    df_full = pd.concat([df_reidx, lut_spatial], axis=1)

    ds = df_full.to_xarray().assign_coords(
        _get_sweep_coords(sweep, radar_info),
    )

    # Promote per-gate spatial vars to coords so the Dataset is plot-ready.
    promote = [c for c in spatial_cols if c in ds.data_vars]
    if promote:
        ds = ds.set_coords(promote)

    # Attach per-sweep gate edge arrays (N_az+1, N_range+1) for accurate
    # pcolormesh(shading="flat") rendering. These use their own dims
    # (azimuth_edge, range_edge) so they coexist with the primary grid.
    if sweep_corners:
        for key in ("x_edges", "y_edges", "z_edges", "lon_edges", "lat_edges"):
            if key in sweep_corners:
                arr = np.asarray(sweep_corners[key])
                ds[key] = (("azimuth_edge", "range_edge"), arr)
    return ds


def reconstruct_datatree(
    df_joined: pl.DataFrame | pd.DataFrame,
    lut_path: str | Path,
    radar_info_path: str | Path,
    label_column: str = "hydrometeor_class",
    max_workers: int = 1,
) -> xr.DataTree:
    """Reconstruct a full DataTree from joined data + LUT + radar info.

    Accepts a polars or pandas ``df_joined``; pandas is used internally because
    this is the xarray seam (see :func:`_to_pandas_frame`).
    """
    df_joined = _to_pandas_frame(df_joined)
    lut_df = pd.read_parquet(str(lut_path), engine="pyarrow")
    with open(str(radar_info_path)) as f:
        radar_info = yaml.safe_load(f)

    # Optional: load per-sweep gate corners from <radar>_corners.npz.
    # These enable pcolormesh(shading="flat") rendering in plot_ppi. Missing
    # file → plots fall back to centroid-based rendering (less accurate).
    lut_dir = Path(lut_path).parent
    radar_name = Path(lut_path).name.split("_")[0]
    corners_file = lut_dir / f"{radar_name}_corners.npz"
    sweep_corners_all: dict[int, dict] = {}
    if corners_file.exists():
        try:
            sweep_corners_all = _parse_corners_npz(corners_file)
        except Exception as exc:
            logger.warning(f"Failed to load sweep corners from {corners_file}: {exc}")

    s_serie = pd.to_numeric(df_joined["sweep"], errors="coerce").dropna().unique()
    sweeps = [int(s) for s in s_serie]
    if not sweeps:
        raise ValueError("No valid sweep numbers found in df_joined.")

    def _rec(sw):
        try:
            return sw, reconstruct_sweep_dataset(
                df_joined,
                sweep=sw,
                lut_df=lut_df,
                radar_info=radar_info,
                label_column=label_column,
                sweep_corners=sweep_corners_all.get(sw),
            )
        except Exception as exc:
            logger.warning(f"Failed to reconstruct sweep {sw}: {exc}")
            return sw, None

    dict_ds = {}
    if max_workers <= 1:
        for sw in sweeps:
            k, ds = _rec(sw)
            if ds is not None:
                dict_ds[f"sweep_{k}"] = ds
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as ex:
            for fut in concurrent.futures.as_completed(
                [ex.submit(_rec, sw) for sw in sweeps],
            ):
                k, ds = fut.result()
                if ds is not None:
                    dict_ds[f"sweep_{k}"] = ds

    if not dict_ds:
        raise ValueError("No sweeps could be reconstructed.")
    return xr.DataTree.from_dict(dict_ds)


# ============================================================================
# Feature addition utilities
# ============================================================================


def add_feature_to_df(
    df: pl.DataFrame | pd.DataFrame,
    feature_name: str,
    compute_fn: callable,
) -> pl.DataFrame | pd.DataFrame:
    """Add a new column to a DataFrame computed from existing columns.

    Accepts polars or pandas and returns the **same kind**, so ``compute_fn``
    receives the frame flavour the caller passed in.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Input DataFrame (e.g. from :func:`parquet_to_dataframe`).
    feature_name : str
        Name of the new column to add.
    compute_fn : callable
        Function that takes ``df`` and returns a Series or array of the
        same length.  Example::

            def my_feature(df):
                return df["ZDR"] + df["DBZH"] * 0.1

    Returns
    -------
    pl.DataFrame or pd.DataFrame
        Copy of ``df`` (same kind) with the new column appended.
    """
    if isinstance(df, pl.DataFrame):
        return df.with_columns(pl.Series(feature_name, np.asarray(compute_fn(df))))
    df = df.copy()
    df[feature_name] = compute_fn(df)
    return df


def add_feature_to_dt(
    dt: xr.DataTree,
    feature_name: str,
    compute_fn: callable,
) -> xr.DataTree:
    """Add a new variable to every sweep in a DataTree.

    Parameters
    ----------
    dt : xarray.DataTree
        Input DataTree with ``sweep_N`` groups.
    feature_name : str
        Name of the new variable to add to each sweep Dataset.
    compute_fn : callable
        Function that takes an ``xr.Dataset`` (one sweep) and returns an
        ``xr.DataArray`` with matching dimensions.  Example::

            def kdp_proxy(ds):
                return (ds["PHIDP"].diff("range") / 0.250).clip(0)

    Returns
    -------
    xr.DataTree
        New DataTree with the computed variable added to every sweep.
    """
    from raddb.helper import list_sweep_names

    sweep_names = list_sweep_names(dt)
    dict_ds = {}
    for sweep_name in sweep_names:
        ds = dt[sweep_name].to_dataset()
        new_var = compute_fn(ds)
        ds = ds.assign({feature_name: new_var})
        dict_ds[sweep_name] = ds
    return xr.DataTree.from_dict(dict_ds)

"""PPI, RHI, CAPPI and vertical-cross-section plots for RadDB.

The four gate-accurate plots are :func:`plot_ppi`, :func:`plot_rhi`,
:func:`plot_cappi` and :func:`plot_vcs`.

A radar gate is not a rectangle: it is a curved frustum whose footprint depends
on range, azimuth, elevation and Earth curvature.  Every plot here draws that
footprint as an explicit polygon, so a filtered, ``sel``-ed or cropped input
renders exactly the gates it still holds — nothing is reindexed onto a full
azimuth x range grid.

Geometry follows the input.  A RadDB or DataFrame reads the LUT lattices; an
``xr.DataTree`` is self-describing and its corners are computed from its own
azimuth/range/elevation with the same 4/3-Earth model that built the LUT; a
GeoDataFrame is treated as a frame.  There is a single geometry path — always the
exact frustum — so what is drawn does not depend on how it was asked for.
"""

from __future__ import annotations

import contextlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

# ============================================================================
# Per-variable plotting defaults
# ============================================================================


def _first_available_cmap(*names: str) -> str:
    """First registered colormap among ``names`` (last is the guaranteed fallback).

    Lets ``DBZH`` prefer Py-ART's ``HomeyerRainbow`` when pyart has been imported
    (registering it), falling back to ``turbo`` otherwise.
    """
    available = set(plt.colormaps())
    for name in names:
        if name in available:
            return name
    return names[-1]


#: Per-variable plot styling, keyed by moment name.  A variable with no entry is drawn
#: with matplotlib's defaults and labelled with its own name, so plotting never depends
#: on being listed here.
#:
#: **This is the extension point for a network's own moments.**  RadDB ships the ODIM
#: moments it can style generically; a package that produces others registers them at
#: import time::
#:
#:     from raddb.viz.plot import PLOT_DEFAULTS
#:
#:     PLOT_DEFAULTS["HCLASS"] = {
#:         "discrete": True,
#:         "classes": ["rain", "wet snow", "graupel", ...],
#:         "colors": ["royalblue", "yellow", "orangered", ...],
#:         "label": "Hydrometeor class",
#:     }
#:
#: Recognised keys: ``cmap``, ``vmin``/``vmax``, ``norm`` (a Normalize, or a zero-argument
#: callable returning a fresh one), ``label`` for the colorbar, and ``discrete`` +
#: ``classes`` + ``colors`` for a classification, which gets a ListedColormap, a
#: BoundaryNorm over 1..n and the class names as colorbar ticks.
#:
#: KDP deliberately departs from the reference figures: they used a cyclic ``twilight``,
#: which is misleading for a signed quantity.
PLOT_DEFAULTS: dict[str, dict] = {
    "DBZH": {"cmap": "HomeyerRainbow", "vmin": 0, "vmax": 60, "label": "Reflectivity [dBz]"},
    "DBZH_raw": {"cmap": "HomeyerRainbow", "vmin": 0, "vmax": 60, "label": "Raw reflectivity [dBz]"},
    "TH": {"cmap": "HomeyerRainbow", "vmin": 0, "vmax": 60, "label": "Total reflectivity [dBz]"},
    "ZDR": {"cmap": "viridis", "vmin": -2, "vmax": 7, "label": "Differential reflectivity [dB]"},
    "ZDR_raw": {"cmap": "viridis", "vmin": -2, "vmax": 7, "label": "Raw differential reflectivity [dB]"},
    "KDP": {"cmap": "plasma", "vmin": -2, "vmax": 5, "label": "Specific differential phase [°/km]"},
    "RHOHV": {"cmap": "cividis", "vmin": 0.5, "vmax": 1.0, "label": "Co-polar correlation [-]"},
    "PHIDP": {"cmap": "twilight", "vmin": -180, "vmax": 180, "label": "Differential phase [deg]"},
    "VRADH": {"cmap": "coolwarm", "label": "Radial velocity [m/s]"},
    "WRADH": {"cmap": "magma", "label": "Spectrum width [m/s]"},
}


# ============================================================================
# Internal helpers
# ============================================================================

_PYART_CMAPS_TRIED = False


def _ensure_cmap_registered(name):
    """Return a usable colormap name; register Py-ART's colormaps on demand.

    Non-string values (Colormap instances) pass through.  A string already known
    to matplotlib is returned as-is.  Otherwise Py-ART is imported **once** — that
    registers its colormaps (e.g. ``HomeyerRainbow``) with matplotlib — and the
    name is re-checked.  If it's still missing (pyart absent or unknown name),
    fall back to ``turbo`` with a warning so plotting never hard-crashes.
    """
    global _PYART_CMAPS_TRIED
    if not isinstance(name, str) or name in plt.colormaps():
        return name
    if not _PYART_CMAPS_TRIED:
        _PYART_CMAPS_TRIED = True
        with contextlib.suppress(Exception):  # pyart is optional
            import pyart  # noqa: F401  # registers Py-ART colormaps with matplotlib
    if name in plt.colormaps():
        return name
    import warnings

    warnings.warn(
        f"colormap {name!r} is unavailable (Py-ART colormaps need pyart installed); " "falling back to 'turbo'.",
        stacklevel=2,
    )
    return "turbo"


def _resolve_plot_kwargs(variable: str, user_kwargs: dict):
    """Merge per-variable defaults with user overrides.

    Returns (plot_kwargs, is_discrete, class_labels, cbar_label). Always sets
    ``cmap.set_bad("none")`` so NaN (filtered) gates render transparent
    instead of picking up a colormap endpoint.
    """
    defaults = PLOT_DEFAULTS.get(variable, {})
    is_discrete = bool(defaults.get("discrete", False))
    class_labels = defaults.get("classes")
    cbar_label = defaults.get("label", variable)

    plot_kwargs = dict(user_kwargs)
    if is_discrete:
        n = len(class_labels)
        class_colors = defaults.get("colors")
        if class_colors is not None:
            cmap = ListedColormap(class_colors[:n])
        else:
            base_cmap = plt.get_cmap(plot_kwargs.pop("cmap", "tab10"), n)
            cmap = ListedColormap([base_cmap(i) for i in range(n)])
        bounds = np.arange(0.5, n + 1.5)
        norm = BoundaryNorm(bounds, cmap.N)
        plot_kwargs.setdefault("cmap", cmap)
        plot_kwargs.setdefault("norm", norm)
    else:
        if "cmap" in defaults:
            plot_kwargs.setdefault("cmap", defaults["cmap"])
        if "norm" in defaults:
            # a default norm (e.g. 0-centered diverging); build a fresh instance so
            # it isn't shared/mutated across figures. Skipped if the caller passed
            # any explicit scale (norm / vmin / vmax) to avoid a matplotlib clash.
            if not any(k in plot_kwargs for k in ("norm", "vmin", "vmax")):
                nrm = defaults["norm"]
                plot_kwargs["norm"] = nrm() if callable(nrm) else nrm
        else:
            for k in ("vmin", "vmax"):
                if k in defaults:
                    plot_kwargs.setdefault(k, defaults[k])

    # Make NaN gates transparent (not bottom-of-colormap colored).
    cmap_val = plot_kwargs.get("cmap")
    if cmap_val is not None:
        cmap_val = _ensure_cmap_registered(cmap_val)  # register pyart cmaps if needed
        cmap_obj = plt.get_cmap(cmap_val).copy() if isinstance(cmap_val, str) else cmap_val.copy()
        cmap_obj.set_bad("none")
        plot_kwargs["cmap"] = cmap_obj

    return plot_kwargs, is_discrete, class_labels, cbar_label


def _maybe_cartopy():
    """Try to import cartopy. Return (ccrs, cfeature) or (None, None)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        return ccrs, cfeature
    except ImportError:
        return None, None


_NE_BORDERS = None  # cache: Natural Earth context lines with their style, in lon/lat
_BORDER_LINES = {}  # cache: (crs, clip box) -> those lines in that frame

# Natural Earth layers drawn by ``context=True``, with the style each gets.
# Coastline and national borders alone leave a radar in the middle of a large
# country — KTLX in Oklahoma, say — with nothing at all to draw, so state and
# province lines are in there too, thinner and paler so they read as secondary.
_NE_LAYERS = (
    ("cultural", "admin_0_boundary_lines_land", {"color": "0.4", "linewidth": 0.6}),
    ("physical", "coastline", {"color": "0.4", "linewidth": 0.6}),
    ("cultural", "admin_1_states_provinces_lines", {"color": "0.65", "linewidth": 0.4}),
)


def _ne_border_lines():
    """Natural Earth 10 m context lines as ``[(geom, style), ...]`` in lon/lat (cached).

    Read through cartopy's shapereader.  A missing cartopy or an uncached
    download warns and yields nothing, so the plot still draws — just without
    context.  Drawn on a plain matplotlib axis, so no GeoAxes / extent quirks.
    """
    global _NE_BORDERS
    if _NE_BORDERS is None:
        try:
            from cartopy.io import shapereader as shpreader

            _NE_BORDERS = []
            for category, name, style in _NE_LAYERS:
                path = shpreader.natural_earth(resolution="10m", category=category, name=name)
                _NE_BORDERS.extend((g, style) for g in shpreader.Reader(path).geometries())
        except Exception as exc:  # - cartopy missing / data not cached
            import warnings

            warnings.warn(
                f"cartopy country borders unavailable ({exc}); plotted without them.",
                stacklevel=2,
            )
            _NE_BORDERS = []
    return _NE_BORDERS


def _border_lines(crs, clip):
    """Context lines clipped to the lon/lat box ``clip``, in ``crs`` (cached).

    ``crs`` is whatever frame the plot is drawn in — an EPSG int, or the proj4
    string of the radar-centered azimuthal frame ``coords="xy"`` uses.  Clipping
    before reprojecting keeps it local, which matters for frames that are only
    valid near their own area (LV95, a single UTM zone).
    """
    key = (str(crs), tuple(round(float(v), 2) for v in clip))
    if key not in _BORDER_LINES:
        import shapely

        from raddb.aoi import _reproject_to_aoi

        box = shapely.box(*clip)
        lines = []
        for geom, style in _ne_border_lines():
            piece = geom.intersection(box)
            if not piece.is_empty:
                lines.append((_reproject_to_aoi(piece, 4326, crs), style))
        _BORDER_LINES[key] = lines
    return _BORDER_LINES[key]


def _draw_borders(ax, mode, epsg, info, reach_deg: float = 3.0):
    """Draw country borders around the radar site, in the plot's own frame.

    Works for every ``coords`` value, not only the Swiss projected one: ``"xy"``
    reprojects through an azimuthal-equidistant frame centered on the radar, which
    is what the LUT's radar-relative meters already are.
    """
    if info is None:
        return
    lon, lat = float(info["longitude"]), float(info["latitude"])
    if mode == "lonlat":
        crs = 4326
    elif mode == "xy":
        crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    else:
        crs = epsg

    def _iter_lines(g):
        if g.geom_type == "LineString":
            yield g
        elif g.geom_type in ("MultiLineString", "GeometryCollection"):
            for sub in g.geoms:
                yield from _iter_lines(sub)

    dlon = reach_deg / max(np.cos(np.radians(lat)), 0.1)
    clip = (lon - dlon, lat - reach_deg, lon + dlon, lat + reach_deg)
    for geom, style in _border_lines(crs, clip):
        for line in _iter_lines(geom):
            ax.plot(*line.xy, zorder=1, **style)


def _add_colorbar(p, ax, is_discrete: bool, class_labels, label: str):
    """Attach either a continuous or a categorical colorbar."""
    if is_discrete and class_labels is not None:
        n = len(class_labels)
        cbar = plt.colorbar(
            p,
            ax=ax,
            ticks=np.arange(1, n + 1),
            boundaries=np.arange(0.5, n + 1.5),
            spacing="uniform",
            fraction=0.046,
            pad=0.04,
        )
        cbar.ax.set_yticklabels(class_labels)
        cbar.set_label(label)
    else:
        cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)
    return cbar


def _draw_range_rings_xy(ax, distances_km=(50, 100, 150), **kwargs):
    """Draw range rings as dashed circles on a cartesian (km) axis."""
    kwargs.setdefault("color", "k")
    kwargs.setdefault("linewidth", 0.5)
    kwargs.setdefault("linestyle", "--")
    theta = np.linspace(0, 2 * np.pi, 361)
    for d in distances_km:
        ax.plot(d * np.cos(theta), d * np.sin(theta), **kwargs)


def _volume_time_str(ds) -> str:
    """Extract a readable timestamp from a sweep Dataset for plot titles."""
    if "time" not in ds.coords and "time" not in ds.data_vars:
        return ""
    vals = np.asarray(ds["time"].values).ravel()
    mask = ~pd.isna(vals)
    if not mask.any():
        return ""
    return str(vals[mask][0])[:19]


# ============================================================================
# Gate geometry straight from the LUT lattices
#
# The four plot entry points (plot_ppi / plot_rhi / plot_cappi / plot_vcs) all
# follow the same three steps:
#
#   1. resolve the input to a polars frame + the archive it came from,
#   2. narrow it to one radar and one volume,
#   3. join the surviving ``gate_id``s onto per-gate corners read from the
#      h_plane / v_plane lattices and hand the vertices to a PolyCollection.
#
# Nothing is reindexed onto a full (azimuth x range) grid on the way, which is
# what lets a cropped or filtered frame plot exactly the gates it still holds.
# ============================================================================

_ARCHIVE_HINT = (
    "pass archive_dir= (the directory holding {radar}/LUT/), or call the method "
    "on a RadDB built with RadDB(archive_dir=...) so it can be inferred."
)


class _Source:
    """What the plots need about their input, resolved once.

    ``kind`` is ``"frame"`` (RadDB / polars / pandas), ``"gdf"`` (a GeoDataFrame,
    whose own geometry is used for rough mode) or ``"datatree"`` (self-describing:
    geometry is computed from its coordinates, no archive involved).
    """

    __slots__ = ("df", "base", "crs", "radar", "tstr", "info", "kind", "dtree", "gdf")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def require_base(self, what: str):
        if self.base is None:
            raise ValueError(f"{what} needs the archive: " + _ARCHIVE_HINT)
        return self.base


def _beamwidth(src):
    """Antenna beamwidth [deg] for a DataTree's *vertical* faces.

    Only reached for DataTree input, where the faces are computed on the fly.
    Archive-backed data needs none: the beamwidth was applied when ``v_plane``
    was generated, and is recorded in ``info.yaml``.

    Inferred exactly as :func:`raddb.lut.generate_lut_from_datatree` infers it —
    from the file's own CfRadial/ODIM attribute, else
    :data:`raddb.lut.DEFAULT_BEAMWIDTH_DEG`.  Neither standard makes that
    attribute mandatory and no volume here carries one, so in practice an RHI or
    CAPPI drawn straight from a DataTree assumes a 1 deg beam.  Archive the
    volume to set it explicitly.
    """
    from raddb.lut import _beamwidth_from_datatree

    return _beamwidth_from_datatree(src.dtree)


def _resolve_frame(data, archive_dir=None):
    """Normalize a plot input to a :class:`_Source`.

    Accepts a :class:`~raddb.main.RadDB`, a polars or pandas frame, a
    GeoDataFrame, or an ``xr.DataTree`` / ``xr.Dataset``.
    """
    from pathlib import Path

    if isinstance(data, (xr.DataTree, xr.Dataset)):
        return _Source(kind="datatree", dtree=data, base=Path(archive_dir) if archive_dir else None)

    crs, base, gdf = None, archive_dir, None

    if hasattr(data, "data") and not isinstance(data, (pl.DataFrame, pd.DataFrame)):
        # A RadDB (avoid importing it: viz is imported from raddb/__init__).
        if base is None:
            base = getattr(data, "archive_dir", None)
        crs = getattr(data, "_crs", None)
        data = data.data

    kind = "frame"
    if isinstance(data, pd.DataFrame):
        if hasattr(data, "geometry") and hasattr(data, "crs"):
            kind, gdf = "gdf", data
            if crs is None and data.crs is not None:
                epsg = data.crs.to_epsg()
                crs = epsg if epsg is not None else None
            data = pd.DataFrame(data.drop(columns=data.geometry.name))
        # A cross-sectioned frame carries shapely objects in `cs_polygon`, which
        # pyarrow cannot convert — WKB-encode them the way the RadDB converters
        # do, and plot_vcs decodes them again on the way out.
        from raddb.main import _encode_geometry

        data = pl.from_pandas(_encode_geometry(data))

    if not isinstance(data, pl.DataFrame):
        raise TypeError(
            f"expected a RadDB, polars/pandas frame, GeoDataFrame or DataTree; " f"got {type(data).__name__}.",
        )
    if data.is_empty():
        raise ValueError("no data to plot (the frame is empty).")
    if "gate_id" not in data.columns:
        raise KeyError("frame has no 'gate_id' column; gate geometry cannot be joined.")
    return _Source(kind=kind, df=data, base=Path(base) if base else None, crs=crs, gdf=gdf)


def _select_radar(df: pl.DataFrame, radar: str | None) -> tuple[pl.DataFrame, str]:
    """Narrow to a single radar, inferring it when the frame holds only one."""
    from raddb.aoi import _radars_from_gate_ids
    from raddb.helper import normalize_radar_name

    if "radar" in df.columns:
        present = sorted(df["radar"].drop_nulls().unique().to_list())
    else:
        present = _radars_from_gate_ids(df["gate_id"])

    if radar is None:
        if len(present) != 1:
            raise ValueError(
                f"data spans radars {present}; pass radar= to pick one " "(one plot draws one radar).",
            )
        return df, present[0]

    radar = normalize_radar_name(radar)
    if "radar" in df.columns:
        out = df.filter(pl.col("radar") == radar)
    else:
        from raddb.lut import GATE_ID_RADAR_BASE, encode_radar_code

        prefix = encode_radar_code(radar) * GATE_ID_RADAR_BASE
        out = df.filter(
            (pl.col("gate_id") >= prefix) & (pl.col("gate_id") < prefix + GATE_ID_RADAR_BASE),
        )
    if out.is_empty():
        raise ValueError(f"no rows for radar {radar!r}; present: {present}.")
    return out, radar


def _select_volume(df: pl.DataFrame, timestep=None, start_time=None, end_time=None):
    """Narrow to a single volume. Returns ``(frame, time label)``.

    ``start_time`` / ``end_time`` restrict the candidates; ``timestep`` then picks
    the nearest volume.  If several volumes still remain and no ``timestep`` was
    given, raise rather than silently drawing them on top of each other.
    """
    col = next((c for c in ("volume_time", "time") if c in df.columns), None)
    if col is None:
        return df, ""

    def _bound(v):
        ts = pd.Timestamp(v)
        dtype = df.schema[col]
        tz = getattr(dtype, "time_zone", None)
        if tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        elif tz is None and ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts

    if start_time is not None:
        df = df.filter(pl.col(col) >= _bound(start_time))
    if end_time is not None:
        df = df.filter(pl.col(col) <= _bound(end_time))
    if df.is_empty():
        raise ValueError("no data left after the start_time/end_time window.")

    vols = sorted(df[col].drop_nulls().unique().to_list())
    if not vols:
        return df, ""
    if timestep is not None:
        target = _bound(timestep)
        chosen = min(vols, key=lambda v: abs(pd.Timestamp(v) - target))
    elif len(vols) == 1:
        chosen = vols[0]
    else:
        raise ValueError(
            f"data holds {len(vols)} volumes ({vols[0]} ... {vols[-1]}); pass "
            "timestep= to pick one, or narrow with start_time=/end_time=.",
        )
    return df.filter(pl.col(col) == chosen), str(chosen)[:19]


# ---------------------------------------------------------------- coordinates

_COORD_ALIASES = {
    "cartesian": "xy",
    "xy": "xy",
    "radar": "xy",
    "geo": "lonlat",
    "lonlat": "lonlat",
    "latlon": "lonlat",
    "wgs": "lonlat",
    "projected": "projected",
    "proj": "projected",
    "swiss": 2056,
    "lv95": 2056,
    "2056": 2056,
}


def _resolve_coords(coords, crs):
    """Map the user's ``coords`` to ``(mode, epsg)``.

    ``mode`` is ``"xy"`` (meters from the radar), ``"lonlat"`` (WGS-84 degrees)
    or ``"projected"`` (the LUT's ``x_<epsg>`` / ``y_<epsg>`` columns).
    """
    if isinstance(coords, (int, np.integer)) and not isinstance(coords, bool):
        return "projected", int(coords)
    key = str(coords).lower()
    if key not in _COORD_ALIASES:
        raise ValueError(
            f"coords must be 'xy', 'lonlat', 'projected' or an EPSG int; got {coords!r}.",
        )
    resolved = _COORD_ALIASES[key]
    if isinstance(resolved, int):
        return "projected", resolved
    if resolved == "projected":
        if crs is None:
            raise ValueError(
                "coords='projected' needs a CRS; build the RadDB with " "RadDB(crs=...) or pass an EPSG int as coords.",
            )
        return "projected", int(crs)
    return resolved, None


def _corner_vertices(tbl: pl.DataFrame, n_corners: int, mode: str, epsg, info):
    """Per-gate corner rings as an ``(n_gates, n_corners, 2)`` float array.

    ``tbl`` is a :func:`raddb.lut.gate_corner_table` result for the ``h_plane``
    (4 corners).  ``mode`` selects the output frame; ``lonlat`` is derived from
    the radar-relative meters with the same spherical model the LUT was built
    with, so it stays consistent with the stored ``latitude``/``longitude``.
    """
    if mode == "projected":
        xs = [f"x_{epsg}_{k}" for k in range(1, n_corners + 1)]
        ys = [f"y_{epsg}_{k}" for k in range(1, n_corners + 1)]
        if not all(c in tbl.columns for c in xs):
            raise KeyError(
                f"the h_plane lattice has no EPSG:{epsg} columns. Regenerate the "
                "LUT with that projection, or use coords='xy' / 'lonlat'.",
            )
    else:
        xs = [f"x_{k}" for k in range(1, n_corners + 1)]
        ys = [f"y_{k}" for k in range(1, n_corners + 1)]

    ring = np.stack(
        [np.stack([tbl[xc].to_numpy(), tbl[yc].to_numpy()], axis=1) for xc, yc in zip(xs, ys, strict=False)],
        axis=1,
    ).astype(np.float64)

    if mode == "lonlat":
        from raddb.lut import cartesian_to_geographic

        lat, lon, _ = cartesian_to_geographic(
            ring[:, :, 0],
            ring[:, :, 1],
            np.zeros(ring.shape[:2]),
            info["latitude"],
            info["longitude"],
            info["altitude"],
        )
        ring = np.stack([lon, lat], axis=2)
    return ring


def _join_corners(df: pl.DataFrame, tbl: pl.DataFrame, variable: str):
    """Align a per-gate corner table with the data frame, dropping unusable rows.

    Returns ``(values, corner table)`` in matching row order.  Gates whose
    variable is NaN, or that the LUT has no geometry for, are dropped — drawing
    them would paint the colormap's "bad" color over real data.
    """
    if variable not in df.columns:
        raise KeyError(f"variable {variable!r} not in the data; have {df.columns}.")

    joined = (
        df.select(["gate_id", variable])
        .join(tbl.drop("sweep"), on="gate_id", how="inner", maintain_order="left")
        .filter(pl.col(variable).is_not_nan() & pl.col(variable).is_not_null())
    )
    if joined.is_empty():
        raise ValueError(
            f"no gates left to draw: every {variable!r} value is NaN, or none of "
            "the gates matched the LUT geometry.",
        )
    return joined[variable].to_numpy(), joined


# ------------------------------------------------- approximate ("rough") gates


def _dt_sweep_names(dt):
    """Sweep group names of a DataTree, ordered by sweep number."""
    if isinstance(dt, xr.Dataset):
        return [None]
    names = [g.lstrip("/") for g in dt.groups if g.lstrip("/").startswith("sweep_")]
    if not names:
        raise ValueError("no sweep_* groups found in the DataTree.")
    return sorted(names, key=lambda s: int(s.split("_")[-1]))


def _dt_sweep(dt, sweep):
    """One sweep Dataset from a DataTree (or a Dataset passed straight through)."""
    if isinstance(dt, xr.Dataset):
        return dt
    name = sweep if isinstance(sweep, str) else f"sweep_{int(sweep)}"
    names = _dt_sweep_names(dt)
    if name not in names:
        raise ValueError(f"sweep {name!r} not in the DataTree; available: {names}.")
    return dt[name].to_dataset()


def _dt_site(ds):
    """Site (lat, lon, alt) from a sweep Dataset, as a radar-info-shaped dict."""
    missing = [k for k in ("latitude", "longitude", "altitude") if k not in ds.variables and k not in ds.coords]
    if missing:
        raise KeyError(
            f"the DataTree sweep has no {missing} coordinate(s), so the radar site "
            "is unknown. xradar volumes normally carry them per sweep.",
        )
    return {k: float(np.asarray(ds[k]).ravel()[0]) for k in ("latitude", "longitude", "altitude")}


def _dt_gate_table(ds, variable):
    """Flatten one DataTree sweep into the per-gate columns the plots consume.

    A DataTree is self-describing: ``azimuth``, ``range`` and ``elevation`` are
    all the geometry needs, so nothing is read from an archive.  Returns a polars
    frame of values plus the antenna columns, ordered ray-major.
    """
    if variable not in ds.variables:
        raise KeyError(
            f"variable {variable!r} not in this sweep; have {sorted(ds.data_vars)}.",
        )
    az = np.asarray(ds["azimuth"].values, dtype=np.float64)
    rng = np.asarray(ds["range"].values, dtype=np.float64)
    el = np.asarray(ds["elevation"].values, dtype=np.float64)
    if el.ndim == 0:
        el = np.full(az.shape, float(el))

    vals = np.asarray(ds[variable].values, dtype=np.float64)
    if vals.shape != (az.size, rng.size):
        vals = vals.T
    n_az, n_rng = az.size, rng.size

    return pl.DataFrame(
        {
            "azimuth": np.repeat(az, n_rng),
            "range": np.tile(rng, n_az),
            "elevation_angle": np.repeat(el, n_rng),
            variable: vals.ravel(),
        },
    )


def _dt_h_vertices(ds, mode, epsg, info):
    """Exact horizontal footprints computed from a DataTree's own coordinates.

    Runs the identical pipeline that built the ``h_plane`` lattice — range edges,
    complex-plane azimuth edges, ``antenna_vectors_to_cartesian`` at ``ke=4/3`` —
    so the result matches the stored geometry without reading any file.

    Takes no beamwidth: the horizontal face sits at the beam *center*, so it is
    beamwidth-independent (verified: 0.8 deg and 1.2 deg give identical nodes).
    """
    from raddb.lut import (
        GATE_RING_OFFSETS,
        antenna_vectors_to_cartesian,
        cartesian_to_geographic,
    )

    az = np.asarray(ds["azimuth"].values, dtype=np.float64)
    rng = np.asarray(ds["range"].values, dtype=np.float64)
    el = np.asarray(ds["elevation"].values, dtype=np.float64)
    if el.ndim == 0:
        el = np.full(az.shape, float(el))

    x, y, _ = antenna_vectors_to_cartesian(rng, az, el, edges=True)
    if mode == "lonlat":
        lat, lon, _ = cartesian_to_geographic(
            x,
            y,
            np.zeros_like(x),
            info["latitude"],
            info["longitude"],
            info["altitude"],
        )
        x, y = lon, lat
    elif mode == "projected":
        x, y = _project_nodes_xy(x, y, epsg, info)

    n_az, n_rng = az.size, rng.size
    ai = np.repeat(np.arange(n_az), n_rng)
    ri = np.tile(np.arange(n_rng), n_az)
    return np.stack(
        [np.stack([x[ai + i, ri + j], y[ai + i, ri + j]], axis=1) for i, j in GATE_RING_OFFSETS],
        axis=1,
    )


def _dt_v_vertices(ds, height, info, beamwidth_deg):
    """Exact vertical faces from a DataTree's own coordinates, in ``(d, z)``.

    Evaluates the beam at ``el ± β`` — the one input a DataTree does not carry,
    hence the ``beamwidth_deg`` argument.
    """
    from raddb.lut import antenna_vectors_to_cartesian

    az = np.asarray(ds["azimuth"].values, dtype=np.float64)
    rng = np.asarray(ds["range"].values, dtype=np.float64)
    el = np.asarray(ds["elevation"].values, dtype=np.float64)
    if el.ndim == 0:
        el = np.full(az.shape, float(el))

    half = float(beamwidth_deg) / 2.0
    faces = {}
    for lvl in (-1, 1):
        x, y, z = antenna_vectors_to_cartesian(rng, az, el + lvl * half, edges=True)
        faces[lvl] = (np.hypot(x, y), z + (0.0 if height == "rel" else info["altitude"]))

    n_az, n_rng = az.size, rng.size
    ai = np.repeat(np.arange(n_az), n_rng)
    ri = np.tile(np.arange(n_rng), n_az)
    # near-bottom, far-bottom, far-top, near-top
    picks = ((-1, 0), (-1, 1), (1, 1), (1, 0))
    return np.stack(
        [np.stack([faces[lvl][0][ai, ri + j], faces[lvl][1][ai, ri + j]], axis=1) for lvl, j in picks],
        axis=1,
    )


def _project_nodes_xy(x, y, epsg, info):
    """Radar-relative meters -> a projected CRS, for DataTree-computed geometry."""
    import pyproj

    from raddb.aoi import _to_pyproj_crs
    from raddb.lut import cartesian_to_geographic

    lat, lon, _ = cartesian_to_geographic(
        x,
        y,
        np.zeros_like(x),
        info["latitude"],
        info["longitude"],
        info["altitude"],
    )
    tf = pyproj.Transformer.from_crs(_to_pyproj_crs(4326), _to_pyproj_crs(epsg), always_xy=True)
    px, py = tf.transform(np.asarray(lon).ravel(), np.asarray(lat).ravel())
    return np.asarray(px).reshape(x.shape), np.asarray(py).reshape(y.shape)


def _nearest_ray_per_sweep(frame: pl.DataFrame, target: float, az_tol: float):
    """Keep, in each sweep, only the rows on that sweep's ray closest to ``target``.

    ``frame`` must carry ``sweep`` and a precomputed ``_off`` column holding the
    absolute angular offset from ``target`` on the circle.  Sweeps whose closest
    ray is further than ``az_tol`` are dropped with a warning; if none qualify,
    raise.  Returns ``(rows, mean azimuth of the chosen rays)``.
    """
    best = frame.group_by("sweep").agg(pl.col("_off").min().alias("_best"))
    within = best.filter(pl.col("_best") <= az_tol)
    if within.is_empty():
        raise ValueError(
            f"no sweep has a ray within ±{az_tol}° of azimuth {target}°; the "
            f"closest is {float(best['_best'].min()):.2f}° away.",
        )
    if within.height < best.height:
        import warnings

        warnings.warn(
            f"{best.height - within.height} of {best.height} sweeps have no ray "
            f"within ±{az_tol}° of azimuth {target}° and are omitted.",
            stacklevel=2,
        )
    picked = frame.join(within, on="sweep", how="inner").filter(
        pl.col("_off") == pl.col("_best"),
    )
    return picked, float(picked["azimuth"].mean())


def _dt_rhi(src, target, variable, az_tol, height, beamwidth_deg):
    """RHI geometry and values from a DataTree, one nearest ray per sweep."""
    all_values, all_verts, rays = [], [], []
    for name in _dt_sweep_names(src.dtree):
        ds = _dt_sweep(src.dtree, name)
        if variable not in ds.variables:
            continue
        az = np.asarray(ds["azimuth"].values, dtype=np.float64)
        off = np.abs(((az - target + 180.0) % 360.0) - 180.0)
        j = int(np.argmin(off))
        if off[j] > az_tol:
            continue
        rays.append(float(az[j]))

        # Geometry is built from the *whole* sweep and the chosen ray selected
        # afterwards: edge interpolation needs the full azimuth and range
        # vectors, so slicing first would move the outermost edges.
        tbl = _dt_gate_table(ds, variable)
        verts = _dt_v_vertices(ds, height, src.info, beamwidth_deg)

        n_rng = np.asarray(ds["range"].values).size
        rows = slice(j * n_rng, (j + 1) * n_rng)  # table is ray-major
        vals = tbl[variable].to_numpy()[rows]
        verts = verts[rows]
        keep = np.isfinite(vals)
        all_values.append(vals[keep])
        all_verts.append(verts[keep])

    if not rays:
        raise ValueError(
            f"no sweep of this DataTree has a ray within ±{az_tol}° of azimuth {target}°.",
        )
    return (np.concatenate(all_values), np.concatenate(all_verts), float(np.mean(rays)))


# `_fill_lowest` is accepted for signature parity with the LUT path but has no
# effect here: a DataTree carries every sweep already, so nothing is missing.
def _dt_cappi(src, altitude, variable, height, overlap, _fill_lowest, mode, epsg, beamwidth_deg):
    """CAPPI geometry and values from a DataTree, with no archive involved.

    Mirrors the LUT path: cut the exact ``(d, z)`` faces at the slice altitude to
    get each range bin's along-beam chord, resolve overlapping sweeps, then trim
    the horizontal footprint to that chord.
    """
    site_alt = float(src.info["altitude"])
    z0 = float(altitude) + (site_alt if height == "rel" else 0.0)

    per_sweep, chords = {}, []
    for name in _dt_sweep_names(src.dtree):
        ds = _dt_sweep(src.dtree, name)
        if variable not in ds.variables:
            continue
        sw = int(str(name).split("_")[-1])
        per_sweep[sw] = ds
        d_near, d_far, rng_idx, dz = _dt_sweep_chords(ds, z0, site_alt, beamwidth_deg)
        if rng_idx.size:
            chords.append(
                pl.DataFrame(
                    {
                        "sweep": np.full(rng_idx.size, sw, dtype=np.int32),
                        "rng_idx": rng_idx.astype(np.int32),
                        "d_near": d_near.astype(np.float32),
                        "d_far": d_far.astype(np.float32),
                        "z_center": np.zeros(rng_idx.size, dtype=np.float32),
                        "dz_center": dz.astype(np.float32),
                    },
                ),
            )
    if not chords:
        raise ValueError(
            f"no beam of this DataTree reaches {altitude} m "
            f"({'ASL' if height == 'asl' else 'above the radar'}); nothing to draw.",
        )
    table = pl.concat(chords, how="vertical")
    if overlap == "nearest":
        table = _resolve_chord_overlap(table)

    all_values, all_verts = [], []
    for sw, ds in per_sweep.items():
        sel = table.filter(pl.col("sweep") == sw)
        if sel.is_empty():
            continue
        ri = sel["rng_idx"].to_numpy()
        dn, df_ = sel["d_near"].to_numpy(), sel["d_far"].to_numpy()

        # Build the whole sweep, then keep the selected range bins: edge
        # interpolation must see the full range vector, so subsetting the
        # Dataset first would shift the outermost gate edges.
        tbl = _dt_gate_table(ds, variable)
        full = _dt_h_vertices(ds, mode, epsg, src.info)
        full_xy = _dt_h_vertices(ds, "xy", None, src.info)

        n_az = np.asarray(ds["azimuth"].values).size
        n_rng = np.asarray(ds["range"].values).size
        rows = np.repeat(np.arange(n_az), ri.size) * n_rng + np.tile(ri, n_az)  # table is ray-major
        verts = _trim_footprints_to_chord(
            full[rows],
            full_xy[rows],
            np.tile(dn, n_az),
            np.tile(df_, n_az),
        )
        vals = tbl[variable].to_numpy()[rows]
        keep = np.isfinite(vals)
        all_values.append(vals[keep])
        all_verts.append(verts[keep])

    if not all_values or sum(len(v) for v in all_values) == 0:
        raise ValueError(f"every {variable!r} value on the {altitude} m slice is NaN.")
    return np.concatenate(all_values), np.concatenate(all_verts)


def _dt_sweep_chords(ds, z0, site_alt, beamwidth_deg):
    """Along-beam chord of the ``z = z0`` cut per range bin, from a DataTree sweep.

    The DataTree counterpart of :func:`raddb.lut.cappi_chords`: identical quad
    clipping, but the ``(d, z)`` faces are computed from the sweep's own
    coordinates instead of read from ``v_plane``.
    """
    from raddb.lut import antenna_vectors_to_cartesian

    # d and z do not depend on azimuth, so one ray describes the sweep — but the
    # edge interpolation needs the whole azimuth vector, so pass it all and keep
    # the first row.
    az = np.asarray(ds["azimuth"].values, dtype=np.float64)
    rng = np.asarray(ds["range"].values, dtype=np.float64)
    el = np.asarray(ds["elevation"].values, dtype=np.float64)
    if el.ndim == 0:
        el = np.full(az.shape, float(el))

    half = float(beamwidth_deg) / 2.0
    faces = {}
    for lvl in (-1, 1):
        x, y, z = antenna_vectors_to_cartesian(rng, az, el + lvl * half, edges=True)
        faces[lvl] = (np.hypot(x, y)[0], z[0] + site_alt)

    ring_d = np.stack([faces[-1][0][:-1], faces[-1][0][1:], faces[1][0][1:], faces[1][0][:-1]], axis=1)
    ring_z = np.stack([faces[-1][1][:-1], faces[-1][1][1:], faces[1][1][1:], faces[1][1][:-1]], axis=1)

    za, zb = ring_z, np.roll(ring_z, -1, axis=1)
    da, db = ring_d, np.roll(ring_d, -1, axis=1)
    sa, sb = za - z0, zb - z0
    crosses = ((sa <= 0) & (sb >= 0)) | ((sa >= 0) & (sb <= 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(zb != za, (z0 - za) / (zb - za), 0.0)
    d_cross = np.where(crosses, da + np.clip(t, 0.0, 1.0) * (db - da), np.nan)

    hit = np.isfinite(d_cross).any(axis=1)
    if not hit.any():
        return (np.empty(0),) * 4
    d_hit = d_cross[hit]
    z_center = ring_z.mean(axis=1)[hit]
    return (np.nanmin(d_hit, axis=1), np.nanmax(d_hit, axis=1), np.flatnonzero(hit), np.abs(z_center - z0))


class _KmFormatter(mticker.Formatter):
    """Tick labels in km, from axis data held in meters.

    The decimals come from the tick spacing matplotlib actually chose.  A fixed
    ``.0f`` silently collapses adjacent labels whenever that step drops below
    1 km: a 1.4-6.0 km cross-section gets 500 m steps and reads
    ``1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6`` — wrong, and worse, plausible-looking.
    (The irregularity is round-half-to-even: 2.5 prints as "2" but 3.5 as "4".)

    ``offset`` subtracts a false origin before scaling, for projected frames such
    as LV95 whose easting starts at 2 000 km.
    """

    def __init__(self, offset: float = 0.0):
        self.offset = float(offset)

    #: Never print more than this many decimals, whatever the ticks ask for.
    MAX_DECIMALS = 4

    def __call__(self, v, _pos=None):
        return f"{(v - self.offset) / 1e3:.{self._decimals()}f}"

    def _decimals(self) -> int:
        """Fewest decimals that write every tick on this axis *exactly*.

        Deriving them from ``log10(step)`` is not enough: matplotlib routinely
        picks steps of 2.5x10**n, where 2500 m would give 0 decimals and print
        0, 2.5, 5, 7.5, 10 as ``0, 2, 5, 8, 10`` — distinct, so it survives a
        duplicate check, but wrong. Asking instead which precision reproduces the
        values handles every step shape.
        """
        if self.axis is None:
            return 0
        locs = np.asarray(self.axis.get_majorticklocs(), dtype=float)
        if locs.size == 0:
            return 0
        km = (locs - self.offset) / 1e3
        for d in range(self.MAX_DECIMALS + 1):
            tol = 1e-6 * np.maximum(1.0, np.abs(km))
            if np.all(np.abs(km - np.round(km, d)) <= tol):
                return d
        return self.MAX_DECIMALS


def _draw_polygons(ax, verts, values, plot_kwargs, edgecolor, rasterized):
    """Add a PolyCollection of gate polygons colored by ``values``."""
    from matplotlib.collections import PolyCollection

    pc = PolyCollection(
        verts,
        array=np.asarray(values, dtype=np.float64),
        edgecolor=edgecolor,
        linewidth=0.1,
    )
    if "cmap" in plot_kwargs:
        pc.set_cmap(plot_kwargs["cmap"])
    if plot_kwargs.get("norm") is not None:
        pc.set_norm(plot_kwargs["norm"])
    else:
        pc.set_clim(plot_kwargs.get("vmin"), plot_kwargs.get("vmax"))
    if rasterized:
        pc.set_rasterized(True)
    ax.add_collection(pc)
    return pc


def _finish_map_axes(ax, mode, epsg, verts, site_xy, xlim, ylim, add_range_rings=True, context=False, info=None):
    """Labels, tick formatting, aspect, range rings and limits for a map plot."""
    if mode == "lonlat":
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        scale = 1.0
    else:
        # Native meters on the axis, km on the tick labels.
        ox, oy = (2e6, 1e6) if epsg == 2056 else (0.0, 0.0)
        ax.xaxis.set_major_formatter(_KmFormatter(ox))
        ax.yaxis.set_major_formatter(_KmFormatter(oy))
        if mode == "xy":
            ax.set_xlabel("East from radar [km]")
            ax.set_ylabel("North from radar [km]")
        else:
            ax.set_xlabel("East [km]")
            ax.set_ylabel("North [km]")
        scale = 1e3

    if context:
        _draw_borders(ax, mode, epsg, info)

    if site_xy is not None:
        ax.plot(*site_xy, "kx", markersize=7, markeredgewidth=2, zorder=5)
        if add_range_rings and mode != "lonlat":
            theta = np.linspace(0, 2 * np.pi, 361)
            for d_km in (50, 100, 150):
                ax.plot(
                    site_xy[0] + d_km * scale * np.cos(theta),
                    site_xy[1] + d_km * scale * np.sin(theta),
                    "k--",
                    linewidth=0.5,
                    zorder=1,
                )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Default view: a square centered on the radar, sized by the furthest gate
    # drawn.  A plain data bounding box would be pulled off-center by a handful
    # of distant echoes, and would differ between variables and time steps —
    # this keeps panels comparable and the radar where the eye expects it.
    if site_xy is not None and (xlim is None or ylim is None):
        reach = float(np.nanmax(np.hypot(verts[:, :, 0] - site_xy[0], verts[:, :, 1] - site_xy[1])))
        auto_x = (site_xy[0] - reach, site_xy[0] + reach)
        auto_y = (site_xy[1] - reach, site_xy[1] + reach)
    else:
        auto_x = (float(verts[:, :, 0].min()), float(verts[:, :, 0].max()))
        auto_y = (float(verts[:, :, 1].min()), float(verts[:, :, 1].max()))

    ax.set_xlim(xlim if xlim is not None else auto_x)
    ax.set_ylim(ylim if ylim is not None else auto_y)


def _site_xy(info, mode, epsg):
    """Radar site position in the requested frame, or None if unavailable."""
    if mode == "xy":
        return (0.0, 0.0)
    if mode == "lonlat":
        return (float(info["longitude"]), float(info["latitude"]))
    import shapely

    from raddb.aoi import _reproject_to_aoi, _to_pyproj_crs

    pt = shapely.Point(float(info["longitude"]), float(info["latitude"]))
    if epsg == 2056:
        p = _reproject_to_aoi(pt, 4326, 2056)
        return (p.x, p.y)
    import pyproj

    tf = pyproj.Transformer.from_crs(_to_pyproj_crs(4326), _to_pyproj_crs(epsg), always_xy=True)
    return tf.transform(pt.x, pt.y)


# ============================================================================
# PPI
# ============================================================================


def plot_aoi_quicklook(
    aoi_geom,
    selected=None,
    radars=None,
    base_path=None,
    context="switzerland",
    ax=None,
    figsize=(9, 9),
    title=None,
    range_rings_km=(100,),
    show_gates=False,
    gate_sample=50_000,
    epsg=None,
    xlim=None,
    ylim=None,
    save_path=None,
):
    """Map an AOI on a country-scale background for a quick sanity-check.

    Answers "**is my AOI where I think it is?**": the AOI footprint (red) drawn on
    a country-scale background with the involved radar sites, in a **square** view.
    Rendered in the archive's own CRS (``epsg=``, default LV95 when unset), axis
    units km.  The default ``"switzerland"`` outline comes from cartopy's cached
    Natural Earth data and is dropped on a non-Swiss frame; everything else is
    dependency-free, so the map still draws (without the outline) if that data is
    missing.

    Parameters
    ----------
    aoi_geom : shapely geometry
        AOI footprint in the AOI frame (Polygon for bbox/polygon/point AOIs;
        LineString for a cross-section line).
    selected : pandas.DataFrame, optional
        Selected gates carrying ``x`` / ``y`` in the AOI CRS.  Only scattered when
        ``show_gates=True`` — off by default so the map stays readable.
    radars : list of str, optional
        Radar names to mark (needs ``base_path`` to load their site coords).
    base_path : str or Path, optional
        RadDB archive base directory, for loading radar site coordinates.
    context : str, shapely geometry, GeoDataFrame, or None
        Map background, resolved into the AOI frame. ``"switzerland"`` (default)
        draws the national outline and is dropped on a non-Swiss frame; ``None``
        draws none; a GeoDataFrame is reprojected from its own ``.crs``, and a
        bare shapely geometry is taken to be in the AOI frame already.
    ax : matplotlib Axes, optional
        Draw into an existing axis instead of creating a figure.
    figsize : tuple
        Default ``(9, 9)`` — square, sized to show the whole country.
    title : str, optional
    range_rings_km : number or iterable of number, optional
        Range-ring radii (km) drawn dashed around each radar site.  Accepts a
        single value (``100``) or several (``(50, 100)``); ``None`` omits them.
    show_gates : bool
        Scatter the selected gate centroids (default False).
    gate_sample : int
        Cap the number of scattered centroids (random subsample); ``None`` = all.
    xlim, ylim : (min, max) in EPSG:2056 meters, optional
        Axis limits.  ``xlim`` defaults to auto (fills the context/AOI extent);
        ``ylim`` defaults to the Swiss north band (1.04-1.31 Mm ~ North 40-310 km).
        Pass ``None`` to either for auto-framing of that axis.
    save_path : str or Path, optional
        If given, save the figure (dpi=150, tight).

    Returns
    -------
    (fig, ax)
    """
    import shapely

    from raddb.aoi import SWISS_EPSG

    # Everything here is drawn in the AOI's own frame.  The default context is
    # the Swiss border, which outside LV95 would draw the wrong country around
    # the AOI, so it is dropped rather than being misleading.
    frame_epsg = SWISS_EPSG if epsg is None else int(epsg)
    if context == "switzerland" and frame_epsg != SWISS_EPSG:
        context = None
    if ylim is None and frame_epsg == SWISS_EPSG:
        # Keep the familiar Swiss band when the frame really is LV95; anywhere
        # else it would put the AOI thousands of km off-screen, so fall through
        # to the computed extent.
        ylim = (1_040_000, 1_310_000)

    from raddb.aoi import _resolve_context

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- context background, in the AOI's own frame ---
    ctx_geom = _resolve_context(context, aoi_epsg=frame_epsg)
    if ctx_geom is not None:
        _draw_context(ax, ctx_geom)

    # --- radar site positions (also used to frame the view) ---
    sites: dict[str, tuple[float, float]] = {}
    if radars and base_path is not None:
        from raddb.aoi import SWISS_EPSG, _reproject_to_aoi
        from raddb.lut import load_radar_info

        for r in radars:
            try:
                info = load_radar_info(r, base_path)
            except Exception:  # - missing info shouldn't kill the quicklook
                continue
            pt = _reproject_to_aoi(shapely.Point(info["longitude"], info["latitude"]), 4326, frame_epsg)
            sites[r] = (pt.x, pt.y)

    # --- optional selected gate centroids ---
    # A cross-sectioned frame carries both x/y (meters from the radar) and
    # x_<epsg>/y_<epsg>; this map is drawn in the projected frame, so prefer
    # those and fall back to plain x/y for a plain crop.
    _xc, _yc = f"x_{frame_epsg}", f"y_{frame_epsg}"
    if selected is not None and _xc not in getattr(selected, "columns", ()):
        _xc, _yc = "x", "y"
    if selected is not None and show_gates and len(selected) and {_xc, _yc}.issubset(selected.columns):
        xs = selected[_xc].to_numpy()
        ys = selected[_yc].to_numpy()
        if gate_sample and len(xs) > gate_sample:
            idx = np.random.default_rng(0).choice(len(xs), gate_sample, replace=False)
            xs, ys = xs[idx], ys[idx]
        ax.scatter(
            xs,
            ys,
            s=2,
            c="tab:blue",
            alpha=0.25,
            linewidths=0,
            label=f"selected gates (n={len(selected):,})",
            zorder=2,
        )

    # --- radar sites (+ dashed range rings) ---
    theta = np.linspace(0, 2 * np.pi, 361)
    # accept None, a single number (e.g. 100), or an iterable of radii
    if range_rings_km is None:
        rings = ()
    elif isinstance(range_rings_km, (int, float)):
        rings = (range_rings_km,)
    else:
        rings = tuple(range_rings_km)
    ring_label = f"range rings ({', '.join(str(int(d)) for d in rings)} km)" if rings else None
    first_site = True
    ring_labeled = False
    for r, (sx, sy) in sites.items():
        for d_km in rings:
            ax.plot(
                sx + d_km * 1e3 * np.cos(theta),
                sy + d_km * 1e3 * np.sin(theta),
                color="0.5",
                lw=0.7,
                ls="--",
                zorder=1,
                label=None if ring_labeled else ring_label,
            )
            ring_labeled = True
        ax.plot(sx, sy, "k^", ms=9, zorder=5, label="radar" if first_site else None)
        ax.annotate(
            r,
            (sx, sy),
            textcoords="offset points",
            xytext=(5, 5),
            fontweight="bold",
            zorder=6,
        )
        first_site = False

    # --- AOI footprint ---
    _draw_aoi_outline(ax, aoi_geom)

    # --- frame: fills the context / AOI / site extent, in the AOI's own CRS ---
    boxes = [aoi_geom.bounds]
    if ctx_geom is not None:
        boxes.append(ctx_geom.bounds)
    boxes += [(x, y, x, y) for x, y in sites.values()]
    xmin = min(b[0] for b in boxes)
    ymin = min(b[1] for b in boxes)
    xmax = max(b[2] for b in boxes)
    ymax = max(b[3] for b in boxes)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        mx = 0.06 * max(xmax - xmin, 1.0)
        ax.set_xlim(xmin - mx, xmax + mx)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        my = 0.06 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - my, ymax + my)

    # --- cosmetics: equal aspect, LV95 km ticks ---
    ax.set_aspect("equal")
    ox, oy = (2e6, 1e6) if frame_epsg == SWISS_EPSG else (0.0, 0.0)
    ax.xaxis.set_major_formatter(_KmFormatter(ox))
    ax.yaxis.set_major_formatter(_KmFormatter(oy))
    ax.set_xlabel("East [km]")
    ax.set_ylabel("North [km]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(title or "Quicklook - Area Of Interest")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def _draw_context(ax, geom, label="Switzerland"):
    """Draw a light-gray filled context outline (country/region) behind the AOI."""
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    first = True
    for g in polys:
        if g.geom_type != "Polygon":
            continue
        ax.fill(*g.exterior.xy, fc="0.93", ec="0.55", lw=0.8, zorder=0, label=label if first else None)
        first = False


def _draw_aoi_outline(ax, geom, color="red"):
    """Draw an AOI geometry outline (Polygon fill+edge, or LineString path)."""
    gt = geom.geom_type
    if gt == "Polygon":
        ax.fill(*geom.exterior.xy, color=color, alpha=0.25, zorder=3)
        ax.plot(*geom.exterior.xy, color=color, lw=2, zorder=4, label="AOI")
    elif gt in ("LineString", "LinearRing"):
        ax.plot(*geom.xy, color=color, lw=2, zorder=4, label="AOI")
    elif gt in ("MultiPolygon", "GeometryCollection"):
        first = True
        for g in geom.geoms:
            lbl = "AOI" if first else None
            if g.geom_type == "Polygon":
                ax.fill(*g.exterior.xy, color=color, alpha=0.25, zorder=3)
                ax.plot(*g.exterior.xy, color=color, lw=2, zorder=4, label=lbl)
            first = False
    else:  # Point or other
        ax.plot(geom.x, geom.y, marker="*", color=color, ms=12, zorder=4, label="AOI")


# ============================================================================
# Vertical cross-section (arbitrary line, from extract_cross_section)
# ============================================================================


def plot_cross_section(
    df_cs,
    variable: str = "DBZH",
    ax=None,
    figsize=(12, 5),
    title: str | None = None,
    add_colorbar: bool = True,
    edgecolor="none",
    xlim=None,
    ylim=None,
    **plot_kwargs,
):
    """Render a vertical cross-section from a :meth:`RadDB.extract_cross_section` result.

    Each row's ``cs_polygon`` — the gate's 4-corner polygon in the
    (distance-along-line, altitude) plane — is drawn as a filled patch colored
    by ``variable`` (per-variable colormap defaults apply, incl. discrete HC
    classes).  Axes: distance along the section line [km] (from ``p1``) vs
    altitude [km ASL].

    Note: pass a **single volume** (filter by ``volume_time``) and ideally a
    single radar — overlapping radars/volumes draw on top of each other.

    Parameters
    ----------
    df_cs : pandas.DataFrame
        Output of :meth:`RadDB.extract_cross_section` (needs ``cs_polygon`` +
        ``variable`` columns).
    variable : str
        Column to color by (default ``"DBZH"``).
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    add_colorbar : bool
    edgecolor : matplotlib color
        Patch edge color (default ``"none"``; e.g. ``"k"`` to outline gates).
    xlim : (dmin_km, dmax_km), optional
        Along-section distance limits in km.
    ylim : (zmin_km, zmax_km), optional
        Altitude limits in km.
    **plot_kwargs
        ``cmap`` / ``vmin`` / ``vmax`` / ``norm`` overrides.

    Returns
    -------
    (fig, ax, collection)
    """
    from matplotlib.collections import PolyCollection

    if "cs_polygon" not in df_cs.columns:
        raise KeyError("df_cs has no 'cs_polygon' column; use RadDB.extract_cross_section first.")
    if variable not in df_cs.columns:
        raise KeyError(f"variable {variable!r} not in df_cs columns.")

    data = df_cs[df_cs[variable].notna() & df_cs["cs_polygon"].notna()]
    if data.empty:
        raise ValueError(f"no non-NaN {variable!r} values on this cross-section.")

    plot_kwargs, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(
        variable,
        plot_kwargs,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # polygons in km on both axes
    verts = [np.asarray(p.exterior.coords)[:, :2] / 1000.0 for p in data["cs_polygon"]]
    pc = PolyCollection(verts, array=data[variable].to_numpy(), edgecolor=edgecolor, linewidth=0.1)
    if "cmap" in plot_kwargs:
        pc.set_cmap(plot_kwargs["cmap"])
    if "norm" in plot_kwargs:
        pc.set_norm(plot_kwargs["norm"])
    else:
        pc.set_clim(plot_kwargs.get("vmin"), plot_kwargs.get("vmax"))
    ax.add_collection(pc)

    d_all = np.concatenate([v[:, 0] for v in verts])
    z_all = np.concatenate([v[:, 1] for v in verts])
    ax.set_xlim(*(xlim if xlim is not None else (d_all.min(), d_all.max())))
    ax.set_ylim(*(ylim if ylim is not None else (max(0.0, z_all.min() - 0.2), z_all.max() + 0.2)))

    ax.set_xlabel("Distance along section [km]")
    ax.set_ylabel("Altitude [km ASL]")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if add_colorbar:
        _add_colorbar(pc, ax, is_discrete, class_labels, cbar_label)
    return fig, ax, pc


# ============================================================================
# RHI (pseudo-RHI from PPI volume)
# ============================================================================

# ============================================================================
# The four gate-accurate plots
# ============================================================================


def _common_prep(data, archive_dir, radar, timestep, start_time, end_time, variable):
    """Resolve the input and narrow it to one radar and one volume.

    Returns a :class:`_Source` with ``df`` / ``radar`` / ``tstr`` / ``info``
    filled in.  A DataTree short-circuits most of this: it describes one volume
    of one radar already, and its site metadata comes from its own coordinates.
    """
    from raddb.lut import load_radar_info

    src = _resolve_frame(data, archive_dir)
    if src.kind == "datatree":
        src.radar = radar or ""
        src.tstr = _volume_time_str(_dt_sweep(src.dtree, _dt_sweep_names(src.dtree)[0]))
        src.info = _dt_site(_dt_sweep(src.dtree, _dt_sweep_names(src.dtree)[0]))
        return src

    src.df, src.radar = _select_radar(src.df, radar)
    src.df, src.tstr = _select_volume(src.df, timestep, start_time, end_time)
    if variable not in src.df.columns:
        raise KeyError(f"variable {variable!r} not in the data; have {src.df.columns}.")
    src.info = load_radar_info(src.radar, src.base) if src.base else None
    if src.crs is None and src.base is not None:
        # The archive knows the CRS it was written with, so reading never requires
        # restating it — coords="projected" resolves it from there.  aoi_epsg also
        # recovers it from the LUT's x_<epsg> columns for archives predating the
        # info.yaml crs block; if there is genuinely none, leave it unset so
        # _resolve_coords can say so.
        from raddb.aoi import aoi_epsg

        try:
            src.crs = aoi_epsg(src.base, src.radar)
        except (ValueError, FileNotFoundError):
            src.crs = None
    return src


def plot_ppi(
    data,
    sweep: int | str = 1,
    variable: str = "DBZH",
    radar: str | None = None,
    timestep=None,
    start_time=None,
    end_time=None,
    coords="xy",
    context: bool = False,
    archive_dir=None,
    ax=None,
    figsize: tuple[float, float] = (6, 6),
    add_colorbar: bool = True,
    add_range_rings: bool = True,
    title: str | None = None,
    xlim=None,
    ylim=None,
    edgecolor="none",
    rasterized: bool | None = None,
    save: str | None = None,
    use_cartopy: bool | None = None,
    **plot_kwargs,
):
    """Plan Position Indicator — one sweep, one plot, one Axes.

    Draws the **exact gate footprints** stored in the ``h_plane`` lattice, so a
    frame that has been filtered, ``sel``-ed or cropped plots precisely the gates
    it still holds.  Gates are joined by ``gate_id``; nothing is reindexed onto a
    full azimuth x range grid on the way.

    To build a multi-panel figure, call this once per panel with ``ax=``::

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, var in zip(axes.ravel(), variables):
            rdf.plot_ppi(sweep=1, variable=var, ax=ax)

    Parameters
    ----------
    data : RadDB, polars/pandas DataFrame, GeoDataFrame, or xr.DataTree
        A DataTree is self-describing: its geometry is computed from its own
        azimuth/range/elevation, so no archive is involved.  A GeoDataFrame is
        treated as a frame — exact mode joins the LUT, approximate mode uses the
        columns it carries.
    sweep : int or str
        Sweep number, or ``"sweep_3"``.
    variable : str
        Column to color by.  Per-variable colormaps and discrete HC class
        colorbars are applied automatically.
    radar : str, optional
        Required only when the frame spans several radars.
    timestep : optional
        Volume to draw.  Required when the frame holds more than one volume;
        the nearest volume is used.
    start_time, end_time : optional
        Restrict the candidate volumes before ``timestep`` is applied.
    coords : {"xy", "lonlat", "projected"} or int
        ``"xy"`` — meters from the radar (ticks in km).  ``"lonlat"`` — WGS-84
        degrees.  ``"projected"`` — the LUT's ``x_<epsg>``/``y_<epsg>`` columns,
        using the RadDB's CRS; an EPSG int selects one directly (``2056`` and
        ``"swiss"`` are the Swiss LV95 frame used by the AOI quicklook).
    context : bool
        Overlay cartopy country borders.  Independent of the coordinate frame.
    archive_dir : str or Path, optional
        Needed only when ``data`` is a bare frame.
    edgecolor : matplotlib color
        Gate outline; ``"k"`` to show the individual gate polygons.
    rasterized : bool, optional
        Rasterize the polygons inside vector output.  Defaults to ``True`` above
        50 000 gates, where an unrasterized PDF/SVG becomes very large.
    save : str, optional
        Path to save the figure to.
    use_cartopy : bool, optional
        Deprecated alias for ``context``.

    Returns
    -------
    matplotlib.collections.PolyCollection
        The artist; ``p.axes`` and ``p.figure`` reach the rest of the plot.
    """
    from raddb.lut import gate_corner_table

    if use_cartopy is not None:
        context = bool(use_cartopy)

    src = _common_prep(data, archive_dir, radar, timestep, start_time, end_time, variable)
    sweep_num = int(str(sweep).split("_")[-1])
    mode, epsg = _resolve_coords(coords, src.crs)

    resolved, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(
        variable,
        plot_kwargs,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if src.kind == "datatree":
        ds = _dt_sweep(src.dtree, sweep_num)
        values = _dt_gate_table(ds, variable)[variable].to_numpy()
        verts = _dt_h_vertices(ds, mode, epsg, src.info)
        keep = np.isfinite(values)
        values, verts = values[keep], verts[keep]
        if values.size == 0:
            raise ValueError(f"every {variable!r} value in sweep {sweep_num} is NaN.")
    else:
        base = src.require_base("plot_ppi")
        tbl = gate_corner_table(src.radar, base, kind="h_plane", sweep=sweep_num)
        if tbl.is_empty():
            raise ValueError(f"radar {src.radar!r} has no sweep {sweep_num} in its LUT.")
        values, joined = _join_corners(src.df, tbl, variable)
        verts = _corner_vertices(joined, 4, mode, epsg, src.info)

    if rasterized is None:
        rasterized = len(values) > 50_000
    p = _draw_polygons(ax, verts, values, resolved, edgecolor, rasterized)

    _finish_map_axes(
        ax,
        mode,
        epsg,
        verts,
        _site_xy(src.info, mode, epsg),
        xlim,
        ylim,
        add_range_rings,
        context,
        src.info,
    )

    if add_colorbar:
        _add_colorbar(p, ax, is_discrete, class_labels, cbar_label)
    ax.set_title(title or _default_title(src.radar, variable, f"sweep {sweep_num}", src.tstr))
    _maybe_save(ax, save, plot_kwargs)
    return p


def plot_rhi(
    data,
    azimuth: float = 0.0,
    variable: str = "DBZH",
    radar: str | None = None,
    timestep=None,
    start_time=None,
    end_time=None,
    height: str = "asl",
    az_tol: float = 1.0,
    archive_dir=None,
    ax=None,
    figsize: tuple[float, float] = (10, 4),
    add_colorbar: bool = True,
    title: str | None = None,
    xlim=None,
    ylim=None,
    max_range_km: float | None = None,
    max_height_km: float | None = None,
    edgecolor="none",
    rasterized: bool | None = None,
    save: str | None = None,
    **plot_kwargs,
):
    """Range-Height Indicator — one azimuth through the whole volume.

    The counterpart of :func:`plot_ppi`: instead of fixing the sweep and sweeping
    azimuth, it fixes the azimuth and stacks every sweep, drawing the gates'
    vertical faces from the ``v_plane`` lattice in the
    ``(ground distance, altitude)`` plane.

    Parameters
    ----------
    azimuth : float
        Target azimuth in degrees (0 = North, clockwise).  The nearest stored ray
        is used; ``az_tol`` bounds how far it may be.
    height : {"asl", "rel"}
        Altitude above sea level (default) or above the radar.
    max_range_km, max_height_km : float, optional
        Convenience clips, equivalent to ``xlim`` / ``ylim``.

    Other parameters are as in :func:`plot_ppi`.

    Returns
    -------
    matplotlib.collections.PolyCollection
    """
    if height not in ("asl", "rel"):
        raise ValueError(f"height must be 'asl' or 'rel'; got {height!r}.")

    from raddb.lut import gate_corner_table, load_radar_lut

    src = _common_prep(data, archive_dir, radar, timestep, start_time, end_time, variable)
    target = float(azimuth) % 360.0
    if src.kind == "datatree":
        values, verts, ray_az = _dt_rhi(src, target, variable, az_tol, height, _beamwidth(src))
    else:
        base = src.require_base("plot_rhi")
        # Nearest ray **per sweep**, compared on the circle so 359.8° and 0.1°
        # are close.  The LUT stores raw antenna azimuths, which jitter by a few
        # tenths of a degree from one sweep to the next, so a single azimuth
        # value matches only one sweep — an RHI needs each sweep's closest ray.
        lut_az = (
            load_radar_lut(src.radar, base)
            .select(["gate_id", "sweep", "azimuth"])
            .with_columns(
                (((pl.col("azimuth") - target + 180.0) % 360.0) - 180.0).abs().alias("_off"),
            )
        )
        picked, ray_az = _nearest_ray_per_sweep(lut_az, target, az_tol)
        tbl = gate_corner_table(src.radar, base, kind="v_plane").join(
            picked.select("gate_id"),
            on="gate_id",
            how="semi",
        )
        values, joined = _join_corners(src.df, tbl, variable)
        z_prefix = "z_asl" if height == "asl" else "z_rel"
        verts = np.stack(
            [
                np.stack([joined[f"d_{k}"].to_numpy(), joined[f"{z_prefix}_{k}"].to_numpy()], axis=1)
                for k in range(1, 5)
            ],
            axis=1,
        ).astype(np.float64)

    if len(values) == 0:
        raise ValueError(f"no gates on the ray at azimuth {ray_az:.1f}° in this input.")

    resolved, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(
        variable,
        plot_kwargs,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if rasterized is None:
        rasterized = len(values) > 50_000

    p = _draw_polygons(ax, verts, values, resolved, edgecolor, rasterized)

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(_KmFormatter())
    ax.set_xlabel("Ground range [km]")
    ax.set_ylabel("Height ASL [km]" if height == "asl" else "Height above radar [km]")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(
        xlim if xlim is not None else (0.0, (max_range_km * 1e3) if max_range_km else float(verts[:, :, 0].max())),
    )
    ax.set_ylim(
        (
            ylim
            if ylim is not None
            else (float(verts[:, :, 1].min()), (max_height_km * 1e3) if max_height_km else float(verts[:, :, 1].max()))
        ),
    )

    if add_colorbar:
        _add_colorbar(p, ax, is_discrete, class_labels, cbar_label)
    ax.set_title(title or _default_title(src.radar, variable, f"azimuth {ray_az:.1f}°", src.tstr))
    _maybe_save(ax, save, plot_kwargs)
    return p


def plot_cappi(
    data,
    altitude: float,
    variable: str = "DBZH",
    radar: str | None = None,
    timestep=None,
    start_time=None,
    end_time=None,
    coords="xy",
    context: bool = False,
    height: str = "asl",
    overlap: str = "nearest",
    fill_lowest: bool = False,
    archive_dir=None,
    ax=None,
    figsize: tuple[float, float] = (6, 6),
    add_colorbar: bool = True,
    add_range_rings: bool = True,
    title: str | None = None,
    xlim=None,
    ylim=None,
    edgecolor="none",
    rasterized: bool | None = None,
    save: str | None = None,
    **plot_kwargs,
):
    """Constant Altitude PPI — a horizontal slice through the volume.

    Where :func:`plot_ppi` fixes the sweep, this fixes the **altitude** and pulls
    from whichever elevation angles actually sample it, which is what makes a
    CAPPI a CAPPI.

    How the geometry is built
    -------------------------
    1. :func:`raddb.lut.cappi_chords` intersects the gates' vertical faces
       (``v_plane``) with the plane ``z = altitude``, giving the along-beam chord
       ``[d_near, d_far]`` of every range bin the surface passes through.  This is
       computed **once per (sweep, range bin)**: ``d`` and ``z`` do not depend on
       azimuth, so the same handful of rows serves all 360 rays.
    2. Each chord trims that bin's ``h_plane`` footprint along the beam, giving
       the output polygon in ``(x, y)``.

    So the result is horizontal — same frame as a PPI — but it is *not* the
    ``h_plane`` face itself: the constant-altitude cut shortens the gate along
    the beam.  ``h_plane`` alone cannot produce it, having no altitude column.

    Because beam thickness at long range (~1.7 km at 100 km) far exceeds the
    height gained across one range bin, each sweep contributes a **wide
    contiguous band** of bins and neighboring sweeps overlap heavily — hence
    ``overlap``.

    Parameters
    ----------
    altitude : float
        Slice altitude in meters, in the reference given by ``height``.
    height : {"asl", "rel"}
        Whether ``altitude`` is above sea level (default) or above the radar.
    overlap : {"nearest", "all"}
        How to resolve sweeps that both sample this altitude at the same ground
        distance.  ``"nearest"`` (default) partitions the ground-distance axis
        and keeps, in each interval, the beam whose center is closest to the
        slice — a real measurement, never an average.  ``"all"`` draws every
        contributing gate, so later sweeps paint over earlier ones.
    fill_lowest : bool
        Beyond the range where even the lowest sweep's beam has climbed above the
        slice, nothing samples that altitude.  ``False`` (default) leaves it
        empty; ``True`` continues along the lowest sweep, the operational
        convention (Stull, *Practical Meteorology* §8.2).

    Other parameters are as in :func:`plot_ppi`.

    Returns
    -------
    matplotlib.collections.PolyCollection
    """
    if overlap not in ("nearest", "all"):
        raise ValueError(f"overlap must be 'nearest' or 'all'; got {overlap!r}.")

    from raddb.lut import _gate_grid_index, cappi_chords, gate_corner_table

    src = _common_prep(data, archive_dir, radar, timestep, start_time, end_time, variable)

    mode, epsg = _resolve_coords(coords, src.crs)

    if src.kind == "datatree":
        values, verts = _dt_cappi(
            src,
            altitude,
            variable,
            height,
            overlap,
            fill_lowest,
            mode,
            epsg,
            _beamwidth(src),
        )
    else:
        base = src.require_base("plot_cappi")
        # The slice is cut against the vertical faces stored in v_plane, i.e.
        # against real quads: a linear beam model would divide by tan(elevation),
        # which blows up on the near-horizontal sweeps supplying most of the far
        # field.
        chords = cappi_chords(src.radar, base, altitude, height=height)
        if chords.is_empty():
            raise ValueError(
                f"no beam of radar {src.radar!r} reaches {altitude} m "
                f"({'ASL' if height == 'asl' else 'above the radar'}); nothing to draw.",
            )
        if overlap == "nearest":
            chords = _resolve_chord_overlap(chords)
        if fill_lowest:
            chords = _extend_lowest_sweep(chords, src.radar, base)

        # Chords are per (sweep, rng_idx) and azimuth-independent -> expand to gates.
        gates = _gate_grid_index(src.radar, base).join(
            chords,
            on=["sweep", "rng_idx"],
            how="inner",
        )
        if gates.is_empty():
            raise ValueError("the constant-altitude surface matched no gates.")
        tbl = gate_corner_table(src.radar, base, kind="h_plane").join(
            gates.select(["gate_id", "d_near", "d_far"]),
            on="gate_id",
            how="inner",
        )
        values, joined = _join_corners(src.df, tbl, variable)
        verts = _trim_footprints_to_chord(
            _corner_vertices(joined, 4, mode, epsg, src.info),
            _corner_vertices(joined, 4, "xy", None, src.info),
            joined["d_near"].to_numpy(),
            joined["d_far"].to_numpy(),
        )

    if len(values) == 0:
        raise ValueError(
            f"no gates at {altitude} m are present in this input — the slice is " "outside the loaded/cropped data.",
        )

    resolved, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(
        variable,
        plot_kwargs,
    )
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if rasterized is None:
        rasterized = len(values) > 50_000

    p = _draw_polygons(ax, verts, values, resolved, edgecolor, rasterized)
    _finish_map_axes(
        ax,
        mode,
        epsg,
        verts,
        _site_xy(src.info, mode, epsg),
        xlim,
        ylim,
        add_range_rings,
        context,
        src.info,
    )

    if add_colorbar:
        _add_colorbar(p, ax, is_discrete, class_labels, cbar_label)
    ref = "m ASL" if height == "asl" else "m above radar"
    ax.set_title(title or _default_title(src.radar, variable, f"{altitude:g} {ref}", src.tstr))
    _maybe_save(ax, save, plot_kwargs)
    return p


def plot_vcs(
    data,
    line=None,
    variable: str = "DBZH",
    radar: str | None = None,
    timestep=None,
    start_time=None,
    end_time=None,
    height: str = "asl",
    crs=None,
    beamwidth_deg: float = 1.0,
    aoi_crs=None,
    archive_dir=None,
    ax=None,
    figsize: tuple[float, float] = (12, 5),
    add_colorbar: bool = True,
    title: str | None = None,
    xlim=None,
    ylim=None,
    edgecolor="none",
    rasterized: bool | None = None,
    save: str | None = None,
    **plot_kwargs,
):
    """Vertical Cross-Section — a vertical slice along an arbitrary line.

    Unlike the other three plots, what is drawn has to be *defined* first: a PPI
    has its sweep and an RHI its azimuth, but a cross-section needs a line.
    Supply it as ``line``, or pass a frame that already went through
    :meth:`~raddb.RadDB.extract_cross_section` (it carries ``cs_polygon``).

    Which combinations are accepted:

    =====================  ==========================  =================================
    ``data``               ``line``                    result
    =====================  ==========================  =================================
    RadDB / frame / gdf    file, points or LineString  section is cut, then drawn
    RadDB / frame / gdf    omitted, has cs_polygon     drawn directly
    RadDB / frame / gdf    omitted, no cs_polygon      **error** — undefined
    RadDB / frame / gdf    given *and* has cs_polygon  **error** — ambiguous
    ``xr.DataTree``        anything                    **error** — archive first
    =====================  ==========================  =================================

    The third row is the common mistake: an AOI crop (rectangle, polygon, marker)
    selects an *area*, not a line, so its result has no section to draw.

    Parameters
    ----------
    line : optional
        The cross-section to cut, given as any of:

        * ``(p1, p2)`` — two ``(x, y)`` points or shapely Points,
        * a shapely ``LineString``,
        * a path to a ``.shp`` / ``.geojson`` holding a line.

        Omit it when ``data`` already carries ``cs_polygon``.
    crs : int or str, optional
        CRS of ``line``.  A file that declares its own CRS wins unless this is
        given explicitly; otherwise the RadDB's CRS is assumed.
    beamwidth_deg : float
        Beamwidth used to give the section its vertical extent.
    height : {"asl", "rel"}
        Altitude reference for the vertical axis.

    Other parameters are as in :func:`plot_ppi`.

    Returns
    -------
    matplotlib.collections.PolyCollection
    """
    if isinstance(data, (xr.DataTree, xr.Dataset)):
        raise TypeError(
            "plot_vcs cannot work from a DataTree: cutting a cross-section needs "
            "the LUT (gate footprints keyed by gate_id), which a DataTree has no "
            "equivalent of. Archive the volume first — it takes a few seconds — "
            "then cut the section on the result:\n"
            "    db.archive(datatree=dt, radar='L')\n"
            "    db.open(radars='L').plot_vcs(line=(p1, p2))",
        )

    # RadDB.columns is a method, a frame's .columns is a property — handle both.
    if isinstance(data, (pl.DataFrame, pd.DataFrame)):
        cols = list(data.columns)
    else:
        inner = getattr(data, "data", None)
        cols = list(inner.columns) if inner is not None else []
    already_cut = "cs_polygon" in cols

    if already_cut and line is not None:
        raise ValueError(
            "both a section line and an already-cut frame were given, so it is "
            "ambiguous which section to draw. Cutting again would intersect two "
            "different sections. Pass the line to an uncut frame, or drop line= "
            "to draw the section this frame already carries.",
        )
    if not already_cut:
        if line is None:
            raise ValueError(
                "no cross-section to draw: this frame carries no 'cs_polygon'. "
                "Pass line=((x1, y1), (x2, y2)), a shapely LineString or a "
                ".shp/.geojson path — or call extract_cross_section() first. "
                "(An AOI crop by rectangle/polygon/point selects an area, not a "
                "line, so its result cannot be drawn as a cross-section.)",
            )
        if not hasattr(data, "extract_cross_section"):
            # A bare frame carries gate_id and the archive carries the geometry,
            # so the section is perfectly cuttable — wrap it, the same way the
            # other three plots read the LUT for a bare frame.
            from raddb.main import RadDB as _RadDB

            probe = _resolve_frame(data, archive_dir)
            probe.require_base("cutting a cross-section from line=")
            data = _RadDB(archive_dir=str(probe.base), crs=probe.crs)._derive(probe.df)
        p1, p2, file_crs = _line_endpoints(line)
        # A file states its own CRS; honor it unless the caller overrode it.
        data = data.extract_cross_section(
            p1,
            p2,
            crs=crs if crs is not None else file_crs,
            beamwidth_deg=beamwidth_deg,
            aoi_crs=aoi_crs,
        )

    src = _common_prep(data, archive_dir, radar, timestep, start_time, end_time, variable)
    from raddb.main import _decode_geometry

    # cs_polygon crosses into polars as WKB, so decode it back to shapely.
    pdf = _decode_geometry(src.df.to_pandas())
    pdf = pdf[pdf[variable].notna() & pdf["cs_polygon"].notna()]
    if pdf.empty:
        raise ValueError(f"no non-NaN {variable!r} values on this cross-section.")

    # cs_polygon lives in (distance along the line, altitude ASL).
    shift = 0.0 if height == "asl" else -float(src.info["altitude"])
    verts = np.stack([np.asarray(poly.exterior.coords)[:4, :2] for poly in pdf["cs_polygon"]]).astype(np.float64)
    verts[:, :, 1] += shift

    resolved, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(
        variable,
        plot_kwargs,
    )
    values = pdf[variable].to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if rasterized is None:
        rasterized = len(values) > 50_000

    p = _draw_polygons(ax, verts, values, resolved, edgecolor, rasterized)

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(_KmFormatter())
    ax.set_xlabel("Distance along section [km]")
    ax.set_ylabel("Altitude [km ASL]" if height == "asl" else "Height above radar [km]")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim if xlim is not None else (float(verts[:, :, 0].min()), float(verts[:, :, 0].max())))
    ax.set_ylim(ylim if ylim is not None else (float(verts[:, :, 1].min()), float(verts[:, :, 1].max())))

    if add_colorbar:
        _add_colorbar(p, ax, is_discrete, class_labels, cbar_label)
    ax.set_title(title or _default_title(src.radar, variable, "cross-section", src.tstr))
    _maybe_save(ax, save, plot_kwargs)
    return p


# ------------------------------------------------------------ plot internals


def _default_title(radar, variable, what, tstr):
    parts = [f"radar {radar}", variable, what]
    if tstr:
        parts.append(tstr)
    return " | ".join(parts)


def _maybe_save(ax, save, kwargs):
    if save:
        ax.figure.savefig(save, bbox_inches="tight", dpi=kwargs.get("dpi", 150))


def _line_endpoints(line):
    """Normalize ``line`` to ``(p1, p2, src_crs)``.

    ``src_crs`` is the CRS the file declared, or ``None`` for points and geometry
    objects, which carry none.  Callers must pass it through: a GeoJSON is
    lon/lat by RFC 7946, and reading those degrees as LV95 meters would place the
    section thousands of kilometers away.
    """
    from pathlib import Path

    import shapely

    src_crs = None
    if isinstance(line, (str, Path)):
        from raddb.aoi import _read_geometry_file

        geom, src_crs = _read_geometry_file(Path(line))
    elif isinstance(line, shapely.geometry.base.BaseGeometry):
        geom = line
    else:
        p1, p2 = line
        to_xy = lambda p: (p.x, p.y) if hasattr(p, "x") else (float(p[0]), float(p[1]))  # noqa: E731
        return to_xy(p1), to_xy(p2), None

    coords = np.asarray(shapely.get_coordinates(geom))
    if len(coords) < 2:
        raise ValueError(f"could not read a line with two endpoints from {line!r}.")
    return tuple(coords[0][:2]), tuple(coords[-1][:2]), src_crs


def _resolve_chord_overlap(chords: pl.DataFrame) -> pl.DataFrame:
    """Partition the ground-distance axis so no two gates cover the same distance.

    Several sweeps typically intersect the slice altitude over overlapping
    ground-distance intervals.  Split the axis at every chord endpoint and give
    each elementary interval to the chord with the smallest ``dz_center`` — the
    beam whose center sits closest to the slice.  Chords are then clipped to the
    intervals they won.
    """
    d_near = chords["d_near"].to_numpy().astype(np.float64)
    d_far = chords["d_far"].to_numpy().astype(np.float64)
    dz = chords["dz_center"].to_numpy().astype(np.float64)

    edges = np.unique(np.concatenate([d_near, d_far]))
    if edges.size < 2:
        return chords
    mid = 0.5 * (edges[:-1] + edges[1:])

    # covers[i, j]: chord i spans elementary interval j.
    covers = (d_near[:, None] <= mid[None, :]) & (mid[None, :] <= d_far[:, None])
    scored = np.where(covers, dz[:, None], np.inf)
    winner = np.argmin(scored, axis=0)
    valid = np.isfinite(scored[winner, np.arange(mid.size)])

    rows: dict[int, list[float]] = {}
    for j in np.flatnonzero(valid):
        w = int(winner[j])
        lo, hi = float(edges[j]), float(edges[j + 1])
        if w in rows:
            rows[w][0] = min(rows[w][0], lo)
            rows[w][1] = max(rows[w][1], hi)
        else:
            rows[w] = [lo, hi]
    if not rows:
        return chords.clear()

    keep = np.fromiter(rows.keys(), dtype=np.int64)
    bounds = np.array([rows[int(i)] for i in keep], dtype=np.float64)
    return chords[keep].with_columns(
        pl.Series("d_near", bounds[:, 0], dtype=pl.Float32),
        pl.Series("d_far", bounds[:, 1], dtype=pl.Float32),
    )


def _extend_lowest_sweep(chords: pl.DataFrame, radar: str, base) -> pl.DataFrame:
    """Follow the lowest sweep past the range where every beam is above the slice.

    The operational CAPPI convention (Stull §8.2): rather than leaving the far
    field empty, keep reading the lowest elevation cone.  Those gates are drawn
    at their full footprint, since they are no longer trimmed by the slice.
    """
    from raddb.lut import load_plane_nodes

    lowest = int(chords["sweep"].min())
    d_end = float(chords["d_far"].max())

    nodes = load_plane_nodes(radar, base, "v_plane", sweep=lowest)
    nodes = nodes.filter(
        (pl.col("az_idx") == pl.col("az_idx").min()) & (pl.col("el_level") == -1),
    ).sort("rng_idx")
    d = nodes["d"].to_numpy()
    if d.size < 2:
        return chords

    beyond = np.flatnonzero(d[1:] > d_end)
    if beyond.size == 0:
        return chords
    extra = pl.DataFrame(
        {
            "sweep": np.full(beyond.size, lowest, dtype=np.int32),
            "rng_idx": beyond.astype(np.int32),
            "d_near": d[beyond].astype(np.float32),
            "d_far": d[beyond + 1].astype(np.float32),
            "z_center": np.zeros(beyond.size, dtype=np.float32),
            "dz_center": np.zeros(beyond.size, dtype=np.float32),
        },
    )
    existing = chords.filter(pl.col("sweep") == lowest)["rng_idx"].to_list()
    return pl.concat(
        [chords, extra.filter(~pl.col("rng_idx").is_in(existing))],
        how="vertical",
    )


def _trim_footprints_to_chord(verts, verts_xy, d_near, d_far):
    """Shorten each h_plane footprint along the beam to ``[d_near, d_far]``.

    ``verts`` is the ``(n, 4, 2)`` ring in the output frame; ``verts_xy`` is the
    same ring in radar-relative meters, where ground distance is simply
    ``hypot(x, y)`` — the frame-independent way to locate the cut.

    Ring order is :data:`raddb.lut.GATE_RING_OFFSETS`: corners 1 and 4 sit on the
    near range edge, corners 2 and 3 on the far one.  So the beam runs 1->2 along
    one azimuth edge and 4->3 along the other, and trimming is a lerp along both.
    """
    d_ring = np.hypot(verts_xy[:, :, 0], verts_xy[:, :, 1])
    d_bin_near = 0.5 * (d_ring[:, 0] + d_ring[:, 3])
    d_bin_far = 0.5 * (d_ring[:, 1] + d_ring[:, 2])

    span = d_bin_far - d_bin_near
    with np.errstate(divide="ignore", invalid="ignore"):
        t_near = np.where(span != 0, (d_near - d_bin_near) / span, 0.0)
        t_far = np.where(span != 0, (d_far - d_bin_near) / span, 1.0)
    t_near = np.clip(np.nan_to_num(t_near, nan=0.0), 0.0, 1.0)[:, None]
    t_far = np.clip(np.nan_to_num(t_far, nan=1.0), 0.0, 1.0)[:, None]

    c1, c2, c3, c4 = verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3]
    return np.stack(
        [
            c1 + t_near * (c2 - c1),  # near edge, azimuth side A
            c1 + t_far * (c2 - c1),  # far  edge, azimuth side A
            c4 + t_far * (c3 - c4),  # far  edge, azimuth side B
            c4 + t_near * (c3 - c4),  # near edge, azimuth side B
        ],
        axis=1,
    )

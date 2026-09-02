"""Area-of-Interest (AOI) selection internals.

These are the **private** building blocks behind the public ``RadDB.crop_*``
methods.  The design is LUT-first: an AOI geometry is intersected once with the
static per-radar LUT **centroids** (in the archive's own projected CRS) to resolve a set
of ``gate_id`` values, and that set then filters any number of dynamic volume
DataFrames — cheaply and identically across timesteps.

Because the intersection is done on gate centroids over **all** sweeps, a single
horizontal footprint selects the whole vertical column above it (every elevation).

Coordinate convention: there is **no built-in CRS**.  Every AOI runs in the CRS the
archive was written with (recorded in ``{radar}_info.yaml``, validated against the
radar site at archive time), and input geometry in another CRS is reprojected into
it via :func:`_reproject_to_aoi` before intersection.  A hardcoded frame was the
source of a silent 17% error on US radars, so nothing here assumes one.

No pyart / geocube dependency — only numpy, polars, pandas, shapely, pyproj.

The centroid tables are **polars** frames (the package-wide default); the spatial
predicate itself runs on plain numpy arrays via shapely, so no geometry objects
are ever materialized for the ~1.7M gates of a full LUT.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import shapely
import shapely.ops

from raddb.lut import add_lut_projection, decode_gate_radars

logger = logging.getLogger(__name__)

#: Swiss LV95.  Used *only* where the subject really is Switzerland — the border
#: overlay on the AOI quicklook, and sniffing a Swiss ``.prj``.  It is never a
#: default for gate geometry; that comes from the archive's own CRS.
SWISS_EPSG: int = 2056

# LUT columns kept for centroid intersection, besides the projected pair, which
# is named after the archive's own EPSG and resolved per radar.
_CENTROID_BASE_COLS = ["gate_id", "sweep", "z", "altitude"]


def aoi_epsg(base_path: str | Path, radar: str) -> int:
    """EPSG the archive stores this radar's projected coordinates in.

    Read from ``{radar}_info.yaml``; recovered from the LUT's own ``x_<epsg>``
    columns for archives written before the ``crs`` block existed.  There is no
    fallback: a CRS is required at archive time precisely so this never has to
    guess.
    """
    from raddb.lut import load_radar_info, lut_file_path

    try:
        info = load_radar_info(radar, base_path)
    except FileNotFoundError:
        info = {}
    epsg = (info.get("crs") or {}).get("epsg")
    if epsg is not None:
        return int(epsg)

    lut_path = lut_file_path(radar, "lut", base_path)
    if lut_path.exists():
        for name in pq.read_schema(lut_path).names:
            if name.startswith("x_") and name[2:].isdigit():
                return int(name[2:])
    raise ValueError(
        f"radar {radar!r} has no projected coordinates in its LUT, so AOI "
        f"operations (crop_*, extract_cross_section, plot_vcs) cannot run. "
        f"Re-archive it with a CRS valid at its site — RadDB(crs=<epsg>) — or "
        f"pass aoi_crs=<epsg> for this call.",
    )


def aoi_epsg_for(base_path: str | Path, radars: list[str], override=None) -> int:
    """The single CRS an AOI spanning ``radars`` runs in.

    All radars must agree, because reprojecting one onto another's frame behind
    the user's back is exactly the kind of implicit choice that made a 17% error
    invisible.  ``override`` (``aoi_crs=``) names a common frame explicitly and
    is validated against every site.
    """
    from raddb.lut import load_radar_info, validate_crs_for_site

    if override is not None:
        for r in radars:
            info = load_radar_info(r, base_path)
            validate_crs_for_site(override, info["longitude"], info["latitude"], r)
        return override

    found = {r: aoi_epsg(base_path, r) for r in radars}
    distinct = set(found.values())
    if len(distinct) > 1:
        pairs = ", ".join(f"{r}=EPSG:{e}" for r, e in sorted(found.items()))
        raise ValueError(
            f"these radars were archived in different CRSs ({pairs}), so there is "
            f"no single frame to run the AOI in. Pass aoi_crs=<epsg> valid for all "
            f"of them, or restrict the selection to radars sharing one.",
        )
    return distinct.pop()


# Per-radar centroid cache: {(base_path, radar): DataFrame}. LUTs are static, so
# a radar's centroid table is loaded from disk at most once per session.
_CENTROID_CACHE: dict[tuple[str, str], pl.DataFrame] = {}


def _radars_from_gate_ids(gate_ids) -> list[str]:
    """Decode the distinct radar names present in a set of ``gate_id`` values.

    Thin alias of :func:`raddb.lut.decode_gate_radars`, kept because it is the
    name the AOI, plotting and RadDB code paths already call.
    """
    return decode_gate_radars(gate_ids)


def _lut_centroids(base_path: str | Path, radars: list[str], epsg=None) -> pl.DataFrame:
    """Load and concatenate LUT centroid tables for the given radars.

    Returns one row per gate with ``[gate_id, radar, sweep, x, y, z, altitude]``,
    where ``x``/``y`` are the archive's projected coordinates renamed to a fixed
    pair so callers need not know the EPSG.  Cached per (base_path, radar, epsg).

    Parameters
    ----------
    base_path : str or Path
        RadDB archive base directory.
    radars : list of str
        Radar names whose LUTs to load (e.g. ``["L", "P"]``).

    Returns
    -------
    pl.DataFrame
    """
    base = Path(base_path)
    epsg = aoi_epsg_for(base, list(radars), override=epsg)
    frames = []
    for radar in radars:
        key = (str(base), radar, int(epsg))
        cached = _CENTROID_CACHE.get(key)
        if cached is None:
            cached = _load_one_centroid_table(base, radar, int(epsg))
            _CENTROID_CACHE[key] = cached
        frames.append(cached)
    if not frames:
        return pl.DataFrame(
            schema={
                "gate_id": pl.Int64,
                "radar": pl.String,
                "sweep": pl.Int32,
                "x": pl.Float64,
                "y": pl.Float64,
                "z": pl.Float64,
                "altitude": pl.Float64,
            },
        )
    return pl.concat(frames, how="vertical")


def _load_one_centroid_table(base: Path, radar: str, epsg: int) -> pl.DataFrame:
    """Load one radar's centroids in ``epsg`` (helper for :func:`_lut_centroids`).

    The projected pair is renamed to plain ``x``/``y`` so the rest of the module
    never has to know which EPSG it is working in.  When the LUT already stores
    that EPSG the columns are read straight off disk; otherwise — an ``aoi_crs=``
    override, or a differently-projected archive — they are computed from the
    ``latitude``/``longitude`` every LUT carries.
    """
    lut_path = base / radar / "LUT" / f"{radar}_LUT.parquet"
    if not lut_path.exists():
        raise FileNotFoundError(
            f"LUT not found at {lut_path}. Cannot resolve AOI for radar {radar!r}.",
        )

    # Column names come from the parquet footer — reading the whole LUT just to
    # inspect `.columns` would load ~100 MB and throw it away.
    available = set(pq.read_schema(lut_path).names)
    xc, yc = f"x_{int(epsg)}", f"y_{int(epsg)}"

    if {xc, yc}.issubset(available):
        cols = [c for c in _CENTROID_BASE_COLS if c in available] + [xc, yc]
        lut = pl.read_parquet(lut_path, columns=cols).rename({xc: "x", yc: "y"})
    else:
        cols = [c for c in (*_CENTROID_BASE_COLS, "latitude", "longitude") if c in available]
        lut = add_lut_projection(pl.read_parquet(lut_path, columns=cols), epsg=int(epsg))
        # Rename first: the select below must see the renamed columns, not x_<epsg>.
        lut = lut.rename({xc: "x", yc: "y"})
        lut = lut.select([c for c in (*_CENTROID_BASE_COLS, "x", "y") if c in lut.columns])

    return lut.with_columns(pl.lit(radar).alias("radar"))


# proj4 definitions for the AOI's canonical frames.  Using proj4 strings (rather
# than EPSG lookups) keeps reprojection working even where the PROJ database is
# unavailable — the same tactic raddb.lut.add_lut_projection uses for WGS-84.
# The LV95 string is the standard 7-parameter CH1903+ definition; it agrees with
# the grid-based EPSG:2056 transform to well under a meter, negligible at the
# kilometer scale of AOI gate selection.
_PROJ4 = {
    4326: "+proj=longlat +datum=WGS84 +no_defs",
    2056: (
        "+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 "
        "+k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel "
        "+towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs"
    ),
}


def _to_pyproj_crs(spec):
    """Build a ``pyproj.CRS``, preferring DB-free proj4 strings for known frames.

    Falls back to :func:`pyproj.CRS` for other specs (which needs a working PROJ
    database); a proj4 string is routed to ``from_proj4`` so custom definitions
    work database-free too.
    """
    import pyproj

    if isinstance(spec, pyproj.CRS):
        return spec
    if isinstance(spec, int) and spec in _PROJ4:
        return pyproj.CRS.from_proj4(_PROJ4[spec])
    if isinstance(spec, str) and "+proj" in spec:
        return pyproj.CRS.from_proj4(spec)
    return pyproj.CRS(spec)


def _reproject_to_aoi(geom, crs: int | str | None, aoi_epsg: int):
    """Reproject a shapely geometry into the AOI's CRS.

    Parameters
    ----------
    geom : shapely geometry
        AOI geometry expressed in ``crs``.
    crs : int, str, or None
        CRS of ``geom``.  ``None`` means "already in the AOI CRS" — the common
        case when the user works in the frame the archive was written with.
    aoi_epsg : int
        The archive's CRS, from :func:`aoi_epsg_for`.

    Returns
    -------
    shapely geometry in ``aoi_epsg``.
    """
    if crs is None:
        return geom
    # Normalize common spellings of "already the AOI CRS" → no pyproj needed.
    if crs in (aoi_epsg, str(aoi_epsg), f"EPSG:{aoi_epsg}", f"epsg:{aoi_epsg}"):
        return geom

    src = _to_pyproj_crs(crs)
    tgt = _to_pyproj_crs(aoi_epsg)
    if src.equals(tgt):
        return geom

    import pyproj

    transformer = pyproj.Transformer.from_crs(src, tgt, always_xy=True)
    return shapely.ops.transform(transformer.transform, geom)


def _resolve_aoi_centroids(centroids: pl.DataFrame, aoi_geom) -> pl.DataFrame:
    """Return the centroid rows whose ``(x, y)`` lie inside an AOI.

    A cheap bounding-box mask pre-filters the (potentially millions of) centroids
    before the vectorized ``shapely.contains`` point-in-geometry test, which keeps
    the intersection at a few milliseconds even for arbitrary polygons.

    Because ``centroids`` spans every sweep, the result covers the full vertical
    column above the AOI footprint.  Returning the rows (not just the ids) lets
    callers clip by ``altitude`` and merge geometry without a second lookup.

    Parameters
    ----------
    centroids : polars.DataFrame
        Output of :func:`_lut_centroids` (needs ``x``, ``y``, ``gate_id``).
    aoi_geom : shapely geometry
        AOI footprint, already in the AOI CRS.

    Returns
    -------
    pl.DataFrame
        Subset of ``centroids`` inside the AOI (may be empty).
    """
    if centroids.is_empty():
        return centroids.clear()

    x = centroids["x"].to_numpy()
    y = centroids["y"].to_numpy()

    minx, miny, maxx, maxy = aoi_geom.bounds
    bbox_mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    if not bbox_mask.any():
        return centroids.clear()

    # Exact point-in-geometry only on the bbox survivors.
    pts = shapely.points(x[bbox_mask], y[bbox_mask])
    inside = shapely.contains(aoi_geom, pts)

    keep = np.zeros(len(x), dtype=bool)
    keep[np.flatnonzero(bbox_mask)[inside]] = True
    return centroids.filter(keep)


def _resolve_gate_ids(centroids: pl.DataFrame, aoi_geom) -> np.ndarray:
    """Resolve the ``gate_id`` set whose centroid lies inside an AOI geometry.

    Thin wrapper over :func:`_resolve_aoi_centroids` returning just the ids.

    Returns
    -------
    np.ndarray of int64
        gate_ids inside the AOI (may be empty).
    """
    sub = _resolve_aoi_centroids(centroids, aoi_geom)
    if sub.is_empty():
        return np.empty(0, dtype=np.int64)
    return sub["gate_id"].to_numpy().astype(np.int64, copy=False)


def _apply_gate_ids(df: pl.DataFrame, gate_ids: np.ndarray) -> pl.DataFrame:
    """Filter a dynamic DataFrame to the rows whose ``gate_id`` is in ``gate_ids``.

    A semi-join: rows are **selected**, never widened — the static LUT geometry
    stays in its own table and is joined only by the ``to_*`` converters.

    Parameters
    ----------
    df : pl.DataFrame
        Dynamic volume data (must have a ``gate_id`` column).
    gate_ids : np.ndarray
        gate_id set from :func:`_resolve_gate_ids`.

    Returns
    -------
    pl.DataFrame
        Subset of ``df`` holding exactly the AOI rows, with ``df``'s columns.
    """
    if "gate_id" not in df.columns:
        raise KeyError("dynamic DataFrame has no 'gate_id' column; cannot apply AOI selection.")
    if len(gate_ids) == 0:
        return df.clear()
    keep = pl.DataFrame({"gate_id": np.asarray(gate_ids, dtype=np.int64)})
    return df.join(keep, on="gate_id", how="semi")


# ============================================================================
# Context geometry (map background for quicklooks)
# ============================================================================

# Loaded-once cache for the reprojected Swiss border ("unset" = not yet tried).
_SWISS_BORDER: object = "unset"


def _swiss_border_2056():
    """Return Switzerland's border as a shapely geometry in EPSG:2056, or None.

    Loaded once per process from cartopy's cached Natural Earth 10 m admin-0
    dataset and reprojected with the DB-free transform.  Returns ``None`` (with a
    warning) if cartopy or the dataset is unavailable, so a quicklook still works
    — just without the country outline.
    """
    global _SWISS_BORDER
    if _SWISS_BORDER != "unset":
        return _SWISS_BORDER
    try:
        from cartopy.io import shapereader as shpreader

        path = shpreader.natural_earth(
            resolution="10m",
            category="cultural",
            name="admin_0_countries",
        )
        geom = None
        for rec in shpreader.Reader(path).records():
            attrs = rec.attributes
            if attrs.get("NAME") == "Switzerland" or attrs.get("ADMIN") == "Switzerland":
                geom = rec.geometry
                break
        # simplify (~300 m) to keep the outline light, then project to LV95.
        _SWISS_BORDER = None if geom is None else _reproject_to_aoi(geom.simplify(0.003), 4326, SWISS_EPSG)
    except Exception as exc:  # - context is optional; never fatal
        logger.warning("Swiss border context unavailable (%s); drawn without it.", exc)
        _SWISS_BORDER = None
    return _SWISS_BORDER


def _resolve_context(context, aoi_epsg: int = SWISS_EPSG):
    """Resolve a map-background context geometry to the AOI's frame, or ``None``.

    Parameters
    ----------
    context : None, str, shapely geometry, or GeoDataFrame/GeoSeries
        - ``None`` → no context.
        - ``"switzerland"`` (or ``"ch"``) → the Swiss border (Natural Earth).
        - a shapely geometry → assumed already in ``aoi_epsg``.
        - a GeoDataFrame/GeoSeries → dissolved and reprojected from its ``.crs``.
    aoi_epsg : int, default ``SWISS_EPSG``
        The frame the quicklook is drawn in — the archive's own CRS, not
        necessarily LV95.  Reprojecting a caller's context to a hardcoded 2056
        would put it thousands of km off-map on any non-Swiss archive.

    Returns
    -------
    shapely geometry in ``aoi_epsg``, or None.
    """
    if context is None:
        return None

    import shapely.geometry.base as _base

    if isinstance(context, str):
        if context.lower() in ("switzerland", "ch", "suisse", "schweiz", "svizzera"):
            border = _swiss_border_2056()
            if border is not None and aoi_epsg != SWISS_EPSG:
                border = _reproject_to_aoi(border, SWISS_EPSG, aoi_epsg)
            return border
        raise ValueError(
            f"unknown context {context!r}; use 'switzerland', None, or a geometry.",
        )
    if isinstance(context, _base.BaseGeometry):
        return context  # assume already in the quicklook frame
    if hasattr(context, "crs"):  # GeoDataFrame / GeoSeries
        try:
            geom = context.union_all() if hasattr(context, "union_all") else context.unary_union
            crs = getattr(context, "crs", None)
            epsg = crs.to_epsg() if crs is not None else None
            return _reproject_to_aoi(geom, epsg, aoi_epsg) if epsg not in (None, aoi_epsg) else geom
        except Exception as exc:
            logger.warning("could not resolve context geometry (%s); drawn without it.", exc)
            return None
    return None


# ============================================================================
# Polygon AOI loading  (shapely / GeoDataFrame / shapefile / GeoJSON)
# ============================================================================


def _load_aoi_polygon(polygon, crs=None, aoi_epsg: int | None = None):
    """Resolve a polygon AOI to a shapely geometry in ``aoi_epsg``.

    Accepts a shapely ``Polygon``/``MultiPolygon``, a GeoDataFrame/GeoSeries, or a
    path to a ``.shp`` or ``.geojson`` / ``.json`` file.  The source CRS is taken
    from ``crs`` when given, else auto-detected (GeoDataFrame ``.crs``; a GeoJSON
    ``crs`` member or WGS-84; a shapefile ``.prj``), else assumed to be
    ``aoi_epsg`` already — see the note on the return statement below.

    File reading is dependency-light: GeoJSON via :mod:`json` + shapely, shapefiles
    via :mod:`shapefile` (pyshp) — no geopandas/pyogrio/fiona needed.
    """
    import shapely.geometry.base as _base

    src_crs = None
    if isinstance(polygon, (str, Path)):
        geom, src_crs = _read_polygon_file(Path(polygon))
    elif isinstance(polygon, _base.BaseGeometry):
        geom = polygon
    elif hasattr(polygon, "crs"):  # GeoDataFrame / GeoSeries
        geom = polygon.union_all() if hasattr(polygon, "union_all") else polygon.unary_union
        src_crs = _crs_to_spec(getattr(polygon, "crs", None))
    else:
        raise TypeError(
            f"crop_polygon: unsupported polygon input {type(polygon).__name__}; pass a "
            "shapely (Multi)Polygon, a GeoDataFrame, or a .shp / .geojson path.",
        )

    if geom is None or geom.is_empty:
        raise ValueError("crop_polygon: the polygon AOI is empty.")
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f"crop_polygon expects a Polygon/MultiPolygon; got {geom.geom_type}.",
        )

    # No fallback CRS: when neither the caller nor the file says, the geometry is
    # taken to be already in the AOI frame rather than silently assumed to be LV95.
    effective = crs if crs is not None else src_crs
    return _reproject_to_aoi(geom, effective, aoi_epsg)


def _crs_to_spec(crs):
    """Reduce a pyproj CRS (or None) to an EPSG int when possible, else pass through."""
    if crs is None:
        return None
    try:
        epsg = crs.to_epsg()
        return epsg if epsg is not None else crs
    except Exception:
        return crs


def _read_geometry_file(path: Path):
    """Read a geometry file → ``(geom, src_crs)``.

    ``.shp`` via pyshp, ``.geojson`` via json.  Polygons define an AOI to crop
    with; lines define a vertical cross-section.  The returned CRS is what the
    file declares — callers must honor it rather than assuming LV95.
    """
    if not path.exists():
        raise FileNotFoundError(f"Geometry file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json"):
        return _read_geojson(path)
    if suffix == ".shp":
        return _read_shapefile(path)
    raise ValueError(f"Unsupported geometry file type {suffix!r}; use .shp or .geojson.")


#: Back-compat alias — the reader handles lines as well as polygons now.
_read_polygon_file = _read_geometry_file


def _read_geojson(path: Path):
    import json

    from shapely.geometry import shape

    data = json.loads(path.read_text())
    kind = data.get("type")
    if kind == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]
    elif kind == "Feature":
        geoms = [shape(data["geometry"])]
    elif kind in (
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
        "LineString",
        "MultiLineString",
        "Point",
        "MultiPoint",
    ):
        # A bare top-level geometry, as hand-written files and some exporters
        # produce.  Lines matter here: a cross-section is defined by one.
        geoms = [shape(data)]
    else:
        raise ValueError(
            f"Unrecognized GeoJSON object type {kind!r}; expected a "
            "FeatureCollection, a Feature, or a bare geometry.",
        )
    if not geoms:
        raise ValueError(f"No geometries found in {path}.")
    return shapely.union_all(geoms), _geojson_crs(data)


def _geojson_crs(data):
    """CRS of a GeoJSON: legacy ``crs`` member if present, else WGS-84 (RFC 7946)."""
    member = data.get("crs")
    if isinstance(member, dict):
        import re

        name = str(member.get("properties", {}).get("name", ""))
        m = re.search(r"EPSG:{1,2}(\d+)", name)
        if m:
            return int(m.group(1))
    return 4326


def _read_shapefile(path: Path):
    try:
        import shapefile  # pyshp
    except ImportError as exc:
        raise ImportError(
            "Reading .shp needs pyshp (or pass a GeoDataFrame / .geojson). " "Install with: pip install pyshp",
        ) from exc
    from shapely.geometry import shape

    reader = shapefile.Reader(str(path))
    geoms = [shape(s.__geo_interface__) for s in reader.shapes() if s.shapeType]
    if not geoms:
        raise ValueError(f"No geometries found in {path}.")
    return shapely.union_all(geoms), _prj_crs(path.with_suffix(".prj"))


def _prj_crs(prj_path: Path):
    """Best-effort CRS from a shapefile ``.prj`` (WKT): EPSG int, WKT string, or None."""
    if not prj_path.exists():
        return None
    wkt = prj_path.read_text().strip()
    if not wkt:
        return None
    try:
        import pyproj

        return pyproj.CRS.from_wkt(wkt).to_epsg() or wkt
    except Exception:  # - broken PROJ db: sniff for Swiss LV95, else pass WKT
        low = wkt.lower()
        if "2056" in wkt or "ch1903+" in low or "lv95" in low:
            return SWISS_EPSG
        return wkt


# ============================================================================
# Vertical cross-section geometry
# ----------------------------------------------------------------------------
# Ported from the RADAR_POLYGONS_CROSS_SECTIONS prototypes:
#   - gate half-dimensions + corner math  <- radDB_csts.get_rad_db /
#     get_gate_corner_xyzd / get_gate_3D_coordinates ("Eo" face = horizontal
#     footprint quad)
#   - line/footprint intersection + beam-following endpoint altitudes +
#     perpendicular +-dE offsets -> per-gate polygon in the
#     (distance-along-line, altitude) plane
#     <- radDB_spatial_plot.get_rad_gdb_vert_cross_section
# Adapted to the current data model (projected x/y + altitude from the LUT) and
# fully vectorized (numpy point-to-segment prefilter replaces the KDTree).
# ============================================================================

_CS_LUT_BASE_COLS = [
    "gate_id",
    "sweep",
    "azimuth",
    "range",
    "elevation_angle",
    "altitude",
]
# Per-radar cross-section geometry cache: {(base_path, radar, beamwidth): df}.
_CS_CACHE: dict = {}


def _lut_cs_table(
    base_path: str | Path,
    radars: list[str],
    beamwidth_deg: float = 1.0,
    epsg=None,
) -> pl.DataFrame:
    """Static per-gate geometry for cross-sections: centers + half-dimensions.

    Half-dimensions (prototype convention):
      - ``dR``: half the radial gate spacing, derived per sweep from the LUT's
        range grid (e.g. 500 m -> 250 m at 500 m range sampling);
      - ``dA`` (= dE): half the across-beam extent, ``range * tan(beamwidth/2)``
        — grows with range.
    """
    epsg = aoi_epsg_for(base_path, list(radars), override=epsg)
    frames = []
    for radar in radars:
        lut_path = Path(base_path) / radar / "LUT" / f"{radar}_LUT.parquet"
        if not lut_path.exists():
            raise FileNotFoundError(
                f"LUT not found at {lut_path}. Cannot build cross-section for radar {radar!r}.",
            )
        # mtime in the key: a LUT regenerated in a live session must invalidate.
        key = (str(base_path), radar, float(beamwidth_deg), int(epsg), lut_path.stat().st_mtime_ns)
        t = _CS_CACHE.get(key)
        if t is None:
            available = set(pq.read_schema(lut_path).names)
            xc, yc = f"x_{int(epsg)}", f"y_{int(epsg)}"
            # The LUT stores two different x/y: meters from the radar, and the
            # projected pair.  The section geometry works in the projected one,
            # which is why it takes the plain `x`/`y` names here; the
            # radar-relative pair rides along as x_rel/y_rel and is renamed back
            # on output (see RadDB.extract_cross_section).
            rel = [c for c in ("x", "y") if c in available]
            if {xc, yc}.issubset(available):
                t = pl.read_parquet(lut_path, columns=[*_CS_LUT_BASE_COLS, *rel, xc, yc])
            else:
                t = add_lut_projection(
                    pl.read_parquet(
                        lut_path,
                        columns=[*_CS_LUT_BASE_COLS, *rel, "latitude", "longitude"],
                    ),
                    epsg=int(epsg),
                ).select([*_CS_LUT_BASE_COLS, *rel, xc, yc])
            t = t.rename({c: f"{c}_rel" for c in rel}).rename({xc: "x", yc: "y"})
            # Radial spacing per sweep from the unique range grid -> dR = spacing/2.
            # `range` is cast to Float64 first so the median-of-diffs matches the
            # float64 arithmetic the pandas implementation used.
            spacing = (
                t.select(["sweep", "range"])
                .unique()
                .sort(["sweep", "range"])
                .group_by("sweep")
                .agg(
                    pl.col("range").cast(pl.Float64).diff().drop_nulls().median().alias("_spacing"),
                )
            )
            t = (
                t.join(spacing, on="sweep", how="left")
                .with_columns(
                    (pl.col("_spacing") / 2.0).cast(pl.Float64).alias("dR"),
                    (pl.col("range").cast(pl.Float64) * float(np.tan(np.deg2rad(beamwidth_deg / 2.0)))).alias("dA"),
                    pl.lit(radar).alias("radar"),
                )
                .drop("_spacing")
            )
            _CS_CACHE[key] = t
        frames.append(t)
    if not frames:
        raise ValueError("no radars given for cross-section geometry.")
    return pl.concat(frames, how="vertical_relaxed")


def _lut_corner_rings(base_path, sub: pd.DataFrame, kind: str, cols: tuple[str, str], epsg: int):
    """Per-gate corner rings read from a LUT lattice, aligned to ``sub``'s rows.

    Returns an ``(n, 4, 2)`` array, or ``None`` when the lattice cannot supply
    the requested columns (e.g. an archive whose LUT carries no EPSG:2056
    projection), leaving the caller to fall back.
    """
    from raddb.lut import gate_corner_table

    gate_ids = sub["gate_id"].to_numpy(dtype=np.int64)
    radars = sorted(set(sub["radar"])) if "radar" in sub.columns else _radars_from_gate_ids(gate_ids)

    frames = []
    for radar in radars:
        try:
            frames.append(gate_corner_table(radar, base_path, kind=kind))
        except (FileNotFoundError, KeyError) as exc:
            logger.warning(
                "radar %s: %s lattice unavailable (%s); falling back to the " "planar gate approximation.",
                radar,
                kind,
                exc,
            )
            return None
    tbl = pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]

    cols = (f"x_{int(epsg)}", f"y_{int(epsg)}") if cols == ("x", "y") else cols
    need = [f"{cols[0]}_{k}" for k in range(1, 5)] + [f"{cols[1]}_{k}" for k in range(1, 5)]
    if not all(c in tbl.columns for c in need):
        logger.warning(
            "the %s lattice has no %s/%s columns; falling back to the planar " "gate approximation.",
            kind,
            cols[0],
            cols[1],
        )
        return None

    # Left-join keeps ``sub``'s row order, which the callers index against.
    aligned = pl.DataFrame({"gate_id": gate_ids}).join(
        tbl.select(["gate_id", *need]),
        on="gate_id",
        how="left",
        maintain_order="left",
    )
    ring = np.stack(
        [
            np.stack([aligned[f"{cols[0]}_{k}"].to_numpy(), aligned[f"{cols[1]}_{k}"].to_numpy()], axis=1)
            for k in range(1, 5)
        ],
        axis=1,
    ).astype(np.float64)
    if not np.isfinite(ring).all():
        logger.warning(
            "%d gate(s) have no %s geometry; falling back to the planar " "approximation for this call.",
            int((~np.isfinite(ring).all(axis=(1, 2))).sum()),
            kind,
        )
        return None
    return ring


def _gate_footprints(sub: pd.DataFrame, half_bw_tan: float, base_path=None, epsg: int | None = None) -> np.ndarray:
    """Horizontal footprint quad per gate (the prototype's ``Eo_xyz`` face).

    Preferred source is the ``h_plane`` lattice, i.e. the same exact curved-beam
    ``ke=4/3`` corners the plots draw, so cross-sections and gridding agree with
    :func:`raddb.viz.plot.plot_ppi` gate for gate.

    Falls back to the original planar construction — center displaced along the
    beam (±dR, foreshortened by cos(el)) and across it (±dA evaluated at
    range±dR) — when the lattice is unavailable, e.g. an archive with no
    EPSG:2056 projection in its LUT.
    """
    if base_path is not None and epsg is not None:
        ring = _lut_corner_rings(base_path, sub, "h_plane", ("x", "y"), epsg)
        if ring is not None:
            return shapely.polygons(ring)

    az = np.deg2rad(sub["azimuth"].to_numpy(dtype=np.float64))
    el = np.deg2rad(sub["elevation_angle"].to_numpy(dtype=np.float64))
    xc = sub["x"].to_numpy(dtype=np.float64)
    yc = sub["y"].to_numpy(dtype=np.float64)
    rng = sub["range"].to_numpy(dtype=np.float64)
    dR = sub["dR"].to_numpy(dtype=np.float64)

    sin_az, cos_az, cos_el = np.sin(az), np.cos(az), np.cos(el)
    rings = []
    for s_r, s_a in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        dr = s_r * dR
        da = s_a * (rng + dr) * half_bw_tan
        rings.append(
            np.stack(
                [
                    xc + dr * cos_el * sin_az + da * cos_az,
                    yc + dr * cos_el * cos_az - da * sin_az,
                ],
                axis=1,
            ),
        )
    return shapely.polygons(np.stack(rings, axis=1))


def _beam_profile(base_path, sub: pd.DataFrame, epsg: int):
    """Per-gate beam profile from the ``v_plane`` lattice, or ``None``.

    Returns ``(d_near, d_far, z_near, z_far, half_thickness)`` — the gate's
    beam-center altitude at its near and far range edge, as a function of ground
    distance, plus half its vertical extent there.  This is the curved-beam
    geometry the plots draw, replacing the flat ``u * tan(el)`` climb.
    """
    ring = _lut_corner_rings(base_path, sub, "v_plane", ("d", "z_asl"), epsg)
    if ring is None:
        return None
    # v_plane ring order: near-bottom, far-bottom, far-top, near-top.
    d_near = 0.5 * (ring[:, 0, 0] + ring[:, 3, 0])
    d_far = 0.5 * (ring[:, 1, 0] + ring[:, 2, 0])
    z_near = 0.5 * (ring[:, 0, 1] + ring[:, 3, 1])
    z_far = 0.5 * (ring[:, 1, 1] + ring[:, 2, 1])
    half_thick = 0.5 * (np.abs(ring[:, 3, 1] - ring[:, 0, 1]) + np.abs(ring[:, 2, 1] - ring[:, 1, 1])) * 0.5
    return d_near, d_far, z_near, z_far, half_thick


def _endpoint_d_z(pt_xy: np.ndarray, sub: pd.DataFrame, origin: tuple[float, float], profile=None):
    """(distance-along-line, altitude) of chord endpoints on the beam surface.

    With a ``profile`` from :func:`_beam_profile` the endpoint's altitude is
    interpolated along the gate's own curved beam between its near and far range
    edges — the same geometry ``v_plane`` stores and the RHI draws.

    Without one, it falls back to the planar approximation: project the
    center->endpoint vector onto the beam direction and climb ``u * tan(el)``
    from the gate-center altitude.
    """
    d = np.hypot(pt_xy[:, 0] - origin[0], pt_xy[:, 1] - origin[1])

    if profile is not None:
        d_near, d_far, z_near, z_far, _ = profile
        # Ground distance of the endpoint from the radar, along the beam.
        rx = pt_xy[:, 0] - sub["x"].to_numpy(dtype=np.float64)
        ry = pt_xy[:, 1] - sub["y"].to_numpy(dtype=np.float64)
        az = np.deg2rad(sub["azimuth"].to_numpy(dtype=np.float64))
        # Along-beam offset of the endpoint from the gate center.
        u = rx * np.sin(az) + ry * np.cos(az)
        d_center = 0.5 * (d_near + d_far)
        span = d_far - d_near
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(span != 0, (d_center + u - d_near) / span, 0.5)
        z = z_near + np.nan_to_num(t, nan=0.5) * (z_far - z_near)
        return d, z

    az = np.deg2rad(sub["azimuth"].to_numpy(dtype=np.float64))
    el = np.deg2rad(sub["elevation_angle"].to_numpy(dtype=np.float64))
    dx = pt_xy[:, 0] - sub["x"].to_numpy(dtype=np.float64)
    dy = pt_xy[:, 1] - sub["y"].to_numpy(dtype=np.float64)
    delta = (2.0 * np.pi - az - np.arctan2(dy, dx) + np.pi / 2.0) % (2.0 * np.pi)
    u = np.hypot(dx, dy) * np.cos(delta)
    z = sub["altitude"].to_numpy(dtype=np.float64) + u * np.tan(el)
    return d, z


def _cross_section_gates(
    cs_t: pl.DataFrame | pd.DataFrame,
    p1: tuple[float, float],
    p2: tuple[float, float],
    beamwidth_deg: float = 1.0,
    min_chord_m: float = 0.5,
    base_path=None,
    epsg: int | None = None,
) -> pd.DataFrame:
    """Gates whose horizontal footprint crosses the line ``p1 -> p2``.

    Accepts the polars geometry table from :func:`_lut_cs_table`.  The body
    works in pandas because the result carries ``cs_polygon`` — a column of
    shapely objects — for which pandas' object dtype is the natural carrier;
    :meth:`raddb.RadDB.extract_cross_section` converts the geometry columns back
    to polars when joining them onto the data frame.

    Returns one row per crossed gate with its cross-section geometry: the gate
    center ``(d_center, z_center)`` and ``cs_polygon`` — the 4-corner shapely polygon
    in the (distance-along-line [m], altitude [m ASL]) plane, built by
    offsetting the chord perpendicularly by ±dA (the vertical half-beamwidth
    extent).  ``d`` is measured from ``p1``.
    """
    if isinstance(cs_t, pl.DataFrame):
        cs_t = cs_t.to_pandas()

    ox, oy = float(p1[0]), float(p1[1])
    ex, ey = float(p2[0]), float(p2[1])
    length = float(np.hypot(ex - ox, ey - oy))
    if length < 1.0:
        raise ValueError("cross-section line is degenerate (< 1 m long).")
    half_bw_tan = float(np.tan(np.deg2rad(beamwidth_deg / 2.0)))

    # --- vectorized point-to-segment prefilter (replaces the KDTree) ---
    px = cs_t["x"].to_numpy(dtype=np.float64)
    py = cs_t["y"].to_numpy(dtype=np.float64)
    diag = np.hypot(cs_t["dR"].to_numpy(dtype=np.float64), cs_t["dA"].to_numpy(dtype=np.float64)) * 1.05
    t_par = ((px - ox) * (ex - ox) + (py - oy) * (ey - oy)) / (length * length)
    t_par = np.clip(t_par, 0.0, 1.0)
    dist = np.hypot(px - (ox + t_par * (ex - ox)), py - (oy + t_par * (ey - oy)))
    sub = cs_t[dist <= diag].reset_index(drop=True)
    if sub.empty:
        return sub.assign(cs_polygon=pd.Series(dtype=object))

    # --- exact footprint / line intersection -> per-gate chord ---
    footprints = _gate_footprints(sub, half_bw_tan, base_path=base_path, epsg=epsg)
    line = shapely.LineString([(ox, oy), (ex, ey)])
    chords = shapely.intersection(footprints, line)
    keep = ~shapely.is_empty(chords)
    sub, chords = sub[keep].reset_index(drop=True), chords[keep]
    if sub.empty:
        return sub.assign(cs_polygon=pd.Series(dtype=object))

    # First/last coordinate of each chord (Points collapse to zero length).
    coords, gidx = shapely.get_coordinates(chords, return_index=True)
    starts = np.searchsorted(gidx, np.arange(len(chords)))
    ends = np.append(starts[1:], len(gidx)) - 1
    p0, p1_ = coords[starts], coords[ends]
    good = np.hypot(*(p1_ - p0).T) > min_chord_m
    sub, p0, p1_ = sub[good].reset_index(drop=True), p0[good], p1_[good]
    if sub.empty:
        return sub.assign(cs_polygon=pd.Series(dtype=object))

    # --- endpoint (d, z) on the beam, ordered near/far along the line ---
    profile = _beam_profile(base_path, sub, epsg) if base_path is not None and epsg is not None else None
    d0, z0 = _endpoint_d_z(p0, sub, (ox, oy), profile)
    d1, z1 = _endpoint_d_z(p1_, sub, (ox, oy), profile)
    swap = d0 > d1
    d_near = np.where(swap, d1, d0)
    d_far = np.where(swap, d0, d1)
    z_near = np.where(swap, z1, z0)
    z_far = np.where(swap, z0, z1)

    # --- perpendicular ±dE offsets -> (d, z) polygon per gate ---
    incl = np.arctan2(z_far - z_near, d_far - d_near)  # chord inclination
    s_i, c_i = np.sin(incl), np.cos(incl)
    # Half the beam's real vertical extent at this gate, from v_plane; without a
    # profile fall back to dA (dE == dA, same beamwidth).
    dE = profile[4] if profile is not None else sub["dA"].to_numpy(dtype=np.float64)
    ring = np.stack(
        [
            np.stack([d_near - s_i * dE, z_near + c_i * dE], axis=1),  # near, top
            np.stack([d_far - s_i * dE, z_far + c_i * dE], axis=1),  # far, top
            np.stack([d_far + s_i * dE, z_far - c_i * dE], axis=1),  # far, bottom
            np.stack([d_near + s_i * dE, z_near - c_i * dE], axis=1),  # near, bottom
        ],
        axis=1,
    )

    out = sub.copy()
    # Only the gate center and its footprint are published.  The chord endpoints
    # d_near/d_far and z_near/z_far are what the polygon is built from, so
    # emitting them as well restated `cs_polygon` in scalar form.
    out["d_center"] = 0.5 * (d_near + d_far)
    out["z_center"] = 0.5 * (z_near + z_far)
    out["cs_polygon"] = shapely.polygons(ring)
    return out

"""Look-Up Table (LUT) generation and loading utilities.

The LUT stores one record per radar gate (azimuth x range x sweep) with
static spatial information (Cartesian coordinates, lat/lon, elevation).
It is generated once per radar and reused for every subsequent volume.

This module is **generic** — it works with any xarray DataTree that
has ``azimuth``, ``range``, and ``elevation`` coordinates per sweep.
No pyart or radar_api dependency.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
import yaml

from raddb.helper import (
    RADAR_ALPHABET,
    RADAR_CODE_LEN,
    list_sweep_names,
    normalize_radar_name,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Radar code — the leading field of a gate_id
# ----------------------------------------------------------------------------

#: Multiplier of the radar field in a ``gate_id`` — everything below it is the
#: gate's own ``sweep``/``azimuth``/``range`` (see :func:`encode_gate_ids`).
GATE_ID_RADAR_BASE: int = 1_000_000_000_000

#: Number of distinct radar codes: ``36**4 - 1`` is the largest, ``0`` the
#: smallest.  ``radar_code * 10**12`` must stay inside int64, which allows
#: ``9_223_371``; base-36 over four characters needs only ``1_679_615``.
MAX_RADAR_CODE: int = 36**RADAR_CODE_LEN - 1

#: Version of the ``gate_id`` encoding written into each radar's info YAML.
#:
#: 1. radar index ``A=0 … Z=25`` — 26 radars, single letters only.
#: 2. radar code = base-36 of the zero-padded 4-character name (``"A"`` ->
#:    ``"000A"`` -> 10, ``"KTLX"`` -> 971493) — 1,679,616 radars.
#:
#: The two disagree for every name (``"L"`` is 11 under v1, 21 under v2).  Only
#: v2 is ever written or read, and nothing detects v1 — a v1 archive loads
#: silently and decodes to the wrong radar.  The migration script that used to
#: convert one has been removed, so re-archive from the source volumes instead.
GATE_ID_VERSION: int = 2

#: The v1 radar index, kept only so the offset between a v1 ``gate_id`` and its
#: v2 replacement stays derivable (and testable) after the migration script was
#: removed.
LEGACY_RADAR_TO_IDX: dict[str, int] = {chr(ord("A") + i): i for i in range(26)}


def encode_radar_code(radar: str) -> int:
    """Base-36 radar code for the leading ``gate_id`` field.

    The name is normalized, right-aligned and zero-padded to
    :data:`~raddb.helper.RADAR_CODE_LEN` characters, then read as a base-36
    integer over :data:`~raddb.helper.RADAR_ALPHABET`.

    Parameters
    ----------
    radar : str
        Radar name, e.g. ``"A"``, ``"MLA"`` or ``"KTLX"``.

    Returns
    -------
    int
        A value in ``[0, MAX_RADAR_CODE]``.

    Examples
    --------
    >>> encode_radar_code("A")
    10
    >>> encode_radar_code("KTLX")
    971493
    """
    name = normalize_radar_name(radar).rjust(RADAR_CODE_LEN, "0")
    code = 0
    for char in name:
        code = code * 36 + RADAR_ALPHABET.index(char)
    return code


def decode_radar_code(code: int) -> str:
    """Inverse of :func:`encode_radar_code`.

    Parameters
    ----------
    code : int
        A radar code in ``[0, MAX_RADAR_CODE]``.

    ``decode_radar_code(encode_radar_code(name)) == name`` for every canonical
    name (all 1,679,580 of them).  The reverse holds for every code the encoder
    can emit; the 36 codes spelling ``"ML0".."MLZ"`` are the exception, because
    :func:`~raddb.helper.normalize_radar_name` resolves that legacy spelling to
    its final letter before encoding, so no ``gate_id`` ever carries one.

    Returns
    -------
    str
        The canonical radar name, without its zero padding.

    Raises
    ------
    ValueError
        If *code* is outside the representable range.

    Examples
    --------
    >>> decode_radar_code(971493)
    'KTLX'
    """
    value = int(code)
    if not 0 <= value <= MAX_RADAR_CODE:
        raise ValueError(
            f"radar code {value} is outside [0, {MAX_RADAR_CODE}] and names no radar.",
        )
    chars = []
    for _ in range(RADAR_CODE_LEN):
        value, rem = divmod(value, 36)
        chars.append(RADAR_ALPHABET[rem])
    return "".join(reversed(chars)).lstrip("0") or "0"


#: Radar code of every single-letter name.
#:
#: .. deprecated::
#:    Superseded by :func:`encode_radar_code`, which is not limited to A-Z.
#:    Kept because it is part of the public API surface; do **not** use it as a
#:    membership test for "is this a usable radar name" — see
#:    :func:`raddb.helper.is_valid_radar_name`.
RADAR_TO_IDX: dict[str, int] = {chr(ord("A") + i): encode_radar_code(chr(ord("A") + i)) for i in range(26)}

#: Antenna 3 dB beamwidth in degrees, used for the gate's angular extent.
#: 1.0 deg is typical of an operational C-band antenna, and matches the
#: reference prototype (which hardcoded ``beta = deg2rad(0.5)`` as the *half*
#: beamwidth).
DEFAULT_BEAMWIDTH_DEG: float = 1.0

#: The five files that make up a complete LUT directory for one radar.
#: ``{radar}`` is substituted with the canonical radar name.
LUT_FILES: dict[str, str] = {
    "lut": "{radar}_LUT.parquet",
    "h_plane": "{radar}_h_plane_LUT.parquet",
    "v_plane": "{radar}_v_plane_LUT.parquet",
    "corners": "{radar}_corners_LUT.parquet",
    "info": "{radar}_info.yaml",
}


def _projection_column_names(df) -> list[str]:
    """``[x_<suffix>, y_<suffix>]`` present on a LUT frame, else ``[]``.

    Local to this module: ``io_core._projection_columns`` does the same job but
    importing it here would be circular (``io_core`` imports ``lut``).
    """
    xs = [c for c in df.columns if re.match(r"^x_\w+$", c)]
    out: list[str] = []
    for xc in xs:
        yc = "y_" + xc[2:]
        if yc in df.columns:
            out += [xc, yc]
    return out


def lut_file_path(radar: str, kind: str, lut_base_path: str | Path) -> Path:
    """Path of one of the five LUT files (see :data:`LUT_FILES`)."""
    if kind not in LUT_FILES:
        raise KeyError(f"unknown LUT file kind {kind!r}; use one of {sorted(LUT_FILES)}.")
    return Path(lut_base_path) / radar / "LUT" / LUT_FILES[kind].format(radar=radar)


# ============================================================================
# Coordinate transforms  (pure numpy, no pyart dependency)
# ============================================================================


def _interpolate_range_edges(ranges: np.ndarray) -> np.ndarray:
    """Interpolate the edges of range gates (PyART's formula).

    Returns an array of size ``ranges.size + 1``: the midpoints between
    consecutive gates, with first and last edges extrapolated by half a
    step and clipped to non-negative values.
    """
    r = np.asarray(ranges, dtype=np.float64)
    edges = np.empty(r.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[0] = r[0] - 0.5 * (r[1] - r[0])
    edges[-1] = r[-1] + 0.5 * (r[-1] - r[-2])
    edges[edges < 0] = 0.0
    return edges


def _interpolate_elevation_edges(elevations: np.ndarray) -> np.ndarray:
    """Interpolate elevation-angle edges (linear, clipped to [-90, 90])."""
    el = np.asarray(elevations, dtype=np.float64)
    edges = np.empty(el.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (el[:-1] + el[1:])
    edges[0] = el[0] - 0.5 * (el[1] - el[0])
    edges[-1] = el[-1] + 0.5 * (el[-1] - el[-2])
    return np.clip(edges, -90.0, 90.0)


def _interpolate_azimuth_edges(azimuths: np.ndarray) -> np.ndarray:
    """Interpolate azimuth-angle edges using complex-plane midpoints.

    Complex-number interpolation handles the 360°→0° wrap-around correctly:
    the midpoint between 359° and 1° is 0°, not 180°.

    Returns
    -------
    edges : np.ndarray
        Shape ``(azimuths.size + 1,)``, values in [0, 360).
    """
    az = np.asarray(azimuths, dtype=np.float64)
    z = np.exp(1j * np.deg2rad(az))
    midpoints = 0.5 * (z[:-1] + z[1:])
    first = z[0] - (midpoints[0] - z[0])
    last = z[-1] + (z[-1] - midpoints[-1])
    edges = np.concatenate(([first], midpoints, [last]))
    return np.rad2deg(np.angle(edges)) % 360.0


def antenna_vectors_to_cartesian(
    ranges: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    ke: float = 4.0 / 3.0,
    edges: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert radar antenna coordinates to Cartesian (x, y, z).

    Replicates the standard radar beam propagation formula used by PyART
    (``pyart.core.transforms.antenna_vectors_to_cartesian``) using only
    numpy.  The 4/3 effective-Earth-radius model is used by default.

    Parameters
    ----------
    ranges : 1-D array, shape (n_gates,)
        Range to each gate in meters.
    azimuths : 1-D array, shape (n_rays,)
        Azimuth angle of each ray in degrees (0 = North, clockwise).
    elevations : 1-D array, shape (n_rays,)
        Elevation angle of each ray in degrees.
    ke : float
        Effective Earth radius scale factor (default 4/3).
    edges : bool
        If True, interpolate ranges/azimuths/elevations to gate edges
        (size N+1) before computing Cartesian coords. Used for
        ``pcolormesh(shading="flat")`` rendering.

    Returns
    -------
    x, y, z : arrays, shape (n_rays, n_gates) — or (n_rays+1, n_gates+1) if ``edges``.
        Cartesian coordinates in meters relative to the radar.
    """
    if edges:
        ranges = _interpolate_range_edges(ranges)
        azimuths = _interpolate_azimuth_edges(azimuths)
        elevations = _interpolate_elevation_edges(elevations)

    r = np.atleast_1d(np.asarray(ranges, dtype=np.float64))
    theta_e = np.deg2rad(np.atleast_1d(np.asarray(elevations, dtype=np.float64)))
    theta_a = np.deg2rad(np.atleast_1d(np.asarray(azimuths, dtype=np.float64)))

    R = 6371.0 * 1000.0 * ke  # effective earth radius [m]

    # Height of each gate above radar (n_rays, n_gates)
    z = (
        np.sqrt(
            r[np.newaxis, :] ** 2 + R**2 + 2.0 * r[np.newaxis, :] * R * np.sin(theta_e[:, np.newaxis]),
        )
        - R
    )

    # Ground-range arc length
    s = R * np.arcsin(
        r[np.newaxis, :] * np.cos(theta_e[:, np.newaxis]) / (R + z),
    )

    # Cartesian
    x = s * np.sin(theta_a[:, np.newaxis])
    y = s * np.cos(theta_a[:, np.newaxis])

    return x, y, z


def cartesian_to_geographic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    radar_alt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert gate Cartesian offsets to geographic coordinates.

    Uses an equirectangular approximation which is accurate for
    typical radar ranges (< 250 km).

    Parameters
    ----------
    x, y, z : arrays
        Cartesian offsets from the radar in meters (x = east, y = north,
        z = height above radar).
    radar_lat, radar_lon : float
        Radar site latitude and longitude in degrees.
    radar_alt : float
        Radar site altitude in meters above sea level.

    Returns
    -------
    lat, lon, alt : arrays (same shape as input)
        Geographic coordinates of each gate.
    """
    R_EARTH = 6_371_000.0  # mean Earth radius [m]
    lat = radar_lat + np.degrees(y / R_EARTH)
    lon = radar_lon + np.degrees(x / (R_EARTH * np.cos(np.radians(radar_lat))))
    alt = radar_alt + z
    return lat, lon, alt


def compute_gate_xyz(
    ranges: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    ke: float = 4.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gate Cartesian coordinates from polar coordinates.

    Thin wrapper around :func:`antenna_vectors_to_cartesian` for
    backwards compatibility.
    """
    return antenna_vectors_to_cartesian(ranges, azimuths, elevations, ke=ke)


# ============================================================================
# The nominal azimuth grid
# ============================================================================

#: Azimuths are stored in ``gate_id`` as ``round(azimuth * 10)`` — tenths of a
#: degree.  A full turn is therefore ``360 * 10`` steps.
AZIMUTH_SCALE: int = 10
AZIMUTH_STEPS: int = 360 * AZIMUTH_SCALE

#: Smallest share of a rotation's grid points that must actually carry a ray.
#: Real volumes drop the odd ray (718 of 720 on WSR-88D, 358 of 360), which is a
#: full rotation with holes; a sector scan is not, and must still be refused.
MIN_ROTATION_COVERAGE: float = 0.95


def _round_half_up(values) -> np.ndarray:
    """``round`` that always breaks .5 upwards, unlike numpy's banker's rounding.

    A nominal grid lands exactly on a half-step whenever the ray spacing is an
    odd multiple of 0.05° — 720-ray NEXRAD sweeps put every ray center on
    ``x.x5``.  Banker's rounding would then alternate down/up and turn a uniform
    0.5° grid into an irregular 0.4°/0.6° one.
    """
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


def nominal_azimuth_grid(azimuths) -> np.ndarray:
    """The canonical ray azimuths of one sweep, as tenths of a degree.

    A radar's scan strategy fixes how many rays a sweep has and how they are
    spaced; what varies volume to volume is only where the antenna *reports*
    itself, which drifts by a few hundredths of a degree.  Rounding those
    measured angles to 0.1° therefore puts the same physical ray in different
    ``gate_id`` bins on different volumes, and every gate whose bin moved has no
    LUT row — measured at 6% of gates per volume on a 360-ray C-band network
    and 35% on WSR-88D.

    So the grid is derived from the scan strategy rather than from one volume's
    measurements: the spacing is the **median gap between neighboring rays**,
    and the offset is the circular mean of the measured residuals (circular
    because the offset is only defined modulo one step).  That gives 1.0° for a
    360-ray sweep and 0.5° for a 720-ray NEXRAD super-resolution sweep, from the
    same rule — which is why no per-network resolution has to be configured.

    The spacing is a **median rather than ``360 / n_rays``** because a real
    sweep drops the odd ray — 718 or 719 of 720 on WSR-88D, 358 of 360 — and
    dividing by the ray count reads that as a 0.5014° strategy, refusing a
    volume that is simply a rotation with holes.  The returned grid always
    covers the whole rotation, so the missing rays keep their place in it and a
    later volume that does record them still joins.

    Parameters
    ----------
    azimuths : array-like
        One sweep's measured ray azimuths [deg], any order.

    Returns
    -------
    np.ndarray of int64
        The rotation's ``n_grid`` sorted azimuths in tenths of a degree, in
        ``[0, 3600)`` — ``n_grid >= n_rays``, and larger when rays are missing.

    Raises
    ------
    ValueError
        If the sweep has no rays, if the spacing is finer than the 0.1°
        ``gate_id`` resolution, if two grid points collide after rounding, or if
        the rays do not cover a full rotation (a sector scan).
    """
    az = np.asarray(azimuths, dtype=np.float64).ravel() % 360.0
    n = az.size
    if n == 0:
        raise ValueError("cannot derive an azimuth grid from a sweep with no rays.")

    az_sorted = np.sort(az)
    # Gaps around the circle, the wrap included, so one full turn is covered.
    gaps = np.diff(np.append(az_sorted, az_sorted[0] + 360.0))
    spacing = float(np.median(gaps))
    if spacing <= 0.0:
        raise ValueError(
            f"the {n} rays of this sweep are not distinct enough to give a ray "
            f"spacing (median gap {spacing:.6f}°).",
        )

    # How many ray slots each gap spans: 1 between neighbors, 2 or more across
    # a hole.  Summing them counts the slots of the whole rotation, so the grid
    # size is the ray count plus whatever is missing.  The median spacing on its
    # own is too noisy to divide 360 by — antenna jitter alone turns a 360-ray
    # sweep into 361 — but it is easily good enough to tell a 1-slot gap from a
    # 2-slot one.
    slots_per_gap = _round_half_up(gaps / spacing).astype(np.int64)
    n_grid = int(slots_per_gap.sum())
    if n_grid < n:
        raise ValueError(
            f"the {n} rays of this sweep do not sit one per slot on a " f"{spacing:.4f}° grid (two rays share a slot).",
        )
    step = 360.0 / n_grid
    if step * AZIMUTH_SCALE < 1.0:
        raise ValueError(
            f"{n_grid} rays give a {step:.4f}° ray spacing, finer than the "
            f"{1 / AZIMUTH_SCALE}° azimuth resolution of gate_id; two rays would "
            f"share one gate_id.",
        )

    # A rotation may have holes but must still be a rotation: a sector scan (or
    # a sweep with a large gap) would otherwise be silently mangled — 90 rays
    # over a 90° sector get a 4° grid and collapse onto 23 of its points.
    if n > n_grid or n < MIN_ROTATION_COVERAGE * n_grid:
        raise ValueError(
            f"these {n} rays do not form a full rotation of evenly spaced rays "
            f"(they span {np.ptp(az_sorted):.1f}° with a derived spacing of "
            f"{step:.4f}°, filling {n}/{n_grid} of the rotation), so they have no "
            f"nominal azimuth grid. Sector scans and irregular sweeps are not "
            f"supported.",
        )

    # Residual of each ray against the slot it occupies on a step grid anchored
    # at 0.  Slots, not ray indices: after a hole the two part company, and ray
    # indices would drag every later residual a full step out.
    slots = np.concatenate([[0], np.cumsum(slots_per_gap[:-1])])
    resid = az_sorted - slots * step

    # The residual is defined only modulo one step, so it is averaged about the
    # first ray's own residual, with the rest wrapped into ±half a step of it.
    # Averaging on the raw period instead would put the mean on the seam exactly
    # when the rays are centered on half-steps — which is every 720-ray NEXRAD
    # sweep — and a 1e-16 wobble there moves the whole grid by a tenth of a
    # degree, differently for a volume that dropped a ray than for one that
    # did not.
    anchor = float(resid[0])
    centered = (resid - anchor + step / 2.0) % step - step / 2.0
    offset = anchor + float(centered.mean())

    grid = (np.arange(n_grid) * step + offset) % 360.0
    az_int = np.sort(_round_half_up(grid * AZIMUTH_SCALE).astype(np.int64) % AZIMUTH_STEPS)
    if np.unique(az_int).size != n_grid:
        raise ValueError(
            f"the {n_grid}-ray azimuth grid collides after rounding to "
            f"{1 / AZIMUTH_SCALE}°; this scan strategy cannot be stored in gate_id.",
        )

    # The grid is only meaningful if it actually fits the rays it came from.
    snapped, dist = snap_azimuths_to_grid(az, az_int)
    tol = azimuth_grid_tolerance(az_int)
    if np.unique(snapped).size != n or (dist.size and dist.max() > tol):
        raise ValueError(
            f"these {n} rays do not form a full rotation of evenly spaced rays "
            f"(they span {np.ptp(az_sorted):.1f}° with a derived spacing of "
            f"{step:.4f}°), so they have no nominal azimuth grid. Sector scans and "
            f"irregular sweeps are not supported.",
        )
    return az_int


def azimuth_grid_tolerance(grid) -> float:
    """Largest snap distance accepted for *grid*, in tenths of a degree.

    Half the ray spacing: beyond that a ray is closer to its neighbor than to
    itself, so it is not antenna drift but a different scan strategy.
    """
    n = len(grid)
    return (AZIMUTH_STEPS / n) / 2.0 if n else float("inf")


def snap_azimuths_to_grid(azimuths, grid) -> tuple[np.ndarray, np.ndarray]:
    """Match measured azimuths onto a sweep's canonical grid.

    Nearest neighbor **on the circle** — a ray reported at 359.97° belongs to
    the grid point at 0.0°, not to the one at 359.5°.

    Parameters
    ----------
    azimuths : array-like
        Measured ray azimuths [deg].
    grid : array-like of int
        The sweep's canonical azimuths in tenths of a degree
        (:func:`nominal_azimuth_grid`).

    Returns
    -------
    (snapped, distance) : np.ndarray
        ``snapped`` is the matched azimuth in tenths of a degree (int64);
        ``distance`` is how far each ray moved, in the same units (float).
    """
    # The comparison runs at full precision, *not* on the 0.1°-rounded azimuth:
    # rounding first would make a ray at 0.02° equidistant from 0.5° and 359.5°
    # and let the tie-break send it the wrong way across the seam.
    az_t = np.asarray(azimuths, dtype=np.float64).ravel() % 360.0 * AZIMUTH_SCALE
    g = np.unique(np.asarray(grid, dtype=np.int64))
    if g.size == 0:
        raise ValueError("cannot snap to an empty azimuth grid.")

    # Wrap the grid once each way so the nearest neighbor of a ray near 0° or
    # 360° is found across the seam.
    ext = np.concatenate([g - AZIMUTH_STEPS, g, g + AZIMUTH_STEPS]).astype(np.float64)
    idx = np.clip(np.searchsorted(ext, az_t), 1, ext.size - 1)
    lo, hi = ext[idx - 1], ext[idx]
    take_lo = (az_t - lo) <= (hi - az_t)
    snapped = np.where(take_lo, lo, hi).astype(np.int64) % AZIMUTH_STEPS
    distance = np.where(take_lo, az_t - lo, hi - az_t)
    return snapped, distance


def load_azimuth_grids(radar: str, base_path: str | Path) -> dict[int, np.ndarray] | None:
    """``{sweep: canonical azimuths}`` for a radar, or ``None`` if it has no LUT.

    The grid is read back out of the LUT parquet itself: its ``azimuth`` column
    already holds the snapped, canonical rays (``generate_lut_from_datatree``
    writes ``snapped / AZIMUTH_SCALE``), so rounding to tenths of a degree
    recovers exactly the values ``gate_id`` carries.  The LUT is therefore the
    single source of the grid, and the info YAML does not restate it.

    Only two of the LUT's thirteen columns are read, and parquet is columnar —
    33-66 ms on radar L's 96 MB / 1.7M-gate LUT, against ~2 s to archive the
    volume it is needed for.

    ``None`` means there is no LUT to read, in which case the caller must keep
    the measured azimuths: there is no grid to snap onto.
    """
    lut_path = lut_file_path(radar, "lut", base_path)
    if not lut_path.exists():
        return None
    az = pl.scan_parquet(lut_path).select(["sweep", "azimuth"]).unique().collect()
    grids = {
        int(sweep): np.sort(
            np.round(
                az.filter(pl.col("sweep") == sweep)["azimuth"].to_numpy() * AZIMUTH_SCALE,
            ).astype(np.int64),
        )
        for sweep in sorted(az["sweep"].unique().to_list())
    }
    return grids or None


# ============================================================================
# Gate ID generation
# ============================================================================


def encode_gate_ids(
    radar: str,
    sweeps: int | np.ndarray,
    azimuths: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    """Vectorized 64-bit gate identifier encoding.

    Encoding: ``radar_code * 10^12 + sweep * 10^10 + az_int * 10^6 + range_int``

    where ``radar_code`` is :func:`encode_radar_code`,
    ``az_int = round(azimuth * 10)`` (1 decimal place precision) and
    ``range_int = int(range_m)`` (integer meters).  This is the single
    canonical implementation, used by LUT generation and volume archiving.

    Parameters
    ----------
    radar : str
        Radar identifier, e.g. ``"A"`` or ``"KTLX"``.
    sweeps : int or array
        Sweep number(s) — a scalar (applied to all gates) or an array
        aligned with ``azimuths`` / ``ranges``.
    azimuths, ranges : arrays (aligned per gate)
        Azimuth [deg] and range [m] of each gate.

    Returns
    -------
    np.ndarray of int64
    """
    radar_code = np.int64(encode_radar_code(radar))
    sweep_v = np.asarray(sweeps, dtype=np.int64)
    az_int = np.round(np.asarray(azimuths, dtype=np.float64) * 10).astype(np.int64)
    rng_int = np.asarray(ranges).astype(np.int64)
    return (
        radar_code * np.int64(GATE_ID_RADAR_BASE)
        + sweep_v * np.int64(10_000_000_000)
        + az_int * np.int64(1_000_000)
        + rng_int
    )


def generate_gate_id(
    radar: str,
    sweep: int,
    azimuth: float,
    range_m: float,
) -> int:
    """Create a unique gate identifier as a 64-bit integer.

    Scalar convenience wrapper around :func:`encode_gate_ids` — see there
    for the encoding definition.
    """
    return int(encode_gate_ids(radar, sweep, azimuth, range_m))


# ============================================================================
# Generic LUT generation from DataTree
# ============================================================================


def _beamwidth_from_datatree(dt: xr.DataTree) -> float:
    """Antenna beamwidth [deg] from the DataTree, else the package default.

    Looks for the CfRadial/xradar attribute names on the root and on each sweep.
    """
    names = ("radar_beam_width_h", "beam_width_h", "beamwidth", "beamwidth_deg")
    candidates = [dt.attrs, *(dt[s].attrs for s in list_sweep_names(dt))]
    for attrs in candidates:
        for nm in names:
            if nm in attrs:
                try:
                    val = float(np.asarray(attrs[nm]).ravel()[0])
                except (TypeError, ValueError):
                    continue
                if 0.0 < val < 20.0:  # sanity: a real antenna beamwidth
                    return val
    return DEFAULT_BEAMWIDTH_DEG


#: Distance error above which a CRS is refused for a radar site, in percent.
#: Every legitimate case measured sits under 0.06%; every wrong one over 20%.
CRS_REFUSE_PCT: float = 1.0
#: Above this, the CRS is accepted but warned about.
CRS_WARN_PCT: float = 0.1


def suggest_crs(longitude: float, latitude: float) -> int:
    """EPSG code of the UTM zone covering a site — a safe default suggestion.

    UTM is not always the best choice (a national grid may fit better, and a
    long-range radar can reach past its zone), but it is defined worldwide and
    accurate to well under a percent near its central meridian, so it is a sound
    thing to put in an error message when the user has to pick one.
    """
    zone = int((float(longitude) + 180.0) // 6) + 1
    return (32600 if float(latitude) >= 0 else 32700) + zone


def crs_distance_error(crs, longitude: float, latitude: float, baseline_m: float = 100_000.0) -> float:
    """Worst relative distance error of ``crs`` at a site, in percent.

    Projects a ``baseline_m`` geodesic in eight directions from the site and
    compares each projected length with the true one.  This is measured rather
    than read from the CRS's declared ``area_of_use``, because that metadata is
    both incomplete (a custom proj4 definition may declare none) and
    insufficient (EPSG:3857 claims the whole world, then reports a 100 km
    baseline as 145 km in Switzerland).
    """
    import pyproj

    from raddb.aoi import _to_pyproj_crs

    geod = pyproj.Geod(ellps="WGS84")
    tf = pyproj.Transformer.from_crs(_to_pyproj_crs(4326), _to_pyproj_crs(crs), always_xy=True)
    x0, y0 = tf.transform(longitude, latitude)
    worst = 0.0
    for azimuth in range(0, 360, 45):
        lon2, lat2, _ = geod.fwd(longitude, latitude, azimuth, baseline_m)
        x1, y1 = tf.transform(lon2, lat2)
        worst = max(worst, abs(np.hypot(x1 - x0, y1 - y0) - baseline_m) / baseline_m)
    return worst * 100.0


def _crs_label(crs) -> str:
    """Human name for a CRS, for error messages.

    ``aoi._to_pyproj_crs`` resolves the common EPSG codes through DB-free proj4
    strings, which lose the name — so look the code up directly when we have one.
    """
    import pyproj

    if isinstance(crs, (int, np.integer)):
        try:
            named = pyproj.CRS.from_epsg(int(crs))
            return f"EPSG:{int(crs)} ({named.name})"
        except Exception:
            return f"EPSG:{int(crs)}"
    name = getattr(crs, "name", None)
    return str(name) if name and name != "unknown" else str(crs)


def validate_crs_for_site(crs, longitude: float, latitude: float, radar: str = "") -> float:
    """Check that ``crs`` can carry radar geometry at this site; return the error %.

    Raises when the CRS is geographic (degrees cannot express a crop radius) or
    when it distorts distance by more than :data:`CRS_REFUSE_PCT` — which is what
    using EPSG:2056 outside Switzerland does, to the tune of 20%.

    Parameters
    ----------
    crs : int or CRS-like
        The CRS the LUT will store projected coordinates in.
    longitude, latitude : float
        Radar site, WGS-84 degrees.
    radar : str, optional
        Used only to make the message concrete.

    Returns
    -------
    float
        Worst distance error at this site, in percent.
    """
    import warnings as _warnings

    import pyproj

    from raddb.aoi import _to_pyproj_crs

    who = f"radar {radar} " if radar else ""
    resolved = _to_pyproj_crs(crs)
    label = _crs_label(crs)
    if not resolved.is_projected:
        raise ValueError(
            f"{label} is a geographic CRS — its units are "
            f"degrees, so it cannot express gate geometry, a crop radius or a "
            f"cross-section distance. Pass a projected CRS; for {who}at "
            f"({longitude:.4f}, {latitude:.4f}) try "
            f"EPSG:{suggest_crs(longitude, latitude)}.",
        )

    err = crs_distance_error(resolved, longitude, latitude)
    if err > CRS_REFUSE_PCT:
        area = getattr(resolved.area_of_use, "name", None)
        if area is None and isinstance(crs, (int, np.integer)):
            import pyproj

            try:
                area = getattr(pyproj.CRS.from_epsg(int(crs)).area_of_use, "name", None)
            except Exception:
                area = None
        where = f", valid for {area}" if area else ""
        raise ValueError(
            f"{label}{where} distorts distance by {err:.1f}% at "
            f"{who}({longitude:.4f}, {latitude:.4f}) — gate geometry, crops and "
            f"cross-sections would all be wrong by that much. Suggested for this "
            f"site: EPSG:{suggest_crs(longitude, latitude)}.",
        )
    if err > CRS_WARN_PCT:
        _warnings.warn(
            f"{label} distorts distance by {err:.2f}% at " f"{who}({longitude:.4f}, {latitude:.4f}).",
            stacklevel=3,
        )
    return err


def generate_lut_from_datatree(
    dt: xr.DataTree,
    radar: str,
    output_base_path: str,
    ke: float = 4.0 / 3.0,
    network: str = "",
    projection_epsg: int | None = None,
    projection_crs=None,
    beamwidth_deg: float | None = None,
) -> str:
    """Generate a LUT from an xarray DataTree.

    Writes the **five** files that make up a complete LUT directory
    (see :data:`LUT_FILES`): the gate-centroid LUT, the horizontal-face and
    vertical-face node lattices, the 3-D corner lattice, and the info YAML.

    This is the **generic** LUT generator — it works with any DataTree
    that has the standard xradar coordinate layout (azimuth, range,
    elevation per sweep).

    Parameters
    ----------
    dt : xarray.DataTree
        DataTree with ``sweep_N`` groups, each containing ``azimuth``,
        ``range``, and ``elevation`` coordinates.
    radar : str
        Radar identifier, e.g. ``"A"`` or ``"KTLX"``.
    output_base_path : str
        Base directory for LUT storage.
    ke : float
        Effective Earth radius scale factor (default 4/3, the standard
        atmosphere model — matches every other ``ke`` in the package).
    network : str, optional
        Network identifier stored in radar info YAML.
    projection_epsg : int, optional
        EPSG code for an additional projected coordinate system to include
        in the LUT (e.g. ``2056`` for CH1903+ / LV95).  Adds
        ``x_{epsg}`` / ``y_{epsg}`` columns via :func:`add_lut_projection`.
    projection_crs : pyproj.CRS or CRS-coercible, optional
        Alternative to ``projection_epsg``.
    beamwidth_deg : float, optional
        Antenna 3 dB beamwidth in degrees, defining the gate's angular extent.
        Read from the DataTree's ``radar_beam_width_h`` / ``beamwidth`` attribute
        when present, else :data:`DEFAULT_BEAMWIDTH_DEG` (1.0).

    Returns
    -------
    str
        Path to the saved LUT parquet file.
    """
    lut_dir = Path(output_base_path) / radar / "LUT"
    lut_path = lut_dir / f"{radar}_LUT.parquet"

    if beamwidth_deg is None:
        beamwidth_deg = _beamwidth_from_datatree(dt)

    # Skip only when *all five* files are present. An archive written before the
    # geometry lattices existed has the LUT parquet + info YAML but not the three
    # plane files, and must still be able to fill them in.
    if all((lut_dir / tmpl.format(radar=radar)).exists() for tmpl in LUT_FILES.values()):
        logger.info(
            "All %d LUT files already exist at %s -- skipping generation.",
            len(LUT_FILES),
            lut_dir,
        )
        return str(lut_path)

    sweep_names = list_sweep_names(dt)
    if not sweep_names:
        raise ValueError("DataTree has no sweep groups (sweep_N).")

    lut_dfs = []
    sweep_meta = {}
    sweep_grids: dict[int, dict] = {}
    radar_lat, radar_lon, radar_alt = None, None, None

    for sweep_name in sweep_names:
        sweep_idx = int(sweep_name.split("_")[-1])
        ds = dt[sweep_name].to_dataset()

        measured_az = np.asarray(ds["azimuth"].values, dtype=np.float64)
        ranges = ds["range"].to_numpy()
        elevations = ds["elevation"].to_numpy()

        # The LUT is the radar's *nominal* scan geometry, not a snapshot of this
        # one volume's antenna readings.  Every later volume snaps onto this same
        # grid, so a gate keeps its gate_id for the life of the archive; keying
        # off the measured angles instead loses 6% (a 360-ray C-band network) to
        # 35% (WSR-88D) of the gates of every volume after the first.
        az_grid = nominal_azimuth_grid(measured_az)
        snapped, snap_dist = snap_azimuths_to_grid(measured_az, az_grid)
        if np.unique(snapped).size != measured_az.size:
            raise ValueError(
                f"radar {radar!r} sweep {sweep_idx}: the {measured_az.size} rays do "
                f"not sit one-per-point on a regular {360 / measured_az.size:.4f}° "
                f"grid (two rays snap together), so this sweep has no nominal "
                f"azimuth grid.",
            )

        elevation_angle = float(ds.coords.get("elevation_angle", np.mean(elevations)))

        # The LUT covers the whole rotation, not only the rays this volume
        # happened to record: a volume that drops a ray still defines the grid
        # point, and a later volume that does record it must find a row to join.
        # Empty for a complete sweep, so nothing changes for one.
        absent = np.setdiff1d(az_grid, snapped)
        azimuths = np.concatenate([snapped, absent]).astype(np.float64) / AZIMUTH_SCALE
        elevations = np.concatenate(
            [np.asarray(elevations, dtype=np.float64), np.full(absent.size, elevation_angle)],
        )
        if absent.size:
            logger.info(
                "sweep %d: %d of %d rays missing from this volume; their grid " "points are still written to the LUT.",
                sweep_idx,
                absent.size,
                az_grid.size,
            )
        n_az, n_rng = len(azimuths), len(ranges)
        logger.debug(
            "sweep %d: %d rays snapped to the nominal grid, max move %.3f°.",
            sweep_idx,
            n_az,
            snap_dist.max() / AZIMUTH_SCALE,
        )

        # Compute Cartesian coordinates
        x_raw, y_raw, z_raw = antenna_vectors_to_cartesian(
            ranges,
            azimuths,
            elevations,
            ke=ke,
        )

        # Extract site coordinates
        site_lat = float(ds.coords.get("latitude", 0.0))
        site_lon = float(ds.coords.get("longitude", 0.0))
        site_alt = float(ds.coords.get("altitude", 0.0))

        if radar_lat is None:
            radar_lat = site_lat
            radar_lon = site_lon
            radar_alt = site_alt

        # Compute per-gate geographic coordinates from Cartesian offsets
        gate_lat, gate_lon, gate_alt = cartesian_to_geographic(
            x_raw,
            y_raw,
            z_raw,
            radar_lat,
            radar_lon,
            radar_alt,
        )

        gate_az = np.repeat(azimuths, n_rng)
        gate_rng = np.tile(ranges, n_az)
        gate_ids = encode_gate_ids(radar, sweep_idx, gate_az, gate_rng)

        lut_dfs.append(
            pl.DataFrame(
                {
                    "gate_id": gate_ids,
                    "sweep": np.full(n_az * n_rng, sweep_idx, dtype=np.int32),
                    "azimuth": gate_az,
                    "range": gate_rng,
                    "elevation_angle": np.full(n_az * n_rng, elevation_angle),
                    "latitude": gate_lat.ravel(),
                    "longitude": gate_lon.ravel(),
                    "altitude": gate_alt.ravel(),
                    "x": x_raw.ravel(),
                    "y": y_raw.ravel(),
                    "z": z_raw.ravel(),
                },
            ),
        )

        rng_res = float(np.median(np.diff(np.sort(ranges).astype(np.float64)))) if n_rng > 1 else float("nan")
        sweep_meta[sweep_idx] = {
            "n_azimuths": n_az,
            "n_ranges": n_rng,
            "n_gates": int(n_az * n_rng),
            "elevation": round(elevation_angle, 2),
            "range_resolution": round(rng_res, 3),
            "range_start": round(float(np.min(ranges)), 3),
        }
        # Grids needed later for the corner/plane lattices — keep them so the
        # lattices are built from the same arrays the centroids came from,
        # rather than re-derived from the written parquet.
        sweep_grids[sweep_idx] = {
            "ranges": np.asarray(ranges, dtype=np.float64),
            "azimuths": np.asarray(azimuths, dtype=np.float64),
            "elevations": np.asarray(elevations, dtype=np.float64),
        }

    df_lut = pl.concat(lut_dfs, how="vertical")
    logger.info(
        "LUT built: %d total gates, %d sweeps.",
        len(df_lut),
        len(sweep_meta),
    )

    # A CRS is required, and must hold at this radar's site.  There is no
    # default: a silently wrong projection corrupts every crop and cross-section
    # downstream, and the archive is the only place to catch it.
    if projection_epsg is None and projection_crs is None:
        raise ValueError(
            f"archiving radar {radar!r} requires a CRS: the LUT stores projected "
            f"gate coordinates, and crops and cross-sections are computed in them. "
            f"There is no default because a wrong one is silently wrong. Radar "
            f"{radar!r} is at ({radar_lon:.4f}, {radar_lat:.4f}); suggested: "
            f"RadDB(crs={suggest_crs(radar_lon, radar_lat)})  "
            f"# UTM zone {int((radar_lon + 180) // 6) + 1}",
        )
    validate_crs_for_site(
        projection_epsg if projection_epsg is not None else projection_crs,
        radar_lon,
        radar_lat,
        radar,
    )
    df_lut = add_lut_projection(df_lut, epsg=projection_epsg, crs=projection_crs)

    crs_info = None
    proj_cols = _projection_column_names(df_lut)
    if proj_cols:
        epsg = None
        if projection_epsg is not None:
            epsg = int(projection_epsg)
        else:
            # suffix of x_<suffix> is the EPSG code when pyproj could detect one
            suffix = proj_cols[0].split("_", 1)[1]
            epsg = int(suffix) if suffix.isdigit() else None
        crs_info = {"epsg": epsg, "columns": proj_cols}

    radar_info = {
        "radar": radar,
        "network": network,
        "latitude": radar_lat,
        "longitude": radar_lon,
        "altitude": radar_alt,
        "crs": crs_info,
        # Recorded for reproducibility: archives built before the ke 1.25 -> 4/3
        # fix carry incompatible geometry, so the file must say which model
        # produced it.
        "ke": float(ke),
        "beamwidth_deg": float(beamwidth_deg),
        "n_sweeps": len(sweep_meta),
        "n_gates": int(len(df_lut)),
        "sweeps": sweep_meta,
    }

    # ---- the three geometry lattices -------------------------------------
    corners_by_sweep: dict[int, dict] = {}
    for sweep_idx, g in sweep_grids.items():
        corners_by_sweep[sweep_idx] = compute_sweep_corners(
            ranges=g["ranges"],
            azimuths=g["azimuths"],
            elevations=g["elevations"],
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            radar_alt=radar_alt,
            ke=ke,
            beamwidth_deg=beamwidth_deg,
        )
    planes = build_gate_planes(
        corners_by_sweep,
        radar_alt=radar_alt,
        projection_epsg=projection_epsg,
        projection_crs=projection_crs,
    )

    return _save_lut_outputs(lut_dir, radar, df_lut, radar_info, planes)


# ============================================================================
# LUT storage helpers
# ============================================================================


def _save_lut_outputs(lut_dir, radar, df_lut, radar_info, planes=None):
    """Save the five LUT files to disk (see :data:`LUT_FILES`).

    ``df_lut`` may be a polars frame (the native format since the LUT layer is
    polars) or a pandas frame (accepted so external callers keep working).

    ``planes`` is the :func:`build_gate_planes` output.  When ``None`` only the
    LUT parquet and the info YAML are written (legacy two-file behavior).

    The idempotence gate checks **all** expected files: an archive written before
    the geometry lattices existed still has its LUT parquet and info YAML, so the
    three new files get filled in without rebuilding the centroids.
    """
    lut_dir = Path(lut_dir)
    lut_dir.mkdir(parents=True, exist_ok=True)
    lut_path = lut_dir / LUT_FILES["lut"].format(radar=radar)
    info_path = lut_dir / LUT_FILES["info"].format(radar=radar)

    plane_paths = {kind: lut_dir / LUT_FILES[kind].format(radar=radar) for kind in ("h_plane", "v_plane", "corners")}
    expected = [lut_path, info_path]
    if planes is not None:
        expected += list(plane_paths.values())

    if all(p.exists() for p in expected):
        logger.info(
            "All %d LUT files already exist at %s -- skipping creation.",
            len(expected),
            lut_dir,
        )
        return str(lut_path)

    if planes is not None:
        for kind, path in plane_paths.items():
            if path.exists():
                continue
            planes[kind].write_parquet(path)
            logger.info("%s lattice saved -> %s", kind, path)

    if lut_path.exists() and info_path.exists():
        # Only the geometry lattices were missing; centroids stay as they are.
        return str(lut_path)

    if isinstance(df_lut, pl.DataFrame):
        df_lut.write_parquet(lut_path)
    else:
        df_lut.to_parquet(lut_path, index=False, engine="pyarrow")
    logger.info("LUT saved -> %s", lut_path)

    with open(info_path, "w") as f:
        yaml.dump(radar_info, f, default_flow_style=False, sort_keys=False)
    logger.info("Radar info saved -> %s", info_path)
    return str(lut_path)


# ============================================================================
# Loaders
# ============================================================================

#: Elevation levels of a gate, as offsets in units of the half beamwidth.
#: ``-1`` = bottom of the beam, ``0`` = beam center, ``+1`` = top.
#: Follows the reference prototype's ``En`` / ``Eo`` / ``Ep`` face naming.
EL_LEVELS: tuple[int, ...] = (-1, 0, 1)


def compute_sweep_corners(
    ranges: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    radar_alt: float,
    ke: float = 4.0 / 3.0,
    beamwidth_deg: float | None = None,
) -> dict:
    """Compute per-sweep gate corner arrays for pcolormesh rendering.

    Uses PyART's edge-interpolation (complex-plane for azimuth wrap-around)
    plus the standard 4/3 Earth beam propagation.

    Parameters
    ----------
    beamwidth_deg : float, optional
        When given, the beam's **vertical extent** is resolved as well: the edge
        mesh is computed at three elevation levels
        (``elevation - beamwidth/2``, ``elevation``, ``elevation + beamwidth/2``)
        and returned under the extra ``levels`` key.  Without it only the
        center-elevation mesh is produced, which has *no* vertical extent — that
        is the legacy behavior and cannot describe a gate's 8 corners.

    Returns
    -------
    dict
        Always contains ``x_edges``, ``y_edges``, ``z_edges``, ``lon_edges``,
        ``lat_edges``, each shape ``(n_az+1, n_range+1)`` — the beam-center mesh,
        kept for backwards compatibility.

        When ``beamwidth_deg`` is given, also contains
        ``levels``: ``{-1: {...}, 0: {...}, 1: {...}}`` with the same five keys
        per level, where the integer is the offset in half-beamwidths
        (see :data:`EL_LEVELS`).
    """
    elevations = np.atleast_1d(np.asarray(elevations, dtype=np.float64))

    def _mesh(el: np.ndarray) -> dict:
        x_e, y_e, z_e = antenna_vectors_to_cartesian(
            ranges,
            azimuths,
            el,
            ke=ke,
            edges=True,
        )
        lat_e, lon_e, _ = cartesian_to_geographic(
            x_e,
            y_e,
            z_e,
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            radar_alt=radar_alt,
        )
        # float64 throughout: these edges are the gate polygon vertices, and gate
        # position precision is a hard requirement (float32 costs ~20 cm, and the
        # error does not shrink with range).
        return {
            "x_edges": x_e.astype(np.float64),
            "y_edges": y_e.astype(np.float64),
            "z_edges": z_e.astype(np.float64),
            "lon_edges": lon_e.astype(np.float64),
            "lat_edges": lat_e.astype(np.float64),
        }

    center = _mesh(elevations)
    if beamwidth_deg is None:
        return center

    half_bw = float(beamwidth_deg) / 2.0
    # dict(center) for level 0 so the `levels` key added below cannot make the
    # structure self-referential.
    levels = {lvl: dict(center) if lvl == 0 else _mesh(elevations + lvl * half_bw) for lvl in EL_LEVELS}
    return {**center, "levels": levels}


# ============================================================================
# Gate plane / corner node lattices  (the h_plane / v_plane / corners files)
# ============================================================================

#: Node-index offsets of a gate's 4 corners within the edge lattice, in ring
#: order. Matches the ring built by :func:`gate_polygons_geoarrow`.
GATE_RING_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 1), (1, 0))


def _project_nodes(lon: np.ndarray, lat: np.ndarray, epsg: int | None, crs=None):
    """Projected easting/northing for lattice nodes, or ``(None, None, None)``."""
    if epsg is None and crs is None:
        return None, None, None
    import pyproj

    if epsg is not None:
        target = pyproj.CRS.from_epsg(epsg)
        suffix = str(epsg)
    else:
        target = crs if isinstance(crs, pyproj.CRS) else pyproj.CRS(crs)
        detected = target.to_epsg()
        suffix = str(detected) if detected is not None else "custom"
    wgs84 = pyproj.CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
    tf = pyproj.Transformer.from_crs(wgs84, target, always_xy=True)
    px, py = tf.transform(lon, lat)
    return np.asarray(px), np.asarray(py), suffix


def build_gate_planes(
    corners_by_sweep: dict[int, dict],
    radar_alt: float,
    projection_epsg: int | None = None,
    projection_crs=None,
) -> dict[str, pl.DataFrame]:
    """Build the h_plane / v_plane / corners **node lattices** as polars frames.

    Input is the output of :func:`compute_sweep_corners` called with
    ``beamwidth_deg`` (so each sweep carries a ``levels`` dict).

    The lattices store *nodes*, not per-gate corners: neighboring gates share
    corner nodes, so a lattice is ~4x smaller for the horizontal face and ~8x
    smaller for the 3-D corners than materializing every gate's corners, and is
    exactly equivalent.  A gate's corners are recovered by indexing
    ``(az_idx + i, rng_idx + j)`` over :data:`GATE_RING_OFFSETS` — see
    :meth:`raddb.RadDB.get_h_plane` / :meth:`raddb.RadDB.get_corners`.

    Returns
    -------
    dict
        ``{"h_plane": pl.DataFrame, "v_plane": pl.DataFrame, "corners": pl.DataFrame}``

        * ``h_plane`` — beam-center level only: ``sweep, az_idx, rng_idx, x, y,
          lon, lat`` (+ ``x_<epsg>, y_<epsg>``).
        * ``v_plane`` — bottom/top levels in the RHI plane: ``sweep, el_level,
          az_idx, rng_idx, d, z_asl, z_rel``.
        * ``corners`` — bottom/top levels in 3-D: ``sweep, el_level, az_idx,
          rng_idx, x, y, z_rel, z_asl, lon, lat``.
    """
    h_parts, v_parts, c_parts = [], [], []

    for sweep_num in sorted(corners_by_sweep):
        entry = corners_by_sweep[sweep_num]
        levels = entry.get("levels")
        if levels is None:
            raise ValueError(
                f"sweep {sweep_num}: compute_sweep_corners must be called with "
                "beamwidth_deg so the vertical levels are available.",
            )

        for lvl in sorted(levels):
            m = levels[lvl]
            xe, ye, ze = m["x_edges"], m["y_edges"], m["z_edges"]
            lone, late = m["lon_edges"], m["lat_edges"]
            n_az_n, n_rng_n = xe.shape  # (n_az+1, n_rng+1) node counts

            az_idx = np.repeat(np.arange(n_az_n, dtype=np.int16), n_rng_n)
            rng_idx = np.tile(np.arange(n_rng_n, dtype=np.int16), n_az_n)
            xf, yf, zf = xe.ravel(), ye.ravel(), ze.ravel()
            lonf, latf = lone.ravel(), late.ravel()
            n = xf.size
            sweep_col = np.full(n, sweep_num, dtype=np.int16)

            if lvl == 0:
                # --- horizontal face (PPI) -----------------------------------
                # lon/lat are deliberately NOT stored: cartesian_to_geographic is
                # a closed form of (x, y) + the site, so storing them would cost
                # 8 B/node for zero information. The accessors derive them.
                cols = {
                    "sweep": sweep_col,
                    "az_idx": az_idx,
                    "rng_idx": rng_idx,
                    "x": xf.astype(np.float32),
                    "y": yf.astype(np.float32),
                }
                px, py, suffix = _project_nodes(lonf, latf, projection_epsg, projection_crs)
                if suffix is not None:
                    cols[f"x_{suffix}"] = px.astype(np.float32)
                    cols[f"y_{suffix}"] = py.astype(np.float32)
                h_parts.append(pl.DataFrame(cols))
                continue

            # --- bottom / top levels: v_plane + 3-D corners ------------------
            lvl_col = np.full(n, lvl, dtype=np.int8)
            z_asl = (zf + radar_alt).astype(np.float32)
            v_parts.append(
                pl.DataFrame(
                    {
                        "sweep": sweep_col,
                        "el_level": lvl_col,
                        "az_idx": az_idx,
                        "rng_idx": rng_idx,
                        # ground distance from the radar: the beam's arc length, which is
                        # exactly hypot(x, y) in this equidistant frame.
                        "d": np.hypot(xf, yf).astype(np.float32),
                        "z_asl": z_asl,
                        "z_rel": zf.astype(np.float32),
                    },
                ),
            )
            # z_asl and lon/lat omitted here for the same reason as in h_plane:
            # z_asl = z_rel + site altitude (a constant), and lon/lat are a
            # closed form of (x, y). Both are derived by the accessors.
            c_parts.append(
                pl.DataFrame(
                    {
                        "sweep": sweep_col,
                        "el_level": lvl_col,
                        "az_idx": az_idx,
                        "rng_idx": rng_idx,
                        "x": xf.astype(np.float32),
                        "y": yf.astype(np.float32),
                        "z_rel": zf.astype(np.float32),
                    },
                ),
            )

    return {
        "h_plane": pl.concat(h_parts, how="vertical_relaxed"),
        "v_plane": pl.concat(v_parts, how="vertical_relaxed"),
        "corners": pl.concat(c_parts, how="vertical_relaxed"),
    }


def load_plane_nodes(
    radar: str,
    lut_base_path: str | Path,
    kind: str,
    sweep: int | None = None,
) -> pl.DataFrame:
    """Load one of the node-lattice files (``h_plane`` / ``v_plane`` / ``corners``).

    ``sweep`` pushes a row filter into the parquet scan.

    A pre-geometry archive (centroid LUT only) is backfilled on first read via
    :func:`ensure_gate_planes` rather than raising.
    """
    path = lut_file_path(radar, kind, lut_base_path)
    if not path.exists():
        ensure_gate_planes(radar, lut_base_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{kind} lattice not found at {path}, and it could not be rebuilt from "
            f"{radar}_LUT.parquet. Regenerate the LUT (generate_lut_from_datatree).",
        )
    lf = pl.scan_parquet(path)
    if sweep is not None:
        lf = lf.filter(pl.col("sweep") == int(sweep))
    return lf.collect()


def _node_grids(
    nodes: pl.DataFrame,
    value_cols: list[str],
    level: int | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Reshape flat lattice rows into ``{sweep: {col: (n_az+1, n_rng+1) array}}``."""
    if level is not None and "el_level" in nodes.columns:
        nodes = nodes.filter(pl.col("el_level") == level)
    out: dict[int, dict[str, np.ndarray]] = {}
    for (sweep_num,), sub in nodes.group_by(["sweep"], maintain_order=True):
        sub = sub.sort(["az_idx", "rng_idx"])
        n_az_n = int(sub["az_idx"].max()) + 1
        n_rng_n = int(sub["rng_idx"].max()) + 1
        out[int(sweep_num)] = {c: sub[c].to_numpy().reshape(n_az_n, n_rng_n) for c in value_cols}
    return out


def gate_corner_table(
    radar: str,
    lut_base_path: str | Path,
    kind: str = "h_plane",
    sweep: int | None = None,
) -> pl.DataFrame:
    """Materialize per-gate corners from a node lattice, keyed by ``gate_id``.

    This is the "hybrid" read side: the archive stores compact node lattices, and
    this expands them to explicit per-gate corners on demand.

    Corner counts and column names by ``kind``:

    * ``h_plane`` — **4** corners, ring order (see :data:`GATE_RING_OFFSETS`):
      ``x_1..x_4``, ``y_1..y_4`` (+ ``x_<epsg>_1..4`` when the lattice is
      projected).
    * ``v_plane`` — **4** corners in the RHI plane: ``d_1..d_4``,
      ``z_asl_1..z_asl_4``, ``z_rel_1..z_rel_4``, ordered
      (near-bottom, far-bottom, far-top, near-top).
    * ``corners`` — **8** corners in 3-D: ``x_1..x_8``, ``y_1..y_8``,
      ``z_rel_1..z_rel_8``.  **1-4 are the near face** (towards the radar),
      **5-8 the far face**; within each face the order is
      (az-, el-), (az+, el-), (az+, el+), (az-, el+).  Because the angular extent
      grows with range, the far face is strictly larger than the near face.

    The join onto ``gate_id`` reuses :func:`_gate_grid_index`, so it inherits the
    same ``searchsorted`` node indexing the lattices were written with.
    """
    if kind not in ("h_plane", "v_plane", "corners"):
        raise ValueError(
            f"kind must be 'h_plane', 'v_plane' or 'corners'; got {kind!r}.",
        )

    idx = _gate_grid_index(radar, lut_base_path)
    if sweep is not None:
        idx = idx.filter(pl.col("sweep") == int(sweep))
    if idx.is_empty():
        return pl.DataFrame(schema={"gate_id": pl.Int64, "sweep": pl.Int32})

    nodes = load_plane_nodes(radar, lut_base_path, kind, sweep=sweep)

    if kind == "h_plane":
        value_cols = [c for c in nodes.columns if c not in ("sweep", "az_idx", "rng_idx", "el_level")]
        grids = {0: _node_grids(nodes, value_cols)}
        # 4 corners, one level, ring order
        picks = [(0, i, j) for (i, j) in GATE_RING_OFFSETS]
    elif kind == "v_plane":
        value_cols = ["d", "z_asl", "z_rel"]
        grids = {lvl: _node_grids(nodes, value_cols, level=lvl) for lvl in (-1, 1)}
        # near-bottom, far-bottom, far-top, near-top
        picks = [(-1, 0, 0), (-1, 0, 1), (1, 0, 1), (1, 0, 0)]
    else:
        value_cols = ["x", "y", "z_rel"]
        grids = {lvl: _node_grids(nodes, value_cols, level=lvl) for lvl in (-1, 1)}
        # near face (rng+0) then far face (rng+1); within a face:
        # (az-, el-), (az+, el-), (az+, el+), (az-, el+)
        picks = [
            (-1, 0, 0),
            (-1, 1, 0),
            (1, 1, 0),
            (1, 0, 0),  # near face
            (-1, 0, 1),
            (-1, 1, 1),
            (1, 1, 1),
            (1, 0, 1),  # far face
        ]

    n = idx.height
    sweeps = idx["sweep"].to_numpy()
    az_i = idx["az_idx"].to_numpy().astype(np.int64)
    rng_i = idx["rng_idx"].to_numpy().astype(np.int64)

    out: dict[str, np.ndarray] = {}
    for k, (lvl, di, dj) in enumerate(picks, start=1):
        for col in value_cols:
            out[f"{col}_{k}"] = np.full(n, np.nan, dtype=np.float64)
        for sw in np.unique(sweeps):
            g = grids[lvl].get(int(sw))
            if g is None:
                logger.warning("sweep %d missing from the %s lattice.", sw, kind)
                continue
            rows = np.flatnonzero(sweeps == sw)
            for col in value_cols:
                arr = g[col]
                out[f"{col}_{k}"][rows] = arr[az_i[rows] + di, rng_i[rows] + dj]

    return pl.DataFrame(
        {
            "gate_id": idx["gate_id"],
            "sweep": idx["sweep"],
            **{c: v.astype(np.float32) for c, v in out.items()},
        },
    )


def ensure_gate_planes(
    radar: str,
    lut_base_path: str | Path,
    ke: float = 4.0 / 3.0,
    beamwidth_deg: float | None = None,
) -> bool:
    """Backfill the three geometry lattices from the centroid LUT if they are missing.

    Archives written before the geometry files existed hold only
    ``{radar}_LUT.parquet`` + ``{radar}_info.yaml``.  Everything the lattices need
    is derivable from those two, so rather than failing, rebuild them in place —
    the centroid LUT is never rewritten (see :func:`_save_lut_outputs`).

    The projection is taken from the ``crs`` block recorded in the info YAML, so a
    backfilled ``h_plane`` carries the same ``x_<epsg>`` / ``y_<epsg>`` columns a
    freshly generated one would.

    Returns
    -------
    bool
        ``True`` if files were written, ``False`` if all three already existed.
    """
    missing = [
        kind for kind in ("h_plane", "v_plane", "corners") if not lut_file_path(radar, kind, lut_base_path).exists()
    ]
    if not missing:
        return False

    info = load_radar_info(radar, lut_base_path)
    if beamwidth_deg is None:
        beamwidth_deg = float(info.get("beamwidth_deg") or DEFAULT_BEAMWIDTH_DEG)
    logger.info(
        "radar %s: geometry lattices %s missing -- rebuilding from the centroid LUT.",
        radar,
        missing,
    )

    corners_by_sweep: dict[int, dict] = {}
    for sweep_num, g in _sweep_grids_from_lut(radar, lut_base_path).items():
        corners_by_sweep[int(sweep_num)] = compute_sweep_corners(
            ranges=g["ranges"],
            azimuths=g["azimuths"],
            elevations=g["elevations"],
            radar_lat=info["latitude"],
            radar_lon=info["longitude"],
            radar_alt=info["altitude"],
            ke=ke,
            beamwidth_deg=beamwidth_deg,
        )

    # Pre-geometry info YAMLs have no ``crs`` block, but the centroid LUT still
    # carries its x_<epsg>/y_<epsg> columns — recover the EPSG from those so a
    # backfilled h_plane is projected exactly like a freshly generated one.
    epsg = (info.get("crs") or {}).get("epsg")
    if epsg is None:
        lut_cols = (
            pl.scan_parquet(
                lut_file_path(radar, "lut", lut_base_path),
            )
            .collect_schema()
            .names()
        )
        for name in _projection_column_names(pl.DataFrame(schema={c: pl.Float64 for c in lut_cols})):
            suffix = name.split("_", 1)[1]
            if name.startswith("x_") and suffix.isdigit():
                epsg = int(suffix)
                logger.info("radar %s: EPSG:%d recovered from the LUT columns.", radar, epsg)
                break

    planes = build_gate_planes(
        corners_by_sweep,
        radar_alt=float(info["altitude"]),
        projection_epsg=epsg,
    )
    lut_dir = Path(lut_base_path) / radar / "LUT"
    for kind in ("h_plane", "v_plane", "corners"):
        path = lut_dir / LUT_FILES[kind].format(radar=radar)
        if not path.exists():
            planes[kind].write_parquet(path)
            logger.info("%s lattice backfilled -> %s", kind, path)
    return True


def cappi_chords(
    radar: str,
    lut_base_path: str | Path,
    altitude: float,
    height: str = "asl",
) -> pl.DataFrame:
    """Where a constant-altitude surface cuts each range bin — the CAPPI slice.

    A CAPPI is a horizontal slice through the volume, so the question it asks of
    the geometry is *vertical*: which gates does the plane ``z = altitude`` pass
    through, and where along the beam does it enter and leave each one.  That is
    answered by the ``v_plane`` lattice, whose ``(d, z)`` quads are the gates'
    vertical faces.  ``h_plane`` has no altitude column at all, so it cannot
    answer it; its role comes afterwards, turning each chord into an ``(x, y)``
    polygon (see :func:`raddb.viz.plot.plot_cappi`).

    The whole computation is **per sweep and range bin, not per gate**: ``d`` and
    ``z`` do not depend on azimuth (elevation is constant across a sweep), which
    is why ``v_plane`` compresses to a fraction of ``h_plane``.  So a few
    thousand rows here serve every azimuth of the volume.

    Parameters
    ----------
    radar : str
    lut_base_path : str or Path
        Archive root (the directory holding ``{radar}/LUT/``).
    altitude : float
        Slice altitude in meters.
    height : {"asl", "rel"}
        Whether ``altitude`` is above sea level (default) or above the radar.

    Returns
    -------
    pl.DataFrame
        One row per ``(sweep, rng_idx)`` the surface intersects:

        * ``d_near`` / ``d_far`` — ground distance [m] where the slice enters and
          leaves the gate.  Interior bins of a band clip to the bin edges; only
          the first and last bin of a band are genuinely trimmed.
        * ``z_center`` — the gate's mid-face altitude, in the same reference as
          ``altitude``.  Used to resolve overlapping sweeps by taking the beam
          whose center is closest to the slice.
        * ``dz_center`` — ``abs(z_center - altitude)``.

        Empty when no beam reaches that altitude.

    Notes
    -----
    Beam thickness grows with range (~1.7 km at 100 km for a 1° beam) and far
    exceeds the height gained across one range bin, so a sweep typically
    intersects a **wide contiguous band** of bins rather than a thin ring, and
    neighboring sweeps overlap heavily.
    """
    if height not in ("asl", "rel"):
        raise ValueError(f"height must be 'asl' or 'rel'; got {height!r}.")
    z_col = "z_asl" if height == "asl" else "z_rel"
    z0 = float(altitude)

    nodes = load_plane_nodes(radar, lut_base_path, "v_plane")
    # Azimuth-independent: one ray of nodes describes every azimuth.
    nodes = nodes.filter(pl.col("az_idx") == pl.col("az_idx").min())

    bottom = _node_grids(nodes, ["d", z_col], level=-1)
    top = _node_grids(nodes, ["d", z_col], level=1)

    parts = []
    for sweep_num in sorted(bottom):
        if sweep_num not in top:
            logger.warning("sweep %d missing the top level of the v_plane lattice.", sweep_num)
            continue
        # Node rows are (n_az+1, n_rng+1); one azimuth was kept, so row 0 is it.
        d_b, z_b = bottom[sweep_num]["d"][0], bottom[sweep_num][z_col][0]
        d_t, z_t = top[sweep_num]["d"][0], top[sweep_num][z_col][0]
        if d_b.size < 2:
            continue

        # Vertical face of bin j, clockwise:
        # near-bottom, far-bottom, far-top, near-top.
        ring_d = np.stack([d_b[:-1], d_b[1:], d_t[1:], d_t[:-1]], axis=1)
        ring_z = np.stack([z_b[:-1], z_b[1:], z_t[1:], z_t[:-1]], axis=1)

        # Clip the quad against z == z0, edge by edge.  Convex quad -> at most
        # one entry and one exit, so min/max of the crossings is the chord.
        za, zb = ring_z, np.roll(ring_z, -1, axis=1)
        da, db = ring_d, np.roll(ring_d, -1, axis=1)
        sa, sb = za - z0, zb - z0
        crosses = ((sa <= 0) & (sb >= 0)) | ((sa >= 0) & (sb <= 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(zb != za, (z0 - za) / (zb - za), 0.0)
        d_cross = np.where(crosses, da + np.clip(t, 0.0, 1.0) * (db - da), np.nan)

        hit = np.isfinite(d_cross).any(axis=1)
        if not hit.any():
            continue
        # Mask before reducing: non-intersecting rows are all-NaN by design.
        d_hit = d_cross[hit]
        d_near = np.nanmin(d_hit, axis=1)
        d_far = np.nanmax(d_hit, axis=1)
        z_center = ring_z.mean(axis=1)[hit]

        parts.append(
            pl.DataFrame(
                {
                    "sweep": np.full(hit.sum(), sweep_num, dtype=np.int32),
                    "rng_idx": np.flatnonzero(hit).astype(np.int32),
                    "d_near": d_near.astype(np.float32),
                    "d_far": d_far.astype(np.float32),
                    "z_center": z_center.astype(np.float32),
                    "dz_center": np.abs(z_center - z0).astype(np.float32),
                },
            ),
        )

    if not parts:
        return pl.DataFrame(
            schema={
                "sweep": pl.Int32,
                "rng_idx": pl.Int32,
                "d_near": pl.Float32,
                "d_far": pl.Float32,
                "z_center": pl.Float32,
                "dz_center": pl.Float32,
            },
        )
    return pl.concat(parts, how="vertical")


def save_sweep_corners(
    corners_by_sweep: dict[int, dict],
    corners_path: str | Path,
) -> str:
    """Save per-sweep corner arrays to a single ``.npz`` file.

    Parameters
    ----------
    corners_by_sweep : dict {sweep_number: {"x_edges", "y_edges", ...}}
    corners_path : str or Path
        Target ``.npz`` file path.
    """
    flat = {}
    for sw, d in corners_by_sweep.items():
        for k, v in d.items():
            flat[f"sweep_{int(sw)}_{k}"] = v
    corners_path = Path(corners_path)
    corners_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(corners_path, **flat)
    logger.info("Sweep corners saved -> %s", corners_path)
    return str(corners_path)


def _sweep_grids_from_lut(
    radar: str,
    lut_base_path: str | Path,
) -> dict[int, dict]:
    """Per-sweep ``(azimuths, ranges, elevation)`` grids read from the LUT parquet.

    Only the four columns needed are pulled, with the projection pushed into the
    parquet reader — reading the whole 13-column LUT here would be ~8x the I/O.

    The grids are ``np.sort(unique(...))``, matching :func:`_gate_grid_index`'s
    ``searchsorted`` and the pandas ``to_xarray()`` MultiIndex order used by
    ``io_core.reconstruct_sweep_dataset``, so node indices are consistent
    everywhere.
    """
    lut_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_LUT.parquet"
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT not found at {lut_path}.")
    lut = (
        pl.scan_parquet(lut_path)
        .select(
            ["sweep", "azimuth", "range", "elevation_angle"],
        )
        .collect()
    )

    grids: dict[int, dict] = {}
    for sweep_num in sorted(lut["sweep"].unique().to_list()):
        sub = lut.filter(pl.col("sweep") == sweep_num)
        azimuths = np.sort(sub["azimuth"].unique().to_numpy()).astype(np.float64)
        ranges = np.sort(sub["range"].unique().to_numpy()).astype(np.float64)
        # elevation_angle is constant per sweep by construction (the fixed
        # antenna angle); broadcast it to one value per ray.
        el = float(sub["elevation_angle"][0])
        grids[int(sweep_num)] = {
            "azimuths": azimuths,
            "ranges": ranges,
            "elevations": np.full(len(azimuths), el, dtype=np.float64),
            "elevation": el,
        }
    return grids


def compute_corners_from_lut(
    radar: str,
    lut_base_path: str | Path,
    ke: float = 4.0 / 3.0,
    beamwidth_deg: float | None = None,
) -> str:
    """Rebuild ``{radar}_corners.npz`` from an existing LUT parquet + info YAML.

    Useful when the corners file is missing (e.g. for LUTs generated before
    corners were introduced). Reads the per-sweep unique (azimuth, range,
    elevation) triples from the LUT, computes corner arrays via
    :func:`compute_sweep_corners`, and saves them.

    ``beamwidth_deg`` defaults to the value recorded in ``{radar}_info.yaml``
    (falling back to :data:`DEFAULT_BEAMWIDTH_DEG`), so the vertical extent of
    the beam is resolved rather than collapsed.

    Returns
    -------
    str : path to the written ``.npz`` file.
    """
    info = load_radar_info(radar, lut_base_path)
    if beamwidth_deg is None:
        beamwidth_deg = float(info.get("beamwidth_deg") or DEFAULT_BEAMWIDTH_DEG)

    corners_by_sweep: dict[int, dict] = {}
    for sweep_num, g in _sweep_grids_from_lut(radar, lut_base_path).items():
        full = compute_sweep_corners(
            ranges=g["ranges"],
            azimuths=g["azimuths"],
            elevations=g["elevations"],
            radar_lat=info["latitude"],
            radar_lon=info["longitude"],
            radar_alt=info["altitude"],
            ke=ke,
            beamwidth_deg=beamwidth_deg,
        )
        # The npz keeps only the beam-center mesh (its consumers — plot_ppi and
        # reconstruct_sweep_dataset — are 2-D). The vertical levels live in the
        # *_corners_LUT.parquet / *_v_plane_LUT.parquet files.
        corners_by_sweep[int(sweep_num)] = {k: v for k, v in full.items() if k != "levels"}
    corners_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_corners.npz"
    save_sweep_corners(corners_by_sweep, corners_path)
    return str(corners_path)


def _parse_corners_npz(corners_path: str | Path) -> dict[int, dict]:
    """Parse a ``*_corners.npz`` file into ``{sweep: {array_name: array}}``."""
    data = np.load(corners_path)
    out: dict[int, dict] = {}
    for key in data.files:
        # key format: "sweep_{N}_{array_name}"
        parts = key.split("_", 2)
        if len(parts) != 3 or parts[0] != "sweep":
            continue
        try:
            sw = int(parts[1])
        except ValueError:
            continue
        out.setdefault(sw, {})[parts[2]] = data[key]
    return out


def load_sweep_corners(
    radar: str,
    lut_base_path: str | Path,
) -> dict[int, dict]:
    """Load per-sweep corner arrays from ``{radar}_corners.npz``.

    Returns
    -------
    dict {sweep_number: {"x_edges", "y_edges", ...}}  — empty dict if the
    corners file does not exist (backwards-compatible).
    """
    corners_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_corners.npz"
    if not corners_path.exists():
        return {}
    return _parse_corners_npz(corners_path)


# ============================================================================
# GeoArrow export  (gate wedge polygons)
# ============================================================================

# Per-radar cache of gate_id -> (sweep, azimuth index, range index) into the
# corner arrays: {(base_path, radar): pl.DataFrame}.  ~27 MB per radar, built
# once per session.
_GRID_CACHE: dict[tuple[str, str, int], pl.DataFrame] = {}


def decode_gate_ids(gate_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`encode_gate_ids` (radar code excluded).

    Use :func:`decode_gate_radars` for the radar names.

    Returns
    -------
    (sweeps, azimuths, ranges) : np.ndarray
        Sweep number (int64), azimuth in degrees (float64, 1 decimal) and range
        in meters (float64), one entry per gate_id.
    """
    ids = np.asarray(gate_ids, dtype=np.int64)
    sweeps = (ids // np.int64(10_000_000_000)) % np.int64(100)
    az_int = (ids // np.int64(1_000_000)) % np.int64(10_000)
    rng_m = ids % np.int64(1_000_000)
    return sweeps, az_int.astype(np.float64) / 10.0, rng_m.astype(np.float64)


def decode_gate_radars(gate_ids) -> list[str]:
    """Sorted distinct radar names encoded in a set of ``gate_id`` values.

    The radar code is the leading field of the encoding
    (``code = gate_id // 10**12``), so the radars a frame spans can be read off
    without a separate ``radar`` column.  Codes that name no radar are skipped
    with a warning rather than raising — a frame is still usable when one of
    its radars cannot be identified.

    Parameters
    ----------
    gate_ids : array-like of int64

    Returns
    -------
    list of str
        e.g. ``["KTLX", "L"]``.
    """
    ids = np.asarray(gate_ids, dtype=np.int64)
    if ids.size == 0:
        return []
    names = []
    for code in np.unique(ids // np.int64(GATE_ID_RADAR_BASE)):
        try:
            names.append(decode_radar_code(int(code)))
        except ValueError:
            logger.warning("gate_id radar code %d names no radar; skipped.", int(code))
    return sorted(names)


def _gate_grid_index(radar: str, lut_base_path: str | Path) -> pl.DataFrame:
    """Map every ``gate_id`` to its position in the per-sweep corner arrays.

    Returns a frame ``[gate_id, sweep, az_idx, rng_idx]``.  The indices are
    computed against ``np.sort(unique(...))`` of the LUT's own azimuth/range
    values — the exact ordering :func:`compute_corners_from_lut` uses — so they
    address :func:`compute_sweep_corners` output directly.

    The lookup is keyed on ``gate_id`` rather than on azimuth/range values
    because ``gate_id`` stores azimuth rounded to 0.1° while the LUT keeps the
    raw antenna azimuth (~0.03° jitter); matching the floats would fail.
    """
    lut_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_LUT.parquet"
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT not found at {lut_path}.")

    # mtime is part of the key: regenerating a LUT in a live session must not
    # keep serving the geometry of the previous one.
    key = (str(lut_base_path), radar, lut_path.stat().st_mtime_ns)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    lut = pl.read_parquet(lut_path, columns=["gate_id", "sweep", "azimuth", "range"])

    parts = []
    for (sweep_num,), sub in lut.group_by(["sweep"]):
        az_grid = np.sort(sub["azimuth"].unique().to_numpy())
        rng_grid = np.sort(sub["range"].unique().to_numpy())
        parts.append(
            sub.select(
                "gate_id",
                pl.lit(int(sweep_num), dtype=pl.Int32).alias("sweep"),
                pl.Series("az_idx", np.searchsorted(az_grid, sub["azimuth"].to_numpy()), dtype=pl.Int32),
                pl.Series("rng_idx", np.searchsorted(rng_grid, sub["range"].to_numpy()), dtype=pl.Int32),
            ),
        )
    table = pl.concat(parts, how="vertical")
    _GRID_CACHE[key] = table
    return table


def gate_polygons_geoarrow(
    radar: str,
    lut_base_path: str | Path,
    gate_ids,
    frame: str = "geographic",
):
    """Build the gate wedge polygons for ``gate_ids`` as a GeoArrow array.

    Each gate becomes a 4-corner ring (closed, 5 vertices) taken from the
    per-sweep edge arrays produced by :func:`compute_sweep_corners` — the exact
    wedge, not a rectangle. The corners file is rebuilt via
    :func:`compute_corners_from_lut` if it does not exist yet.

    Parameters
    ----------
    radar : str
        Radar identifier, e.g. ``"A"`` or ``"KTLX"``.
    lut_base_path : str or Path
        RadDB archive base directory.
    gate_ids : array-like of int64
        Gates to build polygons for, in the order they should appear.
    frame : {"geographic", "cartesian"}
        ``"geographic"`` uses ``lon_edges``/``lat_edges`` (EPSG:4326, the frame
        web maps expect); ``"cartesian"`` uses ``x_edges``/``y_edges`` (meters
        from the radar).

    Returns
    -------
    pyarrow.Array
        A ``geoarrow.polygon`` extension array, one polygon per gate_id.
        Gates whose (azimuth, range) is absent from the LUT grid yield null.
    """
    import pyarrow as pa

    if frame not in ("geographic", "cartesian"):
        raise ValueError(f"frame must be 'geographic' or 'cartesian'; got {frame!r}.")
    xkey, ykey = ("lon_edges", "lat_edges") if frame == "geographic" else ("x_edges", "y_edges")

    corners = load_sweep_corners(radar, lut_base_path)
    if not corners:
        compute_corners_from_lut(radar, lut_base_path)
        corners = load_sweep_corners(radar, lut_base_path)

    ids = np.asarray(gate_ids, dtype=np.int64)
    n = len(ids)
    # Left-join keeps the caller's row order and leaves unknown gates null.
    located = (
        pl.DataFrame({"gate_id": ids, "_ord": np.arange(n, dtype=np.int64)})
        .join(_gate_grid_index(radar, lut_base_path), on="gate_id", how="left")
        .sort("_ord")
    )
    sweeps = located["sweep"].to_numpy()
    az_idx = located["az_idx"].fill_null(0).to_numpy()
    rng_idx = located["rng_idx"].fill_null(0).to_numpy()
    known = located["sweep"].is_not_null().to_numpy()

    # 5 vertices x 2 coordinates per gate; NaN marks a gate we could not place.
    ring_xy = np.full((n, 5, 2), np.nan, dtype=np.float64)

    for sweep_num in np.unique(sweeps[known]):
        sw = int(sweep_num)
        if sw not in corners:
            logger.warning("sweep %d missing from the corners file; gates skipped.", sw)
            continue
        rows = np.flatnonzero(known & (sweeps == sweep_num))
        ai, ri = az_idx[rows], rng_idx[rows]
        xe, ye = corners[sw][xkey], corners[sw][ykey]

        # Ring: (az_i, r_j) -> (az_i, r_j+1) -> (az_i+1, r_j+1) -> (az_i+1, r_j) -> close.
        for k, (ii, jj) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0), (0, 0))):
            ring_xy[rows, k, 0] = xe[ai + ii, ri + jj]
            ring_xy[rows, k, 1] = ye[ai + ii, ri + jj]

    valid = ~np.isnan(ring_xy[:, 0, 0])
    if not valid.all():
        logger.warning(
            "%d of %d gate_ids could not be placed on the LUT grid (null geometry).",
            int((~valid).sum()),
            n,
        )

    # geoarrow.polygon = List<List<FixedSizeList<double>[2]>>: polygon -> rings -> xy.
    # A null in the outer offsets makes that polygon null (unplaceable gate).
    coords = pa.FixedSizeListArray.from_arrays(
        pa.array(ring_xy.reshape(-1), type=pa.float64()),
        2,
    )
    rings = pa.ListArray.from_arrays(np.arange(n + 1, dtype=np.int32) * 5, coords)
    offsets = pa.array(
        np.arange(n + 1, dtype=np.int32),
        mask=np.concatenate([~valid, [False]]),
    )
    return pa.ListArray.from_arrays(offsets, rings)


def geoarrow_field(name: str, dtype, kind: str, crs: str | None = None):
    """Build a ``pyarrow.Field`` tagged as a GeoArrow extension type.

    GeoArrow identifies geometry through *field* metadata, so the tag can only be
    attached where the array is placed into a table/schema.

    Parameters
    ----------
    name : str
        Column name.
    dtype : pyarrow.DataType
        Type of the geometry array (e.g. ``polygons.type``).
    kind : str
        GeoArrow geometry kind, e.g. ``"point"`` or ``"polygon"``.
    crs : str, optional
        CRS identifier such as ``"EPSG:4326"``.
    """
    import pyarrow as pa

    meta = {b"ARROW:extension:name": f"geoarrow.{kind}".encode()}
    if crs:
        meta[b"ARROW:extension:metadata"] = (f'{{"crs":"{crs}","crs_type":"authority_code"}}').encode()
    return pa.field(name, dtype, metadata=meta)


def load_radar_lut(
    radar: str,
    lut_base_path: str | Path,
) -> pl.DataFrame:
    """Load the LUT parquet for a radar as a **polars** DataFrame.

    Call :meth:`polars.DataFrame.to_pandas` on the result if you need pandas.
    """
    lut_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_LUT.parquet"
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT not found at {lut_path}.")
    return pl.read_parquet(lut_path)


def load_radar_info(
    radar: str,
    lut_base_path: str | Path,
) -> dict:
    """Load the radar info YAML for a radar."""
    info_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_info.yaml"
    if not info_path.exists():
        raise FileNotFoundError(f"Info not found at {info_path}.")
    with open(info_path) as f:
        info = yaml.safe_load(f)
    return info


def get_full_sweep_index(
    lut_df: pl.DataFrame | pd.DataFrame,
    sweep: int,
) -> pd.MultiIndex:
    """Get the full (azimuth, range) MultiIndex for a sweep from the LUT.

    Accepts a polars **or** a pandas LUT — :func:`load_radar_lut` returns polars,
    so the polars form is the common one.  The return type stays a pandas
    ``MultiIndex`` because its only consumer is the pandas/xarray reconstruction
    bridge in :func:`raddb.io_core.reconstruct_sweep_dataset`, which reindexes
    against it.
    """
    if isinstance(lut_df, pl.DataFrame):
        sweep_lut = lut_df.filter(pl.col("sweep") == sweep).select(["azimuth", "range"])
        if sweep_lut.is_empty():
            raise ValueError(f"No LUT entries found for sweep={sweep}.")
        return pd.MultiIndex.from_arrays(
            [sweep_lut["azimuth"].to_numpy(), sweep_lut["range"].to_numpy()],
            names=["azimuth", "range"],
        )

    sweep_lut = lut_df[lut_df["sweep"] == sweep]
    if sweep_lut.empty:
        raise ValueError(f"No LUT entries found for sweep={sweep}.")
    return sweep_lut.set_index(["azimuth", "range"]).index


# ============================================================================
# Projection utilities
# ============================================================================


def add_lut_projection(
    lut_df: pl.DataFrame | pd.DataFrame,
    epsg: int | None = None,
    crs=None,
) -> pl.DataFrame | pd.DataFrame:
    """Add projected coordinates to a LUT DataFrame.

    Converts the ``latitude`` / ``longitude`` columns to the target CRS and
    appends ``x_{suffix}`` / ``y_{suffix}`` columns, where ``suffix`` is the
    EPSG code (if available) or ``"custom"``.

    Accepts either a polars or a pandas frame and returns the same kind.  The
    LUT layer is polars end to end (generation and loading), so the polars form
    is the common one; pandas is accepted so external callers keep working.

    Requires **pyproj** (``pip install pyproj``).

    Parameters
    ----------
    lut_df : pl.DataFrame or pd.DataFrame
        LUT DataFrame with ``latitude`` and ``longitude`` columns (degrees,
        WGS-84 / EPSG:4326).
    epsg : int, optional
        EPSG code for the target CRS.
        Example: ``2056`` for CH1903+ / LV95 (Swiss national grid).
    crs : pyproj.CRS or any CRS-coercible object, optional
        Alternative to ``epsg``.  Used when ``epsg`` is ``None``.
        Can be a pyproj ``CRS`` object, a WKT string, or any value accepted
        by ``pyproj.CRS()``.

    Returns
    -------
    pl.DataFrame or pd.DataFrame
        Copy of ``lut_df`` (same kind) with two new columns:
        ``x_{suffix}`` (easting / meters) and ``y_{suffix}`` (northing / meters).

    Raises
    ------
    ValueError
        If neither ``epsg`` nor ``crs`` is provided.
    ImportError
        If pyproj is not installed.

    Examples
    --------
    Add Swiss LV95 (CH1903+) coordinates to a LUT:

    >>> lut_ch = add_lut_projection(lut_df, epsg=2056)
    >>> lut_ch[["x_2056", "y_2056"]].head()

    Use a custom pyproj CRS:

    >>> import pyproj
    >>> my_crs = pyproj.CRS.from_epsg(32632)  # UTM zone 32N
    >>> lut_utm = add_lut_projection(lut_df, crs=my_crs)
    """
    try:
        import pyproj
    except ImportError as exc:
        raise ImportError(
            "pyproj is required for add_lut_projection. " "Install it with: pip install pyproj",
        ) from exc

    if epsg is not None:
        target_crs = pyproj.CRS.from_epsg(epsg)
        col_suffix = str(epsg)
    elif crs is not None:
        target_crs = crs if isinstance(crs, pyproj.CRS) else pyproj.CRS(crs)
        detected_epsg = target_crs.to_epsg()
        col_suffix = str(detected_epsg) if detected_epsg is not None else "custom"
    else:
        raise ValueError("Provide either 'epsg' or 'crs'.")

    # Use proj4 string for WGS-84 to avoid requiring the PROJ database
    wgs84 = pyproj.CRS.from_proj4(
        "+proj=longlat +datum=WGS84 +no_defs",
    )
    transformer = pyproj.Transformer.from_crs(
        wgs84,
        target_crs,
        always_xy=True,
    )
    x_proj, y_proj = transformer.transform(
        lut_df["longitude"].to_numpy(),
        lut_df["latitude"].to_numpy(),
    )

    if isinstance(lut_df, pl.DataFrame):
        return lut_df.with_columns(
            pl.Series(f"x_{col_suffix}", x_proj),
            pl.Series(f"y_{col_suffix}", y_proj),
        )

    lut_out = lut_df.copy()
    lut_out[f"x_{col_suffix}"] = x_proj
    lut_out[f"y_{col_suffix}"] = y_proj
    return lut_out

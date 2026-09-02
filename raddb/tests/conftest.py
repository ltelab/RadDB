"""Shared fixtures for the RadDB test suite.

Every test in this package is **synthetic**: volumes are built in memory and archives are
written under ``tmp_path``.  Nothing reads a machine-local path, so the suite runs
unchanged in CI.

The synthetic site sits at 62.0 N / 27.0 E (Finland), which is why ``crs=3067``
(ETRS89 / TM35FIN, the Finnish national grid) passes RadDB's measured CRS validation.  A
fixture that moves the site must move the CRS with it or
:func:`raddb.lut.generate_lut_from_datatree` refuses to write.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Default radar name used across the suite — an FMI site, four characters so the
# base-36 radar code is exercised over its full width rather than a single letter.
RADAR = "FKUO"

# Second radar, for the multi-radar fixtures.
RADAR_B = "FANJ"

# Default number of azimuth rays in a synthetic sweep.
N_AZ = 12

# Default number of range bins in a synthetic sweep.
N_RNG = 24

# Latitude of the synthetic radar site, in degrees north.
SITE_LAT = 62.0

# Longitude of the synthetic radar site, in degrees east.
SITE_LON = 27.0

# Altitude of the synthetic radar site, in meters above sea level.
SITE_ALT = 1000.0

# The synthetic radar site as ``(longitude, latitude)`` — the argument order every
# RadDB entry point that names a *site* uses.
FI_SITE = (SITE_LON, SITE_LAT)

# ETRS89 / TM35FIN — the projected CRS valid at :data:`FI_SITE`.
FMI_EPSG = 3067


@pytest.fixture(autouse=True)
def _agg_backend():
    """Force a non-interactive matplotlib backend for every test.

    Autouse so no test file has to remember it.  Import is local: matplotlib is only
    needed by the plotting tests and pulling it in at collection time would slow the
    whole suite down.
    """
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - matplotlib is a hard dependency of viz only
        return
    matplotlib.use("Agg", force=True)


def build_datatree(
    n_az: int = N_AZ,
    n_rng: int = N_RNG,
    dbzh_min: float = 1.0,
    dbzh_max: float = 30.0,
    n_sweeps: int = 2,
    vol_time: pd.Timestamp | None = None,
    latitude: float = SITE_LAT,
    longitude: float = SITE_LON,
    altitude: float = SITE_ALT,
) -> xr.DataTree:
    """Build a minimal xradar-layout DataTree with all-positive DBZH.

    All DBZH values are strictly positive so the archive's default ``DBZH > 0``
    clear-sky filter keeps every gate.

    Every sweep also carries a scalar string ``sweep_mode``, the way a real xradar tree
    does.  It is metadata about the scan, not a moment, and must never reach a POL
    parquet — keeping it here means every archiving test exercises that rule.

    Parameters
    ----------
    n_az, n_rng : int
        Number of azimuth rays and range bins per sweep.
    dbzh_min, dbzh_max : float
        Bounds of the uniform random DBZH field, in dBZ.
    n_sweeps : int
        Number of sweeps in the volume.
    vol_time : pandas.Timestamp, optional
        Volume time stamped on every ray.  Defaults to 2024-08-01 12:00:00.
    latitude, longitude, altitude : float
        Radar site position.  Moving it invalidates ``crs=3067``.

    Returns
    -------
    xarray.DataTree
        A volume with sweeps named ``sweep_1`` .. ``sweep_{n_sweeps}``.
    """
    if vol_time is None:
        vol_time = pd.Timestamp("2024-08-01 12:00:00")

    az = np.linspace(0, 360 - 360 / n_az, n_az)
    rng_vals = np.linspace(1000, 20_000, n_rng)
    time_vals = np.array([vol_time] * n_az, dtype="datetime64[ns]")

    dict_ds = {}
    for sweep_idx in range(1, n_sweeps + 1):
        rng_gen = np.random.default_rng(seed=42 + sweep_idx)
        dbzh = rng_gen.uniform(dbzh_min, dbzh_max, (n_az, n_rng)).astype(np.float32)
        ds = xr.Dataset(
            {
                "DBZH": (["azimuth", "range"], dbzh),
                "ZDR": (["azimuth", "range"], np.ones((n_az, n_rng), np.float32)),
                "RHOHV": (["azimuth", "range"], np.full((n_az, n_rng), 0.95, np.float32)),
                "PHIDP": (["azimuth", "range"], np.zeros((n_az, n_rng), np.float32)),
                "VRADH": (["azimuth", "range"], np.full((n_az, n_rng), -3.5, np.float32)),
                "time": (["azimuth"], time_vals),
                # Scalar scan metadata, as a real xradar sweep carries it.
                "sweep_mode": ((), "azimuth_surveillance"),
            },
            coords={
                "azimuth": az,
                "range": rng_vals,
                "elevation": (["azimuth"], np.full(n_az, 0.5 * sweep_idx)),
                "elevation_angle": 0.5 * sweep_idx,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
            },
        )
        ds.attrs["sweep_number"] = sweep_idx
        dict_ds[f"sweep_{sweep_idx}"] = ds

    return xr.DataTree.from_dict(dict_ds)


def relocate(dt: xr.DataTree, longitude: float, latitude: float) -> xr.DataTree:
    """Move a synthetic volume to another place on Earth.

    Used to test the CRS contract: EPSG:3067 is valid at the default Finnish site and
    invalid — by 36.4% — once the volume is moved to Oklahoma.

    Parameters
    ----------
    dt : xarray.DataTree
        Volume to relocate; not modified in place.
    longitude, latitude : float
        New site position, in degrees.

    Returns
    -------
    xarray.DataTree
        A new tree with every sweep's site coordinates replaced.
    """
    out = {}
    for name, node in dt.children.items():
        ds = node.to_dataset().assign_coords(latitude=latitude, longitude=longitude)
        ds.attrs.update(node.attrs)
        out[name] = ds
    return xr.DataTree.from_dict(out)


# Antenna drift measured on a real 360-ray C-band network. Every ray is reported ~0.0327
# degrees past its nominal angle, with a ~0.0069 degree spread. ``gate_id`` resolves azimuth to
# 0.1 degrees, so an unsnapped drifting ray lands in a neighboring bin and its gates match no
# LUT row.
ANTENNA_BIAS, ANTENNA_SPREAD = 0.0327, 0.0069

# WSR-88D antenna drift: zero-mean, spread up to ~0.045 degrees.
NEXRAD_SPREAD = 0.045


def jitter_azimuths(azimuths, rng, bias: float = 0.0, spread: float = ANTENNA_SPREAD):
    """Move an azimuth array the way a real antenna does between rotations.

    Parameters
    ----------
    azimuths : array_like
        Nominal angles, in degrees.
    rng : numpy.random.Generator
        Source of the random spread.
    bias : float
        Systematic offset added to every ray, in degrees.
    spread : float
        Standard deviation of the per-ray noise, in degrees.

    Returns
    -------
    numpy.ndarray
        Drifted angles, wrapped into ``[0, 360)``.
    """
    az = np.asarray(azimuths, float)
    return (az + bias + rng.normal(0, spread, len(az))) % 360.0


def retime(dt: xr.DataTree, when, rng, bias: float = 0.0, spread: float = ANTENNA_SPREAD) -> xr.DataTree:
    """Copy a volume at a new time, with the antenna pointing slightly differently.

    This is what a second rotation of the same radar actually looks like: the same scan
    strategy, reported a few hundredths of a degree away.

    Parameters
    ----------
    dt : xarray.DataTree
        Volume to copy; not modified in place.
    when : pandas.Timestamp
        New volume time, stamped on every ray.
    rng : numpy.random.Generator
        Source of the azimuth drift.
    bias, spread : float
        Passed to :func:`jitter_azimuths`.

    Returns
    -------
    xarray.DataTree
        The retimed, drifted volume.
    """
    out = {}
    for name, node in dt.children.items():
        ds = node.to_dataset()
        n = ds.sizes["azimuth"]
        ds = ds.assign_coords(azimuth=jitter_azimuths(ds["azimuth"].values, rng, bias, spread))
        ds["time"] = ("azimuth", np.array([when] * n, dtype="datetime64[ns]"))
        ds.attrs.update(node.attrs)
        out[name] = ds
    return xr.DataTree.from_dict(out)


# KTLX, Oklahoma — ``(longitude, latitude)``. UTM 14N (EPSG:32614) is valid here.
US_SITE = (-97.2775, 35.3331)

# UTM zone 14N — the projected CRS valid at :data:`US_SITE`.
US_EPSG = 32614


@pytest.fixture
def us_archive_dir(tmp_path, make_datatree):
    """A one-radar archive at KTLX, written in UTM 14N.

    The other-continent counterpart to :func:`archive_dir`: any AOI or plotting behavior
    that silently assumes the primary frame shows up here as a gross error rather than a
    rounding one.

    Returns
    -------
    pathlib.Path
        The archive root.
    """
    from raddb.main import RadDB

    base = tmp_path / "us_archive"
    dt = relocate(make_datatree(n_az=72, n_rng=60, n_sweeps=3), *US_SITE)
    RadDB(archive_dir=str(base), crs=US_EPSG).archive(datatree={RADAR: [dt]})
    return base


@pytest.fixture
def make_datatree():
    """Return the :func:`build_datatree` factory.

    Returns
    -------
    callable
        Same signature as :func:`build_datatree`.
    """
    return build_datatree


@pytest.fixture
def datatree(make_datatree) -> xr.DataTree:
    """A single default synthetic volume.

    Returns
    -------
    xarray.DataTree
        Two sweeps, 12 x 24 gates, 2024-08-01 12:00:00.
    """
    return make_datatree()


@pytest.fixture
def archive_dir(tmp_path, make_datatree):
    """Path to a one-radar, one-volume archive written under ``tmp_path``.

    Radar ``FKUO``, ``crs=3067``, LUT and all four geometry lattices present.

    Returns
    -------
    pathlib.Path
        The archive root, ready for ``RadDB(archive_dir=...)``.
    """
    from raddb.main import RadDB

    base = tmp_path / "archive"
    RadDB(archive_dir=str(base), crs=FMI_EPSG).archive(datatree=make_datatree(), radar=RADAR)
    return base


@pytest.fixture
def archive_dir_two_volumes(tmp_path, make_datatree):
    """Path to a one-radar, two-volume archive (12:00 and 12:05).

    Returns
    -------
    pathlib.Path
        The archive root.
    """
    from raddb.main import RadDB

    base = tmp_path / "archive2"
    volumes = [
        make_datatree(vol_time=pd.Timestamp("2024-08-01 12:00:00")),
        make_datatree(vol_time=pd.Timestamp("2024-08-01 12:05:00")),
    ]
    RadDB(archive_dir=str(base), crs=FMI_EPSG).archive(datatree=volumes, radar=RADAR)
    return base


@pytest.fixture
def archive_dir_two_radars(tmp_path, make_datatree):
    """Path to a two-radar archive (``FKUO`` and ``FANJ``), one volume each.

    Returns
    -------
    pathlib.Path
        The archive root.
    """
    from raddb.main import RadDB

    base = tmp_path / "archive_multi"
    RadDB(archive_dir=str(base), crs=FMI_EPSG).archive(
        datatree={RADAR: [make_datatree()], RADAR_B: [make_datatree()]},
    )
    return base


# Volume shape used by the plotting fixtures. Six sweeps and 60 range bins are the minimum that
# makes the CAPPI and RHI invariants meaningful — a two-sweep volume has no beam overlap to
# resolve.
PLOT_GEOMETRY = {"n_az": 72, "n_rng": 60, "n_sweeps": 6}

# Radar name used by the plotting fixtures.
PLOT_RADAR = "FKOR"


@pytest.fixture(scope="session")
def plot_archive_dir(tmp_path_factory):
    """A 72 x 60 x 6 archive, built **once per session**.

    The plotting tests all read the same static geometry, and rebuilding a 26k-gate
    archive per test would dominate the suite runtime.  Nothing here mutates it.

    Returns
    -------
    pathlib.Path
        The archive root.
    """
    from raddb.main import RadDB

    base = tmp_path_factory.mktemp("plot_archive")
    RadDB(archive_dir=str(base), crs=FMI_EPSG).archive(
        datatree={PLOT_RADAR: [build_datatree(**PLOT_GEOMETRY)]},
    )
    return base


@pytest.fixture(scope="session")
def plot_rdb(plot_archive_dir):
    """A data-carrying RadDB over :func:`plot_archive_dir`.

    Returns
    -------
    raddb.RadDB
        The whole single volume, 25,920 gates.
    """
    from raddb.main import RadDB

    return RadDB(archive_dir=str(plot_archive_dir), crs=FMI_EPSG).open(radars=PLOT_RADAR)


@pytest.fixture(scope="session")
def plot_site(plot_archive_dir):
    """The plotting archive's radar site as ``(x, y)`` in EPSG:3067.

    Returns
    -------
    tuple of float
        Projected easting and northing, in meters.
    """
    import shapely

    from raddb.aoi import _reproject_to_aoi
    from raddb.main import RadDB

    info = RadDB(archive_dir=str(plot_archive_dir)).get_radar_info(PLOT_RADAR)
    point = _reproject_to_aoi(shapely.Point(info["longitude"], info["latitude"]), 4326, FMI_EPSG)
    return (point.x, point.y)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test so a long run does not leak them."""
    yield
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a viz-only dependency
        return
    plt.close("all")


@pytest.fixture
def db(archive_dir):
    """An archive-bound :class:`raddb.RadDB` over :func:`archive_dir`.

    Returns
    -------
    raddb.RadDB
        Opened with ``crs=3067``.
    """
    from raddb.main import RadDB

    return RadDB(archive_dir=str(archive_dir), crs=FMI_EPSG)


@pytest.fixture
def rdb(db):
    """A data-carrying :class:`raddb.RadDB` holding the whole archive.

    Returns
    -------
    raddb.RadDB
        Result of ``db.open(radars="FKUO")``.
    """
    return db.open(radars=RADAR)

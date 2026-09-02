# RadDB: Radar Database

|                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Deployment        | [![PyPI](https://badge.fury.io/py/raddb.svg?style=flat)](https://pypi.org/project/raddb/) [![Conda](https://img.shields.io/conda/vn/conda-forge/raddb.svg?logo=conda-forge&logoColor=white&style=flat)](https://anaconda.org/conda-forge/raddb)                                                                                                                                                                                                                                                                                                                                                                        |
| Activity          | [![PyPI Downloads](https://img.shields.io/pypi/dm/raddb.svg?label=PyPI%20downloads&style=flat)](https://pypi.org/project/raddb/) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/raddb.svg?label=Conda%20downloads&style=flat)](https://anaconda.org/conda-forge/raddb)                                                                                                                                                                                                                                                                                                                                |
| Python Versions   | [![Python Versions](https://img.shields.io/badge/Python-3.11%20%203.12%20%203.13%20%203.14-blue?style=flat)](https://www.python.org/downloads/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Supported Systems | [![Linux](https://img.shields.io/github/actions/workflow/status/ltelab/raddb/.github/workflows/tests.yaml?label=Linux&style=flat)](https://github.com/ltelab/raddb/actions/workflows/tests.yaml) [![macOS](https://img.shields.io/github/actions/workflow/status/ltelab/raddb/.github/workflows/tests.yaml?label=macOS&style=flat)](https://github.com/ltelab/raddb/actions/workflows/tests.yaml) [![Windows](https://img.shields.io/github/actions/workflow/status/ltelab/raddb/.github/workflows/tests_windows.yaml?label=Windows&style=flat)](https://github.com/ltelab/raddb/actions/workflows/tests_windows.yaml) |
| Project Status    | [![Project Status](https://www.repostatus.org/badges/latest/active.svg?style=flat)](https://www.repostatus.org/#active)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Build Status      | [![Tests](https://github.com/ltelab/RadDB/actions/workflows/tests.yaml/badge.svg?style=flat)](https://github.com/ltelab/RadDB/actions/workflows/tests.yaml) [![Lint](https://github.com/ltelab/RadDB/actions/workflows/lint.yaml/badge.svg?style=flat)](https://github.com/ltelab/RadDB/actions/workflows/lint.yaml) [![Docs](https://readthedocs.org/projects/raddb/badge/?version=latest&style=flat)](https://raddb.readthedocs.io/en/latest/)                                                                                                                                                                       |
| Linting           | [![Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat)](https://github.com/astral-sh/ruff) [![Codespell](https://img.shields.io/badge/Codespell-enabled-brightgreen?style=flat)](https://github.com/codespell-project/codespell)                                                                                                                                                                                          |
| Code Coverage     | [![Coveralls](https://coveralls.io/repos/github/ltelab/RadDB/badge.svg?branch=main&style=flat)](https://coveralls.io/github/ltelab/RadDB?branch=main) [![Codecov](https://codecov.io/gh/ltelab/raddb/branch/main/graph/badge.svg?style=flat)](https://codecov.io/gh/ltelab/raddb)                                                                                                                                                                                                                                                                                                                                      |
| Code Quality      | [![Codacy](https://app.codacy.com/project/badge/Grade/d823c50a7ad14268bd347b5aba384623?style=flat)](https://app.codacy.com/gh/ltelab/raddb/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![Codescene](https://codescene.io/projects/83114/status-badges/average-code-health?style=flat)](https://codescene.io/projects/83114)                                                                                                                                                                                                                                                    |
| License           | [![License](https://img.shields.io/github/license/ltelab/raddb?style=flat)](https://github.com/ltelab/raddb/blob/main/LICENSE)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

| Citation          | [![DOI](XXX)](XXX)          |                                                                                                                                                                                                                                                                                                                                                                                                                                                               [**Documentation**](https://raddb.readthedocs.io/en/latest/)

RadDB archives xarray **DataTree** radar volumes as compact Parquet files and
gives you a small, fluent interface to load, filter, crop, extract cross-sections and
plot them.  It is **network-agnostic**: any DataTree with the standard
[xradar](https://docs.openradarscience.org/projects/xradar/) coordinate layout
(ODIM, IRIS, …) can be archived and analyzed.

## Storage model

A radar is stored as a **static LUT** (per-gate geometry, generated once) plus one
**POL parquet per volume** (the variables), linked by an integer `gate_id`:

```
{archive_dir}/{radar}/LUT/{radar}_LUT.parquet          # gate_id, lat/lon/alt, x_<epsg>/y_<epsg>, sweep, …
{archive_dir}/{radar}/{YYYY}/{MM}/{DD}/{radar}_{YYYYMMDD}_{HHMMSS}_POL.parquet
```

No-echo gates are dropped at archive time (default `DBZH > 0`), so the archive
stays small: a 12-sweep WSR-88D volume of 8,791,200 polar gates becomes 8.2 MB.

## Installation

```bash
pip install raddb
pip install "raddb[viz]"       # interactive Jupyter map and cartopy basemaps
```

Core runtime dependencies: `numpy, pandas, polars, geopandas, shapely, pyproj, xarray, pyyaml, pyarrow, matplotlib, netcdf4, zarr`.

## Quick start

`RadDB` is a single, dual-role class:

- **archive-bound** — `db = RadDB(archive_dir, crs=…)`; use it to `archive()` and `open()`.
- **data-carrying** — the object returned by `open()` (call it `rdf`).  It holds the
  data as a **polars** DataFrame (`rdf.data`) and exposes the query / convert /
  crop / cross-section / plot methods.  Each of those returns a **new** `RadDB`,
  so calls chain fluently.

```python
import raddb

db = raddb.RadDB(archive_dir="/data/raddb", crs=3067)
```

A **projected CRS is mandatory to write** an archive and never needed to read one.
There is no default: the wrong projection is silently wrong.
`raddb.lut.suggest_crs(longitude, latitude)` tells you which one to pass.

### Archive

```python
# From saved DataTree files on disk (.zarr / .nc); the LUT is auto-generated:
db.archive(datatree_dir="/data/FMI_datatree")  # radar inferred per file

# ...or archive in-memory DataTrees directly:
db.archive(datatree=dt, radar="FANJ")
db.archive(datatree=[dt1, dt2], radar="FANJ")
db.archive(datatree={"FANJ": [dt1], "FKOR": [dt2]})  # multi-radar
```

Pass `filter=` to decide which gates ever reach the disk — the main control on
archive size:

```python
db.archive(
    datatree=dt, radar="FANJ", filter={"var": "DBZH", "logic": ">", "threshold": 20}
)
```

### Open

```python
rdf = db.open(time_period=("2024-06-17", "2024-06-18"), radars="FANJ")
print(rdf)  # summary: gates, radars, time range, columns

len(rdf), rdf.columns(), rdf.radars()
rdf.start_time(), rdf.end_time()
rdf.extent()  # [xmin, xmax, ymin, ymax] in `crs`
rdf.geographic_extent()  # [lon_min, lon_max, lat_min, lat_max]
rdf.crs(), rdf.geographic_crs()
```

`columns=` and `filters=` are pushed down into the scan, so only the rows you
asked for are ever materialized.

### Filter and convert

```python
filtered_rdf = rdf.filter({"var": "DBZH", "logic": ">", "threshold": 20})
filtered_rdf = rdf.filter(
    [
        {"var": "DBZH", "logic": ">", "threshold": 20},
        {"var": "RHOHV", "logic": ">", "threshold": 0.9},
    ]
)

df = rdf.to_pandas(with_geometry=True)  # pandas + gate coordinates
gdf = rdf.to_geopandas()  # GeoDataFrame (with CRS)
dt = rdf.to_datatree()  # back to xarray
```

Filters are `{"var", "logic", "threshold"}` dicts, where `logic` is one of
`==`, `!=`, `>`, `>=`, `<`, `<=`.  `crs` is an EPSG int (e.g. `3067`), a
CRS object, or `None`.

### Crop to an area of interest

```python
box = rdf.crop_by_bbox(extent=[485_859, 525_859, 6_732_099, 6_772_099])
poly = rdf.crop_by_polygon("catchment.geojson")
disc = rdf.crop_around_point((505_859, 6_752_099), distance=20_000)  # meters
# rdf.interactive_crop()   # draw an AOI on a Jupyter map
```

### Cut a cross-section

```python
cs = rdf.extract_cross_section(p1=(475_859, 6_752_099), p2=(535_859, 6_752_099))
```

### Plot

```python
rdf.plot_ppi(sweep=1, variable="DBZH", save="ppi.png")
rdf.plot_rhi(azimuth=270, variable="DBZH")
rdf.plot_cappi(altitude=3000, variable="DBZH")
cs.plot_vcs(variable="DBZH", save="xsec.png")
```

Each plot draws into one `Axes` and returns the matplotlib artist, so you compose
panels by passing `ax=`.

### Chain them

```python
rdf.filter({"var": "DBZH", "logic": ">", "threshold": 20}).crop_by_bbox(
    extent=rdf.extent()
).plot_ppi(variable="DBZH", save="ppi_plot_example.png")
```

### What is on disk? (archive-bound)

```python
db.inventory()  # radars, volume counts, time ranges, size
db.inventory(detailed=True)  # + LUT info, stored variables, day-by-day counts
db.inventory(datatree_dir="/data/FMI_datatree")  # DataTree files not archived yet
```

### LUT accessors (archive-bound)

```python
db.list_radars()  # radars present in the archive
db.get_lut("FANJ")  # the static LUT (polars)
db.get_radar_info("FANJ")  # site location / sweep geometry
db.add_lut_projection("FANJ", epsg=3067)
```

## Module structure

```
raddb/
├── __init__.py
├── main.py         # the RadDB class
├── io_core.py      # DataTree <-> DataFrame <-> Parquet + archive backends
├── lut.py          # LUT generation / geo projection
├── aoi.py          # AOI / crop / cross-section geometry
├── discovery.py    # find_datatree_files + filename-time parsing
├── helper.py       # filters, radar-name normalization, timers
└── viz/            # plot.py (PPI/RHI/CAPPI/cross-section), interactive.py
```

## Notes

- **Projected coordinates / `crs`.** Generating a LUT with a projection (e.g.
  `crs=3067`) and the projected accessors (`extent`, `to_geopandas`) use `pyproj`,
  which needs the PROJ database.  A `PROJ_DATA` / `PROJ_LIB` inherited from another
  environment (a conda base env, a system PROJ) points at a proj.db of the wrong
  PROJ version and makes every projection fail with *"no database context
  specified"* — `import raddb` detects that and repoints PROJ at the running
  interpreter's own `share/proj` (see `raddb/_proj.py`; `raddb.PROJ_DATA` reports
  what it changed, `None` when nothing had to be).
- **Zarr / NetCDF.** Saved volumes are read back with `raddb.open_any_datatree`
  (engine auto-detected); either format works.

## License

See LICENSE (MIT).

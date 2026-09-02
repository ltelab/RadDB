===========
Quick Start
===========

RadDB turns radar **volumes** into a compact **tabular archive**: one Parquet
file per volume, one row per radar gate.  Once archived, a whole campaign is
queried like a dataframe — filter it, crop it, plot it — without ever
re-reading the source files.

This page walks through the shortest useful path; the notebooks in the
repository's ``tutorial/`` directory cover each step in depth.

What RadDB stores
-----------------

A radar is written as one **static look-up table** (the per-gate geometry,
generated once) plus one **Parquet file per volume** (the variables that change
from scan to scan)::

    {archive_dir}/KTLX/LUT/KTLX_LUT.parquet          # gate geometry, written once
    {archive_dir}/KTLX/LUT/KTLX_info.yaml            # site, CRS, scan strategy
    {archive_dir}/KTLX/2024/06/12/KTLX_20240612_220324_POL.parquet

The two are linked by an integer ``gate_id``.  Because the geometry is stored
once and never repeated, a volume file holds only the measurements — which is
what makes the archive small.

Archive a volume
----------------

RadDB is **network-agnostic**: any
`xarray.DataTree <https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html>`_
with the standard
`xradar <https://docs.openradarscience.org/projects/xradar/en/stable/>`_
coordinate layout can be archived, whether it comes from NEXRAD,
ODIM or IRIS.

.. code-block:: python

    import raddb

    db = raddb.RadDB(archive_dir="/path/to/archive", crs=32614)  # UTM 14N, the KTLX zone

    # a single in-memory volume
    dt = raddb.open_any_datatree("KTLX_20240612_220324.zarr")
    db.archive(datatree=dt, radar="KTLX")

    # or a whole directory of saved volumes, grouped by radar from the filename
    db.archive(datatree_dir="/path/to/datatrees", time_period=("2024-06-01", "2024-07-01"))

.. note::
   A **projected CRS is mandatory to write** an archive and never needed to
   read one.  There is no default, because a wrong projection is silently
   wrong. Use ``raddb.lut.suggest_crs(longitude, latitude)`` if you are
   unsure which one to pass.

Filter while archiving
----------------------

Most gates in a radar volume contain no echo.  The ``filter`` argument decides
which gates ever reach the disk, so it is the main control on archive size.
It takes a ``{"var", "logic", "threshold"}`` dictionary:

.. code-block:: python

    # the default: drop no-echo gates
    db.archive(datatree=dt, radar="KTLX", filter={"var": "DBZH", "logic": ">", "threshold": 0})

    # keep only significant echo — a much smaller archive
    db.archive(datatree=dt, radar="KTLX", filter={"var": "DBZH", "logic": ">", "threshold": 20})

Measured on ``KTLX_20240612_220324``, a 12-sweep WSR-88D volume of
8,791,200 polar gates:

=========================  ==============  ==========
filter                     gates archived  Parquet
=========================  ==============  ==========
``DBZH > 0`` (default)     1,424,223       8.22 MB
``DBZH > 20``              12,642          0.11 MB
=========================  ==============  ==========

The filter is irreversible — discarded gates are not in the archive — so choose
the threshold against what you intend to analyse.

Inspect the archive
-------------------

.. code-block:: python

    db = raddb.RadDB(archive_dir="/path/to/archive")  # reading needs no CRS

    db.list_radars()  # ['FANJ', 'KDVN', 'KLOT', 'KMLB', 'KTLX', ...]
    db.inventory()  # volumes, time range and size per radar
    db.get_radar_info("KTLX")  # site, CRS, beamwidth, sweep geometry

Open and query
--------------

``open()`` returns a **data-carrying** ``RadDB`` holding a
`polars <https://pola.rs/>`_ DataFrame.  Time, radar, columns and gate filters
are all pushed down into the scan, so only the rows you asked for are ever
materialised:

.. code-block:: python

    rdf = db.open(
        radars="KTLX",
        time_period=("2024-06-12", "2024-06-13"),
        columns=["DBZH", "ZDR"],  # fewer columns to read
        filters=[{"var": "DBZH", "logic": ">", "threshold": 20}],  # applied during the scan
    )

    len(rdf)  # 70359
    rdf.columns()  # ['gate_id', 'DBZH', 'ZDR', 'volume_time', 'radar']
    rdf.head()

Every operation returns a **new** ``RadDB``, so calls chain:

.. code-block:: python

    heavy_rain = (
        db.open(radars="KTLX")
        .filter({"var": "DBZH", "logic": ">", "threshold": 30})
        .sel(volume_time="2024-06-12 22:03:24")
        .crop_around_point(point=(-97.278, 35.333), distance=25_000, crs=4326)
    )

Alongside ``crop_around_point`` there are ``crop_by_bbox``, ``crop_by_polygon``
(shapely geometry, GeoDataFrame or a ``.shp`` / ``.geojson`` path) and
``extract_cross_section`` for a vertical slice along an arbitrary line.

Plot
----

Four plots, each drawing into one ``Axes`` and returning the matplotlib artist,
so you compose panels by passing ``ax=``.  They read the exact gate geometry
from the look-up table and draw only the gates the object holds:

.. code-block:: python

    rdf.plot_ppi(sweep=1, variable="DBZH")  # one sweep
    rdf.plot_rhi(azimuth=90)  # one azimuth
    rdf.plot_cappi(altitude=3000)  # constant-ppi slice
    rdf.plot_vcs(line="section.geojson")  # vertical cross-section

Convert
-------

.. code-block:: python

    rdf.to_pandas(with_geometry=True)  # + latitude / longitude / altitude / sweep
    rdf.to_geopandas()  # a GeoDataFrame of gate centroids
    rdf.to_datatree()  # back to xarray, on the full polar grid

Further reading
---------------

If you are new to the ecosystem, the
`xradar <https://docs.openradarscience.org/projects/xradar/en/stable/>`_,
`xarray <https://docs.xarray.dev/en/stable/>`_ and
`polars <https://docs.pola.rs/>`_ documentation are the useful companions to
this one.

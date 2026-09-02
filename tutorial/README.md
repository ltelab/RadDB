# RadDB tutorials

Five notebooks that walk through the whole workflow, in order. Each one is
self-contained.

| #   | notebook                                                | covers                                                                                      |
| --- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 1   | [Archiving](01_archiving.ipynb)                         | the storage model, the CRS contract, `archive()`, what lands on disk                        |
| 2   | [Opening and filtering](02_opening_and_filtering.ipynb) | `open()`, `filter()`, `sel()`, computed columns, converters                                 |
| 3   | [Areas of interest](03_area_of_interest.ipynb)          | bbox / point / polygon crops, cross-sections, the interactive map                           |
| 4   | [Plots](04_plots.ipynb)                                 | PPI, RHI, CAPPI, vertical cross-section                                                     |
| 5   | [Demo pipeline](05_demo_pipeline.ipynb)                 | the whole pipeline on data it downloads itself — two FMI radars, archived, plotted, cropped |

## The data

RadDB is network-agnostic — any xarray `DataTree` with the standard
[xradar](https://docs.openradarscience.org/projects/xradar/) layout works. The
notebooks use Finnish (FMI) volumes stored as Zarr, which FMI publishes as open
data, so every example can be reproduced.

The case they follow is a line of thunderstorms over southern Finland on
**17 June 2024**, sampled every quarter hour from 12:00 to 17:45 UTC on the three
radars `FANJ`, `FKOR` and `FKUO`. It peaks at 17:30, with 13,365 gates above
35 dBZ and a maximum of 56.7 dBZ. Notebook 5 downloads a second case — an even
stronger storm on 10 August 2024 — straight from FMI's public bucket.

One detail matters when you pick your own times: **FMI cycles three different
scan task sets over 15 minutes**, and only `:00`, `:15`, `:30` and `:45` are the
full 13-sweep volume. RadDB stores one geometry per radar, so mixing the three
task sets is refused as a scan-strategy change.

## Running them yourself

Point the notebooks at your own data by editing the configuration cell at the
top of each one:

```python
FMI_DIR = Path("/path/to/FMI_datatree_zarr")  # radars FANJ, FKOR, FKUO
ARCHIVE_DIR = Path("/tmp/raddb_tutorial_archive")  # where to write
```

Then run notebook 1 first — it creates the archive the others read.

Notebook 5 needs no local data at all: it downloads two public volumes over HTTP
and adds them to the same archive.

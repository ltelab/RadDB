"""
RadDB Package — Generic radar data archiving and reconstruction.

RadDB archives xarray DataTree volumes as Parquet files with an efficient
LUT-based layout.  It is network-agnostic: any DataTree with the standard
xradar coordinate layout can be archived and reconstructed (FMI, NEXRAD, ...).

Network-specific ingestion code — readers for a national archive's own raw
format — belongs in a separate package and is never imported here.
"""

from __future__ import annotations

import contextlib
import os
from importlib.metadata import PackageNotFoundError, version

# PROJ data directory — MUST stay the first raddb import: it repairs a
# PROJ_DATA/PROJ_LIB inherited from another environment before pyproj is
# imported (by geopandas, cartopy, raddb.lut, ...).
from raddb._proj import PROJ_DATA

# Discovery functions
from raddb.discovery import find_datatree_files

# Helper functions
# Filtering & archiving functions
from raddb.helper import (
    FILTER_LOGICS,
    RADAR_ALPHABET,
    RADAR_CODE_LEN,
    StageTimer,
    check_dataframe,
    filter_df,
    filter_dt,
    is_valid_radar_name,
    list_sweep_names,
    normalize_radar_name,
    read_parquet_files,
)

# I/O functions
from raddb.io_core import (
    add_feature_to_df,
    add_feature_to_dt,
    archive_multiple_volumes,
    archive_volume,
    archive_volumes_multi_radar,
    dataframe_to_datatree,
    datatree_to_dataframe,
    datatree_to_dataset,
    datatree_to_parquet,
    join_labels_with_lut,
    labels_to_dataframe,
    open_any_datatree,
    parquet_to_dataframe,
    parquet_to_datatree,
    reconstruct_datatree,
    reconstruct_sweep_dataset,
    scan_polar_parquet,
)

# LUT functions
from raddb.lut import (
    AZIMUTH_SCALE,
    DEFAULT_BEAMWIDTH_DEG,
    GATE_ID_RADAR_BASE,
    GATE_ID_VERSION,
    LUT_FILES,
    MAX_RADAR_CODE,
    RADAR_TO_IDX,
    add_lut_projection,
    antenna_vectors_to_cartesian,
    azimuth_grid_tolerance,
    build_gate_planes,
    cappi_chords,
    cartesian_to_geographic,
    compute_gate_xyz,
    compute_sweep_corners,
    decode_gate_radars,
    decode_radar_code,
    encode_radar_code,
    ensure_gate_planes,
    gate_corner_table,
    generate_gate_id,
    generate_lut_from_datatree,
    get_full_sweep_index,
    load_azimuth_grids,
    load_plane_nodes,
    load_radar_info,
    load_radar_lut,
    lut_file_path,
    nominal_azimuth_grid,
    snap_azimuths_to_grid,
)

# High-level interface
from raddb.main import RadDB

# Plotting functions
from raddb.viz.plot import (
    plot_cappi,
    plot_cross_section,
    plot_ppi,
    plot_rhi,
    plot_vcs,
)

__all__ = [
    # High-level API
    "RadDB",
    # Environment
    "PROJ_DATA",
    # Helper functions
    "read_parquet_files",
    "check_dataframe",
    "list_sweep_names",
    "normalize_radar_name",
    "is_valid_radar_name",
    "RADAR_ALPHABET",
    "RADAR_CODE_LEN",
    "StageTimer",
    # I/O functions
    "datatree_to_dataset",
    "datatree_to_dataframe",
    "datatree_to_parquet",
    "open_any_datatree",
    "parquet_to_dataframe",
    "parquet_to_datatree",
    "scan_polar_parquet",
    "dataframe_to_datatree",
    "labels_to_dataframe",
    "join_labels_with_lut",
    "reconstruct_sweep_dataset",
    "reconstruct_datatree",
    "add_feature_to_df",
    "add_feature_to_dt",
    # Discovery functions
    "find_datatree_files",
    # LUT functions
    "RADAR_TO_IDX",
    "GATE_ID_RADAR_BASE",
    "GATE_ID_VERSION",
    "MAX_RADAR_CODE",
    "AZIMUTH_SCALE",
    "nominal_azimuth_grid",
    "snap_azimuths_to_grid",
    "azimuth_grid_tolerance",
    "load_azimuth_grids",
    "encode_radar_code",
    "decode_radar_code",
    "decode_gate_radars",
    "DEFAULT_BEAMWIDTH_DEG",
    "LUT_FILES",
    "antenna_vectors_to_cartesian",
    "build_gate_planes",
    "cappi_chords",
    "cartesian_to_geographic",
    "compute_gate_xyz",
    "compute_sweep_corners",
    "ensure_gate_planes",
    "gate_corner_table",
    "generate_gate_id",
    "generate_lut_from_datatree",
    "load_plane_nodes",
    "load_radar_lut",
    "load_radar_info",
    "lut_file_path",
    "get_full_sweep_index",
    "add_lut_projection",
    # Pipeline functions
    "FILTER_LOGICS",
    "filter_df",
    "filter_dt",
    "archive_volume",
    "archive_multiple_volumes",
    "archive_volumes_multi_radar",
    # Plotting functions
    "plot_ppi",
    "plot_rhi",
    "plot_cappi",
    "plot_vcs",
    "plot_cross_section",
]

_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("raddb")

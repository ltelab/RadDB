"""Tests for :mod:`raddb._proj` — the PROJ data-directory repair.

A ``PROJ_DATA``/``PROJ_LIB`` inherited from another environment passes pyproj's
"does proj.db exist?" check but ships the wrong PROJ version, so every CRS lookup raises
``no database context specified``.  That breaks ``RadDB.crs``, ``to_geopandas``, the
``crop_*`` family and the LUT's ``x_*``/``y_*`` columns — silently, at import time.

Every test here restores the environment through ``monkeypatch`` so the repair does not
leak into the rest of the suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from raddb._proj import PROJ_DATA, fix_foreign_proj_data

# This interpreter's own bundled PROJ data directory (may not exist).
OWN_PROJ_DIR = Path(sys.prefix) / "share" / "proj"


def test_fix_foreign_proj_data(monkeypatch, tmp_path):
    """The four branches of the repair, in one place.

    Unset stays unset; a foreign directory is replaced or dropped; a directory already
    inside ``sys.prefix`` is left untouched.
    """
    # 1. Nothing set -> nothing done.
    monkeypatch.delenv("PROJ_DATA", raising=False)
    monkeypatch.delenv("PROJ_LIB", raising=False)
    assert fix_foreign_proj_data() is None
    assert "PROJ_DATA" not in os.environ

    # 2. A foreign directory is never kept.
    foreign = tmp_path / "foreign_proj"
    foreign.mkdir()
    monkeypatch.setenv("PROJ_DATA", str(foreign))
    monkeypatch.delenv("PROJ_LIB", raising=False)
    result = fix_foreign_proj_data()
    assert os.environ.get("PROJ_DATA") != str(foreign)
    if (OWN_PROJ_DIR / "proj.db").is_file():
        # 3a. Repointed at this interpreter's own data, in both variables.
        assert result == str(OWN_PROJ_DIR)
        assert os.environ["PROJ_DATA"] == os.environ["PROJ_LIB"] == str(OWN_PROJ_DIR)
    else:
        # 3b. No own data to point at, so the foreign variables are dropped entirely
        #     and pyproj falls back to the data it bundles.
        assert result is None
        assert "PROJ_DATA" not in os.environ
        assert "PROJ_LIB" not in os.environ

    # 4. An in-prefix directory is this environment's own -> left alone.
    if (OWN_PROJ_DIR / "proj.db").is_file():
        monkeypatch.setenv("PROJ_DATA", str(OWN_PROJ_DIR))
        assert fix_foreign_proj_data() is None
        assert os.environ["PROJ_DATA"] == str(OWN_PROJ_DIR)


def test_legacy_proj_lib_is_honored(monkeypatch, tmp_path):
    """``PROJ_LIB`` alone (the pre-8.0 name) triggers the repair too."""
    foreign = tmp_path / "legacy_proj"
    foreign.mkdir()
    monkeypatch.delenv("PROJ_DATA", raising=False)
    monkeypatch.setenv("PROJ_LIB", str(foreign))

    fix_foreign_proj_data()

    assert os.environ.get("PROJ_LIB") != str(foreign)


def test_an_unresolvable_directory_does_not_raise(monkeypatch, tmp_path):
    """``Path.resolve()`` on a broken value must not escape as an ``OSError``.

    A symlink loop is the cheap way to make ``resolve()`` raise ``ELOOP``; the repair
    suppresses it and falls through to the "replace the foreign value" branch.
    """
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    monkeypatch.setenv("PROJ_DATA", str(loop))
    monkeypatch.delenv("PROJ_LIB", raising=False)

    fix_foreign_proj_data()  # suppressed internally; reaching here is the assertion

    assert os.environ.get("PROJ_DATA") != str(loop)


def test_module_level_proj_data_matches_the_environment():
    """``raddb.PROJ_DATA`` reports what the import-time repair actually did."""
    assert PROJ_DATA is None or Path(PROJ_DATA).is_dir()
    if PROJ_DATA is not None:
        assert Path(PROJ_DATA).is_relative_to(Path(sys.prefix).resolve())


def test_a_crs_lookup_works_after_import():
    """The point of the whole module: EPSG lookups must resolve."""
    pyproj = pytest.importorskip("pyproj")

    assert pyproj.CRS.from_epsg(4326).name
    assert pyproj.CRS.from_epsg(2056).is_projected

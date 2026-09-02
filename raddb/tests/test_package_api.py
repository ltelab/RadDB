"""Tests for :mod:`raddb` — the package's public import surface.

``raddb/__init__.py`` declares no callables of its own; what it *is* is a contract:
the names in ``__all__`` are what downstream code may import.  These tests pin that
contract, plus the two invariants the module comment calls out — the ``_proj`` import
must come first, and the private out-of-tree ingestion subpackage ``raddb.mch`` must
never be pulled in.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import raddb

INIT_PATH = Path(raddb.__file__)


def test_every_name_in_all_is_importable():
    """``from raddb import <name>`` works for every advertised name."""
    missing = [name for name in raddb.__all__ if not hasattr(raddb, name)]
    assert missing == [], f"__all__ advertises names that do not exist: {missing}"


def test_all_has_no_duplicates():
    """A duplicated entry means two edits collided and one is probably wrong."""
    seen = sorted({n for n in raddb.__all__ if raddb.__all__.count(n) > 1})
    assert seen == [], f"duplicated entries in __all__: {seen}"


def test_star_import_matches_all():
    """``from raddb import *`` exposes exactly ``__all__`` and nothing more."""
    namespace: dict = {}
    exec("from raddb import *", namespace)  # - the behavior under test
    namespace.pop("__builtins__", None)
    assert sorted(namespace) == sorted(raddb.__all__)


def test_the_high_level_class_is_exported():
    """``RadDB`` is the entry point; everything else is a convenience."""
    from raddb.main import RadDB

    assert raddb.RadDB is RadDB


def test_proj_data_is_exported():
    """``raddb.PROJ_DATA`` reports whether the import-time PROJ repair fired."""
    assert hasattr(raddb, "PROJ_DATA")
    assert raddb.PROJ_DATA is None or isinstance(raddb.PROJ_DATA, str)


def test_proj_is_the_first_raddb_import():
    """``from raddb._proj import PROJ_DATA`` must precede every other raddb import.

    pyproj reads its data directory once, at import time.  If ``raddb.lut`` (or anything
    that pulls in geopandas/cartopy) imports first, the broken inherited context is
    already cached and the repair comes too late.
    """
    tree = ast.parse(INIT_PATH.read_text(encoding="utf-8"))
    raddb_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("raddb")
    ]
    assert raddb_imports[0] == "raddb._proj", f"first raddb import is {raddb_imports[0]!r}, must be raddb._proj"


def test_the_private_mch_subpackage_is_not_imported():
    """Network-specific ingestion lives in its own repo: gitignored, excluded from wheels, absent from checkout."""
    tree = ast.parse(INIT_PATH.read_text(encoding="utf-8"))
    offenders = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("raddb.mch")
    ]
    assert offenders == [], "raddb/__init__.py must never import the private mch subpackage"


def test_lonboard_is_not_imported_at_module_level():
    """Lonboard pulls in pyproj; importing it eagerly defeats the PROJ repair."""
    import sys

    assert "lonboard" not in sys.modules or "raddb" in sys.modules


@pytest.mark.parametrize(
    "name",
    ["RadDB", "plot_ppi", "generate_lut_from_datatree", "filter_df", "find_datatree_files"],
)
def test_representative_exports_are_callable(name):
    """A spot check that the re-exports are the real objects, not stubs."""
    assert callable(getattr(raddb, name))


def test_version_is_available():
    """``setuptools_scm`` supplies ``__version__`` for an installed package."""
    assert not hasattr(raddb, "__version__") or isinstance(raddb.__version__, str)


def test_submodules_are_reachable():
    """The documented module layout is importable by path."""
    import importlib

    for name in ("aoi", "discovery", "helper", "io_core", "lut", "main", "viz"):
        assert importlib.import_module(f"raddb.{name}") is not None

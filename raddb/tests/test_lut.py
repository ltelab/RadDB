"""Tests for :mod:`raddb.lut` — LUT generation, gate geometry, ``gate_id`` and CRS checks.

Four themes, each written to pin something that was once silently wrong.

**The base-36 radar code.** ``gate_id`` embeds the zero-padded four-character radar name,
so an archive is self-describing and two archives concatenate.  Encoding v1 numbered
radars ``A=0 .. Z=25``; the two disagree for every name and nothing detects a v1 archive
any more — that is a deliberate trade-off, pinned below.

**The nominal azimuth grid.** The LUT stores a radar's *scan strategy*, not one volume's
measured azimuths.  Half-up rounding, a circular seam, full-precision comparison and a
minimum rotation coverage are all load-bearing; each has its own test.

**Gate geometry.** A gate is a frustum, not a box: corners 5-8 enclose strictly more area
than 1-4.  The first range bin is degenerate — its inner edge clips to r=0, so five
distinct corners instead of eight.

**The CRS contract.** Validity is *measured*, not declared: a 100 km geodesic is projected
in eight directions and compared with the truth.  Metadata alone would pass EPSG:3857,
which reports a 100 km baseline as 216 km in Finland.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyproj
import pytest
import shapely
import yaml

from raddb.helper import normalize_radar_name
from raddb.lut import (
    AZIMUTH_SCALE,
    AZIMUTH_STEPS,
    CRS_REFUSE_PCT,
    DEFAULT_BEAMWIDTH_DEG,
    GATE_ID_RADAR_BASE,
    LEGACY_RADAR_TO_IDX,
    LUT_FILES,
    MAX_RADAR_CODE,
    RADAR_ALPHABET,
    RADAR_CODE_LEN,
    _round_half_up,
    add_lut_projection,
    antenna_vectors_to_cartesian,
    azimuth_grid_tolerance,
    build_gate_planes,
    cappi_chords,
    cartesian_to_geographic,
    compute_corners_from_lut,
    compute_gate_xyz,
    compute_sweep_corners,
    crs_distance_error,
    decode_gate_ids,
    decode_gate_radars,
    decode_radar_code,
    encode_gate_ids,
    encode_radar_code,
    ensure_gate_planes,
    gate_corner_table,
    gate_polygons_geoarrow,
    generate_gate_id,
    generate_lut_from_datatree,
    geoarrow_field,
    get_full_sweep_index,
    load_azimuth_grids,
    load_plane_nodes,
    load_radar_info,
    load_radar_lut,
    load_sweep_corners,
    lut_file_path,
    nominal_azimuth_grid,
    save_sweep_corners,
    snap_azimuths_to_grid,
    suggest_crs,
    validate_crs_for_site,
)
from raddb.tests.conftest import (
    ANTENNA_BIAS,
    FI_SITE,
    FMI_EPSG,
    NEXRAD_SPREAD,
    RADAR,
    US_EPSG,
    US_SITE,
    build_datatree,
    jitter_azimuths,
    relocate,
    retime,
)

N_AZ, N_RNG, N_SWEEPS = 12, 24, 2
N_GATES = N_AZ * N_RNG * N_SWEEPS

REAL_N_AZ, REAL_N_RNG, REAL_N_SWEEPS = 360, 200, 2
REAL_N_GATES = REAL_N_AZ * REAL_N_RNG * REAL_N_SWEEPS


@pytest.fixture
def lut_base(tmp_path, make_datatree):
    """A base path holding radar ``A``'s complete five-file LUT directory."""
    generate_lut_from_datatree(
        make_datatree(),
        radar=RADAR,
        output_base_path=str(tmp_path),
        projection_epsg=FMI_EPSG,
    )
    return tmp_path


@pytest.fixture(scope="session")
def real_lut_base(tmp_path_factory):
    """A realistically-sampled LUT (1 degree azimuths, 200 range bins), built once.

    Needed by two groups of tests.  **Geometry**: a gate footprint is a straight-sided
    quad, so its outer chord cuts inside the true arc; the centroid falls outside its own
    footprint beyond ``r ~ dR*cos(h)/(1-cos(h))``, which is ~11.7 km at the small
    fixture's 30 degree spacing but ~5400 km at 1 degree.  **File sizes**: bytes-per-gate
    is meaningless on a 576-gate file where the fixed parquet footer dominates.
    """
    base = tmp_path_factory.mktemp("realistic_lut")
    generate_lut_from_datatree(
        build_datatree(n_az=REAL_N_AZ, n_rng=REAL_N_RNG, n_sweeps=REAL_N_SWEEPS),
        radar=RADAR,
        output_base_path=str(base),
        projection_epsg=FMI_EPSG,
    )
    return base


def _face_area(table: pl.DataFrame, corners) -> np.ndarray:
    """Planar polygon area of a 4-corner face in 3-D, by Newell's method."""
    pts = np.stack(
        [
            np.stack(
                [table[f"x_{k}"].to_numpy(), table[f"y_{k}"].to_numpy(), table[f"z_rel_{k}"].to_numpy()],
                axis=1,
            )
            for k in corners
        ],
        axis=1,
    )
    normal = np.zeros((pts.shape[0], 3))
    for i in range(4):
        normal += np.cross(pts[:, i], pts[:, (i + 1) % 4])
    return 0.5 * np.linalg.norm(normal, axis=1)


# ---------------------------------------------------------------------------
# encode_radar_code / decode_radar_code
# ---------------------------------------------------------------------------


def test_encode_radar_code():
    """Base-36 over the zero-padded four-character name."""
    assert encode_radar_code("A") == 10  # "000A"
    assert encode_radar_code("L") == 21
    assert encode_radar_code("KTLX") == 971_493
    assert encode_radar_code("0") == 0
    assert encode_radar_code("ZZZZ") == MAX_RADAR_CODE


def test_decode_radar_code():
    """The inverse, with the padding stripped back off."""
    assert decode_radar_code(10) == "A"
    assert decode_radar_code(971_493) == "KTLX"
    assert decode_radar_code(MAX_RADAR_CODE) == "ZZZZ"


def test_the_code_space_is_36_to_the_fourth():
    """Four characters is what the ``gate_id`` layout can hold."""
    assert MAX_RADAR_CODE == 36**RADAR_CODE_LEN - 1 == 1_679_615


def test_every_gate_id_fits_int64():
    """The largest possible id sits 5.5x under the int64 ceiling."""
    largest = MAX_RADAR_CODE * GATE_ID_RADAR_BASE + (GATE_ID_RADAR_BASE - 1)

    assert largest < np.iinfo(np.int64).max
    assert np.int64(largest) == largest


def test_five_characters_would_not_fit():
    """Documents why ``RADAR_CODE_LEN`` is 4: ``36**5`` blows the int64 budget."""
    budget = (np.iinfo(np.int64).max - (GATE_ID_RADAR_BASE - 1)) // GATE_ID_RADAR_BASE

    assert 36**4 - 1 <= budget < 36**5 - 1


def test_decode_is_injective_over_the_whole_code_space():
    """Distinct codes never name the same radar."""
    names = {decode_radar_code(c) for c in range(MAX_RADAR_CODE + 1)}

    assert len(names) == MAX_RADAR_CODE + 1 == 1_679_616


def test_the_round_trip_is_total_over_canonical_names():
    """Every name that is its own canonical form survives encode then decode."""
    names = {decode_radar_code(c) for c in range(MAX_RADAR_CODE + 1)}
    canonical = {n for n in names if normalize_radar_name(n) == n}

    assert len(canonical) == 1_679_580  # all but the 36 ML? aliases
    assert all(decode_radar_code(encode_radar_code(n)) == n for n in canonical)


def test_only_the_ml_aliases_break_the_code_round_trip():
    """``MLA`` normalizes to ``A`` first, so 36 codes can never be emitted."""
    broken = [c for c in range(MAX_RADAR_CODE + 1) if encode_radar_code(decode_radar_code(c)) != c]

    assert len(broken) == 36


def test_zero_padding_is_transparent():
    """Leading zeros are ``gate_id`` padding, never part of the name."""
    assert encode_radar_code("A") == encode_radar_code("000A") == encode_radar_code("0A")


def test_alphabet_positions_define_the_values():
    """Each character's value is its index in ``RADAR_ALPHABET``."""
    for i, char in enumerate(RADAR_ALPHABET):
        assert encode_radar_code(char.rjust(RADAR_CODE_LEN, "0")) == i


@pytest.mark.parametrize("code", [-1, MAX_RADAR_CODE + 1, 10**9])
def test_decode_radar_code_rejects_an_out_of_range_code(code):
    """Outside the 36**4 space there is no name to return."""
    with pytest.raises(ValueError, match="names no radar"):
        decode_radar_code(code)


def test_more_than_26_radars_stay_distinct():
    """The point of encoding v2: no 26-radar ceiling."""
    names = [f"K{a}{b}" for a in "ABCDE" for b in "ABCDEFGHIJ"]  # 50 sites

    codes = [encode_radar_code(n) for n in names]

    assert len(set(codes)) == len(names) == 50
    assert sorted(decode_radar_code(c) for c in codes) == sorted(names)


def test_a_v1_archive_is_read_without_complaint(tmp_path, make_datatree):
    """The deliberate trade-off, pinned so it stays visible.

    ``info.yaml`` used to record ``gate_id_version`` and ``load_radar_info`` raised
    ``OutdatedGateIdError`` on a v1 archive.  Both were removed: only v2 is ever
    produced, so the key carried no information about anything new.

    The cost is real — a v1 archive now loads **silently and decodes to the wrong
    radar**.  This builds one by rewriting the ids back to the v1 encoding and reads it,
    so the trade-off stays visible rather than being rediscovered.
    """
    from raddb.main import RadDB

    base = tmp_path / "archive"
    db = RadDB(archive_dir=str(base), crs=FMI_EPSG)
    db.archive(datatree=make_datatree(), radar="L")

    # Rewrite every gate_id back to the v1 encoding (A=0 .. Z=25), as if written long ago.
    delta = (LEGACY_RADAR_TO_IDX["L"] - encode_radar_code("L")) * GATE_ID_RADAR_BASE
    for path in [base / "L" / "LUT" / "L_LUT.parquet", *sorted((base / "L").rglob("*_POL.parquet"))]:
        pl.read_parquet(path).with_columns((pl.col("gate_id") + delta).alias("gate_id")).write_parquet(path)

    assert "gate_id_version" not in db.get_radar_info("L")
    # No error, no warning — and a v1 'L' (code 11) reads as 'B'.
    assert decode_gate_radars(db.open(radars="L").data["gate_id"].to_numpy()) == ["B"]


# ---------------------------------------------------------------------------
# gate_id encoding and decoding
# ---------------------------------------------------------------------------


def test_generate_gate_id():
    """``radar_code * 1e12 + sweep * 1e10 + az*10 * 1e6 + range_m``, decimal so it reads."""
    gid = generate_gate_id("A", sweep=1, azimuth=45.5, range_m=1000)

    assert gid == 10 * 10**12 + 1 * 10**10 + 455 * 10**6 + 1000


def test_encode_gate_ids():
    """The vectorized form, over parallel arrays."""
    ids = encode_gate_ids("A", np.array([1, 2]), np.array([45.5, 90.0]), np.array([1000.0, 2000.0]))

    assert ids.dtype == np.int64
    assert ids[0] == generate_gate_id("A", 1, 45.5, 1000)
    assert ids[1] == generate_gate_id("A", 2, 90.0, 2000)


def test_decode_gate_ids(lut_base):
    """Decoding inverts the encoding for every gate the LUT holds."""
    lut = load_radar_lut(RADAR, lut_base)

    sweeps, azimuths, ranges = decode_gate_ids(lut["gate_id"].to_numpy())

    assert np.array_equal(sweeps, lut["sweep"].to_numpy().astype(np.int64))
    assert np.allclose(azimuths, np.round(lut["azimuth"].to_numpy() * 10) / 10)
    assert np.allclose(ranges, lut["range"].to_numpy().astype(np.int64))


def test_decode_gate_radars(lut_base):
    """The radar name is recovered from the integers alone, with no registry file."""
    lut = load_radar_lut(RADAR, lut_base)

    assert decode_gate_radars(lut["gate_id"].to_numpy()) == [RADAR]


def test_the_radar_field_is_the_leading_one():
    """``gate_id // GATE_ID_RADAR_BASE`` is exactly the radar code."""
    gid = encode_gate_ids("KTLX", 3, np.array([91.4]), np.array([12_500.0]))

    assert gid[0] // GATE_ID_RADAR_BASE == encode_radar_code("KTLX")


def test_the_low_fields_are_independent_of_the_radar():
    """Only the leading field differs between radars — what the v1 migration relies on.

    ``migrate_gate_id_v2`` is one integer offset per radar precisely because sweep,
    azimuth and range occupy fields the radar code never touches.
    """
    azimuths, ranges = np.array([91.4, 270.0]), np.array([12_500.0, 240_000.0])

    a = encode_gate_ids("A", 3, azimuths, ranges)
    ktlx = encode_gate_ids("KTLX", 3, azimuths, ranges)

    delta = (encode_radar_code("KTLX") - encode_radar_code("A")) * GATE_ID_RADAR_BASE
    assert np.array_equal(ktlx - a, np.full(2, delta))


def test_decode_gate_ids_round_trips_the_extremes():
    """0 degrees, the 359.9 seam, r=0 and a 999,999 m range all survive."""
    azimuths = np.array([0.0, 91.4, 359.9])
    ranges = np.array([0.0, 12_500.0, 999_999.0])
    gid = encode_gate_ids("KTLX", 7, azimuths, ranges)

    sweeps, got_azimuths, got_ranges = decode_gate_ids(gid)

    assert np.array_equal(sweeps, np.full(3, 7))
    assert np.allclose(got_azimuths, azimuths)
    assert np.allclose(got_ranges, ranges)
    assert decode_gate_radars(gid) == ["KTLX"]


def test_decode_gate_radars_on_an_empty_array():
    """An empty frame names no radars, and must not raise."""
    assert decode_gate_radars(np.array([], dtype=np.int64)) == []


def test_decode_gate_radars_skips_an_unknown_code(caplog):
    """A corrupt id is warned about and dropped, not turned into a nonsense name."""
    bogus = np.array([(MAX_RADAR_CODE + 5) * GATE_ID_RADAR_BASE], dtype=np.int64)

    with caplog.at_level("WARNING"):
        assert decode_gate_radars(bogus) == []

    assert "names no radar" in caplog.text


def test_decode_gate_radars_spans_a_concatenated_archive():
    """Two archives can be concatenated because each id names its own radar."""
    ids = np.concatenate(
        [
            encode_gate_ids("A", np.array([1]), np.array([0.5]), np.array([1000.0])),
            encode_gate_ids("KTLX", np.array([1]), np.array([0.5]), np.array([1000.0])),
        ],
    )

    assert sorted(decode_gate_radars(ids)) == ["A", "KTLX"]


# ---------------------------------------------------------------------------
# nominal_azimuth_grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n_rays", "step_tenths"), [(360, 10), (720, 5), (180, 20)])
def test_nominal_azimuth_grid(n_rays, step_tenths):
    """One rule, no per-network constant: the spacing follows the ray count."""
    grid = nominal_azimuth_grid(np.arange(n_rays) * (360 / n_rays) + 0.25)

    assert grid.size == n_rays
    assert np.all(np.diff(grid) == step_tenths)


def test_the_grid_is_recovered_from_jittered_rays():
    """The measured angles drift; the derived strategy does not."""
    nominal = np.arange(360) + 0.5

    grid = nominal_azimuth_grid(jitter_azimuths(nominal, np.random.default_rng(0), ANTENNA_BIAS))

    assert np.array_equal(grid, np.round(nominal * AZIMUTH_SCALE).astype(np.int64))


def test_the_grid_is_stable_across_volumes():
    """The whole point: different rotations must derive the *same* grid."""
    rng = np.random.default_rng(1)
    nominal = np.arange(360) + 0.5

    grids = [nominal_azimuth_grid(jitter_azimuths(nominal, rng, ANTENNA_BIAS)) for _ in range(25)]

    assert all(np.array_equal(g, grids[0]) for g in grids)


def test_rounding_is_half_up_not_bankers():
    """A 720-ray grid puts every center on ``x.x5``.

    numpy's banker's rounding would turn a uniform 0.5 degree grid into an alternating
    0.4/0.6 one.
    """
    grid = nominal_azimuth_grid(np.arange(720) * 0.5 + 0.25)

    assert np.all(np.diff(grid) == 5)


def test_round_half_up_ties_away_from_even():
    """The helper itself, where banker's rounding would differ."""
    np.testing.assert_array_equal(_round_half_up(np.array([0.5, 1.5, 2.5, 3.5])), np.array([1, 2, 3, 4]))


def test_the_offset_is_a_circular_mean():
    """Rays straddling 0 degrees must not drag the offset to mid-step."""
    grid = nominal_azimuth_grid(jitter_azimuths(np.arange(360) * 1.0, np.random.default_rng(2), 0.0, 0.02))

    assert np.array_equal(grid, np.arange(360) * 10)


def test_a_spacing_finer_than_the_resolution_is_refused():
    """``gate_id`` resolves 0.1 degrees; a finer grid could not be represented."""
    with pytest.raises(ValueError, match="finer than"):
        nominal_azimuth_grid(np.arange(7200) * 0.05)


def test_an_empty_sweep_is_refused():
    """There is no strategy to derive from no rays."""
    with pytest.raises(ValueError, match="no rays"):
        nominal_azimuth_grid([])


def test_a_sector_scan_is_refused():
    """90 rays over 90 degrees would silently get a 4 degree grid and collapse to 23."""
    with pytest.raises(ValueError, match="full rotation"):
        nominal_azimuth_grid(np.arange(90, 180, 1.0))


def test_a_sweep_with_a_large_gap_is_refused():
    """Two sectors are not a rotation either."""
    azimuths = np.concatenate([np.arange(0, 120, 1.0), np.arange(240, 360, 1.0)])

    with pytest.raises(ValueError, match="full rotation"):
        nominal_azimuth_grid(azimuths)


@pytest.mark.parametrize("dropped", [[7], [7, 8], [0, 359], [3, 100, 250]])
def test_a_rotation_with_holes_keeps_the_full_grid(dropped):
    """718 of 720 is a rotation with holes, not a 0.5014 degree scan strategy.

    The LUT is written for the whole rotation, missing rays included, so a later volume
    that does record them still joins.
    """
    nominal = np.arange(720) * 0.5 + 0.25

    grid = nominal_azimuth_grid(np.delete(nominal, dropped))

    assert grid.size == 720
    assert np.all(np.diff(grid) == 5)
    assert np.array_equal(grid, nominal_azimuth_grid(nominal))


def test_holes_survive_antenna_drift():
    """WSR-88D drift plus two dropped rays.

    The grid is the same rotation, but not necessarily the same integers: a 720-ray grid
    is centered on ``x.x5``, exactly the 0.1 degree rounding boundary, so the ~0.002
    degrees the two ray sets differ by can tip the whole grid one tenth either way.
    That is a tenth against a half-spacing tolerance of 0.25 degrees, so every ray still
    snaps to its own point.
    """
    nominal = np.arange(720) * 0.5 + 0.25
    recorded = np.delete(jitter_azimuths(nominal, np.random.default_rng(7), 0.0, NEXRAD_SPREAD), [11, 12])

    grid = nominal_azimuth_grid(recorded)

    assert grid.size == 720
    assert np.all(np.diff(grid) == 5)
    shift = (grid - nominal_azimuth_grid(nominal) + AZIMUTH_STEPS // 2) % AZIMUTH_STEPS
    assert np.all(np.abs(shift - AZIMUTH_STEPS // 2) <= 1)
    assert snap_azimuths_to_grid(recorded, grid)[1].max() <= azimuth_grid_tolerance(grid)


def test_too_many_holes_is_not_a_rotation():
    """Past ``MIN_ROTATION_COVERAGE`` it is indistinguishable from a sector scan."""
    recorded = np.delete(np.arange(360) + 0.5, np.arange(0, 100))  # 260 of 360

    with pytest.raises(ValueError, match="full rotation"):
        nominal_azimuth_grid(recorded)


# ---------------------------------------------------------------------------
# snap_azimuths_to_grid / azimuth_grid_tolerance
# ---------------------------------------------------------------------------


@pytest.fixture
def degree_grid():
    """A 360-point grid centered on 0.5, 1.5 ... 359.5 degrees, in tenths."""
    return nominal_azimuth_grid(np.arange(360) + 0.5)


def test_snap_azimuths_to_grid(degree_grid):
    """Each ray moves to the nearest grid point, and the distance comes back with it."""
    snapped, distance = snap_azimuths_to_grid([0.49, 0.51, 1.44, 1.56], degree_grid)

    assert list(snapped) == [5, 5, 15, 15]
    assert np.allclose(distance, [0.1, 0.1, 0.6, 0.6])


@pytest.mark.parametrize(("azimuth", "expected"), [(359.97, 3595), (0.02, 5), (359.60, 3595), (0.60, 5)])
def test_the_seam_is_measured_the_short_way(degree_grid, azimuth, expected):
    """The grid is a circle: distance across 0/360 goes the short way, not through 180."""
    assert snap_azimuths_to_grid([azimuth], degree_grid)[0][0] == expected


def test_a_ray_below_360_can_snap_to_a_grid_point_at_zero():
    """With rays centered on 0, 1, 2 ..., 359.7 degrees belongs to 0.0, not 359.0."""
    grid = nominal_azimuth_grid(np.arange(360) * 1.0)

    snapped, distance = snap_azimuths_to_grid([359.7, 359.4, 0.3], grid)

    assert grid[0] == 0
    assert list(snapped) == [0, 3590, 0]
    assert np.allclose(distance, [3.0, 4.0, 3.0])


def test_full_precision_decides_the_match(degree_grid):
    """Rounding to 0.1 degrees first would make 0.02 an exact tie between 0.5 and 359.5."""
    assert snap_azimuths_to_grid([0.02], degree_grid)[0][0] == 5


def test_snapping_is_a_bijection_under_real_drift(degree_grid):
    """No two rays may collapse onto one grid point, or gates would be lost."""
    drifted = jitter_azimuths(np.arange(360) + 0.5, np.random.default_rng(3), ANTENNA_BIAS)

    snapped, distance = snap_azimuths_to_grid(drifted, degree_grid)

    assert np.unique(snapped).size == 360
    assert distance.max() <= azimuth_grid_tolerance(degree_grid)


def test_snapping_survives_nexrad_scale_drift():
    """720 super-resolution rays at 0.045 degree spread."""
    grid = nominal_azimuth_grid(np.arange(720) * 0.5 + 0.25)
    drifted = jitter_azimuths(np.arange(720) * 0.5 + 0.25, np.random.default_rng(4), 0.0, NEXRAD_SPREAD)

    snapped, distance = snap_azimuths_to_grid(drifted, grid)

    assert np.unique(snapped).size == 720
    assert distance.max() <= azimuth_grid_tolerance(grid)


def test_snapped_output_stays_inside_one_turn(degree_grid):
    """360.0 must not become 3600 tenths."""
    snapped, _ = snap_azimuths_to_grid([0.0, 180.0, 359.999, 360.0], degree_grid)

    assert np.all((snapped >= 0) & (snapped < AZIMUTH_STEPS))


def test_snap_azimuths_to_grid_rejects_an_empty_grid():
    """Nothing to snap onto is an error, not a silent pass-through."""
    with pytest.raises(ValueError, match="empty azimuth grid"):
        snap_azimuths_to_grid([1.0], [])


def test_azimuth_grid_tolerance(degree_grid):
    """Half a ray spacing — the most a gate may legitimately move."""
    assert azimuth_grid_tolerance(degree_grid) == pytest.approx(5.0)
    assert azimuth_grid_tolerance(nominal_azimuth_grid(np.arange(720) * 0.5 + 0.25)) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# load_azimuth_grids
# ---------------------------------------------------------------------------


def test_load_azimuth_grids(tmp_path):
    """Read back from the **LUT parquet**, not from ``info.yaml``, which no longer says."""
    generate_lut_from_datatree(
        build_datatree(n_az=360, n_rng=20, n_sweeps=3),
        radar=RADAR,
        output_base_path=str(tmp_path),
        projection_epsg=FMI_EPSG,
    )

    grids = load_azimuth_grids(RADAR, tmp_path)

    assert grids is not None
    assert set(grids) == {1, 2, 3}
    for grid in grids.values():
        assert grid.size == 360
        assert np.all(np.diff(grid) == 10)

    info = yaml.safe_load((tmp_path / RADAR / "LUT" / f"{RADAR}_info.yaml").read_text())
    assert "azimuths" not in info["sweeps"][1]


def test_load_azimuth_grids_returns_none_without_a_lut(tmp_path):
    """No LUT means no grid to snap onto — the measured azimuths stand."""
    assert load_azimuth_grids(RADAR, tmp_path) is None


def test_the_lut_azimuth_column_holds_the_nominal_grid(tmp_path):
    """The LUT stores the scan strategy, not one volume's measurements."""
    import pandas as pd

    drifted = retime(
        build_datatree(n_az=360, n_rng=20, n_sweeps=2),
        pd.Timestamp("2024-08-01 12:00:00"),
        np.random.default_rng(5),
        ANTENNA_BIAS,
    )
    generate_lut_from_datatree(
        drifted,
        radar=RADAR,
        output_base_path=str(tmp_path),
        projection_epsg=FMI_EPSG,
    )

    lut = load_radar_lut(RADAR, tmp_path)
    azimuths = np.unique(lut.filter(pl.col("sweep") == 1)["azimuth"].to_numpy())

    assert np.allclose(azimuths * AZIMUTH_SCALE, np.round(azimuths * AZIMUTH_SCALE))


# ---------------------------------------------------------------------------
# Beam geometry
# ---------------------------------------------------------------------------


def test_antenna_vectors_to_cartesian():
    """A ray at 0 degrees azimuth points north; at 90 degrees, east."""
    ranges = np.array([10_000.0])

    x_n, y_n, _ = antenna_vectors_to_cartesian(ranges, np.array([0.0]), np.array([0.0]))
    x_e, y_e, _ = antenna_vectors_to_cartesian(ranges, np.array([90.0]), np.array([0.0]))

    assert abs(x_n[0]) < 1.0
    assert y_n[0] == pytest.approx(10_000.0, rel=1e-3)
    assert x_e[0] == pytest.approx(10_000.0, rel=1e-3)
    assert abs(y_e[0]) < 1.0


def test_the_beam_rises_with_elevation_and_earth_curvature():
    """``ke=4/3`` everywhere; a horizontal beam still climbs from the curvature term."""
    _, _, z_flat = antenna_vectors_to_cartesian(np.array([100_000.0]), np.array([0.0]), np.array([0.0]))
    _, _, z_up = antenna_vectors_to_cartesian(np.array([100_000.0]), np.array([0.0]), np.array([5.0]))

    assert z_flat[0] > 0.0  # curvature alone lifts a 0-degree beam
    assert z_up[0] > z_flat[0]


def test_the_default_ke_is_four_thirds():
    """Archives generated with the old 1.25 default carry ~237 m of altitude error."""
    import inspect

    assert inspect.signature(antenna_vectors_to_cartesian).parameters["ke"].default == pytest.approx(4 / 3)
    assert inspect.signature(compute_gate_xyz).parameters["ke"].default == pytest.approx(4 / 3)
    assert inspect.signature(generate_lut_from_datatree).parameters["ke"].default == pytest.approx(4 / 3)


def test_compute_gate_xyz():
    """The meshed form: one position per (azimuth, range) pair."""
    x, y, z = compute_gate_xyz(np.array([1000.0, 2000.0]), np.array([0.0, 90.0]), np.array([0.5, 0.5]))

    assert x.shape == y.shape == z.shape
    assert np.isfinite(x).all()


def test_cartesian_to_geographic():
    """The radar's own position maps back to its own coordinates.

    Note the return order is ``(lat, lon, alt)`` — latitude first, unlike the
    ``(longitude, latitude)`` argument order used everywhere a *site* is named.
    """
    lat, lon, alt = cartesian_to_geographic(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        FI_SITE[1],
        FI_SITE[0],
        1000.0,
    )

    assert lon[0] == pytest.approx(FI_SITE[0], abs=1e-6)
    assert lat[0] == pytest.approx(FI_SITE[1], abs=1e-6)
    assert alt[0] == pytest.approx(1000.0)


def test_geographic_conversion_moves_north_for_positive_y():
    """A 10 km northward offset raises the latitude by ~0.09 degrees."""
    lat, _, _ = cartesian_to_geographic(
        np.array([0.0]),
        np.array([10_000.0]),
        np.array([0.0]),
        FI_SITE[1],
        FI_SITE[0],
        0.0,
    )

    assert lat[0] > FI_SITE[1]
    assert lat[0] - FI_SITE[1] == pytest.approx(0.09, abs=0.01)


# ---------------------------------------------------------------------------
# generate_lut_from_datatree — the five-file directory
# ---------------------------------------------------------------------------


def test_generate_lut_from_datatree(lut_base):
    """All five files are written, and none is empty."""
    lut_dir = lut_base / RADAR / "LUT"

    for kind, template in LUT_FILES.items():
        path = lut_dir / template.format(radar=RADAR)
        assert path.exists(), f"{kind} missing: {path.name}"
        assert path.stat().st_size > 0


def test_the_lut_directory_holds_exactly_the_five_files(lut_base):
    """Nothing else is written beside them."""
    lut_dir = lut_base / RADAR / "LUT"

    assert sorted(f.name for f in lut_dir.iterdir()) == sorted(t.format(radar=RADAR) for t in LUT_FILES.values())


def test_regenerating_backfills_the_lattices_without_rewriting_the_centroids(lut_base, make_datatree):
    """An archive predating the lattices regenerates them, keeping its LUT untouched.

    That matters because an archive's source volumes are often long gone.
    """
    lut_dir = lut_base / RADAR / "LUT"
    centroid_file = lut_dir / LUT_FILES["lut"].format(radar=RADAR)
    stamp = centroid_file.stat().st_mtime_ns
    for kind in ("h_plane", "v_plane", "corners"):
        (lut_dir / LUT_FILES[kind].format(radar=RADAR)).unlink()

    generate_lut_from_datatree(
        make_datatree(),
        radar=RADAR,
        output_base_path=str(lut_base),
        projection_epsg=FMI_EPSG,
    )

    for kind in ("h_plane", "v_plane", "corners"):
        assert (lut_dir / LUT_FILES[kind].format(radar=RADAR)).exists()
    assert centroid_file.stat().st_mtime_ns == stamp


def test_the_info_yaml_records_the_generation_parameters(lut_base):
    """Everything needed to reproduce the geometry, and nothing that went stale."""
    info = load_radar_info(RADAR, lut_base)

    for key in (
        "radar",
        "network",
        "latitude",
        "longitude",
        "altitude",
        "crs",
        "ke",
        "beamwidth_deg",
        "n_sweeps",
        "n_gates",
        "sweeps",
    ):
        assert key in info, f"missing info key {key!r}"
    assert info["ke"] == pytest.approx(4.0 / 3.0)
    assert info["beamwidth_deg"] == DEFAULT_BEAMWIDTH_DEG
    assert (info["n_gates"], info["n_sweeps"]) == (N_GATES, N_SWEEPS)
    assert info["crs"] == {"epsg": FMI_EPSG, "columns": ["x_3067", "y_3067"]}


def test_the_per_sweep_info_block(lut_base):
    """``dR``, ``azimuth_scale`` and ``azimuths`` were dropped deliberately.

    The first two were never read back, and the grid is recovered from the LUT parquet.
    """
    sweep = load_radar_info(RADAR, lut_base)["sweeps"][1]

    for key in ("n_azimuths", "n_ranges", "n_gates", "elevation", "range_resolution", "range_start"):
        assert key in sweep, f"missing per-sweep key {key!r}"
    assert sweep["n_gates"] == sweep["n_azimuths"] * sweep["n_ranges"]
    for key in ("dR", "azimuth_scale", "azimuths"):
        assert key not in sweep, f"per-sweep key {key!r} should no longer be written"


def test_generate_lut_accepts_a_projection_crs_object(tmp_path, make_datatree):
    """A pyproj CRS works where an EPSG int is not available."""
    crs = pyproj.CRS.from_proj4(
        "+proj=utm +zone=35 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs",
    )

    generate_lut_from_datatree(make_datatree(), radar=RADAR, output_base_path=str(tmp_path), projection_crs=crs)

    lut = load_radar_lut(RADAR, tmp_path)
    assert len([c for c in lut.columns if c.startswith("x_")]) == 1
    assert lut[[c for c in lut.columns if c.startswith("x_")][0]].is_not_null().any()


def test_generate_lut_records_an_explicit_beamwidth(tmp_path, make_datatree):
    """The one place beamwidth may be set; no plot takes it."""
    generate_lut_from_datatree(
        make_datatree(),
        radar=RADAR,
        output_base_path=str(tmp_path),
        beamwidth_deg=1.5,
        projection_epsg=FMI_EPSG,
    )

    assert load_radar_info(RADAR, tmp_path)["beamwidth_deg"] == pytest.approx(1.5)


def test_a_wider_beamwidth_makes_a_taller_gate(tmp_path, make_datatree):
    """Only ``v_plane``/``corners`` depend on it — and this is how it shows."""
    heights = {}
    for beamwidth in (1.0, 2.0):
        out = tmp_path / f"bw{beamwidth}"
        generate_lut_from_datatree(
            make_datatree(),
            radar=RADAR,
            output_base_path=str(out),
            beamwidth_deg=beamwidth,
            projection_epsg=FMI_EPSG,
        )
        table = gate_corner_table(RADAR, str(out), kind="corners", sweep=1)
        z = np.stack([table[f"z_rel_{k}"].to_numpy() for k in range(1, 9)], axis=1)
        heights[beamwidth] = float(np.mean(z.max(axis=1) - z.min(axis=1)))

    assert heights[2.0] > heights[1.0] * 1.5


def test_the_horizontal_face_is_beamwidth_independent(tmp_path, make_datatree):
    """A PPI draws the beam *center*, so its footprint cannot depend on beamwidth."""
    nodes = {}
    for beamwidth in (0.8, 1.2):
        out = tmp_path / f"h{beamwidth}"
        generate_lut_from_datatree(
            make_datatree(),
            radar=RADAR,
            output_base_path=str(out),
            beamwidth_deg=beamwidth,
            projection_epsg=FMI_EPSG,
        )
        nodes[beamwidth] = load_plane_nodes(RADAR, str(out), "h_plane").sort(["sweep", "az_idx", "rng_idx"])

    for column in ("x", "y"):
        assert np.abs(nodes[0.8][column].to_numpy() - nodes[1.2][column].to_numpy()).max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The node lattices
# ---------------------------------------------------------------------------


def test_load_plane_nodes(lut_base):
    """One ``(n_az+1) x (n_rng+1)`` node grid per sweep, at the center level."""
    nodes = load_plane_nodes(RADAR, lut_base, "h_plane")

    assert nodes.height == N_SWEEPS * (N_AZ + 1) * (N_RNG + 1)
    assert "el_level" not in nodes.columns


def test_load_plane_nodes_filters_by_sweep(lut_base):
    """``sweep=`` is pushed down, so one sweep costs one sweep's worth of rows."""
    nodes = load_plane_nodes(RADAR, lut_base, "h_plane", sweep=1)

    assert nodes["sweep"].unique().to_list() == [1]
    assert nodes.height == (N_AZ + 1) * (N_RNG + 1)


def test_the_corner_lattice_has_two_elevation_levels(lut_base):
    """Bottom and top of the beam; the horizontal face is the center only."""
    nodes = load_plane_nodes(RADAR, lut_base, "corners")

    assert sorted(nodes["el_level"].unique().to_list()) == [-1, 1]
    assert nodes.height == 2 * N_SWEEPS * (N_AZ + 1) * (N_RNG + 1)


def test_the_lattice_carries_the_projected_columns(lut_base):
    """Otherwise a backfilled lattice would silently lose its projection."""
    assert {"x_3067", "y_3067"} <= set(load_plane_nodes(RADAR, lut_base, "h_plane").columns)


def test_gate_corner_table(lut_base):
    """Nodes expand to four corners per gate on demand."""
    table = gate_corner_table(RADAR, lut_base, kind="h_plane")

    assert table.height == N_GATES
    for k in range(1, 5):
        assert {f"x_{k}", f"y_{k}"} <= set(table.columns)
    assert "x_5" not in table.columns


def test_gate_corner_table_expands_eight_corners(lut_base):
    """The 3-D lattice gives eight corners per gate."""
    table = gate_corner_table(RADAR, lut_base, kind="corners")

    assert table.height == N_GATES
    for k in range(1, 9):
        assert {f"x_{k}", f"y_{k}", f"z_rel_{k}"} <= set(table.columns)
    assert "x_9" not in table.columns


def test_gate_corner_table_gate_ids_match_the_lut(lut_base):
    """The lattices and the centroid LUT describe exactly the same gates."""
    lut_ids = set(load_radar_lut(RADAR, lut_base)["gate_id"].to_list())

    assert set(gate_corner_table(RADAR, lut_base, kind="corners")["gate_id"].to_list()) == lut_ids


def test_a_gate_is_a_frustum_not_a_box(lut_base):
    """Angular half-extents are evaluated at each corner's own range.

    So corners 5-8 (far face) enclose strictly more area than 1-4 (near face).
    """
    table = gate_corner_table(RADAR, lut_base, kind="corners")

    near = _face_area(table, [1, 2, 3, 4])
    far = _face_area(table, [5, 6, 7, 8])

    assert np.all(far > near), f"{int((far <= near).sum())} gate(s) have a far face no larger than the near face"


def test_the_frustum_ratio_stays_physically_sane(lut_base):
    """Excluding the degenerate innermost bin, the growth is bounded."""
    table = gate_corner_table(RADAR, lut_base, kind="corners")
    near = _face_area(table, [1, 2, 3, 4])
    far = _face_area(table, [5, 6, 7, 8])
    keep = near > 1.0  # drop the r~0 near face

    ratio = far[keep] / near[keep]
    assert ratio.min() > 1.0
    assert ratio.max() < 100.0


def test_the_first_range_bin_is_degenerate(lut_base):
    """Its inner edge clips to r=0, so its near face collapses to a point."""
    table = gate_corner_table(RADAR, lut_base, kind="corners")
    pts = np.stack(
        [
            np.stack(
                [table[f"x_{k}"].to_numpy(), table[f"y_{k}"].to_numpy(), table[f"z_rel_{k}"].to_numpy()],
                axis=1,
            )
            for k in range(1, 9)
        ],
        axis=1,
    )

    distinct = np.array([len({tuple(np.round(p, 3)) for p in pts[i]}) for i in range(pts.shape[0])])

    assert set(np.unique(distinct)) <= {5, 8}
    assert (distinct == 8).mean() > 0.9


def test_gate_footprints_are_valid_polygons(lut_base):
    """A self-intersecting quad would break every spatial predicate downstream."""
    table = gate_corner_table(RADAR, lut_base, kind="h_plane")
    ring = np.stack(
        [np.stack([table[f"x_{k}"].to_numpy(), table[f"y_{k}"].to_numpy()], axis=1) for k in (1, 2, 3, 4, 1)],
        axis=1,
    )

    assert shapely.is_valid(shapely.polygons(ring)).all()


def test_a_centroid_lies_inside_its_own_footprint(real_lut_base):
    """Needs realistic 1 degree azimuth sampling — see :func:`real_lut_base`."""
    table = gate_corner_table(RADAR, real_lut_base, kind="h_plane").sort("gate_id")
    lut = load_radar_lut(RADAR, real_lut_base).sort("gate_id")
    ring = np.stack(
        [np.stack([table[f"x_{k}"].to_numpy(), table[f"y_{k}"].to_numpy()], axis=1) for k in (1, 2, 3, 4, 1)],
        axis=1,
    )

    polygons = shapely.polygons(ring)
    points = shapely.points(np.stack([lut["x"].to_numpy(), lut["y"].to_numpy()], axis=1))

    assert shapely.covers(polygons, points).all()


def test_a_centroid_lies_between_its_elevation_levels(lut_base):
    """1 cm tolerance: on a negative-elevation sweep ``z(r)`` has a turning point."""
    table = gate_corner_table(RADAR, lut_base, kind="corners").sort("gate_id")
    lut = load_radar_lut(RADAR, lut_base).sort("gate_id")
    z_center = lut["z"].to_numpy()
    z_corners = np.stack([table[f"z_rel_{k}"].to_numpy() for k in range(1, 9)], axis=1)

    assert (z_center >= z_corners.min(axis=1) - 0.01).all()
    assert (z_center <= z_corners.max(axis=1) + 0.01).all()


def test_the_vertical_lattice_relates_the_two_altitude_references(lut_base):
    """``z_asl`` is ``z_rel`` plus the site altitude, everywhere."""
    site_altitude = load_radar_info(RADAR, lut_base)["altitude"]
    nodes = load_plane_nodes(RADAR, lut_base, "v_plane")

    difference = nodes["z_asl"].to_numpy() - nodes["z_rel"].to_numpy()

    assert np.allclose(difference, site_altitude, atol=1e-3)


def test_ground_distance_is_monotonic_in_range(lut_base):
    """Along a ray, ``d`` must increase — the RHI axis depends on it."""
    nodes = load_plane_nodes(RADAR, lut_base, "v_plane", sweep=1)
    along = nodes.filter((pl.col("el_level") == 1) & (pl.col("az_idx") == 0)).sort("rng_idx")

    assert np.all(np.diff(along["d"].to_numpy()) > 0)


def test_the_geometry_files_stay_smaller_than_the_centroid_lut(real_lut_base):
    """The whole point of storing lattices rather than per-gate corners."""
    centroid = lut_file_path(RADAR, "lut", real_lut_base).stat().st_size
    geometry = sum(lut_file_path(RADAR, k, real_lut_base).stat().st_size for k in ("h_plane", "v_plane", "corners"))

    assert geometry < centroid, f"geometry {geometry / 1e6:.1f} MB vs LUT {centroid / 1e6:.1f} MB"


@pytest.mark.parametrize(("kind", "budget"), [("h_plane", 18.0), ("v_plane", 2.0), ("corners", 20.0)])
def test_each_geometry_file_stays_inside_its_byte_budget(real_lut_base, kind, budget):
    """Bytes per gate, measured on 144k gates so the parquet footer does not dominate.

    ``v_plane`` is startlingly small because ground distance and altitude do not depend
    on azimuth, so parquet run-length-encodes it almost completely away.
    """
    per_gate = lut_file_path(RADAR, kind, real_lut_base).stat().st_size / REAL_N_GATES

    assert per_gate <= budget, f"{kind} is {per_gate:.1f} B/gate, over the {budget} B/gate budget"


def test_the_lattice_beats_per_gate_materialization(real_lut_base):
    """Neighboring gates share corner nodes, which is where the saving comes from."""
    stored = lut_file_path(RADAR, "corners", real_lut_base).stat().st_size
    naive = gate_corner_table(RADAR, real_lut_base, kind="corners").height * 8 * 3 * 4

    assert stored < naive


# ---------------------------------------------------------------------------
# compute_sweep_corners / build_gate_planes / ensure_gate_planes
# ---------------------------------------------------------------------------


def _sweep_corners(beamwidth_deg=DEFAULT_BEAMWIDTH_DEG):
    """A node mesh for the default synthetic sweep geometry."""
    return compute_sweep_corners(
        np.linspace(1000, 20_000, N_RNG),
        np.linspace(0, 330, N_AZ),
        np.full(N_AZ, 0.5),
        radar_lat=FI_SITE[1],
        radar_lon=FI_SITE[0],
        radar_alt=1000.0,
        beamwidth_deg=beamwidth_deg,
    )


def test_compute_sweep_corners():
    """The per-sweep node mesh the three lattices are built from."""
    corners = _sweep_corners()

    assert isinstance(corners, dict)
    assert corners
    # The mesh is one node lattice per elevation level, so it is (n_az+1) x (n_rng+1).
    assert any(np.asarray(v).size == (N_AZ + 1) * (N_RNG + 1) for v in corners.values() if np.ndim(v))


def test_compute_sweep_corners_requires_a_beamwidth_for_the_vertical_levels():
    """Without it there is no top or bottom face to build ``v_plane``/``corners`` from."""
    corners = compute_sweep_corners(
        np.linspace(1000, 20_000, N_RNG),
        np.linspace(0, 330, N_AZ),
        np.full(N_AZ, 0.5),
        radar_lat=FI_SITE[1],
        radar_lon=FI_SITE[0],
        radar_alt=1000.0,
    )

    with pytest.raises(ValueError, match="beamwidth_deg"):
        build_gate_planes({1: corners}, radar_alt=1000.0, projection_epsg=FMI_EPSG)


def test_build_gate_planes():
    """The three lattices come out of one corner mesh, keyed by kind."""
    planes = build_gate_planes({1: _sweep_corners()}, radar_alt=1000.0, projection_epsg=FMI_EPSG)

    assert set(planes) == {"h_plane", "v_plane", "corners"}
    assert all(isinstance(v, pl.DataFrame) and v.height > 0 for v in planes.values())


def test_ensure_gate_planes(lut_base, tmp_path):
    """A two-file archive is backfilled on first read, without re-ingesting anything."""
    old = tmp_path / "old" / RADAR / "LUT"
    old.mkdir(parents=True)
    for kind in ("lut", "info"):
        source = lut_file_path(RADAR, kind, lut_base)
        (old / source.name).write_bytes(source.read_bytes())
    base = tmp_path / "old"

    assert not lut_file_path(RADAR, "h_plane", base).exists()
    assert ensure_gate_planes(RADAR, base) is True

    for kind in ("h_plane", "v_plane", "corners"):
        assert lut_file_path(RADAR, kind, base).exists()
    assert ensure_gate_planes(RADAR, base) is False, "a second call must be a no-op"


def test_the_backfill_recovers_the_projection_from_the_lut(lut_base, tmp_path):
    """Pre-geometry info YAMLs have no ``crs`` block; the LUT's columns still say EPSG."""
    old = tmp_path / "old" / RADAR / "LUT"
    old.mkdir(parents=True)
    for kind in ("lut", "info"):
        source = lut_file_path(RADAR, kind, lut_base)
        (old / source.name).write_bytes(source.read_bytes())
    info_path = old / lut_file_path(RADAR, "info", lut_base).name
    info = yaml.safe_load(info_path.read_text())
    info.pop("crs", None)
    info_path.write_text(yaml.safe_dump(info))

    ensure_gate_planes(RADAR, tmp_path / "old")

    assert {"x_3067", "y_3067"} <= set(load_plane_nodes(RADAR, tmp_path / "old", "h_plane").columns)


def test_save_sweep_corners(lut_base, tmp_path):
    """The legacy ``.npz`` corner store, still read for pre-lattice archives.

    ``np.savez`` is loaded back with ``allow_pickle=False``, so only the flat numeric
    arrays the real producer emits survive the round trip.
    """
    compute_corners_from_lut(RADAR, lut_base)
    original = load_sweep_corners(RADAR, lut_base)

    out = tmp_path / "copy" / RADAR / "LUT" / f"{RADAR}_corners.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_sweep_corners(original, out)

    reloaded = load_sweep_corners(RADAR, tmp_path / "copy")

    assert set(reloaded) == set(original) == {1, 2}
    for sweep, arrays in original.items():
        for key, value in arrays.items():
            np.testing.assert_allclose(reloaded[sweep][key], value)


def test_load_sweep_corners(lut_base):
    """Reads the ``.npz`` back as ``{sweep: {name: array}}``, keyed by sweep number."""
    compute_corners_from_lut(RADAR, lut_base)

    corners = load_sweep_corners(RADAR, lut_base)

    assert set(corners) == {1, 2}
    assert all(isinstance(v, dict) and v for v in corners.values())


def test_load_sweep_corners_is_empty_without_the_file(tmp_path):
    """Backwards-compatible: an archive with no ``.npz`` yields ``{}``, not an error.

    Callers treat an empty result as "use the lattices instead", which is the normal
    path for every archive written since the lattices existed.
    """
    assert load_sweep_corners(RADAR, tmp_path) == {}


def test_compute_corners_from_lut(lut_base):
    """Corners are rebuilt from the centroid LUT alone, with no source volume."""
    path = compute_corners_from_lut(RADAR, lut_base)

    assert Path(path).exists()
    assert set(load_sweep_corners(RADAR, lut_base)) == {1, 2}


# ---------------------------------------------------------------------------
# cappi_chords
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cappi_base(tmp_path_factory):
    """A six-sweep LUT, so a CAPPI slice has overlapping beams to resolve."""
    base = tmp_path_factory.mktemp("cappi_lut")
    generate_lut_from_datatree(
        build_datatree(n_az=72, n_rng=60, n_sweeps=6),
        radar=RADAR,
        output_base_path=str(base),
        projection_epsg=FMI_EPSG,
    )
    return base


def test_cappi_chords(cappi_base):
    """Every reported bin really spans the requested altitude."""
    z0 = 1200.0
    chords = cappi_chords(RADAR, cappi_base, z0)
    assert not chords.is_empty()

    nodes = load_plane_nodes(RADAR, cappi_base, "v_plane")
    nodes = nodes.filter(pl.col("az_idx") == pl.col("az_idx").min())
    for (sweep,), sub in chords.group_by(["sweep"], maintain_order=True):
        bottom = nodes.filter((pl.col("sweep") == sweep) & (pl.col("el_level") == -1)).sort("rng_idx")
        top = nodes.filter((pl.col("sweep") == sweep) & (pl.col("el_level") == 1)).sort("rng_idx")
        zb, zt = bottom["z_asl"].to_numpy(), top["z_asl"].to_numpy()
        j = sub["rng_idx"].to_numpy()
        low = np.minimum.reduce([zb[j], zb[j + 1], zt[j], zt[j + 1]])
        high = np.maximum.reduce([zb[j], zb[j + 1], zt[j], zt[j + 1]])
        assert ((low - 1e-3 <= z0) & (z0 <= high + 1e-3)).all()


def test_chords_stay_inside_their_range_bin(cappi_base):
    """The cut trims a gate along the beam; it never reaches outside it."""
    chords = cappi_chords(RADAR, cappi_base, 1200.0)
    nodes = load_plane_nodes(RADAR, cappi_base, "v_plane")
    nodes = nodes.filter(pl.col("az_idx") == pl.col("az_idx").min())

    for (sweep,), sub in chords.group_by(["sweep"], maintain_order=True):
        bottom = nodes.filter((pl.col("sweep") == sweep) & (pl.col("el_level") == -1)).sort("rng_idx")
        top = nodes.filter((pl.col("sweep") == sweep) & (pl.col("el_level") == 1)).sort("rng_idx")
        db, dt = bottom["d"].to_numpy(), top["d"].to_numpy()
        j = sub["rng_idx"].to_numpy()
        low = np.minimum.reduce([db[j], db[j + 1], dt[j], dt[j + 1]])
        high = np.maximum.reduce([db[j], db[j + 1], dt[j], dt[j + 1]])
        assert (sub["d_near"].to_numpy() >= low - 1e-2).all()
        assert (sub["d_far"].to_numpy() <= high + 1e-2).all()


def test_the_near_chord_edge_is_below_the_far_one(cappi_base):
    """Otherwise the drawn polygon would be inside out."""
    chords = cappi_chords(RADAR, cappi_base, 1200.0)

    assert (chords["d_near"].to_numpy() <= chords["d_far"].to_numpy()).all()


def test_each_sweep_contributes_a_contiguous_band(cappi_base):
    """Beam thickness far exceeds the rise per bin, so the bands have no holes."""
    for (_sweep,), sub in cappi_chords(RADAR, cappi_base, 1200.0).group_by(["sweep"]):
        j = np.sort(sub["rng_idx"].to_numpy())
        assert np.array_equal(j, np.arange(j.min(), j.max() + 1))


def test_an_altitude_above_every_beam_yields_no_chords(cappi_base):
    """Empty, not an error — the caller turns it into the "reaches" message."""
    assert cappi_chords(RADAR, cappi_base, 50_000.0).is_empty()


def test_the_two_height_references_select_the_same_chords(cappi_base):
    """``asl`` at ``z`` is ``rel`` at ``z - site_altitude``."""
    altitude = load_radar_info(RADAR, cappi_base)["altitude"]

    asl = cappi_chords(RADAR, cappi_base, 1200.0, height="asl")
    rel = cappi_chords(RADAR, cappi_base, 1200.0 - altitude, height="rel")

    assert asl.height == rel.height


def test_cappi_chords_rejects_an_unknown_height_reference(cappi_base):
    """Only ``asl`` and ``rel`` exist."""
    with pytest.raises(ValueError):
        cappi_chords(RADAR, cappi_base, 1200.0, height="furlongs")


# ---------------------------------------------------------------------------
# The CRS contract — measured, never declared
# ---------------------------------------------------------------------------


def test_suggest_crs():
    """The UTM zone for a site, quoted in every refusal so the user is told what to pass."""
    assert suggest_crs(*FI_SITE) == 32635  # zone 35N
    assert suggest_crs(*US_SITE) == 32614  # zone 14N


def test_suggest_crs_picks_a_south_zone_below_the_equator():
    """Sydney is zone 56S, not 56N."""
    assert suggest_crs(151.2, -33.9) == 32756


def test_crs_distance_error():
    """The measurement itself: percent error on a projected 100 km geodesic."""
    assert crs_distance_error(FMI_EPSG, *FI_SITE) < 0.1
    assert crs_distance_error(3857, *FI_SITE) > 10.0


@pytest.mark.parametrize(
    ("crs", "site", "accepted"),
    [
        (3067, FI_SITE, True),  # TM35FIN at home
        (32635, FI_SITE, True),  # UTM 35N at home
        (32614, US_SITE, True),  # UTM 14N at KTLX
        (3067, US_SITE, False),  # the bug: a national grid used a continent away
        (3857, FI_SITE, False),  # Web Mercator claims the world and distorts hugely
        (3857, US_SITE, False),
    ],
)
def test_validate_crs_for_site(crs, site, accepted):
    """Validity is measured, because declared metadata is not enough."""
    if accepted:
        assert validate_crs_for_site(crs, *site) < CRS_REFUSE_PCT
    else:
        with pytest.raises(ValueError, match="distorts distance"):
            validate_crs_for_site(crs, *site)


def test_an_area_of_use_check_alone_would_pass_web_mercator():
    """EPSG:3857 declares the whole world, so a bounds check lets it through."""
    area = pyproj.CRS.from_epsg(3857).area_of_use

    assert area.west <= FI_SITE[0] <= area.east
    assert area.south <= FI_SITE[1] <= area.north
    assert crs_distance_error(3857, *FI_SITE) > 10.0


def test_a_geographic_crs_is_refused():
    """Degrees are not meters; EPSG:4326 can never measure a crop radius."""
    with pytest.raises(ValueError, match="geographic"):
        validate_crs_for_site(4326, *FI_SITE)


def test_a_refusal_names_a_usable_replacement():
    """The message must tell the user what to pass, not just complain."""
    with pytest.raises(ValueError, match="32614"):
        validate_crs_for_site(3067, *US_SITE)


def test_generating_a_lut_without_a_crs_is_refused(tmp_path, make_datatree):
    """A CRS is mandatory to write, because a wrong projection is silently wrong."""
    with pytest.raises(ValueError, match="requires a CRS"):
        generate_lut_from_datatree(make_datatree(), radar=RADAR, output_base_path=str(tmp_path))


def test_that_refusal_also_names_a_usable_crs(tmp_path, make_datatree):
    """``RadDB(crs=32635)`` is the UTM zone at the synthetic site."""
    with pytest.raises(ValueError, match=r"RadDB\(crs=32635\)"):
        generate_lut_from_datatree(make_datatree(), radar=RADAR, output_base_path=str(tmp_path))


def test_a_crs_invalid_at_the_site_is_refused(tmp_path, make_datatree):
    """EPSG:3067 outside Finland mis-measures distance by ~36%."""
    dt = relocate(make_datatree(), *US_SITE)

    with pytest.raises(ValueError, match="distorts distance"):
        generate_lut_from_datatree(dt, radar=RADAR, output_base_path=str(tmp_path), projection_epsg=3067)


def test_the_correct_crs_archives_a_us_radar(tmp_path, make_datatree):
    """UTM 14N at KTLX measures to 0.03%."""
    dt = relocate(make_datatree(), *US_SITE)

    generate_lut_from_datatree(dt, radar=RADAR, output_base_path=str(tmp_path), projection_epsg=US_EPSG)

    assert load_radar_info(RADAR, tmp_path)["crs"]["epsg"] == US_EPSG


# ---------------------------------------------------------------------------
# Accessors and converters
# ---------------------------------------------------------------------------


def test_lut_file_path(lut_base):
    """One place that knows the five filenames."""
    for kind, template in LUT_FILES.items():
        assert lut_file_path(RADAR, kind, lut_base).name == template.format(radar=RADAR)


def test_lut_file_path_rejects_an_unknown_kind(lut_base):
    """A typo must not produce a path that will simply not exist."""
    with pytest.raises((KeyError, ValueError)):
        lut_file_path(RADAR, "not_a_kind", lut_base)


def test_load_radar_lut(lut_base):
    """The centroid table, as polars, one row per gate."""
    lut = load_radar_lut(RADAR, lut_base)

    assert isinstance(lut, pl.DataFrame)
    assert lut.height == N_GATES
    assert {"gate_id", "sweep", "azimuth", "range", "latitude", "longitude", "altitude"} <= set(lut.columns)


def test_gate_coordinates_stay_float64(lut_base):
    """Positions are float64 — precision is a hard requirement, not a preference."""
    lut = load_radar_lut(RADAR, lut_base)

    for column in ("latitude", "longitude", "altitude", "x", "y", "z"):
        if column in lut.columns:
            assert lut.schema[column] == pl.Float64, f"{column} lost float64"


def test_load_radar_info(lut_base):
    """The info YAML, as a plain dict."""
    info = load_radar_info(RADAR, lut_base)

    assert info["radar"] == RADAR
    assert info["latitude"] == pytest.approx(FI_SITE[1])
    assert info["longitude"] == pytest.approx(FI_SITE[0])


def test_load_radar_info_raises_for_an_unknown_radar(lut_base):
    """A missing radar is a ``FileNotFoundError``, which callers catch explicitly."""
    with pytest.raises(FileNotFoundError):
        load_radar_info("ZZZZ", lut_base)


def test_get_full_sweep_index(lut_base):
    """A pandas MultiIndex — one of the three deliberate pandas seams (xarray needs it)."""
    import pandas as pd

    index = get_full_sweep_index(load_radar_lut(RADAR, lut_base), sweep=1)

    assert isinstance(index, pd.MultiIndex)
    assert index.names == ["azimuth", "range"]
    assert len(index) == N_AZ * N_RNG


def test_add_lut_projection(lut_base):
    """Projected columns are named after the EPSG, so a frame says which one it is in."""
    lut = load_radar_lut(RADAR, lut_base).drop(["x_3067", "y_3067"])

    out = add_lut_projection(lut, epsg=32635)

    assert {"x_32635", "y_32635"} <= set(out.columns)
    assert np.isfinite(out["x_32635"].to_numpy()).all()


def test_add_lut_projection_returns_the_kind_it_was_given(lut_base):
    """Same-kind-in-same-kind-out, like ``filter_df``."""
    import pandas as pd

    lut = load_radar_lut(RADAR, lut_base).drop(["x_3067", "y_3067"])

    assert isinstance(add_lut_projection(lut, epsg=32635), pl.DataFrame)
    assert isinstance(add_lut_projection(lut.to_pandas(), epsg=32635), pd.DataFrame)


def test_gate_polygons_geoarrow(lut_base):
    """GeoArrow-tagged wedge polygons, for the lonboard path."""
    pytest.importorskip("pyarrow")
    gate_ids = load_radar_lut(RADAR, lut_base)["gate_id"].to_numpy()[:50]

    table = gate_polygons_geoarrow(RADAR, lut_base, gate_ids)

    assert table is not None
    assert len(table) == 50


def test_geoarrow_field():
    """The field metadata is what makes a column readable as geometry."""
    import pyarrow as pa

    field = geoarrow_field("geometry", pa.float64(), "point", crs="EPSG:3067")

    assert field.name == "geometry"
    assert field.metadata

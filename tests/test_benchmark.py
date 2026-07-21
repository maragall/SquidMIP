"""IMA-233: the benchmark harness itself, on synthetic arrays and a synthetic reader.

These tests never touch a real acquisition — they lock the harness's CONTRACTS (the
quality metrics move in the documented direction, the guards refuse before allocating,
the read accounting is restored, the table renders every operator) so that a change to
the harness fails here rather than silently producing a plausible-looking wrong table.
The actual numbers come from ``tools/benchmark.py`` on real data; a benchmark asserted
against synthetic data would be measuring the fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip import _benchmark as bm


# --- quality metrics -------------------------------------------------------------------

def test_relative_gradient_energy_rises_with_structure():
    flat = np.full((64, 64), 100.0, dtype=np.float32)
    noisy = flat.copy()
    noisy[::2, :] += 50.0
    assert bm.relative_gradient_energy(flat) == pytest.approx(0.0)
    assert bm.relative_gradient_energy(noisy) > 0.1


def test_relative_gradient_energy_is_scale_invariant():
    """Normalising by the mean is the whole point: doubling the exposure must not read as
    a sharper image."""
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)).astype(np.float32) + 1.0
    assert bm.relative_gradient_energy(a * 2) == pytest.approx(
        bm.relative_gradient_energy(a), rel=1e-5)


def test_relative_gradient_energy_handles_empty_and_dark():
    assert np.isnan(bm.relative_gradient_energy(np.zeros((8, 8))))
    assert np.isnan(bm.relative_gradient_energy(np.zeros((0, 0))))


def test_block_uniformity_flat_is_one_vignetted_is_lower():
    flat = np.full((64, 64), 500.0, dtype=np.float32)
    y, x = np.mgrid[0:64, 0:64]
    vignette = flat * (1.0 - 0.6 * ((y - 32) ** 2 + (x - 32) ** 2) / (2 * 32 ** 2))
    assert bm.block_uniformity(flat) == pytest.approx(1.0)
    assert bm.block_uniformity(vignette) < 0.95


def test_block_uniformity_rejects_non_2d_and_tiny():
    assert np.isnan(bm.block_uniformity(np.zeros((4, 4))))
    assert np.isnan(bm.block_uniformity(np.zeros((8, 8, 3))))


def test_overlap_ncc_is_one_at_the_true_offset():
    tilefusion = pytest.importorskip("tilefusion.registration")
    assert tilefusion is not None
    rng = np.random.default_rng(1)
    full = rng.random((256, 400)).astype(np.float32)
    # Two tiles 256x256 sharing a 112 px strip: tile_j sits 144 px to the right of tile_i.
    tile_i, tile_j = full[:, :256], full[:, 144:400]
    assert bm.overlap_ncc(tile_i, tile_j, 0, 144) == pytest.approx(1.0, abs=1e-6)
    # A wrong placement scores near zero on white noise. This is the property that makes
    # the metric usable as a seam score at all.
    assert abs(bm.overlap_ncc(tile_i, tile_j, 0, 100)) < 0.3


# --- guards ----------------------------------------------------------------------------

_META = {
    "frame_shape": (2084, 2084),
    "dtype": "uint16",
    "channels": [{"name": "c0"}, {"name": "c1"}],
    "z_levels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "n_t": 1,
    "fovs_per_region": {"A1": list(range(27))},
    "pixel_size_um": 0.752,
}


def test_expected_output_bytes_counts_z_for_a_plane_op():
    """A plane-op keeps z at full depth; a z-reducer collapses it. The Nz factor between
    them is exactly the term whose omission fills memory."""
    reducer = bm.expected_output_bytes(_META, kind="fov", regions=["A1"], n_fovs=1,
                                       consumes_z=True)
    plane_op = bm.expected_output_bytes(_META, kind="fov", regions=["A1"], n_fovs=1,
                                        consumes_z=False)
    assert plane_op == reducer * len(_META["z_levels"])
    assert reducer == 2084 * 2084 * 2 * 2   # Y * X * n_channels * itemsize


def test_expected_output_bytes_returns_zero_without_a_frame_shape():
    assert bm.expected_output_bytes({}, kind="fov", regions=[], n_fovs=1,
                                    consumes_z=True) == 0


def test_guard_memory_refuses_an_impossible_run():
    with pytest.raises(bm.BenchmarkGuardError) as exc:
        bm.guard_memory(1 << 60, what="a preposterous run")
    assert "preposterous" in str(exc.value)


def test_guard_memory_allows_a_small_run():
    assert bm.guard_memory(1024, what="a small run")["checked"] in (True, False)


def test_persist_estimate_is_overlap_aware_for_region_operators():
    """A 27-FOV well fused into one mosaic must cost less than 27 separate frames — that
    is the whole reason ``estimate_write_bytes`` grew ``region_operator``."""
    per_fov = bm.persist_estimate(_META | {"fov_positions_um": _positions()},
                                  kind="fov", regions=["A1"], n_fovs=None)
    region = bm.persist_estimate(_META | {"fov_positions_um": _positions()},
                                 kind="region", regions=["A1"], n_fovs=None)
    assert 0 < region < per_fov


def _positions():
    """A 27-FOV 5x6-ish grid at the 10x acquisition's measured 1410.45 um step."""
    step = 1410.45
    return {("A1", i): ((i % 6) * step, (i // 6) * step) for i in range(27)}


# --- read accounting -------------------------------------------------------------------

class _FakeReader:
    def __init__(self):
        self.calls = 0

    def read(self, *_args, **_kwargs):
        self.calls += 1
        return np.zeros((4, 4), dtype=np.uint16)


def test_read_recorder_accumulates_and_restores():
    reader = _FakeReader()
    original = reader.read
    rec = bm._ReadRecorder()
    with rec.wrap(reader):
        reader.read("A1", 0, "c0", 0, 0)
        reader.read("A1", 0, "c0", 1, 0)
    assert rec.calls == 2
    assert rec.nbytes == 2 * 4 * 4 * 2
    assert rec.ms >= 0.0
    assert reader.calls == 2
    # The wrapper must come off: a reader left permanently instrumented would make every
    # later measurement include this run's bookkeeping.
    assert reader.read == original or reader.read.__func__ is original.__func__


def test_read_recorder_restores_after_an_exception():
    reader = _FakeReader()
    with pytest.raises(ValueError):
        with bm._ReadRecorder().wrap(reader):
            raise ValueError("boom")
    assert "read" not in vars(reader)


# --- reporting -------------------------------------------------------------------------

def _result(op="mip", **kw):
    r = bm.OperatorResult(operator=op, kind=kw.pop("kind", "fov"), dataset="/tmp/ds")
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_as_row_and_derived_rates():
    r = _result(wall_ms=2000.0, read_ms=500.0, out_megapixels=100.0)
    assert r.compute_ms == pytest.approx(1500.0)
    assert r.mpix_per_s == pytest.approx(50.0)
    assert r.as_row()["operator"] == "mip"


def test_compute_ms_never_goes_negative():
    """Read time is accumulated across threads, so at workers > 1 it can exceed the wall
    clock. A negative 'compute' column would be nonsense, not a measurement."""
    assert _result(wall_ms=100.0, read_ms=400.0).compute_ms == 0.0


def test_format_table_lists_every_operator_including_failures():
    results = [_result("mip", wall_ms=10.0, wells=1),
               _result("stitch", kind="region", error="KeyError: nope")]
    table = bm.format_table(results)
    assert "mip" in table and "stitch" in table
    assert "! stitch: KeyError: nope" in table
    assert "seam_ncc" in bm.QUALITY_NOTES["stitch"]


def test_format_stages_reports_the_unattributed_residual():
    """Stages never sum to the wall clock; the harness must SAY so rather than let the
    reader misattribute the gap."""
    r = _result("stitch", kind="region", wall_ms=1000.0,
                stage_ms={"project": 400.0, "fuse": 500.0})
    out = bm.format_stages([r])
    assert "(other)" in out and "100.0 ms" in out


def test_write_csv_and_json_round_trip(tmp_path):
    import csv
    import json

    results = [_result("mip", wall_ms=1.0, wells=1, quality={"sharp_gain": 1.5})]
    csv_path = bm.write_csv(results, tmp_path / "b.csv")
    rows = list(csv.DictReader(open(csv_path)))
    assert rows[0]["operator"] == "mip"

    json_path = bm.write_json(results, tmp_path / "b.json", meta={"dataset": "x"})
    payload = json.loads(json_path.read_text())
    assert payload["results"][0]["operator"] == "mip"
    assert payload["machine"]["machine"]          # the machine is part of the measurement
    assert payload["meta"]["dataset"] == "x"


def test_default_operators_are_all_real_registry_entries():
    from squidmip import available_projectors, available_region_operators

    known = set(available_projectors()) | set(available_region_operators())
    assert set(bm.DEFAULT_OPERATORS) <= known


def test_every_default_operator_documents_its_quality_direction():
    """A quality number whose desired direction the reader has to guess is not a
    measurement."""
    assert set(bm.DEFAULT_OPERATORS) <= set(bm.QUALITY_NOTES)


# ---------------------------------------------------------------------------------------
# IMA-259: per-region decomposition and the registration-scope probe
# ---------------------------------------------------------------------------------------
# These reuse test_stitch's synthetic mosaic reader rather than a real acquisition, for the
# same reason the rest of this file does: the CONTRACTS are what must not drift (the phase
# split adds up, a region's solve is unaffected by its neighbours). The real milliseconds
# come from `tools/tile_bench.py` on real data, and a wall-clock asserted against a 256 px
# fixture would only be measuring the fixture.


def _stitch_fixture(regions=("A1",), error_px=None):
    """The 2x2 / 64 px-overlap synthetic mosaic reader from ``tests/test_stitch.py``."""
    pytest.importorskip("tilefusion")
    pytest.importorskip("profiling")
    from tests.test_stitch import _FakeReader, _master

    return _FakeReader(_master(), regions=regions, error_px=error_px)


_FAST = {"blend_px": 24, "block_px": 512, "max_workers": 2}


def test_time_region_decomposes_the_wall_clock_into_phases():
    """Every phase must be accounted for: the parts cannot exceed the whole, and the
    stage the reader is told dominates must be a real measurement, not a residual."""
    reader = _stitch_fixture()
    t = bm.time_region(reader, "A1", [0, 1, 2, 3], channels=[0], **_FAST)

    assert t.tiles == 4
    assert t.total_ms > 0
    # 4 FOVs x 2 CHANNELS, even though channels=[0] was asked for: `stitch_region` calls
    # `project_well` (which has no channel selector) and slices the result afterwards, so a
    # one-channel mosaic still pays full-channel I/O and full-channel z-reduction. That is a
    # measured inefficiency, not an accident of this fixture — pinned here so the day it is
    # fixed, this number drops and the test says so.
    assert t.read_calls == 4 * 2
    assert t.io_ms > 0                          # the fixture still goes through reader.read
    assert t.project_ms >= t.io_ms              # I/O happens INSIDE the project stage
    assert t.register_ms > 0 and t.fuse_ms > 0
    # mip is the project stage net of I/O, not an independent stopwatch
    assert t.mip_ms == pytest.approx(t.project_ms - t.io_ms)
    # the phases are disjoint spans of one wall clock
    assert t.project_ms + t.register_ms + t.optimize_ms + t.fuse_ms <= t.total_ms + 1e-6


def test_time_region_normalises_per_tile_and_per_megapixel():
    reader = _stitch_fixture()
    t = bm.time_region(reader, "A1", [0, 1, 2, 3], channels=[0], **_FAST)

    assert t.mosaic_shape == (448, 448)          # 2x2 tiles of 256 at step 192
    assert t.mosaic_megapixels == pytest.approx(448 * 448 / 1e6)
    assert t.ms_per_tile == pytest.approx(t.total_ms / 4)
    assert t.mpix_per_s == pytest.approx(t.mosaic_megapixels / (t.total_ms / 1000.0))
    assert sum(t.shares().values()) <= 1.0 + 1e-9


def test_shares_are_fractions_of_this_regions_own_wall_clock():
    t = bm.RegionTiming(region="A1", tiles=4, channels=1, n_z=1, total_ms=1000.0,
                        io_ms=400.0, project_ms=500.0, register_ms=200.0,
                        optimize_ms=10.0, fuse_ms=250.0)
    s = t.shares()
    assert s == {"io": 0.4, "mip": 0.1, "register": 0.2, "optimize": 0.01, "fuse": 0.25}
    assert t.registration_ms == 210.0


def test_benchmark_regions_runs_every_region_every_repeat():
    reader = _stitch_fixture(regions=("A1", "A2"))
    seen = []
    out = bm.benchmark_regions(reader, regions=None, channels=[0], repeats=2,
                               warmup=False, on_region=seen.append, **_FAST)
    assert [t.region for t in out] == ["A1", "A2", "A1", "A2"]
    assert seen == out                           # the callback sees each region as it lands


def test_benchmark_regions_refuses_an_empty_selection():
    reader = _stitch_fixture()
    with pytest.raises(ValueError, match="no regions"):
        bm.benchmark_regions(reader, regions=["nope"], channels=[0], warmup=False)


def test_registration_solves_per_region_not_across_the_plate():
    """The Q2 answer, asserted rather than asserted-about.

    Two regions with DIFFERENT injected stage error. If any information crossed the region
    boundary — one pooled pose graph, one shared anchor — then A1's offsets would depend on
    whether A2 was in the run. They must not.
    """
    reader = _stitch_fixture(regions=("A1", "A2"), error_px={3: (6.0, -4.0)})
    probe = bm.registration_scope_probe(reader, channels=[0], **_FAST)

    assert probe["scope"] == "per-region"
    assert probe["per_region"] is True
    assert max(probe["max_abs_diff"].values()) == 0.0
    # each region carries its OWN gauge: its own tile 0 is pinned, not one plate-wide tile
    assert probe["anchored_per_region"] is True


def test_scope_probe_recovers_the_injected_error_it_is_probing():
    """A probe that reported "per-region" while registering nothing would be worthless, so
    the offsets it compares must be real: tile 3 was displaced by a known 6 px."""
    reader = _stitch_fixture(regions=("A1",), error_px={3: (6.0, -4.0)})
    probe = bm.registration_scope_probe(reader, channels=[0], **_FAST)
    offsets = np.asarray(probe["solo"]["A1"])
    assert offsets[3] == pytest.approx((-6.0, 4.0), abs=0.5)   # correction undoes the error


def test_format_scope_probe_calls_a_pooled_solve_a_bug():
    """The negative branch must be reachable and must say the word, or nobody will notice
    the day it changes."""
    text = bm.format_scope_probe({
        "regions": ["A1", "A2"], "scope": "cross-region (POOLED)",
        "per_region": False, "anchored_per_region": False,
        "max_abs_diff": {"A1": 3.5, "A2": 3.5},
        "solo": {}, "in_plate": {},
    })
    assert "CROSS-REGION (POOLED)" in text
    assert "bug" in text


def test_format_region_timings_reports_per_tile_and_the_phase_split():
    timings = [bm.RegionTiming(region="A1", tiles=4, channels=1, n_z=1,
                               mosaic_shape=(448, 448), total_ms=1000.0, io_ms=400.0,
                               project_ms=500.0, register_ms=200.0, optimize_ms=10.0,
                               fuse_ms=250.0)]
    out = bm.format_region_timings(timings)
    assert "ms_per_tile" in out and "250 ms/tile" in out
    assert "io=40%" in out and "fuse=25%" in out

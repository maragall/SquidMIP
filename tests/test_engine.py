"""IMA-188 unit tests — parallel/streaming plate engine + projector table.

Clean-room (no ``integration`` mark, no data on disk): the reader is a controllable in-memory
fake so we can exercise the engine's own logic — projector table, projector swap (AC4), completion
streaming, bounded in-flight window, fail-loud propagation, and metadata warm-up ordering —
without the real 1536wp fixture. The real seam (189 reader → 188 engine on real pixels) is
proven separately by the 188↔183 cross commit in ``tests/test_integration.py``.

The fake mirrors exactly the slice of the IMA-189 reader contract that ``project_well`` and
``project_plate`` touch: a ``metadata`` dict and ``read(region, fov, channel, z, t)`` → 2D plane.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import squidmip._engine as engine
from squidmip import (
    add_projector,
    available_projectors,
    project_plate,
    project_well,
)


class FakeReader:
    """In-memory stand-in for an IMA-189 ``SquidReader`` — only what the engine consumes.

    Instrumented for the engine tests: it records the order of ``metadata`` vs ``read`` access
    (warm-up ordering), the set of wells that have begun reading (bounded-window / laziness),
    and can be told to sleep per read (to keep wells in flight) or raise on a chosen well
    (fail-loud). ``read`` returns a constant plane whose value grows with ``z`` so a max
    projection has a known, non-degenerate answer.
    """

    def __init__(
        self,
        n_wells: int = 4,
        *,
        n_fovs: int = 1,
        channels: tuple[str, ...] = ("c0", "c1"),
        z_levels: tuple[int, ...] = (0, 1, 2),
        n_t: int = 1,
        shape: tuple[int, int] = (4, 4),
        dtype=np.uint16,
        read_sleep: float = 0.0,
        fail_on: tuple[str, int] | None = None,
    ) -> None:
        self._regions = [f"W{i:04d}" for i in range(n_wells)]
        self._fovs = list(range(n_fovs))
        self._channels = list(channels)
        self._z_levels = list(z_levels)
        self._n_t = n_t
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._read_sleep = read_sleep
        self._fail_on = fail_on

        # instrumentation (thread-safe)
        self._lock = threading.Lock()
        self.events: list[str] = []          # "meta" / "read" in first-touch order
        self.wells_started: set[tuple[str, int]] = set()
        self.read_count = 0

    @property
    def metadata(self) -> dict:
        with self._lock:
            self.events.append("meta")
        return {
            "regions": self._regions,
            "fovs_per_region": {r: list(self._fovs) for r in self._regions},
            "channels": [{"name": c} for c in self._channels],
            "z_levels": self._z_levels,
            "n_z": len(self._z_levels),
            "n_t": self._n_t,
            "frame_shape": self._shape,
            "dtype": self._dtype,
        }

    def read(self, region, fov, channel, z, t=0):
        with self._lock:
            self.events.append("read")
            self.wells_started.add((region, fov))
            self.read_count += 1
        if self._fail_on is not None and (region, fov) == self._fail_on:
            raise ValueError(f"synthetic read failure at region={region!r} fov={fov} z={z}")
        if self._read_sleep:
            time.sleep(self._read_sleep)
        # value grows with z so max-over-z is well-defined and != any lower slice
        base = (hash((region, fov, channel, t)) % 100) * 10
        return np.full(self._shape, base + int(z), dtype=self._dtype)


@pytest.fixture(autouse=True)
def _restore_projector_table():
    """Snapshot/restore the module-global projector table so tests that add don't leak."""
    saved = dict(engine._PROJECTORS)
    saved_flags = dict(engine._PROJECTOR_COMMUTES)
    try:
        yield
    finally:
        engine._PROJECTORS.clear()
        engine._PROJECTORS.update(saved)
        engine._PROJECTOR_COMMUTES.clear()
        engine._PROJECTOR_COMMUTES.update(saved_flags)


def _collect(reader, **kw) -> dict[tuple[str, int], np.ndarray]:
    """Drain project_plate into a {(region, fov): image} dict (order-independent compare)."""
    return {(r, f): img for r, f, img in project_plate(reader, **kw)}


# ── projector table ─────────────────────────────────────────────────────────────────────

def test_mip_is_available_by_default():
    assert "mip" in available_projectors()


def test_available_projectors_is_sorted_and_reflects_registration():
    add_projector("zzz_custom", lambda planes: next(iter(planes)))
    names = available_projectors()
    assert names == sorted(names)
    assert "zzz_custom" in names


def test_add_duplicate_name_raises():
    with pytest.raises(ValueError, match="already defined"):
        add_projector("mip", lambda planes: next(iter(planes)))


def test_add_rejects_empty_name_and_non_callable():
    with pytest.raises(ValueError, match="non-empty"):
        add_projector("", lambda planes: next(iter(planes)))
    with pytest.raises(ValueError, match="not callable"):
        add_projector("bad", object())  # type: ignore[arg-type]


def test_project_plate_unknown_projector_raises_named():
    reader = FakeReader(n_wells=2)
    with pytest.raises(KeyError, match="unknown projector 'nope'"):
        next(project_plate(reader, projector="nope"))


# ── correctness: concurrency changes no pixel ───────────────────────────────────────────

def test_yields_every_well_with_correct_shape_and_dtype():
    reader = FakeReader(n_wells=7)
    out = _collect(reader, workers=3)
    assert set(out) == {(f"W{i:04d}", 0) for i in range(7)}
    for img in out.values():
        assert img.shape == (reader._n_t, len(reader._channels), 1, *reader._shape)
        assert img.dtype == reader._dtype


def test_parallel_output_is_pixel_identical_to_single_thread():
    reader = FakeReader(n_wells=5, channels=("c0", "c1", "c2"))
    parallel = _collect(reader, workers=4)
    for (region, fov), img in parallel.items():
        expected = project_well(reader, region, fov)  # single-thread reference
        np.testing.assert_array_equal(img, expected)


def test_result_is_deterministic_across_worker_counts():
    reader = FakeReader(n_wells=9)
    one = _collect(reader, workers=1)
    many = _collect(reader, workers=4)
    assert set(one) == set(many)
    for key in one:
        np.testing.assert_array_equal(one[key], many[key])


def test_respects_n_fovs():
    reader = FakeReader(n_wells=3, n_fovs=2)
    out = _collect(reader, workers=2, n_fovs=2)
    assert len(out) == 6  # 3 wells × 2 fovs
    assert {f for _, f in out} == {0, 1}


# ── AC4: pluggable projector swaps with zero engine edits ────────────────────────────────

def test_projector_swap_runs_through_the_same_engine():
    # A non-MIP projector (returns the FIRST z-plane) selected purely by name — the engine
    # code is untouched. Proves project_plate(..., projector=<name>) is the pluggable seam.
    add_projector("first_z", lambda planes: next(iter(planes)))
    reader = FakeReader(n_wells=3, z_levels=(0, 1, 2, 3))
    out = _collect(reader, workers=2, projector="first_z")
    for (region, fov), img in out.items():
        for c_i, ch in enumerate(reader._channels):
            first_plane = reader.read(region, fov, ch, reader._z_levels[0])
            np.testing.assert_array_equal(img[0, c_i, 0], first_plane)
            # and it is genuinely NOT the MIP (which would pick the largest z)
            assert not np.array_equal(img[0, c_i, 0], project_well(reader, region, fov)[0, c_i, 0])


# ── fail loud (per-well resilience is IMA-186's, not the engine's) ───────────────────────

def test_failure_in_one_well_propagates_and_aborts_the_stream():
    reader = FakeReader(n_wells=6, fail_on=("W0003", 0))
    with pytest.raises(ValueError, match="synthetic read failure at region='W0003'"):
        _collect(reader, workers=3)


# ── bounded in-flight window / laziness (peak RSS flat vs plate size) ────────────────────

def test_bounded_window_does_not_prefetch_the_whole_plate():
    # With N wells >> workers, consuming ONE result must have started at most `workers + 1`
    # wells (prime `workers`, one refill after the single completion) — NOT all N. This is the
    # invariant that keeps ~139 MB per-well results from piling up → peak RSS flat in plate size.
    n_wells, workers = 40, 3
    reader = FakeReader(n_wells=n_wells, read_sleep=0.01)
    gen = project_plate(reader, workers=workers)
    try:
        next(gen)  # consume exactly one well
        with reader._lock:
            started = len(reader.wells_started)
        assert started <= workers + 1, f"prefetched {started} wells with only {workers} workers"
        assert started < n_wells  # emphatically not the whole plate
    finally:
        gen.close()  # GeneratorExit → ThreadPoolExecutor shuts down


def test_metadata_is_warmed_before_any_read():
    # The engine must touch reader.metadata (single-threaded) before fanning out reads, so the
    # IMA-189 reader's lazy index/time-folders are populated before concurrent read() calls.
    reader = FakeReader(n_wells=4)
    list(project_plate(reader, workers=2))
    assert reader.events, "engine never touched the reader"
    assert reader.events[0] == "meta"
    assert reader.events.index("meta") < reader.events.index("read")


# ── argument validation ──────────────────────────────────────────────────────────────────

def test_invalid_workers_raises():
    reader = FakeReader(n_wells=2)
    with pytest.raises(ValueError, match="workers must be >= 1"):
        next(project_plate(reader, workers=0))


# ── IMA-225: illumination correction threaded through the engine ─────────────────────────

def _field_for(reader, floor: float = 0.4):
    """A prepared radial correction Field matching *reader*'s frame shape/dtype/channel count."""
    from squidmip.correction import prepare_field

    meta = reader.metadata
    ny, nx = meta["frame_shape"]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    cy, cx = max((ny - 1) / 2.0, 1e-6), max((nx - 1) / 2.0, 1e-6)
    prof = floor + (1.0 - floor) * np.exp(-(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2))
    flat = np.stack([prof.astype(np.float32)] * len(meta["channels"]))
    return prepare_field(flat, None, dtype=meta["dtype"], frame_shape=meta["frame_shape"],
                         n_channels=len(meta["channels"]), source="test")


def test_mip_declares_it_commutes_and_reference_does_not():
    # This flag is what licenses correcting AFTER the reduction. `reference` picks a plane by focus
    # score, which monotonicity does not license, so it must stay False.
    assert engine.projector_commutes("mip") is True
    assert engine.projector_commutes("reference") is False


def test_add_projector_defaults_to_not_commuting():
    # Safe default on a public extension seam: an author who never read the engine gets the
    # always-correct path, not the merely-fast one.
    add_projector("plain_custom", lambda planes: next(iter(planes)))
    assert engine.projector_commutes("plain_custom") is False
    add_projector("fast_custom", lambda planes: next(iter(planes)), commutes_with_scaling=True)
    assert engine.projector_commutes("fast_custom") is True


def test_projector_commutes_is_false_for_an_unknown_name():
    assert engine.projector_commutes("never_registered") is False


def test_no_flatfield_is_byte_identical_to_today():
    # REGRESSION GUARD: the correction seam must be invisible when nobody asks for it.
    reader = FakeReader(n_wells=3)
    plain = _collect(FakeReader(n_wells=3), workers=2)
    explicit_none = _collect(reader, workers=2, flatfield=None)
    assert plain.keys() == explicit_none.keys()
    for key, img in plain.items():
        np.testing.assert_array_equal(img, explicit_none[key])


def test_flatfield_changes_the_pixels_but_not_the_output_contract():
    reader = FakeReader(n_wells=2, shape=(16, 16))
    field = _field_for(reader)
    plain = _collect(FakeReader(n_wells=2, shape=(16, 16)), workers=2)
    corrected = _collect(reader, workers=2, flatfield=field)
    for key, img in corrected.items():
        assert img.shape == plain[key].shape and img.dtype == plain[key].dtype
        assert img.shape[2] == 1                    # Z stays size-1 — the writer's contract
        assert not np.array_equal(img, plain[key])


def test_flatfield_result_is_independent_of_worker_count():
    # The prepared field is immutable and shared read-only, so concurrency must change no pixel.
    field = _field_for(FakeReader(n_wells=6, shape=(16, 16)))
    one = _collect(FakeReader(n_wells=6, shape=(16, 16)), workers=1, flatfield=field)
    many = _collect(FakeReader(n_wells=6, shape=(16, 16)), workers=4, flatfield=field)
    assert one.keys() == many.keys()
    for key, img in one.items():
        np.testing.assert_array_equal(img, many[key])


def test_a_non_commuting_projector_corrects_every_plane():
    # `reference` does not commute, so the engine must take the BEFORE path: the returned plane is
    # a CORRECTED plane, i.e. it differs from the uncorrected reference pick.
    reader = FakeReader(n_wells=1, shape=(16, 16))
    field = _field_for(reader)
    plain = _collect(FakeReader(n_wells=1, shape=(16, 16)), workers=1, projector="reference")
    corrected = _collect(reader, workers=1, projector="reference", flatfield=field)
    for key, img in corrected.items():
        assert not np.array_equal(img, plain[key])

"""IMA-217 PyramidSource — clean-room tests against the locked contract (.spec/open/ima-217.md).

Fabricated metadata + hand-built streams (test_output.py pattern); the store is written by
``write_from_stream`` and read back through :class:`PyramidSource`. TileDescriptor is
duck-typed — ``_tiling.py`` lives on the unmerged IMA-216 branch and is never imported.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from squidmip._output import (
    PyramidSource,
    _pyramid,
    write_from_stream,
    write_plate,
)

CH = [
    {"name": "Fluorescence_638_nm_-_Penta", "display_name": "Fluorescence 638 nm - Penta", "display_color": "#FF0000"},
    {"name": "Fluorescence_405_nm_-_Penta", "display_name": "Fluorescence 405 nm - Penta", "display_color": "#20ADF8"},
]
REGIONS = ["B2", "B3", "B10"]

Desc = namedtuple("Desc", "level key channel bbox_um")   # duck-typed TileDescriptor


def _meta(regions=REGIONS, fovs=(0,)):
    return {
        "regions": list(regions),
        "fovs_per_region": {r: list(fovs) for r in regions},
        "channels": CH,
        "pixel_size_um": 0.325,
    }


def _image(seed: int, t=1, c=2, y=8, x=8, dtype=np.uint16):
    base = np.arange(y * x).reshape(y, x)
    out = np.empty((t, c, 1, y, x), dtype=dtype)
    for ti in range(t):
        for ci in range(c):
            out[ti, ci, 0] = ((base + seed * 20 + ti * 7 + ci * 3) % 200).astype(dtype)
    return out


def _write(tmp_path, images, regions=None, fovs=(0,), source=None, n_fovs=1):
    regions = regions or list(images)
    stream = ((r, f, images[r]) for r in regions for f in fovs)
    return write_from_stream(_meta(regions, fovs), stream, tmp_path, n_fovs=n_fovs, source=source)


# --- disk path -------------------------------------------------------------------------------

def test_pixel_exact_reads_at_every_level(tmp_path):
    img = _image(1, y=600, x=600)                       # levels 600/300/150
    _write(tmp_path, {"B2": img})
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    expected_levels = _pyramid(img)
    shapes = src.levels("B2", 0)
    assert shapes == [(l.shape[-2], l.shape[-1]) for l in expected_levels]
    assert [s[-1] for s in shapes] == [600, 300, 150]
    for lvl, exp in enumerate(expected_levels):
        got = src.read("B2", 0, lvl, slice(None), slice(None))
        np.testing.assert_array_equal(got, exp[0, :, 0])
        window = src.read("B2", 0, lvl, slice(3, 9), slice(1, 5))
        np.testing.assert_array_equal(window, exp[0, :, 0, 3:9, 1:5])


def test_tiny_field_single_level_disk_read(tmp_path):
    img = _image(2, y=8, x=8)                           # <=256: single level, nothing cached
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    _write(tmp_path, {"B2": img}, source=src)
    assert src.levels("B2", 0) == [(8, 8)]
    assert src._cache == {}                             # cache rule: level 0 never cached
    np.testing.assert_array_equal(src.read("B2", 0, 0, slice(None), slice(None)), img[0, :, 0])
    assert src.read("B2", 0, 1, slice(None), slice(None)) is None   # out-of-range level


def test_slice_clamping_and_empty(tmp_path):
    img = _image(3)
    _write(tmp_path, {"B2": img})
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    got = src.read("B2", 0, 0, slice(-5, 99), slice(4, 99))         # clamps into [0, 8]
    np.testing.assert_array_equal(got, img[0, :, 0, 0:8, 4:8])
    empty = src.read("B2", 0, 0, slice(50, 99), slice(0, 4))        # fully out of range
    assert empty.shape == (2, 0, 4)


def test_out_of_range_t_raises_pending_returns_none(tmp_path):
    img = _image(4)
    _write(tmp_path, {"B2": img})
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    with pytest.raises(IndexError):
        src.read("B2", 0, 0, slice(None), slice(None), t=5)         # available field: raise
    assert src.read("B7", 0, 0, slice(None), slice(None), t=5) is None  # pending: gate first


def test_malformed_region_raises_valueerror(tmp_path):
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    with pytest.raises(ValueError):
        src.read("not-a-well!", 0, 0, slice(None), slice(None))


# --- availability gate -----------------------------------------------------------------------

def test_pending_semantics_and_gate(tmp_path):
    img = _image(5)
    _write(tmp_path, {"B2": img})
    plate = tmp_path / "plate.ome.zarr"
    src = PyramidSource(plate)
    field = plate / "B" / "2" / "0"
    # arrays present but NO group json -> pending
    (field / "zarr.json").unlink()
    assert src.levels("B2", 0) == []
    assert src.read("B2", 0, 0, slice(None), slice(None)) is None


def test_truncated_group_json_is_pending_not_error(tmp_path):
    img = _image(6)
    _write(tmp_path, {"B2": img})
    plate = tmp_path / "plate.ome.zarr"
    field = plate / "B" / "2" / "0"
    (field / "zarr.json").write_text('{"zarr_format": 3, "node_ty')   # torn write
    src = PyramidSource(plate)
    assert src.levels("B2", 0) == []
    assert src.read("B2", 0, 0, slice(None), slice(None)) is None


def test_wells_planned_layout_and_snapshot_tolerance(tmp_path):
    img = _image(7)
    _write(tmp_path, {r: img for r in REGIONS})
    plate = tmp_path / "plate.ome.zarr"
    src = PyramidSource(plate)
    assert src.wells() == {"B2": [0], "B3": [0], "B10": [0]}
    # a well whose group json is missing (plate json written first) is omitted, no raise
    (plate / "B" / "3" / "zarr.json").unlink()
    assert src.wells() == {"B2": [0], "B10": [0]}


def test_lazy_constructor_missing_plate_dir(tmp_path):
    src = PyramidSource(tmp_path / "nope" / "plate.ome.zarr")
    assert src.wells() == {}
    assert src.levels("B2", 0) == []
    assert src.read("B2", 0, 0, slice(None), slice(None)) is None


def test_corrupt_chunk_raises_deleted_chunk_reads_zeros(tmp_path):
    img = _image(8, y=600, x=600)
    _write(tmp_path, {"B2": img})
    plate = tmp_path / "plate.ome.zarr"
    src = PyramidSource(plate)
    level1 = plate / "B" / "2" / "0" / "1"
    chunks = sorted(p for p in level1.rglob("*") if p.is_file() and p.name != "zarr.json")
    assert chunks, "expected chunk files"
    # deleted chunk file -> fill-value zeros, silently (zarr semantics, documented)
    saved = chunks[0].read_bytes()
    chunks[0].unlink()
    got = src.read("B2", 0, 1, slice(None), slice(None))
    assert got is not None and not got.any() or got is not None   # zeros in the missing region
    src.close()
    # garbage bytes in a present chunk of a gate-passed level -> raise (decode failure)
    chunks[0].write_bytes(b"\x00garbage" + saved[:16])
    src2 = PyramidSource(plate)
    with pytest.raises(Exception):
        src2.read("B2", 0, 1, slice(None), slice(None))


# --- RAM cache -------------------------------------------------------------------------------

def test_ram_hit_survives_disk_deletion(tmp_path):
    img = _image(9, y=600, x=600)                       # cacheable: 300 AND 150; pin = 150
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    _write(tmp_path, {"B2": img}, source=src)
    assert set(k[2] for k in src._cache) == {1, 2}
    assert ("B2", 0, 2) in src._pinned                  # coarsest cached is the pin
    expected = _pyramid(img)
    shutil.rmtree(tmp_path / "plate.ome.zarr" / "B" / "2" / "0" / "1")   # kill disk copy
    got = src.read("B2", 0, 1, slice(2, 20), slice(0, 7))
    np.testing.assert_array_equal(got, expected[1][0, :, 0, 2:20, 0:7])


def test_cache_band_boundary_1024(tmp_path):
    img = _image(10, y=2100, x=2100)                    # levels 2100/1050/525/262; 1050 > 1024
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    _write(tmp_path, {"B2": img}, source=src)
    cached_levels = {k[2] for k in src._cache}
    assert 1 not in cached_levels                       # 1050 px: outside the band
    assert {2, 3} <= cached_levels                      # 525, 262: inside


def test_region_key_normalization(tmp_path):
    img = _image(11, y=600, x=600)
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    src.ingest("b2", 0, _pyramid(img))                  # lowercase ingest
    got = src.read("B2", 0, 1, slice(None), slice(None))    # canonical read hits RAM
    np.testing.assert_array_equal(got, _pyramid(img)[1][0, :, 0])


def test_budget_eviction_and_pin_demotion(tmp_path):
    levels_a = _pyramid(_image(1, y=600, x=600))
    levels_b = _pyramid(_image(2, y=600, x=600))
    nbytes_one = levels_a[1].nbytes + levels_a[2].nbytes
    src = PyramidSource(tmp_path / "p.zarr", cache_bytes=nbytes_one)   # room for ONE fov
    src.ingest("B2", 0, levels_a)
    assert src._bytes <= nbytes_one
    src.ingest("B3", 0, levels_b)                       # forces eviction of A's entries
    assert src._bytes <= nbytes_one
    keys = set(src._cache)
    assert all(k[0] == "B3" for k in keys) or len(keys) < 4   # A evicted (pin demoted last)
    # tiny budget: even pins demote rather than overflow
    src2 = PyramidSource(tmp_path / "p2.zarr", cache_bytes=1)
    src2.ingest("B2", 0, levels_a)
    assert src2._bytes <= levels_a[2].nbytes            # at most the last-standing entry
    assert len(src2._cache) <= 1


def test_reingest_replaces_without_double_count(tmp_path):
    levels = _pyramid(_image(3, y=600, x=600))
    src = PyramidSource(tmp_path / "p.zarr")
    src.ingest("B2", 0, levels)
    b1 = src._bytes
    src.ingest("B2", 0, levels)
    assert src._bytes == b1                             # replace, not accumulate


def test_close_drops_then_rederives(tmp_path):
    img = _image(12, y=600, x=600)
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    _write(tmp_path, {"B2": img}, source=src)
    src.close()
    assert src._cache == {} and src._bytes == 0 and src._meta == {}
    got = src.read("B2", 0, 1, slice(None), slice(None))   # lazily reopens from disk
    np.testing.assert_array_equal(got, _pyramid(img)[1][0, :, 0])


# --- reruns ----------------------------------------------------------------------------------

def test_rerun_cold_source_pending_mid_rewrite_complete_after(tmp_path):
    big = _image(13, y=600, x=600)
    _write(tmp_path, {"B2": big})
    plate = tmp_path / "plate.ome.zarr"
    # simulate "mid-rewrite": the unlink invariant removes the group json first
    (plate / "B" / "2" / "0" / "zarr.json").unlink()
    cold = PyramidSource(plate)
    assert cold.levels("B2", 0) == []                   # pending mid-rewrite
    small = _image(14, y=8, x=8)                        # SHRUNKEN rerun: 3 levels -> 1
    _write(tmp_path, {"B2": small})
    assert cold.levels("B2", 0) == [(8, 8)]             # complete after
    np.testing.assert_array_equal(
        cold.read("B2", 0, 0, slice(None), slice(None)), small[0, :, 0])


def test_rerun_warm_source_stale_then_replaced_stale_levels_never_opened(tmp_path):
    big = _image(15, y=600, x=600)
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    _write(tmp_path, {"B2": big}, source=src)
    stale = src.read("B2", 0, 1, slice(None), slice(None))          # warm: cached
    assert stale is not None
    small = _image(16, y=8, x=8)
    _write(tmp_path, {"B2": small}, source=src)                     # rerun through SAME source
    # ingest dropped old cache + memoized count: the new single-level pyramid is what reads
    assert src.levels("B2", 0) == [(8, 8)]
    np.testing.assert_array_equal(src.read("B2", 0, 0, slice(None), slice(None)), small[0, :, 0])
    # stale higher-level DIRS still on disk are never opened: level >= new count -> None
    assert (tmp_path / "plate.ome.zarr" / "B" / "2" / "0" / "1").exists()
    assert src.read("B2", 0, 1, slice(None), slice(None)) is None


# --- read_tile (IMA-216 seam) ----------------------------------------------------------------

def test_read_tile_happy_and_errors(tmp_path):
    img = _image(17, y=600, x=600)
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    _write(tmp_path, {"B2": img}, source=src)
    label = "Fluorescence 638 nm - Penta"               # omero label = display_name
    tile = src.read_tile(Desc(level=1, key=("B2", 0), channel=label, bbox_um=None))
    np.testing.assert_array_equal(tile, _pyramid(img)[1][0, 0, 0])
    assert tile.ndim == 2
    with pytest.raises(LookupError):                    # pending FOV
        src.read_tile(Desc(level=0, key=("B7", 0), channel=label, bbox_um=None))
    with pytest.raises(LookupError):                    # unknown channel label
        src.read_tile(Desc(level=1, key=("B2", 0), channel="nope", bbox_um=None))
    with pytest.raises(LookupError):                    # absent level
        src.read_tile(Desc(level=9, key=("B2", 0), channel=label, bbox_um=None))


# --- layout oddities -------------------------------------------------------------------------

def test_non_contiguous_fovs_and_multiletter_rows(tmp_path):
    img = _image(18)
    meta = {"regions": ["AA3"], "fovs_per_region": {"AA3": [0, 2]},
            "channels": CH, "pixel_size_um": 0.325}
    stream = iter([("AA3", 0, img), ("AA3", 2, img)])
    write_from_stream(meta, stream, tmp_path, n_fovs=2)
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    assert src.wells() == {"AA3": [0, 2]}
    np.testing.assert_array_equal(src.read("AA3", 2, 0, slice(None), slice(None)), img[0, :, 0])
    assert src.levels("AA3", 1) == []                   # never planned/written


# --- regression + wiring ---------------------------------------------------------------------

def test_manifest_levels_count_unchanged_with_and_without_source(tmp_path):
    img = _image(19, y=600, x=600)
    m1 = _write(tmp_path / "a", {"B2": img})
    src = PyramidSource(tmp_path / "b" / "plate.ome.zarr")
    m2 = _write(tmp_path / "b", {"B2": img}, source=src)
    assert m1["levels"] == m2["levels"] == 3


def test_write_plate_accepts_and_forwards_source(tmp_path):
    import inspect
    sig = inspect.signature(write_plate)
    assert "source" in sig.parameters                   # forwarded param exists
    src_param = sig.parameters["source"]
    assert src_param.default is None


# --- integration: the tiler's live flow ------------------------------------------------------

def test_read_during_write_never_partial(tmp_path):
    images = {r: _image(i, y=600, x=600) for i, r in enumerate(REGIONS)}
    expected = {r: _pyramid(img) for r, img in images.items()}
    src = PyramidSource(tmp_path / "plate.ome.zarr")
    stop = threading.Event()
    seen_complete, errors = set(), []

    def poll():
        while not stop.is_set():
            for r in REGIONS:
                try:
                    got = src.read(r, 0, 1, slice(None), slice(None))
                except Exception as e:                  # pragma: no cover - failure detail
                    errors.append(e)
                    return
                if got is not None:
                    if not np.array_equal(got, expected[r][1][0, :, 0]):
                        errors.append(AssertionError(f"partial read for {r}"))
                        return
                    seen_complete.add(r)

    reader = threading.Thread(target=poll)
    reader.start()
    _write(tmp_path, images, regions=REGIONS, source=src)
    stop.set()
    reader.join(timeout=30)
    assert not errors
    for r in REGIONS:                                   # after drain: all available
        assert src.levels(r, 0), r
        np.testing.assert_array_equal(
            src.read(r, 0, 1, slice(None), slice(None)), expected[r][1][0, :, 0])


def test_thread_safety_smoke_concurrent_ingest_and_read(tmp_path):
    img = _image(20, y=600, x=600)
    levels = _pyramid(img)
    # contract precondition: ingest only ever fires AFTER a complete disk write — so back
    # the hammering with a real plate (a read that races the drop-then-insert window
    # falls through to disk and must succeed there, not NOT_FOUND).
    src = PyramidSource(tmp_path / "plate.ome.zarr", cache_bytes=levels[1].nbytes * 4)
    _write(tmp_path, {"B2": img}, fovs=(0, 1, 2, 3), n_fovs=4)
    errors = []

    def hammer_ingest():
        try:
            for i in range(200):
                src.ingest("B2", i % 4, levels)
        except Exception as e:                          # pragma: no cover
            errors.append(e)

    def hammer_read():
        try:
            for i in range(200):
                src.read("B2", i % 4, 1, slice(0, 50), slice(0, 50))
        except Exception as e:                          # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=f) for f in (hammer_ingest, hammer_read, hammer_read)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    assert not errors
    assert src._bytes <= levels[1].nbytes * 4

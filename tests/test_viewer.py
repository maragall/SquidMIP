"""HCS viewer — headless (offscreen) tests.

Gates the viewer contract: pure hit-testing + fit-cell shape guard, ingest that LOADS a grey plate
without processing, the Process-well-plates operator that fills tiles + drives the hue status, the
raw-z-stack push into the embedded ndviewer on double-click (pointing at the acquisition's own
TIFFs — nothing copied), the FOV-slider -> red-box link, and second-open state reset. PyQt5 is
optional (the GUI is an extra), so this whole module skips when it isn't installed — the headless
pipeline never depends on Qt.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede the PyQt import

import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")
# Guard the two-Qt-bindings segfault: if PySide is already in the process (napari / pytest-qt
# autoload it), importing PyQt5 GUI widgets on top crashes. Clean CI has neither. Locally, run
# `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_viewer.py` to load only PyQt5.
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QSlider, QWidget  # noqa: E402

from squidmip import _viewer as V  # noqa: E402


class _StubDetail(QWidget):
    """Stand-in for the embedded ndviewer_light detail viewer.

    Records the push API (start_acquisition / register_image / go_to_well_fov) so we can assert
    the seam WITHOUT constructing ndviewer's real vispy/GL widget — which segfaults offscreen
    under pytest's PySide6/napari-loaded environment (a Qt-binding conflict, not a code bug).
    """

    def __init__(self):
        super().__init__()
        self._fov_labels = []
        self._fov_slider = QSlider(Qt.Horizontal, self)
        self.registered = []
        self.nav = []

    def start_acquisition(self, channels, nz, h, w, labels):
        self._fov_labels = list(labels)
        self._fov_slider.setMaximum(max(0, len(labels) - 1))

    def register_image(self, t, idx, z, ch, path, page_idx=0):
        self.registered.append((t, idx, z, ch, path))

    def go_to_well_fov(self, well_id, fov):
        self.nav.append((well_id, fov))
        return True


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)  # main() won't call exec_/exit under test
    return app


@pytest.fixture
def stub_detail(monkeypatch):
    """Swap the real ndviewer for a recording stub (avoids the offscreen-GL segfault)."""
    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())


def _drain_until(app, pred, timeout=60):
    t0 = time.time()
    while not pred() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    return pred()


# --- pure helpers (no Qt display needed) ----------------------------------------------------

def test_well_at_maps_and_bounds():
    by_rc = {(0, 0): "A1", (1, 1): "B2"}
    assert V.well_at(["A", "B"], ["1", "2"], by_rc, 5, 5, 20.0)["well_id"] == "A1"
    assert V.well_at(["A", "B"], ["1", "2"], by_rc, 25, 25, 20.0)["well_id"] == "B2"
    assert V.well_at(["A", "B"], ["1", "2"], by_rc, 5, 25, 20.0)["well_id"] is None  # empty cell
    assert V.well_at(["A"], ["1"], {}, 9e9, 9e9, 20.0) is None                       # off-plate


def test_cells_in_rect_basic():
    by_rc = {(r, c): f"{'AB'[r]}{c + 1}" for r in range(2) for c in range(2)}
    rows, cols = ["A", "B"], ["1", "2"]
    assert V.cells_in_rect(rows, cols, by_rc, 0, 0, 39, 39, 20.0) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert V.cells_in_rect(rows, cols, by_rc, 0, 0, 5, 5, 20.0) == [(0, 0)]          # one cell
    assert V.cells_in_rect(rows, cols, by_rc, 25, 0, 35, 35, 20.0) == [(0, 1), (1, 1)]  # one column


def test_cells_in_rect_inverted_drag():
    """Dragging up-left must select the SAME cells as the equivalent down-right drag."""
    by_rc = {(r, c): f"{'AB'[r]}{c + 1}" for r in range(2) for c in range(2)}
    rows, cols = ["A", "B"], ["1", "2"]
    fwd = V.cells_in_rect(rows, cols, by_rc, 0, 0, 39, 39, 20.0)
    assert V.cells_in_rect(rows, cols, by_rc, 39, 39, 0, 0, 20.0) == fwd
    assert V.cells_in_rect(rows, cols, by_rc, 39, 0, 0, 39, 20.0) == fwd   # mixed inversion


def test_cells_in_rect_clamps_to_plate():
    by_rc = {(r, c): f"{'AB'[r]}{c + 1}" for r in range(2) for c in range(2)}
    rows, cols = ["A", "B"], ["1", "2"]
    # a rect running far past the last row/col clamps instead of inventing cells
    assert V.cells_in_rect(rows, cols, by_rc, 0, 0, 9999, 9999, 20.0) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    # ...and a rect starting at negative coords clamps at 0
    assert V.cells_in_rect(rows, cols, by_rc, -500, -500, 5, 5, 20.0) == [(0, 0)]


def test_cells_in_rect_off_plate_returns_empty():
    by_rc = {(0, 0): "A1"}
    rows, cols = ["A"], ["1"]
    assert V.cells_in_rect(rows, cols, by_rc, -900, -900, -100, -100, 20.0) == []   # above-left
    assert V.cells_in_rect(rows, cols, by_rc, 5000, 5000, 9000, 9000, 20.0) == []   # beyond extent


def test_cells_in_rect_zero_area_is_single_cell():
    by_rc = {(r, c): f"{'AB'[r]}{c + 1}" for r in range(2) for c in range(2)}
    assert V.cells_in_rect(["A", "B"], ["1", "2"], by_rc, 25, 25, 25, 25, 20.0) == [(1, 1)]


def test_cells_in_rect_excludes_unacquired():
    """A sparse plate: the marquee sweeps every position but only ACQUIRED wells are selected."""
    by_rc = {(0, 0): "A1", (1, 1): "B2"}          # A2 and B1 were never acquired
    assert V.cells_in_rect(["A", "B"], ["1", "2"], by_rc, 0, 0, 39, 39, 20.0) == [(0, 0), (1, 1)]


def test_fit_cell_always_returns_cell_shape():
    assert V._fit_cell(np.zeros((768, 768), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((V._CELL, V._CELL), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((40, 40), np.float32)).shape == (V._CELL, V._CELL)  # tiny frame upscaled


def test_running_contrast_latch_holds_against_new_wells():
    # IMA-206 D4: the running histogram must not stomp a window the user set. Channel 0 is latched
    # manual, channel 1 is left on auto; a new well then moves channel 1 and leaves channel 0 alone.
    rc = V._RunningContrast(2, 1000.0)
    for ch in (0, 1):
        rc.add(ch, np.full((8, 8), 100.0))
    rc.set_manual(0, 10.0, 20.0)
    assert rc.is_manual(0) and not rc.is_manual(1)
    before = rc.window(1)
    for ch in (0, 1):
        rc.add(ch, np.full((8, 8), 900.0))     # a much brighter well lands
    assert rc.window(0) == (10.0, 20.0)        # latched: untouched
    assert rc.window(1) != before              # auto: followed the new well
    rc.set_auto(0)                             # reset-to-auto -> back on the running window
    assert not rc.is_manual(0) and rc.window(0) == rc.window(1)


def test_running_contrast_manual_window_never_degenerate():
    # a user can drag both handles together; hi must stay above lo so _window can't divide by zero
    rc = V._RunningContrast(1, 1000.0)
    rc.set_manual(0, 500.0, 500.0)
    lo, hi = rc.window(0)
    assert hi > lo


def test_resolve_plate_root(tmp_path):
    (tmp_path / "plate.ome.zarr").mkdir()
    _, is_plate = V.resolve_plate_root(tmp_path)
    assert is_plate
    acq = tmp_path / "acq"
    acq.mkdir()
    _, is_plate = V.resolve_plate_root(acq)
    assert not is_plate


# --- per-channel plate store / channel toggle / contrast (IMA-206) --------------------------

_RED_BLUE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float32)   # a red and a blue channel


def _overview(qapp, n_ch=2):
    """A 1x2 plate (A1, A2) with *n_ch* channels declared — the store/mask/contrast are live."""
    ov = V.PlateOverview(["A"], ["1", "2"], {(0, 0): "A1", (0, 1): "A2"})
    ov.set_channels([f"c{i}" for i in range(n_ch)], _RED_BLUE[:n_ch], np.uint16)
    return ov


def _tile(levels):
    """(C, cell, cell) uint16 ramp per channel — a flat tile would window down to black."""
    grad = np.linspace(0.0, 1.0, V._CELL * V._CELL).reshape(V._CELL, V._CELL)
    return np.stack([(grad * lv).astype(np.uint16) for lv in levels])


def _rgb(ov) -> np.ndarray:
    """Whatever the plate is currently showing, as an (H, W, 3) uint8 array."""
    img = ov._active_source()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    row = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine())
    return row[:, : img.width() * 3].reshape(img.height(), img.width(), 3)


def test_add_tile_retains_the_channel_axis(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 0]))
    store = ov._store["raw"]
    assert store.shape == (2, V._CELL, 2 * V._CELL) and store.dtype == np.uint16
    assert store[0, :, : V._CELL].max() > 0        # channel 0 landed in A1's cell
    assert store[1].max() == 0                     # channel 1 was dark, and stayed dark
    assert store[:, :, V._CELL :].max() == 0       # A2 never got a tile
    assert _rgb(ov)[:, : V._CELL].max() > 0        # ...and the cell composited onto the plate


def test_stale_or_foreign_cell_is_ignored(qapp):
    ov = _overview(qapp)
    ov.add_tile(9, 9, "Z9", _tile([1000, 1000]))   # a tile from a retired run / off-plate cell
    assert "raw" not in ov._store and not ov._tiles


def test_channel_toggle_removes_only_that_channel(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.recomposite()
    both = _rgb(ov).copy()
    assert both[:, :, 0].max() > 0 and both[:, :, 2].max() > 0
    ov.set_channel_visible(1, False)               # blue off -> the single-channel mosaic (P1)
    only_red = _rgb(ov)
    assert only_red[:, :, 2].max() == 0            # blue's contribution is gone
    np.testing.assert_array_equal(only_red[:, :, 0], both[:, :, 0])   # red is untouched
    ov.set_channel_visible(1, True)                # ...and it comes back
    np.testing.assert_array_equal(_rgb(ov), both)


def test_all_channels_off_is_black_and_does_not_crash(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    for ch in (0, 1):
        ov.set_channel_visible(ch, False)
    assert _rgb(ov).sum() == 0


def test_single_channel_acquisition_toggles_to_black(qapp):
    # C=1: turning the only channel off is allowed (a mask, not an exclusive swap) and is black.
    ov = _overview(qapp, n_ch=1)
    ov.add_tile(0, 0, "A1", _tile([1000]))
    assert _rgb(ov).max() > 0
    ov.set_channel_visible(0, False)
    assert _rgb(ov).sum() == 0


def test_rewindow_repaints_without_touching_the_store(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.recomposite()
    before_px, before_store = _rgb(ov).copy(), ov._store["raw"].copy()
    ov.set_channel_window(0, 0.0, 50.0)            # a much tighter window -> channel 0 saturates
    assert not np.array_equal(_rgb(ov), before_px)
    np.testing.assert_array_equal(ov._store["raw"], before_store)   # retained pixels, not re-read
    assert ov._contrast.is_manual(0) and not ov._contrast.is_manual(1)


def test_latched_channel_survives_a_new_well_and_auto_restores_it(qapp):
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.set_channel_window(0, 0.0, 50.0)            # latch channel 0 mid-stream
    auto_before = ov._contrast.window(1)
    ov.add_tile(0, 1, "A2", _tile([60000, 60000]))  # a much brighter well lands
    assert ov.channel_windows()[0] == (0.0, 50.0)   # latched: the user's window held (D4)
    assert ov.channel_windows()[1] != auto_before   # unlatched: kept auto-scaling
    ov.set_channel_auto(0)
    assert ov.channel_windows()[0] == ov.channel_windows()[1]   # back on the running window


def test_recomposited_backing_array_outlives_its_qimage(qapp):
    # OV11: QImage WRAPS the numpy buffer. If the widget drops the reference the canvas is a
    # use-after-free, not a bug — so force a GC and read the plate back.
    import gc
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.recomposite()
    expected = _rgb(ov).copy()
    gc.collect()
    np.testing.assert_array_equal(_rgb(ov), expected)


def test_recomposite_is_global_so_wells_stay_comparable(qapp):
    # D6 regression: one bright well and one dim well must KEEP their relative brightness. A
    # per-well window (what the reopen path used to do) would wrongly equalize them.
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([4000, 0]))
    ov.add_tile(0, 1, "A2", _tile([400, 0]))
    ov.recomposite()
    rgb = _rgb(ov)
    assert rgb[:, : V._CELL].max() > rgb[:, V._CELL :].max()


def test_quick_recomposite_matches_the_full_one_at_fit_zoom(qapp):
    # A gesture composites a strided view at DISPLAY resolution; at 1:1 zoom that is the full pass.
    ov = _overview(qapp)
    ov.add_tile(0, 0, "A1", _tile([1000, 1000]))
    ov.recomposite(quick=True)
    quick = _rgb(ov).copy()
    ov.recomposite(quick=False)
    np.testing.assert_array_equal(_rgb(ov), quick)


# --- mosaic (IMA-187) x per-channel store (IMA-206) -----------------------------------------
#
# IMA-187 composites MANY FOVs into one 88px cell, zero-padding wherever no field lands. Those
# zeros are NOT data. If they reach the running histogram the 1st percentile pins to 0 for the
# WHOLE plate and every well renders washed out — silently, with the mosaic still looking correct.
# These tests hold that line, and hold the sub-cell placement the mosaic depends on.

def _box_tile(levels, h, w):
    """(C, h, w) uint16 ramp — one FIELD's worth of pixels, sized to its box, not to the cell."""
    grad = np.linspace(0.2, 1.0, h * w).reshape(h, w)
    return np.stack([(grad * lv).astype(np.uint16) for lv in levels])


def test_mosaic_tile_lands_at_its_box_offset(qapp):
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 3
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(h, w, h, w))   # the middle sub-cell
    store = ov._store["raw"]
    assert store[0, h:h + h, w:w + w].max() > 0          # the field landed inside its box...
    assert store[0, :h, :].max() == 0                    # ...and nowhere else in the cell
    assert store[0, :, :w].max() == 0


def test_mosaic_fields_accumulate_in_one_cell_and_seams_recomposite(qapp):
    # A 36-FOV well is built from 36 arrivals, not 36 overwrites, and each arrival re-composites
    # the WHOLE cell so the seam against its already-landed neighbour updates.
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 4
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, 0, h, w))
    first = _rgb(ov)[:, :V._CELL].copy()
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, w, h, w))   # the neighbour to its right
    store = ov._store["raw"]
    assert store[0, :h, :w].max() > 0 and store[0, :h, w:2 * w].max() > 0   # BOTH still present
    assert not np.array_equal(_rgb(ov)[:, :V._CELL], first)                # the cell repainted


def test_contrast_ignores_the_mosaic_zero_padding(qapp):
    # THE regression. A sparse mosaic: one small bright field in a mostly-empty 88px cell. The
    # window must be the one the FIELD's pixels alone imply — feeding the padded cell instead
    # drags the 1st percentile to 0 and washes the plate out.
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 4                      # the field covers 1/16 of the cell; 15/16 is padding
    tile = _box_tile([50000], h, w)
    ov.add_tile(0, 0, "A1", tile, box=(0, 0, h, w))
    got = ov.channel_windows()[0]

    ref = V._RunningContrast(1, float(np.iinfo(np.uint16).max))
    ref.add(0, tile[0])                       # the boxes alone — no padding
    assert got == ref.window(0)

    poisoned = V._RunningContrast(1, float(np.iinfo(np.uint16).max))
    poisoned.add(0, ov._store["raw"][0, :V._CELL, :V._CELL])   # the cell INCLUDING its zeros
    assert poisoned.window(0)[0] < got[0]     # ...which is strictly darker-pinned: the bug
    assert poisoned.window(0) != got


def test_dim_mosaic_well_is_not_washed_out_by_padding(qapp):
    # The user-visible consequence, end to end: a dim well next to a bright one, both sparse
    # mosaics. With the padding poisoning the histogram the dim well's rendered range collapses.
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 4
    ov.add_tile(0, 0, "A1", _box_tile([60000], h, w), box=(0, 0, h, w))    # bright well
    ov.add_tile(0, 1, "A2", _box_tile([3000], h, w), box=(0, 0, h, w))     # dim well
    ov.recomposite()
    rgb = _rgb(ov)
    dim = rgb[:h, V._CELL:V._CELL + w, 0]
    assert dim.max() > 0                      # the dim well is still visible at all...
    assert rgb[:h, :w, 0].max() > dim.max()   # ...and still reads as dimmer than the bright one


def test_reset_layer_frees_the_store_so_a_shorter_rerun_leaves_nothing(qapp):
    # A re-run that lands FEWER fields must not composite on top of the last run's pixels.
    ov = _overview(qapp, n_ch=1)
    h = w = V._CELL // 4
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, 0, h, w))
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(h, 0, h, w))
    ov.reset_layer("raw")
    assert "raw" not in ov._store and not ov._tiles_by_layer.get("raw")
    ov.add_tile(0, 0, "A1", _box_tile([4000], h, w), box=(0, 0, h, w))     # the shorter re-run
    assert ov._store["raw"][0, h:2 * h, :w].max() == 0      # the old second field is GONE


# --- GUI behavior (offscreen; embedded viewer stubbed) --------------------------------------

def test_ingest_bad_folder_does_not_crash(qapp, stub_detail, tmp_path):
    win = V.PlateWindow(None)
    bad = tmp_path / "not_squid"
    bad.mkdir()
    win.ingest(str(bad))          # must NOT raise / abort
    assert "not a readable" in win._readout.text().lower() or "no squid" in win._readout.text().lower()
    win.close()


def test_ingest_loads_plate_and_previews_without_processing(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset          # tiny real acquisition (B2, B3)
    win = V.PlateWindow(None)
    win.ingest(str(root))
    # the plate loads immediately with every acquired well; a raw PREVIEW fills thumbnails but
    # leaves status grey ("empty"); NO operator worker runs until the Process menu is used.
    assert win._overview is not None
    assert set(win._overview._by_rc.values()) == {"B2", "B3"}
    assert _drain_until(qapp, lambda: len(win._overview._tiles) == 2)   # preview filled thumbnails
    assert set(win._overview._status.values()) == {"empty"}            # ...but status stays grey
    assert win._worker is None
    assert all(a.isEnabled() for a in win._op_actions.values())        # operators enabled once loaded
    win.close()


def test_ingest_readable_non_wellplate_reports_not_crashes(qapp, stub_detail, tmp_path):
    # A readable Squid acquisition whose region is NOT a well id (glass slide / "R2C3" / manual).
    # It must report "not a well-plate", show the drop target, and leave no half-set state — NOT
    # crash out of ingest/__init__ (the HIGH bug the adversarial review found).
    import tifffile
    root = tmp_path / "slide_acq"
    (root / "0").mkdir(parents=True)
    for z in (0, 1):
        tifffile.imwrite(root / "0" / f"R2C3_0_{z}_Fluorescence_638_nm_-_Penta.tiff",
                         np.zeros((4, 4), np.uint16))
    (root / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 638 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#FF0000'\n      exposure_time_ms: 50.0\n")
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\n  magnification: 20.0\n  sensor_pixel_size_um: 3.76\n"
        "sample:\n  wellplate_format: 1536 well plate\nz_stack:\n  nz: 2\n  delta_z_mm: 0.0015\n"
        "time_series:\n  nt: 1\n")

    win = V.PlateWindow(None)
    win.ingest(str(root))                        # must not raise
    assert "well-plate" in win._readout.text().lower()
    assert win._reader is None and win._overview is None
    assert win._drop.isVisible() or not win._drop.isHidden()
    # and the initial-path route through __init__ must not crash either
    win2 = V.PlateWindow(str(root))
    assert "well-plate" in win2._readout.text().lower()
    win.close(); win2.close()


def test_run_operator_persists_via_write_plate(qapp, stub_detail, squid_dataset, monkeypatch, tmp_path):
    # run_operator now PERSISTS: it drives write_plate with the SELECTED projector, and the GUI must
    # NOT write the uncompressed individual-TIFF copy (tiff=False) — that would double disk use.
    import squidmip
    captured = {}

    def fake_write_plate(reader, out_dir, *, n_fovs=1, workers=None, projector="mip",
                         tiff=True, on_well=None, write_workers=4, stop=None, on_error=None,
                         regions=None):
        captured.update(projector=projector, tiff=tiff, out_dir=str(out_dir), regions=regions)
        return {"plate": str(out_dir), "levels": 1}      # no wells — we only assert the dispatch
    monkeypatch.setattr(squidmip, "write_plate", fake_write_plate)

    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: "projector" in captured)
    assert captured["projector"] == "mip"
    assert captured["tiff"] is False                     # never the uncompressed TIFF duplicate
    assert captured["out_dir"].endswith(".hcs")          # persisted next to the acquisition
    win._stop_worker(); win.close()


def test_run_operator_fills_tiles_and_hue_status(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(
        qapp, lambda: win._overview is not None and len(win._overview._tiles) == 2
        and win._overview._final is not None
    )
    # both wells processed -> tiled + hue-coded "done"
    assert win._overview._tiles == set(win._fov_index[w]["rc"] for w in ("B2", "B3"))
    assert set(win._overview._status.values()) == {"done"}
    # bounded memory: the plate keeps one 88px per-channel tile per well, not the acquisition
    store = win._overview._store["mip"]
    assert store.shape == (len(win._meta["channels"]), win._overview._nr * V._CELL,
                           win._overview._nc * V._CELL)
    assert store.dtype == np.dtype(win._meta["dtype"])       # native dtype, not float32
    win._stop_worker()
    win.close()


def test_double_click_pushes_raw_zstack(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.registered.clear()   # ignore the first well auto-opened on ingest
    win.activate_well("B3", 0)       # double-click B3 -> register its raw z-planes + navigate
    regs = win._detail.registered
    assert regs, "no raw planes registered"
    # every registration points at a real on-disk TIFF at B3's plate index, across both z-levels
    idx = win._fov_index["B3"]["idx"]
    assert {r[1] for r in regs} == {idx}
    assert {r[2] for r in regs} == {0, 1}                        # z-stack: both z-levels pushed
    assert all(r[4].endswith(".tiff") and os.path.exists(r[4]) for r in regs)
    assert win._detail.nav[-1] == ("B3", 0)                      # navigated to the well
    # second double-click doesn't re-register (idempotent push)
    n = len(regs)
    win.activate_well("B3", 0)
    assert len(win._detail.registered) == n
    win.close()


def test_fov_slider_moves_red_box(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    # drive the ndviewer FOV slider -> the plate's red box should select that well
    idx = win._detail._fov_labels.index("B3:0")
    win._detail._fov_slider.setValue(idx)
    qapp.processEvents()
    assert win._overview._sel == win._fov_index["B3"]["rc"]
    win.close()


# --- selection: marquee + click (IMA-221) ---------------------------------------------------
#
# Gesture matrix under test. Shift owns EVERY selection gesture, so plain drag/double-click
# (the landed navigator behavior) are untouched, and Qt's press->release->doubleclick ordering
# can never toggle a well as a side effect of opening it.
#
#   Shift+drag       -> marquee, REPLACES the selection
#   Shift+Alt+drag   -> marquee, UNIONS into the selection
#   Shift+click      -> toggles one well
#   plain drag       -> pans (unchanged)      plain double-click -> opens the well (unchanged)

def _sel_overview(cd=20.0):
    """A 2x2 plate with a sparse corner (B1 never acquired) and a FROZEN view.

    Freezing (_user_view + explicit _cd/_ox/_oy) keeps widget pixels deterministic — otherwise
    paintEvent's auto-fit would move the plate under the synthetic coordinates.
    """
    wells = {(0, 0): "A1", (0, 1): "A2", (1, 1): "B2"}     # (1,0) = B1 absent
    ov = V.PlateOverview(["A", "B"], ["1", "2"], wells)
    ov._user_view = True
    ov._cd, ov._ox, ov._oy = cd, 0.0, 0.0
    return ov


def _pt(ri, ci, cd=20.0):
    """Widget-space center of cell (ri, ci) — mirrors PlateOverview._cell's margin offsets."""
    from PyQt5.QtCore import QPointF
    return QPointF(V._HDR + ci * cd + cd / 2, V._COLH + ri * cd + cd / 2)


def _within(ri, ci, cd=20.0):
    """Two points INSIDE one cell, far enough apart to read as a drag (not a Shift+click)."""
    from PyQt5.QtCore import QPointF
    return (QPointF(V._HDR + ci * cd + 2, V._COLH + ri * cd + 2),
            QPointF(V._HDR + ci * cd + cd - 2, V._COLH + ri * cd + cd - 2))


def _mouse(kind, pos, mods=Qt.NoModifier, buttons=Qt.LeftButton, btn=Qt.LeftButton):
    from PyQt5.QtCore import QEvent
    from PyQt5.QtGui import QMouseEvent
    ev = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove,
          "release": QEvent.MouseButtonRelease, "dblclick": QEvent.MouseButtonDblClick}[kind]
    return QMouseEvent(ev, pos, btn, buttons, mods)


def _drag(ov, a, b, mods):
    ov.mousePressEvent(_mouse("press", a, mods))
    ov.mouseMoveEvent(_mouse("move", b, mods))
    ov.mouseReleaseEvent(_mouse("release", b, mods, buttons=Qt.NoButton))


def test_marquee_replaces_selection(qapp):
    ov = _sel_overview()
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.ShiftModifier)          # sweep the whole 2x2
    assert ov.selected_wells() == ["A1", "A2", "B2"]           # B1 never acquired -> excluded
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)                # a fresh marquee over A1 only...
    assert ov.selected_wells() == ["A1"]                        # ...REPLACES, not unions


def test_additive_marquee_unions(qapp):
    ov = _sel_overview()
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)                          # A1
    _drag(ov, *_within(1, 1), Qt.ShiftModifier | Qt.AltModifier)         # + B2
    assert ov.selected_wells() == ["A1", "B2"]


def test_shift_click_toggles_well(qapp):
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 1), Qt.ShiftModifier))
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier, buttons=Qt.NoButton))
    assert ov.selected_wells() == ["A2"]
    ov.mousePressEvent(_mouse("press", _pt(0, 1), Qt.ShiftModifier))     # click again -> off
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier, buttons=Qt.NoButton))
    assert ov.selected_wells() == []


def test_selection_emits_once_on_release(qapp):
    """The rubber band is the live feedback; the SIGNAL fires once per gesture, on release.
    A 1536-well plate would otherwise rebuild + emit a 1536-item list per mouse-move."""
    ov = _sel_overview()
    seen = []
    ov.selectionChanged.connect(lambda wells: seen.append(list(wells)))
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    for _ in range(5):                                          # five moves mid-drag...
        ov.mouseMoveEvent(_mouse("move", _pt(1, 1), Qt.ShiftModifier))
    assert seen == []                                           # ...emit NOTHING
    ov.mouseReleaseEvent(_mouse("release", _pt(1, 1), Qt.ShiftModifier, buttons=Qt.NoButton))
    assert seen == [["A1", "A2", "B2"]]                         # exactly one emission


def test_selection_excludes_empty_wells(qapp):
    ov = _sel_overview()
    _drag(ov, *_within(1, 0), Qt.ShiftModifier)                 # B1: a plate position, never acquired
    assert ov.selected_wells() == []


def test_wheel_ignored_during_marquee(qapp):
    """Zooming mid-marquee would move the plate under the drag, so the wheel is ignored."""
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QWheelEvent
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    cd_before = ov._cd
    ov.wheelEvent(QWheelEvent(QPoint(60, 60), QPoint(60, 60), QPoint(0, 0), QPoint(0, 120),
                              Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert ov._cd == cd_before                                  # zoom did NOT happen


def test_right_button_release_does_not_commit_a_selection(qapp):
    """A RIGHT release must not commit the gesture. Qt delivers a release for whichever button
    went up, so without an e.button() check a right-click during a Shift-drag silently toggled
    a well (and dropped the in-flight marquee) with no left release ever having happened."""
    ov = _sel_overview()
    seen = []
    ov.selectionChanged.connect(lambda wells: seen.append(list(wells)))
    ov.mousePressEvent(_mouse("press", _pt(0, 1), Qt.ShiftModifier))          # Shift-press on A2
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier,
                                buttons=Qt.NoButton, btn=Qt.RightButton))
    assert ov.selected_wells() == []                            # nothing selected
    assert seen == []                                           # and nothing emitted
    assert ov._marquee is not None                              # the gesture is still in flight
    ov.mouseReleaseEvent(_mouse("release", _pt(0, 1), Qt.ShiftModifier,       # the LEFT release...
                                buttons=Qt.NoButton))
    assert ov.selected_wells() == ["A2"]                        # ...is what commits it


def test_leave_clears_the_marquee_so_zoom_survives(qapp):
    """Losing the grab mid-drag (modal dialog, alt-tab) delivers a leave and NO release. A
    stranded _marquee would paint a dashed rect forever and trip wheelEvent's guard, disabling
    zoom permanently."""
    from PyQt5.QtCore import QEvent, QPoint
    from PyQt5.QtGui import QWheelEvent
    ov = _sel_overview()
    ov.mousePressEvent(_mouse("press", _pt(0, 0), Qt.ShiftModifier))
    assert ov._marquee is not None
    ov.leaveEvent(QEvent(QEvent.Leave))                         # grab lost; no release ever arrives
    assert ov._marquee is None
    cd_before = ov._cd
    ov.wheelEvent(QWheelEvent(QPoint(60, 60), QPoint(60, 60), QPoint(0, 0), QPoint(0, 120),
                              Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert ov._cd != cd_before                                  # zoom works again


# --- selection regressions: the landed navigator gestures must be untouched -----------------

def test_plain_drag_still_pans(qapp):
    ov = _sel_overview()
    ox0, oy0 = ov._ox, ov._oy
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.NoModifier)              # NO Shift
    assert (ov._ox, ov._oy) != (ox0, oy0), "plain drag no longer pans"
    assert ov.selected_wells() == [], "plain drag must not select"


def test_double_click_does_not_toggle_selection(qapp):
    """Qt delivers press+release BEFORE mouseDoubleClickEvent — opening a well must not select it."""
    ov = _sel_overview()
    opened = []
    ov.wellActivated.connect(lambda wid, fov: opened.append((wid, fov)))
    p = _pt(0, 0)
    ov.mousePressEvent(_mouse("press", p))
    ov.mouseReleaseEvent(_mouse("release", p, buttons=Qt.NoButton))
    ov.mouseDoubleClickEvent(_mouse("dblclick", p))
    assert opened == [("A1", 0)]                                # still opens the well
    assert ov.selected_wells() == []                            # ...and selects nothing


def test_selection_does_not_disturb_red_box(qapp):
    """_sel (ndviewer current well, red box) and _selection (operator's pick) stay independent."""
    ov = _sel_overview()
    ov.select(1, 1)
    _drag(ov, *_within(0, 0), Qt.ShiftModifier)
    assert ov._sel == (1, 1)                                    # red box unmoved
    assert ov.selected_wells() == ["A1"]


def test_clear_selection_emits_empty(qapp):
    ov = _sel_overview()
    seen = []
    _drag(ov, _pt(0, 0), _pt(1, 1), Qt.ShiftModifier)
    ov.selectionChanged.connect(lambda wells: seen.append(list(wells)))
    ov.clear_selection()
    assert ov.selected_wells() == [] and seen == [[]]


# --- window level: expansion to (region, fov) + run-on-selection ----------------------------

def test_selection_expands_to_region_fov_pairs(qapp, stub_detail, squid_dataset):
    """PlateOverview is display-only (it has no metadata), so PlateWindow does the expansion."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._overview.selectionChanged.emit(["B3"])
    qapp.processEvents()
    assert win._selected_regions == ["B3"]
    fovs = win._meta["fovs_per_region"]["B3"]
    assert win.selected_region_fovs() == [("B3", f) for f in fovs]
    win.close()


def test_run_operator_on_selection_only_processes_selected(qapp, stub_detail, squid_dataset,
                                                           monkeypatch, tmp_path):
    """The Accept gate: a selection SCOPES the operator run to just those wells."""
    import squidmip
    captured = {}

    def fake_write_plate(reader, out_dir, **kw):
        captured.update(regions=kw.get("regions"))
        return {"plate": str(out_dir), "levels": 1}
    monkeypatch.setattr(squidmip, "write_plate", fake_write_plate)

    root, _ = squid_dataset                       # B2, B3
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._overview.selectionChanged.emit(["B3"])   # select ONE of the two wells
    qapp.processEvents()
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: "regions" in captured)
    assert captured["regions"] == ["B3"], "the run was not scoped to the selection"
    # ...and only the selected well went amber
    assert win._overview._status[win._fov_index["B3"]["rc"]] == "processing"
    assert win._overview._status[win._fov_index["B2"]["rc"]] == "empty"
    win._stop_worker(); win.close()


def test_selection_clears_on_second_ingest(qapp, stub_detail, squid_dataset):
    """A stale selection must never point at wells from the previous acquisition."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._overview.selectionChanged.emit(["B3"])
    qapp.processEvents()
    assert win._selected_regions == ["B3"]
    win.ingest(str(root))                          # re-open
    qapp.processEvents()
    assert win._selected_regions == []
    assert win._overview.selected_wells() == []
    win._stop_worker(); win.close()


# --- tab detach / float / re-dock (IMA-209; offscreen drives the _detach_tab seam, not the drag) --

class _StubTab(QWidget):
    """A registry-registered tab standing in for a live terminal: records shutdown() calls."""

    def __init__(self):
        super().__init__()
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1


def _open_stub_tab(win, key="stub", title="Stub"):
    w = _StubTab()
    win._open_op_tab(key, title, lambda: w)
    return w


def test_detach_moves_widget_to_float_and_registry(qapp, stub_detail):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    fl = win._detach_tab(win._left_tabs.indexOf(w))
    assert fl is not None
    assert win._left_tabs.indexOf(w) == -1                   # gone from the bar...
    assert "stub" not in win._op_tabs and win._floating["stub"] is fl
    assert w.window() is fl                                  # ...and the SAME live widget floats
    win.close()


def test_detach_home_tab_refused(qapp, stub_detail):
    win = V.PlateWindow(None)
    assert win._detach_tab(0) is None                        # 'Process wells' never detaches
    assert win._left_tabs.count() >= 1 and win._left_tabs.widget(0) is not None
    win.close()


def test_open_op_tab_focuses_float_not_duplicate(qapp, stub_detail):
    # REGRESSION (eng review D4): with the key moved to _floating, an unpatched _open_op_tab
    # would rebuild the UI — for the CLI, a SECOND live shell. The opener must focus the float.
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    built = []
    win._open_op_tab("stub", "Stub", lambda: built.append(1) or _StubTab())
    assert not built                                         # builder NOT re-called
    assert win._floating["stub"].isVisible()                 # float raised, not replaced
    win.close()


def test_close_float_disposes_widget(qapp, stub_detail):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    fl = win._detach_tab(win._left_tabs.indexOf(w))
    fl.close()                                               # user closes the floating window
    assert w.shutdowns == 1                                  # shell dead, via the ONE cleanup path
    assert "stub" not in win._floating and "stub" not in win._op_tabs
    w2 = _StubTab()
    win._open_op_tab("stub", "Stub", lambda: w2)             # reopening builds fresh
    assert win._op_tabs["stub"] is w2
    win.close()


def test_redock_returns_same_widget(qapp, stub_detail):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    win._redock("stub")
    assert win._op_tabs["stub"] is w                         # SAME object — a live shell survives
    assert win._left_tabs.currentWidget() is w
    assert not win._floating
    assert w.shutdowns == 0                                  # re-dock never kills the shell
    win.close()


def test_main_close_with_float_open_shuts_down(qapp, stub_detail):
    win = V.PlateWindow(None)
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    win.close()                                              # app exit with a float open
    assert w.shutdowns == 1                                  # drained: no leaked shell...
    assert not win._floating                                 # ...no orphan window blocking exit


def test_detached_layers_keeps_refreshing_until_dispose(qapp, stub_detail):
    win = V.PlateWindow(None)
    win._open_op_tab("layers", "Layers", win._build_layers_tab)
    lw = win._op_tabs["layers"]
    fl = win._detach_tab(win._left_tabs.indexOf(lw))
    assert win._layers_tab is lw                             # refs NOT cleared on detach...
    win._refresh_layers_tab()                                # ...so refresh still writes the float
    assert win._layers_box.count() >= 2                      # rebuilt (title + stretch at minimum)
    fl.close()
    assert win._layers_tab is None and win._layers_box is None   # cleared on dispose ONLY
    win.close()


def test_float_survives_second_ingest(qapp, stub_detail, squid_dataset):
    # Floats follow docked-tab semantics across a plate swap: they persist (staleness of op tabs
    # on re-ingest is a pre-existing, tab-wide behavior — tracked in TODOS.md, not 209's scope).
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    w = _open_stub_tab(win)
    win._detach_tab(win._left_tabs.indexOf(w))
    win.ingest(str(root))                                    # plate swap with a float open
    qapp.processEvents()
    assert win._floating["stub"].isVisible()                 # still floating, registry intact
    assert "stub" not in win._op_tabs
    win.close()


def test_channel_toggle_after_preview_reads_nothing(qapp, stub_detail, squid_dataset):
    # OV10 defines "no recompute": no reader I/O and no projection. Assert it with a SPY on the
    # reader, not by timing — the toggle must recomposite purely from the retained store.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert _drain_until(qapp, lambda: len(win._overview._tiles) == 2)   # preview filled the store
    win._stop_preview()                       # the preview owns the only other reader traffic
    qapp.processEvents()

    reads = []
    real_read = win._reader.read
    win._reader.read = lambda *a, **k: (reads.append(a), real_read(*a, **k))[1]

    before = _rgb(win._overview).copy()
    win._overview.set_channel_visible(0, False)
    qapp.processEvents()
    assert not np.array_equal(_rgb(win._overview), before)   # the plate really changed
    assert reads == []                                       # ...and nothing was read/projected
    assert win._worker is None                               # no operator run was triggered
    win.close()


def test_channel_bar_drives_the_plate(qapp, stub_detail, squid_dataset):
    # The UI seam: one row per channel, checkbox -> mask, slider -> latched manual window, auto ->
    # back to the running one. The bar is built from the acquisition's RESOLVED display_color.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    bar = win._channel_bar
    assert bar is not None and len(bar._rows) == len(win._meta["channels"])

    box, s_lo, s_hi = bar._rows[0]
    box.setChecked(False)
    assert win._overview._mask[0] == False        # noqa: E712 — numpy bool, not python bool
    box.setChecked(True)
    s_hi.setValue(s_hi.value() // 2)              # dragging a handle latches the channel manual
    assert win._overview._contrast.is_manual(0)
    bar._auto(0)
    assert not win._overview._contrast.is_manual(0)
    win.close()


def test_channel_store_survives_an_operator_run(qapp, stub_detail, squid_dataset, tmp_path):
    # D3: the store lives in the widget, so the toggle works on the operator layer too — not just
    # on the raw preview. Both layers keep their own (C, H, W) store.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: "mip" in win._overview._store
                        and len(win._overview._tiles_by_layer.get("mip", ())) == 2)
    assert set(win._overview._store) >= {"mip"}        # the operator layer has its own store
    before = _rgb(win._overview).copy()
    win._overview.set_channel_visible(0, False)
    assert not np.array_equal(_rgb(win._overview), before)
    win._stop_worker()
    win.close()


def test_second_ingest_resets_state(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: win._overview is not None and len(win._overview._tiles) == 2)
    win.ingest(str(root))            # second open: must stop the old worker + reset state
    qapp.processEvents()
    time.sleep(0.1)
    qapp.processEvents()
    assert len(win._fov_index) == 2                              # rebuilt, not accumulated
    assert len(win.findChildren(V.PlateOverview)) == 1           # one overview, not stacked
    assert set(win._overview._status.values()) == {"empty"}     # fresh grey plate
    win._stop_worker()
    win.close()


# --- IMA-187 wiring guard -------------------------------------------------------------
# The mosaic half of IMA-187 shipped DEAD: `_OperatorWorker` was constructed without
# `n_fovs`, so it defaulted to 1 and `_boxes` was always {}; and `set_mosaic_boxes` had
# zero callers in the repo. Every inherited viewer test still passed, because they only
# exercise the single-tile path. These fail on that dead wiring, so the 227 -> 206 -> 187
# rebase cannot silently drop the feature again.

def test_operator_worker_is_constructed_for_multi_fov_not_defaulted_to_one(
        qapp, stub_detail, squid_dataset, tmp_path, monkeypatch):
    """run_operator must hand the worker a multi-FOV n_fovs, or the mosaic is unreachable."""
    seen = {}
    real_init = V._OperatorWorker.__init__

    def spy(self, *a, **kw):
        seen["n_fovs"] = kw.get("n_fovs", "NOT-PASSED")
        return real_init(self, *a, **kw)

    monkeypatch.setattr(V._OperatorWorker, "__init__", spy)
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: "n_fovs" in seen)

    try:
        assert seen.get("n_fovs") != "NOT-PASSED", (
            "run_operator constructed _OperatorWorker without n_fovs, so it defaults to 1, "
            "_boxes is always {}, and the coordinate-placed mosaic can never render.")
        assert seen["n_fovs"] != 1, (
            f"n_fovs={seen['n_fovs']!r}; the mosaic path requires n_fovs != 1 "
            "(_OperatorWorker: `_boxes = _mosaic_boxes(meta) if n_fovs != 1 else {}`).")
    finally:
        win._stop_worker(); win.close()


def test_set_mosaic_boxes_is_actually_called_by_the_viewer(
        qapp, stub_detail, squid_dataset, tmp_path, monkeypatch):
    """PlateOverview.set_mosaic_boxes exists but nothing calls it -- boxes never reach paint."""
    calls = []
    monkeypatch.setattr(V.PlateOverview, "set_mosaic_boxes",
                        lambda self, boxes: calls.append(boxes))
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    _drain_until(qapp, lambda: bool(calls))

    try:
        assert calls, (
            "set_mosaic_boxes was never called. PlateOverview._boxes stays empty, so _fov_at() "
            "always returns FOV 0 and the mosaic is invisible to hit-testing and paint.")
    finally:
        win._stop_worker(); win.close()

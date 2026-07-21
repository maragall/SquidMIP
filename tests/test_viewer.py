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


def test_fit_cell_always_returns_cell_shape():
    assert V._fit_cell(np.zeros((768, 768), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((V._CELL, V._CELL), np.float32)).shape == (V._CELL, V._CELL)
    assert V._fit_cell(np.zeros((40, 40), np.float32)).shape == (V._CELL, V._CELL)  # tiny frame upscaled


def test_resolve_plate_root(tmp_path):
    (tmp_path / "plate.ome.zarr").mkdir()
    _, is_plate = V.resolve_plate_root(tmp_path)
    assert is_plate
    acq = tmp_path / "acq"
    acq.mkdir()
    _, is_plate = V.resolve_plate_root(acq)
    assert not is_plate


# --- carrier background (IMA-220) -----------------------------------------------------------

def _overview(qapp, carrier=None, rows=("A", "B"), cols=("1", "2"), wells=None):
    ov = V.PlateOverview(list(rows), list(cols),
                         {(0, 0): "A1"} if wells is None else wells, carrier=carrier)
    ov.resize(600, 500)
    return ov


def _render(ov):
    """Rasterise the widget offscreen and return it as a QImage we can sample."""
    from PyQt5.QtGui import QImage, QPainter
    img = QImage(ov.width(), ov.height(), QImage.Format_ARGB32)
    img.fill(Qt.black)
    p = QPainter(img)
    ov.render(p)
    p.end()
    return img


def test_montage_canvas_is_transparent_so_a_background_can_show_through(qapp):
    """REGRESSION + the core IMA-220 premise. With the old Format_RGB888 canvas an opaque
    montage covered the whole plate rect, so ANY background layer was invisible regardless
    of paint order."""
    from PyQt5.QtGui import QImage
    ov = _overview(qapp)
    assert ov._canvas.format() == QImage.Format_ARGB32_Premultiplied
    assert ov._canvas.hasAlphaChannel()
    assert qapp.instance() is not None
    assert ov._canvas.pixelColor(5, 5).alpha() == 0          # un-imaged cell -> transparent
    assert ov._canvas_for("op1").pixelColor(5, 5).alpha() == 0


def test_add_tile_still_paints_an_opaque_tile(qapp):
    """REGRESSION. Tiles must stay fully opaque on the now-transparent canvas, otherwise the
    carrier bleeds through real image data."""
    ov = _overview(qapp)
    rgb = np.full((V._CELL, V._CELL, 3), 200, np.uint8)
    ov.add_tile(0, 0, "A1", rgb)
    assert ov._canvas.pixelColor(V._CELL // 2, V._CELL // 2).alpha() == 255
    assert (0, 0) in ov._tiles
    assert ov._canvas.pixelColor(V._CELL + 5, 5).alpha() == 0   # untouched cell still clear


def test_carrier_none_leaves_geometry_identical_to_the_lattice(qapp):
    """REGRESSION. Every existing construction path passes no carrier and must be unchanged."""
    ov = _overview(qapp, carrier=None)
    ov._fit()
    w, h = ov.width(), ov.height()
    expect_cd = max(2.0, min((w - V._HDR - 2 * V._PAD) / ov._nc,
                             (h - V._COLH - 2 * V._PAD) / ov._nr))
    assert ov._cd == pytest.approx(expect_cd)
    assert ov._ox == pytest.approx(max(V._PAD, (w - V._HDR - ov._nc * ov._cd) / 2))
    assert ov._oy == pytest.approx(max(V._PAD, (h - V._COLH - ov._nr * ov._cd) / 2))
    assert ov._extent_cells() == (0.0, 0.0, float(ov._nc), float(ov._nr))


def _wells_384():
    return {(r, c): f"{V._row_letter(r)}{c + 1}" for r in range(16) for c in range(24)}


def test_fit_with_a_carrier_keeps_the_whole_artwork_on_screen(qapp):
    """The skirt extends left of and above A1, so a lattice-only fit clipped it into the
    label gutters and the zoom-out floor could never reveal it."""
    from squidmip._plate import carrier_for, carrier_placement
    spec = carrier_for("384")
    ov = _overview(qapp, carrier=spec, rows=[V._row_letter(i) for i in range(16)],
                   cols=[str(i + 1) for i in range(24)], wells=_wells_384())
    ov._fit()
    ax, ay = ov._ox + V._HDR, ov._oy + V._COLH
    _, dx, dy, dw, dh = carrier_placement(spec, ov._cd, ax, ay)
    assert dx >= -0.5 and dy >= -0.5                       # not clipped off the left/top
    assert dx + dw <= ov.width() + 0.5
    assert dy + dh <= ov.height() + 0.5
    assert dx < ax and dy < ay                             # artwork really does start before A1


def test_carrier_fit_is_smaller_than_lattice_fit(qapp):
    """Pins the accepted trade-off: honouring the skirt costs ~16% of displayed data, and
    because _fit_cd is also the zoom-out floor that cost is permanent."""
    from squidmip._plate import carrier_for
    rows, cols = [V._row_letter(i) for i in range(16)], [str(i + 1) for i in range(24)]
    bare = _overview(qapp, None, rows, cols, _wells_384())
    with_c = _overview(qapp, carrier_for("384"), rows, cols, _wells_384())
    assert with_c._fit_cd() < bare._fit_cd()
    assert with_c._fit_cd() / bare._fit_cd() == pytest.approx(1 / 1.1875, abs=0.02)


def test_carrier_is_actually_visible_behind_empty_wells(qapp):
    """End to end: render the widget and prove non-background pixels appear where the carrier
    is, and do not when there is none."""
    from squidmip._plate import carrier_for
    rows, cols = [V._row_letter(i) for i in range(16)], [str(i + 1) for i in range(24)]
    bg = V.QColor(V._BG).rgb()

    def _nonbg(ov):
        img = _render(ov)
        return sum(img.pixel(x, y) not in (bg, V.QColor(Qt.black).rgb())
                   for x in range(0, ov.width(), 7) for y in range(0, ov.height(), 7))

    with_c = _nonbg(_overview(qapp, carrier_for("384"), rows, cols, _wells_384()))
    without = _nonbg(_overview(qapp, None, rows, cols, _wells_384()))
    assert with_c > without * 1.5, (with_c, without)


def test_carrier_pixmap_never_exceeds_the_viewport(qapp):
    """The obvious implementation (scale the whole PNG to the destination, like _scaled does)
    allocates ~2.3 GB at the zoom clamp on a 1536wp. The crop must stay widget-bounded."""
    from squidmip._plate import carrier_for
    rows, cols = [V._row_letter(i) for i in range(32)], [str(i + 1) for i in range(48)]
    wells = {(r, c): f"{V._row_letter(r)}{c + 1}" for r in range(32) for c in range(48)}
    ov = _overview(qapp, carrier_for("1536"), rows, cols, wells)
    ov._fit()
    ov._cd = ov._fit_cd() * 40          # the wheelEvent zoom clamp
    ov._user_view = True
    _render(ov)                          # must not allocate at plate scale
    _at, pm = ov._carrier_scaled
    assert pm.width() <= ov.width() + 1 and pm.height() <= ov.height() + 1


def test_carrier_cache_holds_across_hover_but_rebuilds_on_zoom(qapp):
    """Hover repaints fire on every cross-cell move; rescaling the artwork each time would be
    a visible regression against an already-optimised paint path."""
    from squidmip._plate import carrier_for
    ov = _overview(qapp, carrier_for("384"), [V._row_letter(i) for i in range(16)],
                   [str(i + 1) for i in range(24)], _wells_384())
    _render(ov)
    key1 = ov._carrier_key
    ov._hover = (2, 3)
    _render(ov)
    assert ov._carrier_key == key1               # hover: cache survives
    ov._user_view = True
    ov._cd *= 2
    _render(ov)
    assert ov._carrier_key != key1               # zoom: rebuilt


def test_carrier_artwork_is_not_rotated(qapp):
    """A well lattice is centro-symmetric, so 'A1 lands on cell (0,0) and the last well lands
    on the last cell' is satisfied whether or not the artwork is upside-down. These pixels sit
    near the A1 corner marking and differ from their 180-degree counterparts, so they are what
    actually pins orientation."""
    from PyQt5.QtGui import QImage
    from squidmip._plate import carrier_for
    for fmt, (x, y), bright in (("384", (44, 32), True), ("1536", (122, 85), False)):
        img = QImage(str(carrier_for(fmt).image_path()))
        assert not img.isNull()
        here = img.pixelColor(x, y).value()
        there = img.pixelColor(img.width() - 1 - x, img.height() - 1 - y).value()
        assert (here > there) is bright, f"{fmt} artwork looks rotated ({here} vs {there})"


def test_set_final_marks_only_rendered_cells_opaque(qapp, stub_detail, tmp_path):
    """REGRESSION + the _tiles_by_layer trap. The alpha must come from the set the montage was
    built from; _tiles_by_layer is never cleared between a preview and a full run and drops
    tiles the montage keeps, which would paint solid black squares onto the carrier."""
    from PyQt5.QtGui import QImage
    win = V.PlateWindow.__new__(V.PlateWindow)          # bypass __init__: only _set_final under test
    win._overview = _overview(qapp)
    win._active_op_key = None
    win._final_arr = None
    nr, nc = win._overview._nr, win._overview._nc
    rgb = np.full((nr * V._CELL, nc * V._CELL, 3), 77, np.uint8)
    win._overview._tiles_by_layer["raw"] = {(1, 1)}      # deliberately WRONG / stale
    win._set_final((rgb, {(0, 0)}))                      # the montage really only has (0,0)
    img = win._overview._final
    assert img.format() == QImage.Format_ARGB32_Premultiplied
    assert img.pixelColor(V._CELL // 2, V._CELL // 2).alpha() == 255          # rendered cell
    assert img.pixelColor(V._CELL + V._CELL // 2, V._CELL // 2).alpha() == 0  # stale claim ignored
    assert win._final_arr is not None and win._final_arr.shape[2] == 4        # buffer kept alive


def test_set_final_preserves_colour_channels(qapp, stub_detail):
    """REGRESSION. The RGB->ARGB32 byte-order swap must not silently swap red and blue."""
    win = V.PlateWindow.__new__(V.PlateWindow)
    win._overview = _overview(qapp)
    win._active_op_key = None
    win._final_arr = None
    nr, nc = win._overview._nr, win._overview._nc
    rgb = np.zeros((nr * V._CELL, nc * V._CELL, 3), np.uint8)
    rgb[:, :, 0] = 200          # pure red
    win._set_final((rgb, {(0, 0)}))
    c = win._overview._final.pixelColor(V._CELL // 2, V._CELL // 2)
    assert (c.red(), c.green(), c.blue()) == (200, 0, 0)


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
    # bounded memory: the worker keeps one 88px tile per well, not the acquisition
    assert len(win._worker._raw) == 2
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

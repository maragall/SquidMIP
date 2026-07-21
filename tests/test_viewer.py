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


def _tiles(win, layer):
    """Cells that have an image on *layer* (each layer owns its own canvas + tile set)."""
    st = win._overview._layers.get(layer) if win._overview is not None else None
    return st.tiles if st is not None else set()


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
    assert _drain_until(qapp, lambda: len(_tiles(win, "raw")) == 2)     # preview filled thumbnails
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
        qapp, lambda: len(_tiles(win, "mip")) == 2
        and win._overview._layers["mip"].final is not None
    )
    # both wells processed -> tiled on the MIP layer + hue-coded "done"
    assert _tiles(win, "mip") == set(win._fov_index[w]["rc"] for w in ("B2", "B3"))
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
    _drain_until(qapp, lambda: len(_tiles(win, "mip")) == 2)
    win.ingest(str(root))            # second open: must stop the old worker + reset state
    qapp.processEvents()
    time.sleep(0.1)
    qapp.processEvents()
    assert len(win._fov_index) == 2                              # rebuilt, not accumulated
    assert len(win.findChildren(V.PlateOverview)) == 1           # one overview, not stacked
    assert set(win._overview._status.values()) == {"empty"}     # fresh grey plate
    win._stop_worker()
    win.close()


# --- layer stack -> what the plate renders (IMA-227) ----------------------------------------

def _solid(v: int) -> np.ndarray:
    """A plate-cell tile of one grey value, so the rendered layer is identifiable by one pixel."""
    return np.full((V._CELL, V._CELL, 3), v, np.uint8)


def _shown(ov) -> int:
    """The grey value of the image the plate actually paints."""
    return ov._active_source().pixelColor(0, 0).red()


def test_every_layer_keeps_its_own_canvas_and_empty_state(qapp):
    # Nothing here is MIP-specific: three operator layers (one of them a stand-in for a future
    # operator) each keep their own pixels, and the plate shows whichever one is active.
    ov = V.PlateOverview(["A"], ["1", "2"], {(0, 0): "A1", (0, 1): "A2"})
    for layer, val in (("raw", 10), ("mip", 120), ("stitched", 240)):
        ov.add_tile(0, 0, "A1", _solid(val), layer=layer)
    for layer, val in (("raw", 10), ("mip", 120), ("stitched", 240)):
        ov.set_active_layer(layer)
        assert _shown(ov) == val
        assert ov._layers[layer].tiles == {(0, 0)}       # per-layer grey dots, not a shared set
    ov.set_active_layer("never_run")     # a layer nothing has been written to: empty, NOT stale
    assert ov._active_source() is V._blank_image()
    assert "never_run" not in ov._layers                 # ...and showing it allocates no canvas
    ov.set_active_layer(None)            # every layer unticked: the same defined empty state
    assert ov._active_source() is V._blank_image()
    ov.set_active_layer("mip")
    assert _shown(ov) == 120                             # and the real layers survived it


def test_reset_layer_drops_only_that_layer(qapp):
    ov = V.PlateOverview(["A"], ["1"], {(0, 0): "A1"})
    ov.add_tile(0, 0, "A1", _solid(10), layer="raw")
    ov.add_tile(0, 0, "A1", _solid(200), layer="mip")
    ov.reset_layer("mip")
    ov.set_active_layer("mip")
    assert ov._active_source() is V._blank_image()       # cleared, not carrying old pixels
    ov.set_active_layer("raw")
    assert _shown(ov) == 10                              # the base layer is untouched


def test_run_operator_sets_up_the_layer_state(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._raw_btn.isHidden()                       # raw on open: nothing to return from
    win.run_operator("mip", out_parent=str(tmp_path))
    assert win._running_key == "mip"                     # the worker streams into the MIP layer
    assert [ly.key for ly in win._op_stack.layers()] == ["raw", "mip"]
    assert win._op_stack.top_enabled().key == "mip"
    assert win._overview._active == "mip"                # ...which is what the plate shows
    assert win._plate_title.text() == "acq   ·   Maximum Intensity Projection"
    assert not win._raw_btn.isHidden()
    win._stop_worker(); win.close()


def test_toggle_and_reorder_pick_the_right_layer_for_any_operator(qapp, stub_detail, squid_dataset,
                                                                  tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: len(_tiles(win, "mip")) == 2)
    win._stop_worker()
    # A SECOND operator layer. No stitcher exists in the repo yet, so register one by hand — the
    # point is that the stack, the plate and the title are operator-agnostic.
    rc = win._fov_index["B2"]["rc"]
    win._op_stack.add("stitched", "Stitched")
    win._overview.add_tile(*rc, "B2", _solid(200), layer="stitched")
    win._apply_layers()
    assert win._overview._active == "stitched"
    assert win._plate_title.text().endswith("Stitched")

    win._on_layer_toggle("stitched", False)              # untick the top -> the one below shows
    assert win._overview._active == "mip"
    assert win._plate_title.text().endswith("Maximum Intensity Projection")
    win._on_layer_toggle("stitched", True)
    win._on_layer_move("mip", +1)                        # "↑": MIP now sits above stitched
    assert win._op_stack.layers()[-1].key == "mip"
    assert win._overview._active == "mip"
    win._on_layer_toggle("mip", False)                   # ...and unticking it falls to stitched
    assert win._overview._active == "stitched"
    win._on_layer_toggle("stitched", False)              # both operators off -> back to raw
    assert win._overview._active == "raw"
    assert win._plate_mode == "raw" and win._raw_btn.isHidden()
    win.close()


def test_all_layers_off_gives_a_defined_empty_plate(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert _drain_until(qapp, lambda: len(_tiles(win, "raw")) == 2)
    win._on_layer_toggle("raw", False)                   # the last enabled layer off
    assert win._op_stack.top_enabled() is None
    assert win._overview._active is None
    assert win._overview._active_source() is V._blank_image()   # empty, not the stale raw montage
    assert win._plate_mode == "no layers"
    assert "no layers" in win._plate_title.text()
    win._on_layer_toggle("raw", True)                    # ...and it comes straight back
    assert win._overview._active == "raw"
    win._stop_preview(); win.close()


def test_return_to_raw_routes_through_the_stack(qapp, stub_detail, squid_dataset, tmp_path):
    # The button and the Layers tab must be ONE path: after "Return to raw view" the tab cannot
    # still show MIP ticked on top while the plate renders raw.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: len(_tiles(win, "mip")) == 2)
    win._return_to_raw()
    assert win._running_key is None
    assert [ly.enabled for ly in win._op_stack.layers() if ly.key == "mip"] == [False]
    assert win._op_stack.top_enabled().key == "raw"
    assert win._overview._active == "raw"
    assert win._plate_mode == "raw" and win._raw_btn.isHidden()
    win._on_layer_toggle("mip", True)                    # re-ticking MIP brings the result back
    assert win._overview._active == "mip"
    assert not win._raw_btn.isHidden()
    win._stop_preview(); win._stop_worker(); win.close()


def test_toggle_mid_run_does_not_redirect_the_worker(qapp, stub_detail, squid_dataset, tmp_path):
    # Display and worker routing are separate concerns: unticking the layer being computed changes
    # what is SHOWN, never where the in-flight tiles land (nor which mode the detail view is in).
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    win._on_layer_toggle("mip", False)                   # untick mid-run
    assert win._overview._active == "raw"                # the plate follows the stack ...
    assert win._running_key == "mip"                     # ... the worker keeps its own layer
    assert _drain_until(qapp, lambda: len(_tiles(win, "mip")) == 2)
    assert _tiles(win, "mip") == set(win._fov_index[w]["rc"] for w in ("B2", "B3"))
    win._detail.registered.clear()
    win.activate_well("B3", 0)                           # the detail stays in processed mode
    assert win._detail.nav[-1] == ("B3", 0)
    assert not win._detail.registered                    # no raw z-stack push behind the results
    win._stop_worker(); win.close()


def test_rerun_with_preview_limit_leaves_no_stale_tiles(qapp, stub_detail, squid_dataset, tmp_path):
    # Bug repro: a full run followed by a 1-well re-run used to leave the first run's pixels in the
    # untouched wells, with no grey dot — the plate looked fully computed when it wasn't.
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    assert _drain_until(qapp, lambda: len(_tiles(win, "mip")) == 2)
    assert _drain_until(qapp, lambda: not win._worker.isRunning())
    win.run_operator("mip", preview_limit=1, save=False)
    assert _drain_until(qapp, lambda: len(_tiles(win, "mip")) == 1)
    assert _tiles(win, "mip") == {win._fov_index["B2"]["rc"]}   # only the re-run well has an image
    win._stop_worker(); win.close()


def test_open_computed_routes_through_the_stack(qapp, stub_detail, squid_dataset, tmp_path,
                                                monkeypatch):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))      # write a real .hcs to re-open
    assert _drain_until(qapp, lambda: win._processed_plate is not None)
    assert _drain_until(qapp, lambda: not win._worker.isRunning())
    monkeypatch.setattr(V.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path / "acq.hcs")))
    win._open_computed()
    assert [ly.key for ly in win._op_stack.layers()] == ["raw", "computed"]
    assert win._running_key == "computed"
    assert win._overview._active == "computed"
    assert "computed MIP" in win._plate_title.text()
    assert win._raw_btn.isHidden()                         # no raw reader behind a computed plate
    win._on_layer_toggle("computed", False)                # tab and plate never disagree
    assert win._overview._active == "raw"
    win._stop_worker(); win.close()

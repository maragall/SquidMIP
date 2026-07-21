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
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QPushButton, QSlider, QSpinBox, QWidget,
)

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
        self.arrays = []          # (t, idx, z, ch) of every register_array push (IMA-205)
        self.nav = []
        self.acquisitions = []    # one entry per start_acquisition — the slider's label list

    def start_acquisition(self, channels, nz, h, w, labels):
        self._fov_labels = list(labels)
        self._fov_slider.setMaximum(max(0, len(labels) - 1))
        self.acquisitions.append(list(labels))

    def register_image(self, t, idx, z, ch, path, page_idx=0):
        self.registered.append((t, idx, z, ch, path))

    def register_array(self, t, idx, z, ch, plane):
        """Record computed-well pushes. The real ndviewer indexes its slider by ``idx``, so a push
        whose idx exceeds the current label list would land out of range — recording it here is what
        makes the global->subset remap assertable at all (the push path was previously unobserved)."""
        self.arrays.append((t, idx, z, ch))

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


# --- IMA-205: exploration tabs ---------------------------------------------------------------
#
# The tab is a multi-instance container scoped to a region subset. Identity is content-addressed,
# operator results are filed per-tab, and closing a tab must stop its run and free its canvas.

def test_exploration_tab_key_is_order_independent_and_set_based():
    k = V.exploration_tab_key("acq", ["B3", "B2"])
    assert k == V.exploration_tab_key("acq", ["B2", "B3"])        # drag order must not matter
    assert k == V.exploration_tab_key("acq", ["B2", "B3", "B2"])  # duplicates collapse
    assert k != V.exploration_tab_key("acq", ["B2"])              # a different set is a different tab
    assert k.startswith("exp:")


def test_exploration_tab_key_includes_acquisition_identity():
    # the SAME well ids on a DIFFERENT plate must never dedupe onto a stale tab
    assert V.exploration_tab_key("plate_a", ["B2"]) != V.exploration_tab_key("plate_b", ["B2"])


def test_exploration_tab_key_rejects_empty():
    with pytest.raises(ValueError):
        V.exploration_tab_key("acq", [])


def test_exploration_tab_key_stable_at_plate_scale():
    many = [f"B{i}" for i in range(1, 1537)]
    k = V.exploration_tab_key("acq", many)
    assert k == V.exploration_tab_key("acq", list(reversed(many)))
    assert len(k) < 32                       # bounded — it is a tab key, not a serialized set


def test_exploration_tab_label_is_human_readable():
    assert V.exploration_tab_label(["B2"]) == "B2"
    assert V.exploration_tab_label(["B5", "B2", "B3"]) == "B2–B5 (3)"
    assert "exp:" not in V.exploration_tab_label(["B2", "B3"])   # never show the hash as a title


def test_operator_layer_key_namespaces_only_when_scoped():
    assert V.operator_layer_key("mip", None) == "mip"             # plate-wide: unchanged behavior
    assert V.operator_layer_key("mip", "exp:ab12") == "mip@exp:ab12"


def test_open_exploration_tab_lists_exactly_the_selection(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    assert key is not None
    tab = win._op_tabs[key]
    assert isinstance(tab, V._ExplorationTab)
    assert tab.regions == ["B3"]
    assert tab.listing.text() == "B3"                 # the tab shows exactly what it is scoped to
    assert win._left_tabs.currentWidget() is tab
    win.close()


def test_open_exploration_tab_same_selection_focuses_not_duplicates(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n0 = win._left_tabs.count()
    k1 = win.open_exploration_tab(["B2", "B3"])
    k2 = win.open_exploration_tab(["B3", "B2"])       # same SET, different order
    assert k1 == k2
    assert win._left_tabs.count() == n0 + 1           # one tab, not two
    k3 = win.open_exploration_tab(["B3"])             # a different set DOES open another
    assert k3 != k1
    assert win._left_tabs.count() == n0 + 2
    win.close()


def test_open_exploration_tab_rejects_empty_and_unknown(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n0 = win._left_tabs.count()
    assert win.open_exploration_tab([]) is None
    assert "empty selection" in win._readout.text().lower()
    assert win.open_exploration_tab(["ZZ99"]) is None          # named, not a raw KeyError
    assert "not in this acquisition" in win._readout.text().lower()
    assert win._left_tabs.count() == n0
    win.close()


def test_open_exploration_tab_needs_an_acquisition(qapp, stub_detail):
    win = V.PlateWindow(None)
    assert win.open_exploration_tab(["B2"]) is None
    assert "acquisition" in win._readout.text().lower()
    win.close()


def test_run_operator_rejects_empty_and_unknown_regions(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path), regions=[])
    assert "empty selection" in win._readout.text().lower()
    assert win._worker is None
    win.run_operator("mip", out_parent=str(tmp_path), regions=["ZZ99"])
    assert "not in this acquisition" in win._readout.text().lower()
    assert win._worker is None                                  # never started
    win.close()


def test_subset_run_scopes_slider_and_remaps_push_index(qapp, stub_detail, squid_dataset):
    """The regression that decision 3 would have introduced without the remap.

    B3 is plate index 1, but in a ['B3'] subset its slider position is 0. The worker emits the
    GLOBAL index, so an unremapped push would address slot 1 of a 1-entry slider."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    assert win._fov_index["B3"]["idx"] == 1                     # global index is 1
    win._detail.arrays.clear()
    win.run_operator("mip", regions=["B3"], save=False)
    assert win._detail._fov_labels == ["B3:0"]                  # slider is the SUBSET, not the plate
    assert _drain_until(qapp, lambda: len(win._detail.arrays) > 0)
    pushed = {a[1] for a in win._detail.arrays}
    assert pushed == {0}, f"push landed at {pushed}, expected subset position 0"
    assert max(pushed) < len(win._detail._fov_labels)           # never out of range
    win._stop_worker(); win.close()


def test_whole_plate_run_keeps_identity_indexing(qapp, stub_detail, squid_dataset, tmp_path):
    """Regression guard: the remap must not disturb the shipped whole-plate path."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.arrays.clear()
    win.run_operator("mip", out_parent=str(tmp_path))
    assert win._push_index is None                              # identity for a full plate
    assert _drain_until(qapp, lambda: len(win._detail.arrays) >= 2)
    assert {a[1] for a in win._detail.arrays} == {0, 1}         # both plate indices, unchanged
    win._stop_worker(); win.close()


def test_preview_spinner_still_runs_first_n_wells(qapp, stub_detail, squid_dataset, monkeypatch):
    """REGRESSION for the preview_limit -> regions= collapse: the shipped spinner call site."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    seen = {}
    real = V.PlateWindow.run_operator

    def spy(self, key, out_parent=None, regions=None, save=True, tab_key=None):
        seen["regions"] = regions
        return real(self, key, out_parent=out_parent, regions=regions, save=save, tab_key=tab_key)
    monkeypatch.setattr(V.PlateWindow, "run_operator", spy)

    tab = win._build_run_tab(V._OPERATIONS_BY_KEY["mip"])       # the real MIP tab
    prev = [b for b in tab.findChildren(QPushButton) if b.text() == "Preview"][0]
    spin = tab.findChildren(QSpinBox)[0]
    spin.setValue(1)
    prev.click()
    assert seen["regions"] == ["B2"], "preview must still run the FIRST N wells"
    win._stop_worker(); win.close()


def test_operator_tab_opened_twice_is_one_tab(qapp, stub_detail, squid_dataset):
    """REGRESSION: exploration tabs are multi-instance, operator tabs must stay singletons."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n0 = win._left_tabs.count()
    win._activate_operator("mip")
    win._activate_operator("mip")
    assert win._left_tabs.count() == n0 + 1
    win.close()


def test_two_tabs_same_operator_get_separate_layers(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    k1 = win.open_exploration_tab(["B2"])
    win.run_operator("mip", regions=["B2"], save=False, tab_key=k1)
    assert _drain_until(qapp, lambda: not win._busy())
    k2 = win.open_exploration_tab(["B3"])
    win.run_operator("mip", regions=["B3"], save=False, tab_key=k2)
    assert _drain_until(qapp, lambda: not win._busy())
    keys = {ly.key for ly in win._op_stack.layers()}
    assert f"mip@{k1}" in keys and f"mip@{k2}" in keys          # distinct layers, no collision
    assert f"mip@{k1}" in win._overview._op_canvas
    assert f"mip@{k2}" in win._overview._op_canvas
    win._stop_worker(); win.close()


def test_closing_tab_mid_run_stops_worker_and_frees_canvas(qapp, stub_detail, squid_dataset):
    """CRITICAL gap: no test, no error handling, and silent memory growth before this change."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B2", "B3"])
    win.run_operator("mip", regions=["B2", "B3"], save=False, tab_key=key)
    layer = f"mip@{key}"
    assert win._active_op_key == layer
    idx = next(i for i in range(win._left_tabs.count())
               if win._left_tabs.widget(i) is win._op_tabs[key])
    win._close_op_tab(idx)                                       # close it, possibly mid-run
    assert _drain_until(qapp, lambda: not win._busy())
    assert layer not in {ly.key for ly in win._op_stack.layers()}   # layer dropped
    assert layer not in win._overview._op_canvas                    # ~plate-sized canvas freed
    assert layer not in win._overview._op_final
    assert win._active_op_key is None
    win.close()


def test_closing_idle_exploration_tab_is_clean(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B2"])
    n = win._left_tabs.count()
    idx = next(i for i in range(win._left_tabs.count())
               if win._left_tabs.widget(i) is win._op_tabs[key])
    win._close_op_tab(idx)
    assert win._left_tabs.count() == n - 1
    assert key not in win._op_tabs
    win.close()


def test_home_tab_is_never_closable(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    n = win._left_tabs.count()
    win._close_op_tab(0)
    assert win._left_tabs.count() == n
    win.close()


def test_busy_guard_covers_retired_workers(qapp, stub_detail, squid_dataset, tmp_path):
    """_stop_worker clears self._worker while the retired thread drains — the guard must still
    refuse a new run, or closing a tab lets two workers hit the same reader at once."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win.run_operator("mip", out_parent=str(tmp_path))
    win._stop_worker()
    assert win._worker is None
    if win._busy():                                    # still draining -> a new run must be refused
        win.run_operator("mip", out_parent=str(tmp_path))
        assert win._worker is None
        assert "already processing" in win._readout.text().lower()
    assert _drain_until(qapp, lambda: not win._busy())
    win.close()


def test_tab_switch_repoints_detail_and_home_restores_plate(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B3"])
    qapp.processEvents()
    assert win._detail._fov_labels == ["B3:0"]                  # follows the exploration tab
    assert win._active_exploration is win._op_tabs[key]
    win._left_tabs.setCurrentIndex(0)                           # back to "Process wells"
    qapp.processEvents()
    assert win._detail._fov_labels == ["B2:0", "B3:0"]          # whole plate restored
    assert win._active_exploration is None
    win.close()


def test_subset_tab_registers_raw_paths_at_subset_positions(qapp, stub_detail, squid_dataset):
    """The raw bulk-register path indexes the slider too — it must use subset positions, not the
    global plate index, or B3 (plate idx 1) would register past the end of a 1-entry slider."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    win._detail.registered.clear()
    win.open_exploration_tab(["B3"])
    qapp.processEvents()
    if win._detail.registered:
        assert {r[1] for r in win._detail.registered} == {0}
    win.activate_well("B3", 0)
    assert all(r[1] < len(win._detail._fov_labels) for r in win._detail.registered)
    win.close()


def test_ingest_closes_exploration_tabs(qapp, stub_detail, squid_dataset):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B2", "B3"])
    assert key in win._op_tabs
    win.ingest(str(root))                              # re-open: tabs belong to the old _fov_index
    qapp.processEvents()
    assert key not in win._op_tabs
    assert not [i for i in range(win._left_tabs.count())
                if isinstance(win._left_tabs.widget(i), V._ExplorationTab)]
    assert win._active_exploration is None
    win.close()


def test_subset_save_is_disk_guarded(qapp, stub_detail, squid_dataset, monkeypatch, tmp_path):
    """The guard used to be skipped entirely for subsets (`if not ok and regions is None`)."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    class _Tiny:
        free = 1                                        # a byte free: everything must be refused
    monkeypatch.setattr("shutil.disk_usage", lambda p: _Tiny())
    win.run_operator("mip", out_parent=str(tmp_path), regions=["B3"], save=True)
    assert win._worker is None, "a subset save must be blocked when the disk can't hold it"
    assert "free space" in win._readout.text().lower()
    win.close()


def test_check_disk_scales_with_subset_size(qapp, stub_detail, squid_dataset, tmp_path):
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    _, full_gb, _ = win._check_disk(tmp_path / "x.hcs")
    _, one_gb, _ = win._check_disk(tmp_path / "x.hcs", n_regions=1)
    assert one_gb < full_gb                             # a 1-well run is not a whole-plate estimate
    assert one_gb == pytest.approx(full_gb / 2, rel=0.01)   # 1 of 2 wells
    win.close()


def test_note_partial_output_marks_a_stopped_plate(qapp, stub_detail, squid_dataset, tmp_path):
    """A save run that is stopped mid-write leaves a real-looking plate.ome.zarr holding only some
    wells. Mark it, so 'Open a computed MIP' can refuse it instead of showing a truncated plate as
    a finished one."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    out = tmp_path / "acq.hcs"
    (out / "plate.ome.zarr").mkdir(parents=True)
    win._run_out_dir = str(out)
    win._note_partial_output()
    assert (out / "INCOMPLETE").exists()
    assert win._run_out_dir is None                     # consumed, so it can't leak to a later run
    win.close()


def test_open_computed_refuses_an_incomplete_plate(qapp, stub_detail, tmp_path, monkeypatch):
    base = tmp_path / "acq.hcs"
    (base / "plate.ome.zarr").mkdir(parents=True)
    (base / "plate.ome.zarr" / "zarr.json").write_text("{}")
    (base / "INCOMPLETE").write_text("stopped\n")
    win = V.PlateWindow(None)
    monkeypatch.setattr(V.QFileDialog, "getExistingDirectory", lambda *a, **k: str(base))
    win._open_computed()
    assert "incomplete" in win._readout.text().lower()
    win.close()


def test_completed_save_run_is_not_marked_incomplete(qapp, stub_detail, squid_dataset, tmp_path):
    """The other half of the invariant: a run that finishes must NOT be flagged."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    key = win.open_exploration_tab(["B2", "B3"])
    win.run_operator("mip", out_parent=str(tmp_path), regions=["B2", "B3"], save=True, tab_key=key)
    assert _drain_until(qapp, lambda: not win._busy(), timeout=90)
    out = tmp_path / f"{win._acq_name}.hcs"
    idx = next(i for i in range(win._left_tabs.count())
               if win._left_tabs.widget(i) is win._op_tabs[key])
    win._close_op_tab(idx)                              # close AFTER it finished
    assert not (out / "INCOMPLETE").exists(), "a completed plate must not be flagged incomplete"
    win.close()


def test_operation_stack_remove_and_remove_suffix():
    from squidmip._layers import OperationStack
    st = OperationStack()
    st.add("mip@exp:a", "MIP · a")
    st.add("mip@exp:b", "MIP · b")
    st.add("mip", "MIP")
    assert st.remove_suffix("@exp:a") == ["mip@exp:a"]
    keys = {ly.key for ly in st.layers()}
    assert keys == {"raw", "mip@exp:b", "mip"}
    assert st.remove("raw") is False                    # the base layer is never removable
    assert st.remove("mip") is True
    assert st.remove("mip") is False


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

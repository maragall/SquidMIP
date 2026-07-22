"""The AGAVE 3D view as a Qt tab in PANE 3 — headless (offscreen).

Pane 3 is the SUPPLEMENTARY pane: operator results belong in the plate view and the centre
viewer as toggleable layers, and pane 3 is for things like this. So these tests assert WHERE the
tab lands as hard as they assert what it does.

The AGAVE process and its websocket never appear here: the tab is driven through an injected
worker, so this whole module is green on a machine with no AGAVE installed.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded — Qt binding conflict; run with "
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.", allow_module_level=True)

from PyQt5.QtCore import QObject, Qt, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QSlider, QSplitter  # noqa: E402

from squidmip import _agave as A  # noqa: E402
from squidmip import _agave_pane as P  # noqa: E402
from squidmip import _viewer as V  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    return app


def _meta(n_t=3):
    return {
        "regions": ["B2", "B3", "C4"],
        "fovs_per_region": {r: [0] for r in ("B2", "B3", "C4")},
        "fov_positions_um": {(r, 0): (0.0, 0.0) for r in ("B2", "B3", "C4")},
        "channels": [{"name": "Fluorescence_488_nm_Ex", "display_color": "#1FFF00"}],
        "n_z": 4, "z_levels": [0, 1, 2, 3], "dz_um": 1.5, "pixel_size_um": 0.75,
        "frame_shape": (8, 8), "dtype": np.dtype("uint16"), "n_t": n_t,
    }


class _FakeWorker(QObject):
    """Stands in for _AgaveWorker: same signals, same request methods, no thread and no AGAVE."""

    rendered = pyqtSignal(bytes)
    problem = pyqtSignal(str)
    note = pyqtSignal(str)

    def __init__(self, meta=None):
        super().__init__()
        self.calls = []
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def request_region(self, region, t):
        self.calls.append(("region", region, t))

    def request_time(self, t):
        self.calls.append(("time", t))

    def request_orbit(self, dtheta, dphi):
        self.calls.append(("orbit", dtheta, dphi))

    def request_frame(self, w, h):
        self.calls.append(("frame", w, h))

    def shutdown(self):
        self.stopped += 1

    @property
    def regions(self):
        return ["B2", "B3", "C4"]


def _tab(qapp, meta=None, worker=None):
    w = worker or _FakeWorker()
    tab = P.AgaveTab(reader=object(), meta=meta or _meta(), acq_path="/data/acq",
                     worker=w)
    return tab, w


def _all_text(widget) -> str:
    return "\n".join(lab.text() for lab in widget.findChildren(QLabel))


# --- the sliders: the unit of navigation is the REGION, and both sliders are LAZY -------------

def test_tab_offers_every_region_and_opens_on_the_first(qapp):
    tab, w = _tab(qapp)
    assert tab.region_slider.minimum() == 0
    assert tab.region_slider.maximum() == 2          # 3 regions
    assert ("region", "B2", 0) in w.calls
    assert "B2" in _all_text(tab)
    tab.shutdown()


def test_moving_the_region_slider_asks_for_that_regions_volume(qapp):
    tab, w = _tab(qapp)
    w.calls.clear()
    tab.region_slider.setValue(2)
    assert ("region", "C4", 0) in w.calls
    assert "C4" in _all_text(tab)
    tab.shutdown()


def test_region_slider_navigates_regions_never_fovs(qapp):
    """A region is a MOSAIC of FOVs and is the only unit this app navigates."""
    tab, w = _tab(qapp)
    tab.region_slider.setValue(1)
    kinds = {c[1] for c in w.calls if c[0] == "region"}
    assert kinds <= {"B2", "B3", "C4"}               # region ids, not fov indices
    assert not any(c[0] == "fov" for c in w.calls)
    tab.shutdown()


def test_moving_the_timepoint_slider_is_lazy_and_never_reloads_the_volume(qapp):
    tab, w = _tab(qapp)
    w.calls.clear()
    tab.time_slider.setValue(2)
    assert ("time", 2) in w.calls
    assert not any(c[0] == "region" for c in w.calls)     # set_time, not a re-load
    tab.shutdown()


def test_timepoint_slider_spans_the_acquisitions_timepoints(qapp):
    tab, _ = _tab(qapp, meta=_meta(n_t=5))
    assert (tab.time_slider.minimum(), tab.time_slider.maximum()) == (0, 4)
    tab.shutdown()


def test_a_single_timepoint_acquisition_disables_the_slider_and_says_why(qapp):
    tab, _ = _tab(qapp, meta=_meta(n_t=1))
    assert tab.time_slider.isEnabled() is False
    assert "single timepoint" in _all_text(tab).lower()
    tab.shutdown()


# --- NO SILENT FAILURES -----------------------------------------------------------------------

def test_a_missing_agave_is_a_named_refusal_in_the_tab_not_an_empty_pane(qapp):
    tab, w = _tab(qapp)
    w.problem.emit("AGAVE is not installed on this machine, so the 3D view cannot open.")
    text = _all_text(tab)
    assert "AGAVE is not installed on this machine" in text
    assert tab.has_problem is True
    tab.shutdown()


def test_a_server_that_will_not_start_is_reported_verbatim(qapp):
    tab, w = _tab(qapp)
    w.problem.emit("AGAVE did not start listening on port 1246 within 20s.")
    assert "port 1246" in _all_text(tab)
    tab.shutdown()


def test_a_volume_that_cannot_be_written_is_reported_verbatim(qapp):
    tab, w = _tab(qapp)
    w.problem.emit("B2: no stage positions / pixel size in this acquisition.")
    assert "no stage positions" in _all_text(tab)
    tab.shutdown()


def test_a_problem_never_leaves_a_stale_frame_pretending_to_be_the_current_one(qapp):
    tab, w = _tab(qapp)
    w.rendered.emit(_png(40, 30))
    assert tab.canvas.pixmap() is not None
    w.problem.emit("the render socket closed")
    assert tab.canvas.pixmap() is None or tab.canvas.pixmap().isNull()
    tab.shutdown()


# --- frames ------------------------------------------------------------------------------------

def _png(w, h) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(w, h) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="JPEG")
    data = buf.getvalue()
    assert data[:3] == b"\xff\xd8\xff"
    return data


def test_a_rendered_frame_is_shown_at_its_own_size(qapp):
    tab, w = _tab(qapp)
    w.rendered.emit(_png(64, 48))
    pm = tab.canvas.pixmap()
    assert pm is not None and (pm.width(), pm.height()) == (64, 48)
    assert tab.has_problem is False
    tab.shutdown()


def test_an_undecodable_frame_says_so_rather_than_showing_nothing(qapp):
    tab, w = _tab(qapp)
    w.rendered.emit(b"not an image")
    assert "could not be decoded" in _all_text(tab).lower()
    assert tab.has_problem is True
    tab.shutdown()


def test_a_jpeg_frame_is_shown_because_agave_streams_jpeg_not_png(qapp):
    """MEASURED against AGAVE 1.10.0: the server sends JPEG (ff d8 ff e0). A decoder pinned to
    "PNG" would reject every real frame and show a permanent 'could not be decoded'."""
    tab, w = _tab(qapp)
    w.rendered.emit(_jpeg(64, 48))
    pm = tab.canvas.pixmap()
    assert pm is not None and (pm.width(), pm.height()) == (64, 48)
    assert tab.has_problem is False
    tab.shutdown()


def test_dragging_on_the_canvas_orbits_the_volume(qapp):
    tab, w = _tab(qapp)
    w.calls.clear()
    tab.drag(12, -7)
    assert any(c[0] == "orbit" for c in w.calls)
    tab.shutdown()


# --- lifecycle: never an orphan -----------------------------------------------------------------

def test_shutdown_stops_the_worker(qapp):
    tab, w = _tab(qapp)
    tab.shutdown()
    assert w.stopped == 1


def test_shutdown_twice_is_safe_because_tab_close_and_window_close_both_call_it(qapp):
    tab, w = _tab(qapp)
    tab.shutdown()
    tab.shutdown()
    assert w.stopped == 1


def test_the_tab_exposes_shutdown_so_the_windows_duck_typed_teardown_frees_it(qapp):
    tab, w = _tab(qapp)
    assert hasattr(tab, "shutdown")
    tab.shutdown()


# --- WHERE it lands: pane 3, embedded, never a top-level window ---------------------------------

@pytest.fixture
def win(qapp, monkeypatch, squid_dataset):
    from .test_viewer import _StubDetail

    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())
    monkeypatch.setattr(P, "_make_worker", lambda *a, **k: _FakeWorker())
    w = V.PlateWindow()
    w.ingest(str(squid_dataset[0]))
    qapp.processEvents()
    yield w
    w.close()


def _agave_button(win):
    for b in win.findChildren(QPushButton):
        if "3D" in b.text() and "AGAVE" in b.text().upper():
            return b
    return None


def test_pane_one_offers_the_three_d_view(win):
    assert _agave_button(win) is not None


def test_the_button_opens_the_three_d_view_as_a_tab_in_pane_three(win, qapp):
    before = win._explore_tabs.count()
    _agave_button(win).click()
    qapp.processEvents()
    assert win._explore_tabs.count() == before + 1
    tab = win._explore_tabs.widget(win._explore_tabs.count() - 1)
    assert isinstance(tab, P.AgaveTab)
    assert win._explore_tabs.currentWidget() is tab


def test_the_three_d_view_is_embedded_never_a_separate_top_level_window(win, qapp):
    _agave_button(win).click()
    qapp.processEvents()
    tab = win._op_tabs[V.AGAVE_KEY]
    assert tab.isWindow() is False
    assert tab.window() is win
    assert win._floating == {}


def test_the_three_d_view_never_lands_in_pane_one(win, qapp):
    """Pane 1 is the operator console; pane 3 is the supplementary pane AGAVE belongs to."""
    _agave_button(win).click()
    qapp.processEvents()
    in_pane_one = [win._left_tabs.widget(i) for i in range(win._left_tabs.count())]
    assert not any(isinstance(w, P.AgaveTab) for w in in_pane_one)


def test_clicking_twice_focuses_the_same_tab_rather_than_opening_a_second_server(win, qapp):
    _agave_button(win).click()
    qapp.processEvents()
    n = win._explore_tabs.count()
    _agave_button(win).click()
    qapp.processEvents()
    assert win._explore_tabs.count() == n


def test_closing_the_three_d_tab_shuts_its_worker_down(win, qapp):
    _agave_button(win).click()
    qapp.processEvents()
    tab = win._op_tabs[V.AGAVE_KEY]
    worker = tab._worker
    idx = win._explore_tabs.indexOf(tab)
    win._close_op_tab(idx, win._explore_tabs)
    assert worker.stopped == 1
    assert V.AGAVE_KEY not in win._op_tabs


def test_closing_the_window_shuts_the_three_d_view_down_leaving_no_orphan(win, qapp):
    _agave_button(win).click()
    qapp.processEvents()
    worker = win._op_tabs[V.AGAVE_KEY]._worker
    win.close()
    assert worker.stopped == 1


# --- the splitter: pane 3 must be draggable to FULL SCREEN --------------------------------------

def test_pane_three_can_be_dragged_to_take_the_whole_window(win, qapp):
    """Julio asked for this explicitly. Panes 1 and 2 must be COLLAPSIBLE, not min-width-locked
    against pane 3 — a fixed neighbour is dead space he cannot reclaim."""
    split = win._split
    assert isinstance(split, QSplitter)
    assert split.count() == 3
    assert split.childrenCollapsible() is True
    for i in range(split.count()):
        assert split.isCollapsible(i) is True

    total = sum(split.sizes())
    split.setSizes([0, 0, total])
    qapp.processEvents()
    sizes = split.sizes()
    assert sizes[0] == 0 and sizes[1] == 0            # neighbours actually collapsed
    assert sizes[2] == sum(sizes)                      # pane 3 holds ALL the content width
    # ...and that content width is the window minus only the splitter handles: genuinely full
    # screen, not "as wide as the neighbours' minimum sizes allow".
    assert sizes[2] >= total - split.handleWidth() * (split.count() - 1)


def test_pane_three_full_screen_survives_the_three_d_tab_being_open(win, qapp):
    _agave_button(win).click()
    qapp.processEvents()
    split = win._split
    total = sum(split.sizes())
    split.setSizes([0, 0, total])
    qapp.processEvents()
    assert split.sizes()[:2] == [0, 0]
    assert split.sizes()[2] == sum(split.sizes())
    assert split.sizes()[2] >= total - split.handleWidth() * (split.count() - 1)


# --- napari's own 3D button is NOT aliased; it only gains a signpost ---------------------------

def test_naparis_3d_button_gets_a_tooltip_pointing_at_agave(qapp):
    from PyQt5.QtWidgets import QPushButton

    from squidmip import _napari_pane as NP

    btn = QPushButton("3D")
    clicked = []
    btn.clicked.connect(lambda *_: clicked.append(1))
    NP.apply_ndisplay_tooltip(btn)

    tip = btn.toolTip()
    assert "AGAVE" in tip
    assert "exploration pane" in tip
    btn.click()
    assert clicked == [1]          # a TOOLTIP only: what the button does is untouched


def test_the_tooltip_does_not_claim_naparis_button_opens_agave(qapp):
    """Julio: 'let's not alias the button, that's bad design.' The button keeps doing napari 3D;
    the tooltip says where a better render lives and names the separate control that opens it."""
    from squidmip import _napari_pane as NP

    tip = NP.NDISPLAY_TOOLTIP
    assert tip.lower().startswith("3d view (napari)")
    assert "3D view (AGAVE)" in tip                  # names the SEPARATE affordance
    assert "coarsest" in tip.lower()                 # and says WHY, honestly


def test_the_agave_entry_is_a_separate_control_from_naparis_3d_button(win, qapp):
    from squidmip import _napari_pane as NP

    btn = _agave_button(win)
    assert btn is not None
    assert btn is not getattr(getattr(win, "_mosaic", None), "ndisplay_button", None)
    assert NP.NDISPLAY_TOOLTIP not in btn.toolTip()


# --- pane 3 tab hygiene: "all the tabs that we can open / close and so on" ----------------------

def test_closing_every_pane_three_tab_leaves_no_agave_worker_running(win, qapp):
    """Julio calls pane 3 'the exploration pane where we have all the tabs that we can open /
    close and so on'. Emptying it must not leak an AGAVE server process."""
    _agave_button(win).click()
    qapp.processEvents()
    worker = win._op_tabs[V.AGAVE_KEY]._worker
    for i in range(win._explore_tabs.count() - 1, -1, -1):
        win._close_op_tab(i, win._explore_tabs)
    assert worker.stopped == 1
    assert V.AGAVE_KEY not in win._op_tabs


def test_reopening_after_a_close_builds_a_fresh_worker_not_the_dead_one(win, qapp):
    _agave_button(win).click()
    qapp.processEvents()
    first = win._op_tabs[V.AGAVE_KEY]._worker
    win._close_op_tab(win._explore_tabs.indexOf(win._op_tabs[V.AGAVE_KEY]), win._explore_tabs)
    _agave_button(win).click()
    qapp.processEvents()
    second = win._op_tabs[V.AGAVE_KEY]._worker
    assert second is not first
    assert second.stopped == 0


def test_floating_the_three_d_tab_out_and_closing_the_float_still_stops_the_worker(win, qapp):
    """Pane 3's tabs are detachable. The float-close path must free it exactly as a tab close."""
    _agave_button(win).click()
    qapp.processEvents()
    tab = win._op_tabs[V.AGAVE_KEY]
    worker = tab._worker
    floated = win._detach_tab(win._explore_tabs.indexOf(tab), win._explore_tabs)
    assert floated is not None
    win._on_float_closed(V.AGAVE_KEY)
    assert worker.stopped == 1


def test_the_three_d_view_says_no_when_there_is_no_acquisition_open(qapp, monkeypatch):
    """A named refusal in pane 1's status line, not a tab that opens onto nothing."""
    from .test_viewer import _StubDetail

    monkeypatch.setattr(V.PlateWindow, "_make_detail_viewer", lambda self: _StubDetail())
    monkeypatch.setattr(P, "_make_worker", lambda *a, **k: _FakeWorker())
    w = V.PlateWindow()
    try:
        before = w._explore_tabs.count()
        _agave_button(w).click()
        qapp.processEvents()
        assert w._explore_tabs.count() == before
        assert "no acquisition" in w._readout.text().lower()
    finally:
        w.close()


# --- the log panel gets it too (stdlib logging), without replacing the in-pane message ---------

def test_a_problem_is_logged_as_well_as_shown_in_the_pane(qapp, caplog):
    import logging

    tab, w = _tab(qapp)
    with caplog.at_level(logging.WARNING, logger="squidmip._agave_pane"):
        w.problem.emit("AGAVE is not installed on this machine.")
    assert any("not installed" in r.getMessage() for r in caplog.records)
    assert "not installed" in _all_text(tab)      # and STILL where the user is looking
    tab.shutdown()

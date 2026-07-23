"""Decentralized viewer windows: one INDEPENDENT napari window per selection.

WHY THIS EXISTS. The app was one locked window that owned a plate view, a central napari pane
and a right "exploration" pane, wired together in a splitter so the whole thing moved as a slab.
Spencer's brief (2026-07-23 call) is the opposite: the plate is the ROOT, and clicking a
selection opens an INDEPENDENT napari window that floats on the desktop. Many wells become ONE
window with a region slider, not many windows. Every open window is tracked by ID in an "Open
View list" so the user can raise it. That is what this module builds.

Nothing here reinvents napari. Each window is a ``MosaicPane`` — the same full napari window the
central pane was — placed in its own ``QMainWindow``. Navigation is the same ``RegionCursor`` +
``RegionSlider`` the central pane used. The mosaic load is the same ``_MosaicWorker`` fusing FOVs
off the GUI thread. The only new thing is that these pieces are now instanced PER WINDOW instead
of once for a locked central pane, and a registry tracks the windows.

The reader is stateless (``reader.read(region, fov, channel, z)`` is a pure keyed read), so every
window SHARES the one reader/meta the root opened. No window reopens the dataset.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger("squidmip.regionviewer")

#: Debounce before a settled region is fused, matching the central pane's 140 ms. The red frame /
#: slider move instantly; only the expensive fuse waits for the slider to stop, so a drag across
#: ten regions fuses ONE mosaic instead of ten. See _region_nav for why the region is not an axis.
_REGION_LOAD_DEBOUNCE_MS = 140

#: Processing layer key for the raw fused mosaic (mirrors _viewer's "raw"). Operators that write
#: an OME-Zarr will add their own op key as a second visibility layer; not needed for exploration.
_RAW_OP = "raw"


class RegionViewer(QMainWindow):
    """ONE independent napari window over a subset of regions.

    Owns its own napari pane, its own region cursor + slider, and its own mosaic-load pipeline.
    Shares the app's single ``reader``/``meta`` (stateless reads). Closing it stops its worker and
    joins its slider's animation thread so a close during playback cannot abort the process.
    """

    closed = pyqtSignal(object)   # emits self, so the registry can drop it

    def __init__(
        self,
        reader: Any,
        meta: dict,
        regions: Sequence[str],
        *,
        window_id: int,
        title: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._reader = reader
        self._meta = meta
        self._regions = [str(r) for r in regions]
        self.window_id = int(window_id)
        self._worker = None
        self._pending_region: Optional[str] = None
        self._load_timer: Optional[QTimer] = None
        self._pane = None
        self._slider = None
        self._cursor = None
        self._native3d = None      # keeps a spawned 3D popout viewer alive

        # Name the window by the regions it holds (the deck shows the slider as "<> A1, B6, C3"),
        # not "N regions" — Julio: "'2 regions' is a bad name". Truncate a long list so the title
        # bar stays readable, keeping the count only as an overflow tail.
        label = title or self._region_label(self._regions)
        self.setWindowTitle(f"[{self.window_id}] {label}")
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # A modest, cascaded window — the deck's windows are small tiles, not full-screen slabs.
        # Cascade by ID so several opened in a row do not land exactly on top of one another.
        self.resize(860, 720)
        off = 28 * ((self.window_id - 1) % 8)
        self.move(120 + off, 90 + off)

        self._build()

    @staticmethod
    def _region_label(regions: "list[str]", limit: int = 3) -> str:
        if not regions:
            return "(empty)"
        if len(regions) <= limit:
            return ", ".join(regions)
        return ", ".join(regions[:limit]) + f", +{len(regions) - limit}"

    # -- construction -------------------------------------------------------------------
    def _build(self) -> None:
        from squidmip._napari_pane import make_pane
        from squidmip._region_nav import RegionCursor, RegionSlider

        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        pane, mode, message = make_pane(show_docks=True)
        if pane is None or not getattr(pane, "ok", False):
            # No napari here. Say why, out loud, in the window — never a blank floater.
            msg = QLabel(f"napari viewer unavailable — {message}")
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            msg.setStyleSheet("color:#ffd7d7;background:#3a2020;padding:16px;font-size:13px;")
            lay.addWidget(msg, 1)
            self.setCentralWidget(central)
            return
        self._pane = pane

        # PER-WINDOW TOOLBAR. 3D is a per-window action (the deck's "2D 3D" per window): pull THIS
        # window's current region into a native-resolution napari 3D popout (gallery-view recipe),
        # carrying the contrast/colormap on screen. Operators live here later (deferred per Spencer).
        tb = QToolBar("view", self)
        tb.setMovable(False)
        act3d = tb.addAction("3D (native)")
        act3d.triggered.connect(self._open_3d)
        act3d.setToolTip("Open a native-resolution napari 3D view of this region's centre FOV "
                         "(fits the GPU texture), carrying the contrast on screen.")
        self.addToolBar(Qt.TopToolBarArea, tb)

        lay.addWidget(pane, 1)

        # THE REGION SLIDER — napari's own dims slider driven by our region cursor. One owner of
        # "which region is current"; the slider and the loader are subscribers, never opinions.
        self._cursor = RegionCursor()
        self._cursor.on_problem(self._say)
        self._cursor.subscribe(self._on_region_changed)
        self._slider = RegionSlider()
        self._slider.on_problem(self._say)
        self._slider.bind(self._cursor)
        lay.addWidget(self._slider)

        self.setCentralWidget(central)

        # Seed the cursor: this announces region 0 to the loader, so the first mosaic loads now.
        self._cursor.set_order(self._regions)
        if self._cursor.index is None and self._regions:
            self._cursor.set_index(0)

    # -- navigation ---------------------------------------------------------------------
    def _on_region_changed(self, index: int, region: str) -> None:
        """Current region moved. Debounce the fuse; the slider label already moved instantly."""
        if getattr(self, "_load_timer", None) is None:
            self._load_timer = QTimer(self)
            self._load_timer.setSingleShot(True)
            self._load_timer.timeout.connect(
                lambda: self._load_mosaic(self._pending_region))
        self._pending_region = region
        self._load_timer.start(_REGION_LOAD_DEBOUNCE_MS)

    def _load_mosaic(self, region: Optional[str]) -> None:
        """Fuse one region's FOVs into this window's napari pane, one layer per channel."""
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return
        if self._reader is None or self._meta is None or not region:
            return
        from squidmip._viewer import _MosaicWorker

        prior = self._worker
        if prior is not None and prior.isRunning():
            prior.stop()
            prior.wait(2000)

        pane.mosaic.remove_op(_RAW_OP)
        channels = [c["name"] for c in self._meta["channels"]]
        w = _MosaicWorker(self._reader, self._meta, region, channels, z_index=0, parent=self)
        w.ready.connect(lambda r, ch, levels, bbox: self._on_plane(r, ch, levels, bbox))
        w.problem.connect(self._say)
        w.finished_count.connect(lambda n: self._on_done(region, n))
        self._worker = w
        w.start()

    def _on_plane(self, region: str, channel: str, levels, bbox_um) -> None:
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return
        if self._cursor is not None and self._cursor.region != region:
            return                                  # a later region won the race; drop this one
        from squidmip._napari_pane import _colormap_for

        pane.mosaic.add_mosaic(
            _RAW_OP, channel, levels,
            colormap=_colormap_for(channel),
            multiscale=True,
            bbox_um=bbox_um,
            z_scale_um=(self._meta or {}).get("dz_um"),
        )

    def _on_done(self, region: str, n: int) -> None:
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return
        if n == 0:
            pane.say(f"{region}: no mosaic could be built (see the message above).")
            self._frame_done()
            return
        pane.say("")
        try:
            pane.mosaic.show_op(_RAW_OP)
            pane.mosaic.model.reset_view()
        except Exception:                            # noqa: BLE001 - view framing is cosmetic
            pass
        self._frame_done()

    def _frame_done(self) -> None:
        """Open the playback gate: this region is on screen, the next may be requested."""
        if self._slider is not None:
            self._slider.frame_done()

    # -- 2D -> 3D, per window -----------------------------------------------------------
    def _open_3d(self) -> None:
        """Open a native-resolution napari 3D popout of this window's current region (gallery-view
        recipe, ``_napari3d.open_native_3d``), carrying the per-channel contrast/colormap on screen.

        A popout, not an embedded toggle: 3D is the whole region's centre FOV at NATIVE resolution
        (fits the GPU texture where a fused mosaic cannot), so it is its own window the user can
        place beside the 2D one — exactly the compare-two-views flow Spencer described."""
        region = self._cursor.region if self._cursor is not None else (
            self._regions[0] if self._regions else None)
        if region is None or self._reader is None or self._meta is None:
            self._say("no region to render in 3D.")
            return
        contrast_by: dict = {}
        colormap_by: dict = {}
        if self._pane is not None and self._pane.mosaic is not None:
            for c in self._meta.get("channels", []):
                name = c["name"]
                layer = self._pane.mosaic.find(_RAW_OP, name)
                if layer is None:
                    continue
                try:
                    contrast_by[name] = tuple(layer.contrast_limits)
                except Exception:                    # noqa: BLE001 - contrast is a nicety
                    pass
                try:
                    cmap = layer.colormap
                    colormap_by[name] = getattr(cmap, "name", cmap)
                except Exception:                    # noqa: BLE001
                    pass
        from squidmip._napari3d import open_native_3d

        try:
            # Keep a ref so the popout viewer is not garbage-collected the instant this returns.
            self._native3d = open_native_3d(
                self._reader, self._meta, region,
                contrast_by_channel=contrast_by or None,
                colormap_by_channel=colormap_by or None,
            )
        except Exception as exc:                     # noqa: BLE001 - named to the window, never silent
            self._say(f"3D view could not open: {exc}")

    def _say(self, text: str) -> None:
        if self._pane is not None and getattr(self._pane, "ok", False):
            self._pane.say(text)
        elif text:
            log.warning("[window %s] %s", self.window_id, text)

    # -- render-halt: a window not being manipulated must not keep drawing ----------------
    def set_active(self, active: bool) -> None:
        """Halt draw/refresh on windows the user is not touching (Spencer's memory brief).

        A window that is not the active one stops its playback so it is not fusing regions in the
        background and competing for the GPU with the window the user is actually looking at.
        """
        if active or self._slider is None:
            return
        try:
            if self._slider.is_playing:
                self._slider.stop()
        except Exception:                            # noqa: BLE001 - best effort
            pass

    def changeEvent(self, event):                    # noqa: N802 - Qt naming
        from PyQt5.QtCore import QEvent

        if event.type() == QEvent.ActivationChange:
            self.set_active(self.isActiveWindow())
        super().changeEvent(event)

    # -- teardown -----------------------------------------------------------------------
    def closeEvent(self, event):                     # noqa: N802 - Qt naming
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(2000)
        except Exception:                            # noqa: BLE001
            pass
        try:
            if self._slider is not None:
                self._slider.shutdown()
        except Exception:                            # noqa: BLE001
            pass
        self.closed.emit(self)
        super().closeEvent(event)


class ViewerManager(QObject):
    """Registry of open :class:`RegionViewer` windows, keyed by a monotonic ID.

    The root plate window owns one of these. It is the single source of "what windows are open",
    so the Open View list is a pure VIEW of it and can never drift from the real set of windows.
    Memory is polled here (not per window) so one warning speaks for the whole app.
    """

    windowsChanged = pyqtSignal()          # the set of open windows changed
    memoryChanged = pyqtSignal(float)      # process RSS as a fraction 0..1 of total RAM

    def __init__(self, reader: Any = None, meta: Optional[dict] = None,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._reader = reader
        self._meta = meta
        self._windows: "dict[int, RegionViewer]" = {}
        self._next_id = 1

        self._mem_timer = QTimer(self)
        self._mem_timer.setInterval(2000)
        self._mem_timer.timeout.connect(self._poll_memory)
        self._mem_timer.start()

    def set_dataset(self, reader: Any, meta: dict) -> None:
        self._reader, self._meta = reader, meta

    @property
    def windows(self) -> "list[RegionViewer]":
        return list(self._windows.values())

    def open(self, regions: Sequence[str], *, title: Optional[str] = None) -> Optional[RegionViewer]:
        """Open ONE independent window over *regions*. Many regions => one window with a slider."""
        if self._reader is None or self._meta is None:
            log.warning("open() called before a dataset was loaded; ignoring.")
            return None
        regions = [str(r) for r in regions if r]
        if not regions:
            return None
        wid = self._next_id
        self._next_id += 1
        win = RegionViewer(self._reader, self._meta, regions, window_id=wid, title=title)
        win.closed.connect(self._on_window_closed)
        self._windows[wid] = win
        win.show()
        win.raise_()
        win.activateWindow()
        self.windowsChanged.emit()
        return win

    def focus(self, window_id: int) -> None:
        win = self._windows.get(int(window_id))
        if win is not None:
            win.showNormal()
            win.raise_()
            win.activateWindow()

    def close(self, window_id: int) -> None:
        win = self._windows.get(int(window_id))
        if win is not None:
            win.close()

    def close_all(self) -> None:
        for win in list(self._windows.values()):
            win.close()

    def _on_window_closed(self, win: "RegionViewer") -> None:
        self._windows.pop(getattr(win, "window_id", -1), None)
        self.windowsChanged.emit()

    def _poll_memory(self) -> None:
        frac = _process_memory_fraction()
        if frac is not None:
            self.memoryChanged.emit(frac)


class OpenViewList(QWidget):
    """The "Open View list": every open window by ID, plus a live memory bar.

    Clicking a row raises that window to the front of the desktop — the meeting's "give it an ID,
    click it to pop it forward". A flat list of IDs is dev-step one; parent/child nesting (a
    selection's regions, then its ROIs) is the next step and slots onto the same tree.
    """

    def __init__(self, manager: ViewerManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._manager = manager

        # DARK THEME. Without an explicit stylesheet this widget renders WHITE against the dark app
        # (Julio: "Open views window still white") — QTreeWidget/QProgressBar do not inherit the
        # app palette on macOS. Match the plate's palette (#0b0e14 bg, #c9d1d9 text) here.
        self.setStyleSheet(
            "QWidget{background:#0b0e14;color:#c9d1d9;}"
            "QTreeWidget{background:#0d1117;border:1px solid #232b3a;border-radius:4px;"
            "outline:none;}"
            "QTreeWidget::item{padding:4px 6px;}"
            "QTreeWidget::item:selected{background:#1f6feb;color:#ffffff;}"
            "QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
            "border-radius:4px;padding:4px 10px;}"
            "QPushButton:hover{background:#21262d;}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        header = QLabel("Open views")
        header.setStyleSheet("color:#c9d1d9;font-size:13px;font-weight:600;border:none;")
        lay.addWidget(header)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(False)
        self._tree.itemActivated.connect(self._on_activated)
        self._tree.itemClicked.connect(self._on_activated)
        lay.addWidget(self._tree, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        close_btn = QPushButton("Close view")
        close_btn.clicked.connect(self._close_selected)
        row.addWidget(close_btn)
        row.addStretch(1)
        lay.addLayout(row)

        self._mem_label = QLabel("Memory")
        self._mem_label.setStyleSheet("color:#8b949e;font-size:11px;border:none;")
        lay.addWidget(self._mem_label)
        self._mem_bar = QProgressBar(self)
        self._mem_bar.setRange(0, 100)
        self._mem_bar.setTextVisible(True)
        self._mem_bar.setFixedHeight(14)
        lay.addWidget(self._mem_bar)

        manager.windowsChanged.connect(self.refresh)
        manager.memoryChanged.connect(self._on_memory)
        self.refresh()

    def refresh(self) -> None:
        self._tree.clear()
        for win in self._manager.windows:
            item = QTreeWidgetItem([win.windowTitle()])
            item.setData(0, Qt.UserRole, int(win.window_id))
            self._tree.addTopLevelItem(item)

    def _on_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        wid = item.data(0, Qt.UserRole)
        if wid is not None:
            self._manager.focus(int(wid))

    def _close_selected(self) -> None:
        item = self._tree.currentItem()
        if item is not None:
            wid = item.data(0, Qt.UserRole)
            if wid is not None:
                self._manager.close(int(wid))

    def _on_memory(self, frac: float) -> None:
        pct = max(0, min(100, int(round(frac * 100))))
        self._mem_bar.setValue(pct)
        # Warn out loud past 85%: Spencer wanted a memory bar AND a warning, not a silent cap.
        warn = pct >= 85
        self._mem_label.setText("Memory — HIGH, close a view" if warn else "Memory")
        color = "#f85149" if warn else "#3fb950"
        self._mem_bar.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;}"
            f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"
        )


def _process_memory_fraction() -> Optional[float]:
    """This process's RSS as a fraction of total system RAM, or None if it can't be measured.

    Tries psutil (accurate, cross-platform incl. the Windows target); falls back to resource +
    a best-effort total. Returns None rather than a fake number when neither is available — a
    memory bar that invents a value is worse than one that is honestly absent.
    """
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        return float(proc.memory_info().rss) / float(psutil.virtual_memory().total)
    except Exception:                                # noqa: BLE001 - psutil optional
        pass
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in bytes, Linux in kilobytes.
        import sys

        rss = float(rss_kb) if sys.platform == "darwin" else float(rss_kb) * 1024.0
        import os

        total = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        return rss / total if total > 0 else None
    except Exception:                                # noqa: BLE001
        return None

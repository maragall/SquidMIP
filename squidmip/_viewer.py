"""HCS viewer — a post-acquisition, well-plate viewer for Squid acquisitions (IMA-185).

A single professional Qt window, isolated from the Squid acquisition software. This tool runs on
acquisitions that are ALREADY on disk (post-acquisition), so there is no live-follow machinery — it
opens a completed scan and lets you navigate it and apply post-processing operators to it.

    drop a Squid acquisition folder
      -> LEFT  (<= half the display): a low-resolution PLATE OVERVIEW — one cell per well, laid out
               in true plate row-major (A,B,...,Z,AA,...). Each well is HUE-CODED by its PROCESSING
               status (Hongquan Li's record-zstack-viewer palette): grey = not processed, amber =
               processing, blue = done, red-x = failed. Row/column labels, black grid, red hover box.
      -> RIGHT (>= half): ndviewer_light EMBEDDED (dark-themed) — the per-FOV 4D detail. DOUBLE-CLICK
               a well and its RAW z-stack (all z, all channels) opens here by pointing ndviewer at the
               acquisition's existing TIFFs (register_image with the raw paths) — zero bytes copied,
               nothing written to disk. The z / t sliders are the real acquisition axes.
      -> PROCESS WELL-PLATES menu -> "Maximum Intensity Projection": run the operator over every well.
               The plate fills with MIP thumbnails as each well completes, hue tracking the run.

The plate is the spatial navigator; ndviewer handles the per-FOV z-stack. "Processing" here means
post-processing: MIP is operator #1, and more operators stack behind the same menu (the moment a
second operator lands this is a general HCS viewer, not just a MIP tool).

Design notes:
- ndviewer_light is the embedded detail viewer (its LightweightViewer QWidget + push API); PyQt5 to
  match its stack. PyQt5 is imported here, never in squidmip/__init__, so the pipeline stays Qt-free.
- Nothing is written to the user's disk: the detail view reads the acquisition's own read-only
  TIFFs. Memory is NOT one-well-at-a-time on the plate side: the MIP run retains one downsampled
  88x88xC float32 tile PER WELL (_OperatorWorker._raw) for the final global-contrast montage, so
  the plate-side footprint is O(n_wells x C) (~190 MB for a 1536wp, C=4), plus a grid-sized RGB
  canvas (~36 MB) and a transient float32 montage buffer at run end. Bounded by the plate format
  (<=1536 wells), not by z/frame size. What IS one-well-at-a-time is project_plate's producer
  (workers x one ~139 MB well) and the detail viewer's LRU-bounded decoded planes.
- Hit-testing / cell fitting are pure functions (unit-testable); widgets run headless under
  QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QRect, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QAction, QApplication, QLabel, QMainWindow, QSplitter, QVBoxLayout, QWidget,
)

from squidmip._engine import _default_workers
from squidmip._montage import _area_downsample, _hex_to_rgb01, _window
from squidmip._output import parse_well_id

_CELL = 88                 # per-well px in the low-res overview (1536wp -> ~4224x2816)
_HDR, _COLH = 46, 30       # left / top label margins (px)
_PAD = 16                  # breathing room around the plate
_VIEWER_WORKERS = min(6, _default_workers())   # adapt to the machine, but CAP at 6: the producer's
                           # peak RAM is ~workers x one-well (~139 MB each on a 1536wp), and projection
                           # throughput scales only sublinearly past ~6 threads — so more workers buys
                           # little speed for linearly more memory. 6 balances both, leaves GUI cores.
_BG = "#070a0f"
_GRID, _RED, _MUTED, _ACCENT = QColor(0, 0, 0), QColor("#ff2d2d"), QColor("#8b98ad"), QColor("#58a6ff")

# Processing-status hue coding, adopted from Hongquan Li's record-zstack-viewer plate navigator.
# Deliberately colorblind-safe (blue/amber, never red/green) with a shape cue for failure (the x).
_STATUS = {
    "empty":      QColor("#b7bcc4"),   # not yet processed
    "processing": QColor("#f59e0b"),   # amber — running now
    "done":       QColor("#3b82f6"),   # blue — MIP computed
    "failed":     QColor("#ef4444"),   # red outline + x cross
}
_NDV_DARK = (  # ndviewer defaults to light; theme its Qt chrome dark (bg AND text) to match
    "QWidget{background:#0b0e14;color:#e6edf3;}"
    "QLabel{color:#e6edf3;background:transparent;}"
    "QSlider::groove:horizontal{background:#232b3a;height:4px;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#58a6ff;width:12px;margin:-5px 0;border-radius:6px;}"
    "QPushButton{background:#131824;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;padding:3px 8px;}"
)

# Post-processing operators. MIP is operator #1; add an entry here and the menu grows. Each maps a
# label to the streaming generator that yields (region, fov, projected-image) per well.
_OPERATORS = {
    "mip": {"label": "Maximum Intensity Projection", "menu": "&Maximum Intensity Projection"},
}


# --- pure geometry (unit-testable, no Qt display) -------------------------------------------

def well_at(rows, cols, by_rc, px: float, py: float, cell_disp: float) -> Optional[dict]:
    """Map a plate pixel (px, py) at *cell_disp* px/well to a cell, or None if out of bounds.

    ``by_rc`` maps (row_index, col_index) -> well_id for acquired wells (else the cell is 'empty').
    Pixels are relative to the plate's top-left (label margins already removed by the caller).
    """
    if px < 0 or py < 0:
        return None
    ci, ri = int(px // cell_disp), int(py // cell_disp)
    if ci >= len(cols) or ri >= len(rows):
        return None
    return {"row_index": ri, "col_index": ci, "row": rows[ri], "col": cols[ci],
            "well_id": by_rc.get((ri, ci))}


def _fit_cell(a: np.ndarray) -> np.ndarray:
    """Resize a 2D plane to EXACTLY (_CELL, _CELL) for the montage tile.

    Area-downsample when larger (the common case: a ~768px tile -> 88); nearest-upscale a tiny
    frame so the tile shape is always (_CELL, _CELL) (guards the <88px-frame crash the review found).
    """
    if a.shape == (_CELL, _CELL):
        return a
    if a.shape[0] >= _CELL and a.shape[1] >= _CELL:
        return _area_downsample(a, _CELL, _CELL)
    yi = (np.arange(_CELL) * a.shape[0]) // _CELL
    xi = (np.arange(_CELL) * a.shape[1]) // _CELL
    return a[yi][:, xi].astype(np.float32)


def resolve_plate_root(path) -> tuple[Path, bool]:
    """(path, is_plate): is_plate True when *path* already holds an OME-zarr plate (not a raw
    acquisition); False for a raw acquisition (the case this viewer opens)."""
    p = Path(path)
    if (p / "plate.ome.zarr").is_dir() or (p.name.endswith(".zarr") and (p / "zarr.json").exists()):
        return p, True
    return p, False


class _RunningContrast:
    """Per-channel global contrast that updates as wells stream in (histogram over tiles so far)."""

    def __init__(self, n_ch: int, dmax: float, pct=(1.0, 99.8), bins=512):
        self._bins, self._dmax, self._pct = bins, max(1.0, float(dmax)), pct
        self._hist = [np.zeros(bins, dtype=np.int64) for _ in range(n_ch)]

    def add(self, ch: int, tile: np.ndarray):
        idx = np.clip((tile.ravel() / self._dmax * self._bins).astype(int), 0, self._bins - 1)
        self._hist[ch] += np.bincount(idx, minlength=self._bins)

    def window(self, ch: int) -> tuple[float, float]:
        h = self._hist[ch]
        tot = h.sum()
        if tot == 0:
            return 0.0, self._dmax
        cdf = np.cumsum(h) / tot
        lo = np.searchsorted(cdf, self._pct[0] / 100.0) / self._bins * self._dmax
        hi = np.searchsorted(cdf, self._pct[1] / 100.0) / self._bins * self._dmax
        return float(lo), float(max(hi, lo + 1))


# --- plate overview widget (one cell per well; hue-coded status; fit-to-view) ---------------

class PlateOverview(QWidget):
    """The low-res plate: an RGB canvas of MIP tiles, plus a per-well status hue and a red box."""

    hovered = pyqtSignal(str)              # region id (or "" off-plate), for the window's readout
    wellActivated = pyqtSignal(str, int)   # (well_id, fov_index) double-clicked -> load in ndviewer

    def __init__(self, rows, cols, wells: dict):
        """``wells``: (row_index, col_index) -> well_id for every acquired well (drawn grey until
        processed). Tiles/status arrive as an operator runs."""
        super().__init__()
        self._rows, self._cols = list(rows), list(cols)
        self._nr, self._nc = len(self._rows), len(self._cols)
        self._by_rc: dict[tuple, str] = dict(wells)            # every acquired well (for status + hit-test)
        self._status: dict[tuple, str] = {rc: "empty" for rc in wells}
        self._tiles: set[tuple] = set()                        # cells that have a MIP tile painted
        self._canvas = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
        self._canvas.fill(QColor(_BG))
        self._final = None            # crisp global-contrast montage swapped in when the run ends
        self._cd = float(_CELL)       # displayed px/well (recomputed to fit the widget)
        self._ox = self._oy = _PAD    # top-left of the plate within the widget (centered)
        self._hover = None
        self._sel = None              # well selected from the ndviewer FOV slider
        self.setMouseTracking(True)
        self.setMinimumSize(240, 200)

    # -- data in --
    def add_tile(self, ri: int, ci: int, well_id: str, rgb: np.ndarray):
        if (ri, ci) not in self._by_rc:    # ignore a stale tile from a retired run / foreign cell
            return
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        img = QImage(rgb.data, _CELL, _CELL, 3 * _CELL, QImage.Format_RGB888)
        p = QPainter(self._canvas)
        p.drawImage(ci * _CELL, ri * _CELL, img)
        p.end()
        self._tiles.add((ri, ci))
        self.update()

    def set_status(self, ri: int, ci: int, state: str):
        if (ri, ci) not in self._status:   # never let a foreign/stale key leak into the status map
            return
        self._status[(ri, ci)] = state
        self.update()

    def set_all_status(self, state: str):
        for rc in self._status:
            self._status[rc] = state
        self.update()

    def set_final(self, img: QImage):
        self._final = img
        self.update()

    def select(self, ri: int, ci: int):
        """Move the red box to a well (driven by the ndviewer FOV slider)."""
        self._sel = (ri, ci)
        self.update()

    # -- fit-to-view (single resolution: the whole plate always fits this widget, centered) --
    def _layout(self):
        if self._nr == 0 or self._nc == 0:
            return                                       # empty grid — nothing to fit (guard div-by-zero)
        w, h = self.width(), self.height()
        cd = min((w - _HDR - 2 * _PAD) / self._nc, (h - _COLH - 2 * _PAD) / self._nr)
        self._cd = max(2.0, cd)
        pw, ph = self._nc * self._cd, self._nr * self._cd
        self._ox = max(_PAD, (w - _HDR - pw) / 2)
        self._oy = max(_PAD, (h - _COLH - ph) / 2)

    def resizeEvent(self, e):
        self._layout()
        self.update()

    # -- mouse --
    def _cell(self, x, y):
        px, py = x - (self._ox + _HDR), y - (self._oy + _COLH)
        return well_at(self._rows, self._cols, self._by_rc, px, py, self._cd)

    def mouseMoveEvent(self, e):
        c = self._cell(e.x(), e.y())
        self._hover = (c["row_index"], c["col_index"]) if c else None
        self.hovered.emit((c["well_id"] or (c["row"] + c["col"] + "  ·  empty")) if c else "")
        self.update()

    def leaveEvent(self, e):
        self._hover = None
        self.hovered.emit("")
        self.update()

    def mouseDoubleClickEvent(self, e):
        c = self._cell(e.x(), e.y())
        if c and c["well_id"]:
            self.wellActivated.emit(c["well_id"], 0)   # 1 FOV/well (IMA-183); IMA-187 will pick the FOV

    # -- paint --
    def paintEvent(self, _):
        self._layout()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(_BG))
        cd, nr, nc = self._cd, self._nr, self._nc
        ax, ay = self._ox + _HDR, self._oy + _COLH   # plate top-left (after label margins)
        W, H = nc * cd, nr * cd
        p.drawImage(QRect(int(ax), int(ay), int(W), int(H)), self._final or self._canvas)

        # per-well STATUS hue: a filled dot on empty/processing cells, a colored ring on tiled cells,
        # and a red x on failures. Drawn under the grid so the black grid still reads.
        ring = max(1.5, cd * 0.09)
        for (ri, ci), state in self._status.items():
            x0, y0 = ax + ci * cd, ay + ri * cd
            col = _STATUS[state]
            if state == "failed":
                p.setPen(QPen(col, max(1.5, cd * 0.08)))
                p.drawRect(int(x0), int(y0), int(cd), int(cd))
                p.drawLine(int(x0), int(y0), int(x0 + cd), int(y0 + cd))
                p.drawLine(int(x0 + cd), int(y0), int(x0), int(y0 + cd))
            elif (ri, ci) in self._tiles:
                p.setPen(QPen(col, ring))               # status ring around the MIP thumbnail
                p.drawRect(int(x0 + ring / 2), int(y0 + ring / 2), int(cd - ring), int(cd - ring))
            else:
                p.setPen(Qt.NoPen)                      # not-yet / processing: a centered status dot
                p.setBrush(col)
                d = max(3.0, cd * 0.42)
                p.drawEllipse(int(x0 + (cd - d) / 2), int(y0 + (cd - d) / 2), int(d), int(d))
        p.setBrush(Qt.NoBrush)

        p.setPen(QPen(_GRID, 3))       # black grid lines between wells (room for multi-FOV, IMA-187)
        for c in range(nc + 1):
            p.drawLine(int(ax + c * cd), int(ay), int(ax + c * cd), int(ay + H))
        for r in range(nr + 1):
            p.drawLine(int(ax), int(ay + r * cd), int(ax + W), int(ay + r * cd))
        p.setFont(QFont("Helvetica Neue", 11, QFont.DemiBold))
        for c in range(nc):
            p.setPen(_ACCENT if self._hover and self._hover[1] == c else _MUTED)
            p.drawText(int(ax + c * cd), int(self._oy), int(cd), _COLH, Qt.AlignCenter, str(self._cols[c]))
        for r in range(nr):
            p.setPen(_ACCENT if self._hover and self._hover[0] == r else _MUTED)
            p.drawText(int(self._ox), int(ay + r * cd), _HDR, int(cd), Qt.AlignCenter, str(self._rows[r]))
        box = self._hover or self._sel     # hover wins; else the slider-selected well
        if box:
            ri, ci = box
            p.setPen(QPen(_RED, 2))
            p.drawRect(int(ax + ci * cd), int(ay + ri * cd), int(cd), int(cd))
        p.end()


# --- operator worker: stream a projection over the plate, fill row-major -------------------

class _OperatorWorker(QThread):
    """Runs a post-processing operator (MIP) over the plate, filling one tile per well as it
    completes (no reorder buffer — a buffer would stall behind slow low-index wells). Memory:
    project_plate's PRODUCER is bounded (workers x one ~139 MB well) and each full-res well is
    dropped after we downsample it, so full-res frames never accumulate. But we DO retain one
    88x88xC float32 tile per well in ``_raw`` for the final global-contrast montage, so this
    object's footprint is O(n_wells x C) (~190 MB for 1536wp/C=4), not flat. ``_final_montage``
    additionally spikes ~3x the grid-sized float32 canvas transiently (see its note). Nothing is
    written to disk.
    """

    tileReady = pyqtSignal(int, int, str, object)   # (row_index, col_index, well_id, rgb tile)
    progress = pyqtSignal(int, int)                 # (done, total)
    finalReady = pyqtSignal(object)                 # final global-contrast montage (H, W, 3) uint8
    failed = pyqtSignal(str)
    finished_ok = pyqtSignal()

    def __init__(self, operator: str, reader, meta, fov_index: dict, nr: int, nc: int):
        super().__init__()
        self._operator = operator
        self._reader, self._meta = reader, meta
        self._fov_index, self._nr, self._nc = fov_index, nr, nc
        self._channels = [c["name"] for c in meta["channels"]]
        self._colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in meta["channels"]])
        self._dtype = np.dtype(meta["dtype"])
        self._contrast = _RunningContrast(len(self._channels), float(np.iinfo(self._dtype).max))
        self._raw: dict[tuple, np.ndarray] = {}   # (ri,ci) -> (C, _CELL, _CELL) tiles, for the final montage
        self._stop = threading.Event()            # set by the window to end the run cleanly

    def stop(self):
        """Ask the run to stop; run() returns after the current well. Call wait() after."""
        self._stop.set()

    def run(self):
        try:
            from squidmip import project_plate   # operator #1; dispatch here as more operators land

            total = len(self._meta["regions"])
            done = 0
            for region, fov, image in project_plate(self._reader, n_fovs=1, workers=_VIEWER_WORKERS,
                                                    projector=self._operator):
                if self._stop.is_set():
                    return  # window closing / re-opening; drop out cleanly (no final emit)
                info = self._fov_index[region]
                ri, ci, well_id = *info["rc"], info["well_id"]
                well = image[0, :, 0]  # (C, Y, X)
                raw = np.empty((well.shape[0], _CELL, _CELL), self._dtype)  # native dtype: half the
                #                                    RAM of float32 for the retained per-well tiles
                rgb = np.zeros((_CELL, _CELL, 3), np.float32)
                for c_i in range(len(self._channels)):
                    ds = _fit_cell(well[c_i])
                    raw[c_i] = ds
                    self._contrast.add(c_i, ds)
                    lo, hi = self._contrast.window(c_i)
                    rgb += _window(ds, lo, hi)[:, :, None] * self._colors[c_i][None, None, :]
                self._raw[(ri, ci)] = raw
                self.tileReady.emit(ri, ci, well_id, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
                done += 1
                self.progress.emit(done, total)
            self.finalReady.emit(self._final_montage())
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _final_montage(self) -> np.ndarray:
        wins = [self._contrast.window(ch) for ch in range(len(self._channels))]
        canvas = np.zeros((self._nr * _CELL, self._nc * _CELL, 3), np.float32)
        for (ri, ci), raw in self._raw.items():
            y0, x0 = ri * _CELL, ci * _CELL
            for ch in range(raw.shape[0]):
                lo, hi = wins[ch]
                canvas[y0:y0 + _CELL, x0:x0 + _CELL] += _window(raw[ch], lo, hi)[:, :, None] * self._colors[ch][None, None, :]
        # clip/scale IN PLACE — avoids 3 grid-sized float32 copies (a ~430 MB transient on a 1536wp)
        np.clip(canvas, 0, 1, out=canvas)
        canvas *= 255
        return canvas.astype(np.uint8)


# --- main window: plate overview | embedded ndviewer ----------------------------------------

class PlateWindow(QMainWindow):
    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("HCS viewer")
        self.resize(1600, 950)
        self._worker = None
        self._retired = []            # workers asked to stop; kept alive until they actually finish
        self._overview = None
        self._reader = None
        self._meta = None
        self._fov_index = {}
        self._pushed = set()          # wells whose raw z-stack is already registered in the detail viewer
        self._final_arr = None        # keep the final montage array alive for its QImage

        # Process-well-plates menu (operators). MIP is #1; disabled until an acquisition is open.
        self._op_actions = {}
        proc_menu = self.menuBar().addMenu("&Process well-plates")
        for key, spec in _OPERATORS.items():
            act = QAction(spec["menu"], self)
            act.setEnabled(False)
            act.triggered.connect(lambda _=False, k=key: self.run_operator(k))
            proc_menu.addAction(act)
            self._op_actions[key] = act

        self._readout = QLabel("drop a Squid acquisition")
        self._readout.setStyleSheet("color:#e6edf3;font-size:22px;font-weight:800;padding:8px 18px;")
        bar = QWidget()
        bar.setStyleSheet(f"background:{_BG};border-bottom:1px solid #232b3a;")
        QVBoxLayout(bar).addWidget(self._readout)

        self._drop = QLabel("Drop a Squid acquisition folder here\n\nProcess well-plates ▸ Maximum Intensity Projection")
        self._drop.setAlignment(Qt.AlignCenter)
        self._drop.setStyleSheet("color:#8b98ad;font-size:17px;border:2px dashed #232b3a;border-radius:12px;margin:40px;")
        left = QWidget()
        left.setStyleSheet(f"background:{_BG};")
        self._left_l = QVBoxLayout(left)
        self._left_l.setContentsMargins(0, 0, 0, 0)
        self._left_l.addWidget(self._drop, 1)   # the plate overview replaces this on ingest

        self._detail = self._make_detail_viewer()
        if self._detail is not None:   # connect the FOV slider -> red box ONCE (not per ingest)
            slider = getattr(self._detail, "_fov_slider", None)
            if slider is not None:
                slider.valueChanged.connect(self._on_fov_slider)

        split = QSplitter(Qt.Horizontal)
        split.setStyleSheet("QSplitter::handle{background:#232b3a;width:1px;}")
        split.setChildrenCollapsible(False)
        split.addWidget(left)
        if self._detail is not None:
            split.addWidget(self._detail)
            split.setSizes([760, 840])                 # plate <= half the display (Hongquan's note)
            split.setStretchFactor(0, 1)
            split.setStretchFactor(1, 1)
        self._split = split

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(bar)
        cl.addWidget(split, 1)
        self.setCentralWidget(central)

        self.setAcceptDrops(True)
        if initial_path:
            self.ingest(initial_path)

    def _make_detail_viewer(self):
        try:
            from ndviewer_light.core import LightweightViewer
            v = LightweightViewer()       # empty -> push mode (we register raw z-planes on demand)
            v.setStyleSheet(_NDV_DARK)    # ndviewer defaults to light; match the plate view
            return v
        except Exception as e:
            self._readout.setText(f"ndviewer_light unavailable: {e}")
            return None

    # -- drag & drop --
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.ingest(urls[0].toLocalFile())

    # -- open an acquisition (no processing yet — that's the Process menu) --
    def ingest(self, path: str):
        from squidmip import open_reader
        from squidmip._output import plate_metadata

        p, is_plate = resolve_plate_root(path)
        if is_plate:
            self._readout.setText("this is already a written plate — drop a raw Squid acquisition")
            return
        # stop any in-flight run and clear prior state before opening a new acquisition
        self._stop_worker()
        self._reader = self._meta = None
        self._fov_index = {}
        self._pushed = set()
        self._final_arr = None
        for act in self._op_actions.values():
            act.setEnabled(False)
        if self._overview is not None:
            self._overview.setParent(None)
            self._overview.deleteLater()
            self._overview = None
        self._readout.setText("scanning acquisition …")
        QApplication.processEvents()
        try:
            reader = open_reader(str(p))
            meta = reader.metadata
        except Exception as e:   # not a Squid acquisition / unreadable -> report, don't crash the app
            self._readout.setText(f"not a readable Squid acquisition: {e}")
            self._drop.show()
            return
        self._reader, self._meta = reader, meta

        # Order wells in TRUE plate row-major (A,B,...,Z,AA,...). NOT lexicographic ("AA" < "B").
        # This parses region ids as well ids — guard it: a readable acquisition whose regions are
        # NOT well-plate ids (glass slide, manual/coordinate names, "R2C3", "0") must report, not
        # crash. parse_well_id / plate_metadata raise ValueError on those. (contract: never crash.)
        try:
            plate = plate_metadata(meta["regions"], field_count=1)["plate"]
            rows = [r["name"] for r in plate["rows"]]
            cols = [c["name"] for c in plate["columns"]]
            row_of = {r: i for i, r in enumerate(rows)}   # plate order: A=0,B=1,...,Z=25,AA=26,...
            col_of = {c: i for i, c in enumerate(cols)}

            def _rc(region):
                rr, cc = parse_well_id(region)
                return (row_of[rr], col_of[cc])
            order = sorted(meta["regions"], key=_rc)
        except (ValueError, KeyError) as e:
            self._reader = self._meta = None
            self._readout.setText(
                f"not a well-plate acquisition — regions like {list(meta['regions'])[:3]} aren't "
                f"well ids (e.g. B2); the HCS viewer needs a well plate. ({type(e).__name__})")
            self._drop.show()
            return
        wells = {}
        for idx, region in enumerate(order):
            rc = _rc(region)
            self._fov_index[region] = {"idx": idx, "well_id": region, "rc": rc}
            wells[rc] = region

        self._overview = PlateOverview(rows, cols, wells)
        self._overview.hovered.connect(self._on_hover)
        self._overview.wellActivated.connect(self.activate_well)
        self._drop.hide()
        self._left_l.addWidget(self._overview, 1)   # fills the pane and self-fits — no scrollbars

        if self._detail is not None:
            # push mode over the RAW acquisition: full z (real z-stack) and full frame; the detail
            # reads the acquisition's own TIFFs (register_image) — nothing copied.
            h, w = meta["frame_shape"]
            self._detail.start_acquisition([c["name"] for c in meta["channels"]], meta["n_z"],
                                           h, w, [f"{r}:0" for r in order])

        for act in self._op_actions.values():
            act.setEnabled(True)
        self._readout.setText(
            f"{len(self._fov_index)} wells  ·  double-click a well for its z-stack  ·  "
            "Process well-plates ▸ Maximum Intensity Projection")

    # -- run a post-processing operator over the whole plate --
    def run_operator(self, key: str):
        if self._reader is None or self._overview is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._readout.setText("already processing — let the current run finish first")
            return
        self._overview.set_all_status("processing")          # amber across the plate
        label = _OPERATORS[key]["label"]
        self._worker = _OperatorWorker(key, self._reader, self._meta, self._fov_index,
                                       self._overview._nr, self._overview._nc)
        self._worker.tileReady.connect(self._on_tile)
        self._worker.progress.connect(lambda d, t: self._readout.setText(f"{label}  {d}/{t} wells"))
        self._worker.finalReady.connect(self._set_final)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(lambda: self._readout.setText(
            f"{label} complete  ·  {len(self._fov_index)} wells  ·  double-click a well for its z-stack"))
        self._worker.start()

    def _on_tile(self, ri, ci, well_id, rgb):
        if self._overview is None:
            return
        self._overview.add_tile(ri, ci, well_id, rgb)
        self._overview.set_status(ri, ci, "done")           # blue

    def _on_failed(self, msg):
        if self._overview is not None:
            for rc, state in list(self._overview._status.items()):
                if state == "processing":
                    self._overview.set_status(*rc, "failed")  # red x on wells that didn't finish
        self._readout.setText(f"failed: {msg}")

    def _set_final(self, rgb):
        if self._overview is None:
            return
        self._final_arr = np.ascontiguousarray(rgb)
        h, w, _ = self._final_arr.shape
        self._overview.set_final(QImage(self._final_arr.data, w, h, 3 * w, QImage.Format_RGB888))

    # -- navigation links --
    def _on_hover(self, text: str):
        if text:
            self._readout.setText(text)

    def activate_well(self, well_id: str, fov_index: int):
        """Double-click -> open the well's RAW z-stack in the embedded ndviewer. Registers the raw
        TIFF paths lazily (once per well): the detail is the true z-stack, zero bytes copied."""
        if self._detail is None or self._reader is None or well_id not in self._fov_index:
            return
        idx = self._fov_index[well_id]["idx"]
        if well_id not in self._pushed:
            fov = self._meta["fovs_per_region"][well_id][0]
            for z_i, z in enumerate(self._meta["z_levels"]):
                for ch in (c["name"] for c in self._meta["channels"]):
                    try:
                        path = self._reader.plane_path(well_id, fov, ch, z)
                        self._detail.register_image(0, idx, z_i, ch, str(path))
                    except (KeyError, IndexError, OSError, RuntimeError):
                        continue   # a genuinely-missing plane / closed viewer shouldn't block the rest
            self._pushed.add(well_id)
        row, col = parse_well_id(well_id)
        try:
            self._detail.go_to_well_fov(f"{row}{col}", fov_index)
        except Exception:
            pass

    def _on_fov_slider(self, flat_idx: int):
        """ndviewer FOV slider moved -> move the red box on the plate to that well."""
        if self._detail is None or self._overview is None:
            return
        labels = getattr(self._detail, "_fov_labels", None)
        if not labels or not (0 <= flat_idx < len(labels)):
            return
        info = self._fov_index.get(labels[flat_idx].split(":")[0])
        if info:
            self._overview.select(*info["rc"])

    def _stop_worker(self):
        """Retire the operator thread WITHOUT ever destroying a running QThread (that aborts the
        app). First disconnect its signals so a tile already queued before the stop can't paint onto
        a freshly-opened plate (the cross-plate corruption the review found); then keep a reference
        alive until it actually finishes (stop() returns after the current well, which is bounded)."""
        w = self._worker
        self._worker = None
        if w is None:
            return
        for sig in (w.tileReady, w.progress, w.finalReady, w.failed, w.finished_ok):
            try:
                sig.disconnect()
            except TypeError:
                pass                 # nothing connected — fine
        if w.isRunning():
            w.stop()
            self._retired.append(w)
            w.finished.connect(lambda: self._retired.remove(w) if w in self._retired else None)

    def closeEvent(self, e):
        self._stop_worker()          # stop the run cleanly; nothing on disk to clean up (no cache)
        for w in list(self._retired):
            w.wait()                 # join before exit — never leave a QThread running at teardown
        super().closeEvent(e)


def main(dataset_path: str = None):
    path = dataset_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    app = QApplication.instance() or QApplication(sys.argv)
    win = PlateWindow(path)
    win.show()
    if not app.property("_squidmip_test"):
        sys.exit(app.exec_())
    return win


if __name__ == "__main__":
    main()

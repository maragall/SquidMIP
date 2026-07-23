"""The AGAVE 3D view as a tab in PANE 3 — the exploration (supplementary) pane.

WHY PANE 3. Julio re-specified what pane 3 is for: "on the right pane we show results is maybe
not the best design, because when I run the MIP or background sub or flatfield correction or
stitcher or decon, like these are also reflected in the plate view and in my central viewer and
that's why I turn layers on and off. This mean that the exploration pane is a supplementary pane
to for example embed agave". Operator RESULTS stay in the plate view and the centre viewer as
toggleable layers; pane 3 hosts supplementary views, and AGAVE is exactly one.

WHY A WORKER THREAD. Every AGAVE call blocks: the server takes ~2 s to start, fusing a region's
volume takes seconds, and a path-traced frame takes ~1.3 s. All of it happens on ONE long-lived
thread that owns the websocket and the AGAVE process, so the GUI never freezes and the socket is
never touched from two threads.

    AgaveTab  --request_*-->  queue  -->  _AgaveWorker.run  -->  AgaveEngine  -->  agave --server
       ^                                        |
       +--------- rendered / problem -----------+   (Qt signals, back on the GUI thread)

THE THREAD IS ALWAYS JOINED. This codebase has a hard-won rule (``_join_retired`` /
``_stop_mosaic_worker`` in ``_viewer``): a QThread destroyed while still running ABORTS the
process. ``shutdown()`` therefore posts the sentinel and ``wait()``s, and it is idempotent
because the tab close AND the window close both call it.

NO SILENT FAILURES. Every ``problem`` the worker emits is printed in the tab, verbatim and
selectable, and the frame is cleared so a stale render can never pass for the current one. A
missing AGAVE is a named refusal in this pane, never an empty pane.
"""

from __future__ import annotations

import logging
import queue
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from squidmip._agave import DEFAULT_MAX_PX, AgaveEngine

#: Not 1235. AGAVE's own default is what a hand-started server uses, and adopting one whose
#: lifetime we do not own is a lifecycle we cannot end cleanly.
DEFAULT_PANE_PORT = 1246

#: Frame size and quality. 96 iterations at 900x700 measured 1.3 s on the owner's machine; 64 is
#: the interactive default and still path-traced.
FRAME_SIZE = (900, 700)
FRAME_ITERATIONS = 64

#: Degrees of orbit per pixel dragged.
ORBIT_PER_PX = 0.4

_BG = "#0d1117"

#: stdlib logging, so the 3D view's progress and its failures appear in the app's log panel for
#: free. Never INSTEAD of the in-pane message: a failure the user cannot see where they are
#: looking is the silent failure this pane exists to prevent.
log = logging.getLogger(__name__)


class _AgaveWorker(QThread):
    """The one thread that talks to AGAVE. Jobs in, frames (or named problems) out."""

    rendered = pyqtSignal(bytes)
    problem = pyqtSignal(str)
    note = pyqtSignal(str)

    def __init__(self, reader, meta, acq_path, *, port: int = DEFAULT_PANE_PORT,
                 max_px: int = DEFAULT_MAX_PX, iterations: int = FRAME_ITERATIONS,
                 parent=None) -> None:
        super().__init__(parent)
        self._args = (reader, meta, acq_path)
        self._port = int(port)
        self._max_px = int(max_px)
        self._iterations = int(iterations)
        self._size = FRAME_SIZE
        self._q: "queue.Queue" = queue.Queue()
        self._alive = True

    # -- requests (called from the GUI thread) -------------------------------------------
    def request_region(self, region: str, t: int = 0) -> None:
        self._put(("region", str(region), int(t)))

    def request_time(self, t: int) -> None:
        self._put(("time", int(t)))

    def request_orbit(self, dtheta: float, dphi: float) -> None:
        self._put(("orbit", float(dtheta), float(dphi)))

    def request_zoom(self, notches: float) -> None:
        self._put(("zoom", float(notches)))

    def request_frame(self, width: int, height: int) -> None:
        self._put(("frame", int(width), int(height)))

    def request_launch_app(self) -> None:
        self._put(("launch_app",))

    def _put(self, job) -> None:
        if self._alive:
            self._q.put(job)

    # -- the thread ------------------------------------------------------------------------
    def run(self) -> None:                                    # pragma: no cover - needs AGAVE
        reader, meta, acq_path = self._args
        engine = AgaveEngine(reader, meta, acq_path, port=self._port, max_px=self._max_px)
        try:
            engine.open()
        except Exception as exc:                              # noqa: BLE001 - NAMED, not swallowed
            log.error("the 3D view could not open: %s", exc)
            self.problem.emit(str(exc))
            engine.close()
            return
        self.note.emit(f"AGAVE server running on port {engine.port}.")
        try:
            while True:
                job = self._q.get()
                if job is None:
                    break
                try:
                    self._do(engine, job)
                except Exception as exc:                      # noqa: BLE001
                    log.error("3D view job %r failed: %s: %s", job[0], type(exc).__name__, exc)
                    self.problem.emit(f"{type(exc).__name__}: {exc}")
        finally:
            engine.close()                                    # server killed, cache deleted

    def _do(self, engine, job) -> None:                       # pragma: no cover - needs AGAVE
        kind = job[0]
        if kind == "region":
            info = engine.show_region(job[1], job[2])
            mb = info["bytes"] / 1e6
            self.note.emit(
                f"{info['region']}: {info['shape'][0]}z x {info['shape'][1]}c x "
                f"{info['shape'][3]}x{info['shape'][2]} px, {mb:.0f} MB "
                f"({'cached' if info['cached'] else f'{info['seconds']:.1f}s to fuse'}), "
                f"voxel {info['voxel_um'][0]:.2f}/{info['voxel_um'][1]:.2f}/"
                f"{info['voxel_um'][2]:.2f} um."
            )
        elif kind == "time":
            engine.set_time(job[1])
        elif kind == "orbit":
            engine.orbit(job[1], job[2])
        elif kind == "zoom":
            engine.zoom(job[1])
        elif kind == "launch_app":
            self.note.emit("fusing a higher-resolution volume and launching the full AGAVE app…")
            info = engine.open_in_app()
            self.note.emit(
                f"opened the full AGAVE app on {info['region']} at ~{info['max_px']} px "
                f"(all controls, native full screen). If it starts empty, open this file: "
                f"{info['path']}")
            return                                    # launching does not change the embedded frame
        elif kind == "frame":
            self._size = (job[1], job[2])
        w, h = self._size
        self.rendered.emit(engine.frame(w, h, self._iterations))

    # -- teardown ---------------------------------------------------------------------------
    def shutdown(self) -> None:
        """Idempotent, and it JOINS: a QThread destroyed while running aborts the process."""
        if not self._alive and not self.isRunning():
            return
        self._alive = False
        self._q.put(None)
        if self.isRunning():
            self.wait(20000)


def _make_worker(reader, meta, acq_path, **kw) -> _AgaveWorker:
    """The seam the tests replace, so no test ever starts a real thread or a real AGAVE."""
    return _AgaveWorker(reader, meta, acq_path, **kw)


class AgaveTab(QWidget):
    """Pane 3's 3D view: a region slider, a timepoint slider, a frame, and a status line."""

    def __init__(self, reader, meta: dict, acq_path, parent=None, *, worker=None) -> None:
        super().__init__(parent)
        self._meta = dict(meta or {})
        self.regions = [str(r) for r in (self._meta.get("regions") or [])]
        self.n_timepoints = max(1, int(self._meta.get("n_t") or 1))
        self.has_problem = False
        self._down = False
        self._fs_window = None                    # the full-screen top-level, when active
        self._prev_parent = None
        self.setFocusPolicy(Qt.StrongFocus)       # so Esc reaches keyPressEvent in full screen

        self.setStyleSheet(f"background:{_BG};")
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 10)
        v.setSpacing(6)

        self.title = QLabel("3D — AGAVE path-traced volume")
        self.title.setStyleSheet("color:#c9d1d9;font-size:12px;font-weight:700;")
        # The embedded pane paints AGAVE frames into a QLabel: quick, but it exposes NONE of AGAVE's
        # transfer-function / lighting / material controls and is capped at the interactive volume
        # resolution. "Open in AGAVE" launches the real desktop app on a higher-res fuse, which is
        # where all the controls, max resolution and native full screen actually live.
        self.launch_btn = QPushButton("Open in AGAVE (all controls)")
        self.launch_btn.setToolTip("Launch the full AGAVE application on a higher-resolution volume "
                                   "of the current region: every control, max detail, native full "
                                   "screen. The embedded view here stays the quick preview.")
        self.launch_btn.setStyleSheet("color:#c9d1d9;font-size:11px;padding:2px 8px;")
        self.launch_btn.clicked.connect(self._launch_full_app)
        self.fs_btn = QPushButton("Full screen")
        self.fs_btn.setToolTip("Full-screen the embedded preview (Esc to exit). "
                               "Scroll to zoom, drag to orbit. For all controls use 'Open in AGAVE'.")
        self.fs_btn.setStyleSheet("color:#c9d1d9;font-size:11px;padding:2px 8px;")
        self.fs_btn.clicked.connect(self.toggle_fullscreen)
        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.launch_btn)
        header.addWidget(self.fs_btn)
        v.addLayout(header)

        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumHeight(240)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # FILL the pane (and the full-screen window). Without this the fixed-size AGAVE pixmap sat
        # small and centred on a big black canvas, which is why "full screen doesn't work" -- the
        # window went full screen but the image did not grow. Scale the pixmap to the label instead.
        self.canvas.setScaledContents(True)
        self.canvas.setStyleSheet("background:#000;border:1px solid #232b3a;")
        v.addWidget(self.canvas, 1)

        # A REGION is a mosaic of FOVs and is the unit of navigation — never a single FOV.
        self.region_label = QLabel("region —")
        self.region_slider = QSlider(Qt.Horizontal)
        self.region_slider.setMinimum(0)
        self.region_slider.setMaximum(max(0, len(self.regions) - 1))
        self.region_slider.setEnabled(len(self.regions) > 1)
        self.region_slider.valueChanged.connect(self._on_region)
        v.addLayout(_row(self.region_label, self.region_slider))

        self.time_label = QLabel("t 0")
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(max(0, self.n_timepoints - 1))
        self.time_slider.setEnabled(self.n_timepoints > 1)
        self.time_slider.valueChanged.connect(self._on_time)
        if self.n_timepoints <= 1:
            self.time_label.setText("t 0 — single timepoint acquisition")
        v.addLayout(_row(self.time_label, self.time_slider))

        self.status = QLabel("starting AGAVE…")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)   # errors are copyable
        self.status.setStyleSheet("color:#8b98ad;font-size:11px;")
        v.addWidget(self.status)

        self._worker = worker if worker is not None else _make_worker(reader, self._meta, acq_path)
        self._worker.rendered.connect(self._on_frame)
        self._worker.problem.connect(self._on_problem)
        self._worker.note.connect(self._on_note)
        self._stopped = False
        self._worker.start()
        if self.regions:
            self.region_label.setText(f"region {self.regions[0]}")
            self._worker.request_region(self.regions[0], 0)
        else:
            self._on_problem("this acquisition declares no regions, so there is nothing to render "
                             "in 3D.")

    # -- sliders (both LAZY) -------------------------------------------------------------
    def _on_region(self, index: int) -> None:
        if not self.regions:
            return
        region = self.regions[max(0, min(int(index), len(self.regions) - 1))]
        self.region_label.setText(f"region {region}")
        self.status.setText(f"fusing {region}…")
        self._worker.request_region(region, int(self.time_slider.value()))

    def _on_time(self, t: int) -> None:
        """SET_TIME only. The loaded volume already holds every timepoint, so this never
        re-fuses — that is what makes the timepoint slider the cheap one."""
        self.time_label.setText(f"t {int(t)}")
        self._worker.request_time(int(t))

    def drag(self, dx: float, dy: float) -> None:
        """Orbit by a mouse drag. Exposed as a method so the offscreen tests drive the real path."""
        self._worker.request_orbit(float(dx) * ORBIT_PER_PX, float(dy) * ORBIT_PER_PX)

    def mousePressEvent(self, e):                             # pragma: no cover - GUI gesture
        self._down = True
        self._last = e.pos()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):                              # pragma: no cover - GUI gesture
        if self._down:
            p = e.pos()
            self.drag(p.x() - self._last.x(), p.y() - self._last.y())
            self._last = p
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):                           # pragma: no cover - GUI gesture
        self._down = False
        super().mouseReleaseEvent(e)

    def zoom(self, notches: float) -> None:
        """Zoom the volume. Exposed as a method so the offscreen tests drive the real path."""
        self._worker.request_zoom(float(notches))

    def wheelEvent(self, e):                                  # pragma: no cover - GUI gesture
        # One wheel notch is 120 in Qt's angleDelta units. Positive (wheel up) = zoom IN. This is
        # the gesture the user reported missing ("when I click on the agave window I can't zoom").
        notches = e.angleDelta().y() / 120.0
        if notches:
            self.zoom(notches)
        e.accept()

    def _launch_full_app(self) -> None:
        """Hand off to the worker: fuse a higher-res volume and launch the full AGAVE app on it."""
        self.status.setText("preparing the full AGAVE app (fusing a higher-resolution volume)…")
        self.status.setStyleSheet("color:#8b98ad;font-size:11px;")
        self._worker.request_launch_app()

    def toggle_fullscreen(self) -> None:
        """Full-screen the 3D view so it leverages the whole app for a fancy render; restore on a
        second press or Esc. Reparents to a top-level window rather than maximising the tab, so it
        truly fills the display, and asks for a bigger, higher-iteration frame while there."""
        if self._fs_window is not None:
            self._exit_fullscreen()
            return
        self._prev_parent = self.parentWidget()
        self._fs_window = QWidget()
        self._fs_window.setWindowTitle("AGAVE — 3D (full screen)  ·  Esc to exit")
        self._fs_window.setStyleSheet(f"background:{_BG};")
        lay = QVBoxLayout(self._fs_window)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self)                       # reparents self into the full-screen window
        self.fs_btn.setText("Exit full screen")
        # Request a frame sized to the actual screen so the fancy render is crisp, not a 1600px
        # image stretched to fill. Fall back to a large default if the screen size is unavailable.
        try:
            from PyQt5.QtWidgets import QApplication
            scr = QApplication.primaryScreen().size()
            self._worker.request_frame(min(scr.width(), 2560), min(scr.height(), 1600))
        except Exception:                    # noqa: BLE001 - a bigger frame is a nicety
            self._worker.request_frame(2200, 1400)
        self._fs_window.showFullScreen()

    def _exit_fullscreen(self) -> None:
        if self._fs_window is None:
            return
        parent = self._prev_parent
        if parent is not None and parent.layout() is not None:
            parent.layout().addWidget(self)
        self.fs_btn.setText("Full screen")
        self._worker.request_frame(*FRAME_SIZE)
        self._fs_window.close()
        self._fs_window = None

    def keyPressEvent(self, e):                               # pragma: no cover - GUI gesture
        if e.key() == Qt.Key_Escape and self._fs_window is not None:
            self._exit_fullscreen()
            return
        super().keyPressEvent(e)

    # -- from the worker ------------------------------------------------------------------
    def _on_frame(self, data: bytes) -> None:
        # NO format hint: AGAVE 1.10.0 streams JPEG, not PNG, and a hard-coded "PNG" makes
        # loadFromData fail on every single frame. Qt sniffs the format from the bytes.
        pm = QPixmap()
        if not pm.loadFromData(bytes(data)):
            self._on_problem("the frame AGAVE returned could not be decoded as an image.")
            return
        self.has_problem = False
        self.canvas.setPixmap(pm)

    def _on_problem(self, message: str) -> None:
        """Say it BY NAME, and drop the frame: a stale render must never pass for the current one."""
        self.has_problem = True
        log.warning("3D view: %s", message)
        self.canvas.clear()
        self.status.setText(str(message))
        self.status.setStyleSheet("color:#ff7b72;font-size:11px;")

    def _on_note(self, message: str) -> None:
        if self.has_problem:
            self.has_problem = False
            self.status.setStyleSheet("color:#8b98ad;font-size:11px;")
        self.status.setText(str(message))

    # -- teardown -------------------------------------------------------------------------
    def shutdown(self) -> None:
        """Duck-typed by ``PlateWindow._dispose_tab_widget`` (tab close, float close, app exit)."""
        if self._stopped:
            return
        self._stopped = True
        self._worker.shutdown()

    def closeEvent(self, e):
        self.shutdown()
        super().closeEvent(e)


def _row(label: QLabel, slider: QSlider) -> QHBoxLayout:
    label.setStyleSheet("color:#8b98ad;font-size:11px;")
    label.setMinimumWidth(120)
    h = QHBoxLayout()
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(label)
    h.addWidget(slider, 1)
    return h

"""The grouped layer tree, and the 2D/3D button that has to be reachable on a small screen.

Why a separate file. ``tests/test_napari_view.py`` is being edited by several agents at once;
everything this branch adds lives here so the two never collide.

Why subprocesses. napari's canvas is vispy/GL and the gate runs the suite under
``QT_QPA_PLATFORM=offscreen``, which ships no GL — constructing a canvas under it does not
raise, it SEGFAULTS the session. ``test_napari_view.py`` already solved this: run the Qt part
in a clean subprocess with the platform plugin left alone, so a crash is a test failure rather
than a dead run, and a genuinely GL-less box skips with the reason attached. The pure-logic
parts below need neither Qt nor napari and run in-process.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

napari = pytest.importorskip("napari")

REPO = pathlib.Path(__file__).resolve().parent.parent


def _run_qt(script_body: str, tmp_path, marker: str):
    """Run *script_body* in a clean Qt process and return the dict it printed after *marker*.

    An exception inside OUR code prints ``<marker>FAIL`` and FAILS the test. Only a box with no
    GL at all produces no marker line and skips. A skip and a bug must never look the same —
    that is how the embedding check read green for its whole life while asserting nothing.
    """
    script = tmp_path / f"{marker.lower()}_check.py"
    script.write_text(_PREAMBLE.replace("__MARKER__", marker) + script_body + _POSTAMBLE.replace("__MARKER__", marker))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # The gate exports offscreen for the whole suite and offscreen has no GL, so inheriting it
    # guarantees a segfault and a permanent skip. Let Qt pick the real platform.
    env.pop("QT_QPA_PLATFORM", None)
    # squidmip imports PyQt5; qtpy defaults to PySide6 here and loading both aborts the process
    # long before any assertion runs.
    env["QT_API"] = "pyqt5"

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=300, cwd=str(REPO), env=env,
    )
    failed = [ln for ln in proc.stdout.splitlines() if ln.startswith(marker + "FAIL ")]
    if failed:
        pytest.fail("Qt check raised:\n" + json.loads(failed[0][len(marker) + 5:]))
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith(marker + " ")]
    if not line:
        pytest.skip(
            f"napari's Qt canvas could not be constructed here (rc={proc.returncode}); "
            f"stderr tail: {proc.stderr[-400:]}"
        )
    return json.loads(line[0][len(marker) + 1:])


_PREAMBLE = r"""
import json, os, sys, traceback
os.environ.setdefault("QT_API", "pyqt5")
import numpy as np
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
app = QApplication.instance() or QApplication([])
out = {}
try:
    from squidmip._napari_pane import MosaicPane
"""

_POSTAMBLE = r"""
    print("__MARKER__ " + json.dumps(out))
except BaseException:
    print("__MARKER__FAIL " + json.dumps(traceback.format_exc()))
sys.stdout.flush()
os._exit(0)
"""


# ---------------------------------------------------------------- the 2D/3D button
#
# The button is not missing from napari — it is napari's own ``QtViewerButtons.ndisplayButton``,
# and a probe of the embedded window found it present and visible at y=752 inside a 900 px host:
# the LAST row of the left dock column, under a layer list that grows with every layer. Julio is
# on a small monitor and has asked for a visible 3D toggle twice. So the fix is not to build a
# button, it is to put NAPARI'S button somewhere that does not scroll off: a fixed row at the top
# of pane 2.

_NDISPLAY_SCRIPT = r"""
    host = QWidget()
    host.resize(1440, 900)          # Julio's monitor is small; check at the width he uses.
    lay = QVBoxLayout(host)
    pane = MosaicPane()
    lay.addWidget(pane)
    host.show()
    app.processEvents()

    from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton

    btn = pane.ndisplay_button
    top_of_pane = btn.mapTo(pane, btn.rect().topLeft()).y() if btn is not None else -1

    out["is_napari_widget_class"] = isinstance(btn, QtViewerPushButton)
    out["visible"] = bool(btn.isVisible())
    # "Visible" is not enough: the napari one is visible too, 752 px down. It has to be near the
    # top, where a short pane still shows it.
    out["y_within_pane"] = top_of_pane
    out["pane_height"] = pane.height()

    before = int(pane.mosaic.model.dims.ndisplay)
    btn.click(); app.processEvents()
    after = int(pane.mosaic.model.dims.ndisplay)
    checked_in_3d = bool(btn.isChecked())
    btn.click(); app.processEvents()
    back = int(pane.mosaic.model.dims.ndisplay)

    out["toggle"] = [before, after, back]
    out["checked_follows_dims"] = checked_in_3d
    # napari's dims is the ONE owner of 2D/3D. Move it from the model and our button must follow
    # without anybody hand-syncing it.
    pane.mosaic.model.dims.ndisplay = 3
    app.processEvents()
    out["follows_model_write"] = bool(btn.isChecked())
    pane.mosaic.model.dims.ndisplay = 2
    app.processEvents()
    out["unchecks_on_model_write"] = bool(btn.isChecked())
"""


def test_the_3d_button_is_naparis_own_and_sits_where_a_short_pane_shows_it(tmp_path):
    """A 2D/3D toggle Julio can actually see, built out of napari's own button.

    Asked for twice. napari HAS the button — bottom of the left dock column, below a layer list
    that grows with every layer added, so on a small screen it is simply not on screen. Lifting
    napari's own widget into a fixed row at the top of the pane fixes reachability without
    inventing a second control: the button we show and the one napari docks drive the same
    ``viewer.dims.ndisplay``, so they cannot disagree.
    """
    got = _run_qt(_NDISPLAY_SCRIPT, tmp_path, "NDISPLAY")

    assert got["is_napari_widget_class"] is True, "we rebuilt a button instead of reusing napari's"
    assert got["visible"] is True
    assert 0 <= got["y_within_pane"] <= 80, (
        f"the 3D button is {got['y_within_pane']} px down a {got['pane_height']} px pane — "
        "that is the same 'present but off the bottom' failure it was meant to fix"
    )
    before, after, back = got["toggle"]
    assert [before, after, back] == [2, 3, 2], "clicking it does not actually change ndisplay"
    assert got["checked_follows_dims"] is True
    # One owner: dims. The button READS it, it does not keep a second copy.
    assert got["follows_model_write"] is True
    assert got["unchecks_on_model_write"] is False

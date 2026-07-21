#!/usr/bin/env python3
"""Headless acceptance gate: drive the REAL widget on the REAL acquisitions.

Why this exists. Every defect this project has shipped passed a green unit suite,
because nothing drove the application. The backend was solid, the GUI wiring was
dead, and the test doubles agreed with each other. Two examples, both real:

  * ``minerva_selection()`` probed only ``PlateOverview`` while the selection it
    wanted lived on ``PlateWindow``. It reached the right answer by accident
    through a fallback, and every test passed.
  * The viewer refused every glass-slide acquisition outright. The unit suite was
    green the whole time, because a test asserted the refusal.

So: run this after every land, on both real datasets, before believing anything.

    QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python tools/acceptance.py

Exit code is 0 only if every case passes. Both env vars are required: without
PYTEST_DISABLE_PLUGIN_AUTOLOAD the PyQt5 tests silently skip against PySide.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

# Run from anywhere: import the repo this file lives in, not whatever `squidmip`
# happens to be installed. The mac filesystem is case-insensitive, so an invoker
# sitting in .../CEPHLA/ instead of .../Cephla/ otherwise resolves a different tree.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The two acquisitions the product is actually demoed on. READ ONLY - never copy
# or convert them; copying the 18 GB set is how this machine hit 0 bytes free.
TISSUE = ("/Users/julioamaragall/Downloads/"
          "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")
PLATE = "/Users/julioamaragall/Downloads/synthetic_2x2_wellplate"

# (label, path, expected regions, expected fov_positions_um entries)
CASES = [
    ("tissue (glass slide, freeform regions)", TISSUE, ["manual0", "manual1"], 55),
    ("2x2 well plate", PLATE, ["A1", "A2", "B1", "B2"], 144),
]


_APP = None


def check(label, path, want_regions, want_positions):
    from PyQt5.QtWidgets import QApplication
    import squidmip._viewer as V

    # Keep a module-level reference: a QApplication with no Python owner is garbage
    # collected, and the next QWidget aborts with 'Must construct a QApplication first'.
    global _APP
    _APP = QApplication.instance() or QApplication([])
    win = V.PlateWindow(None)
    fails = []
    try:
        win.ingest(path)
    except Exception as e:
        return [f"ingest raised {type(e).__name__}: {e}"]

    readout = (getattr(getattr(win, "_readout", None), "text", lambda: "")() or "")
    if win._reader is None:
        fails.append(f"reader is None; readout: {readout!r}")
        return fails
    if win._overview is None:
        fails.append(f"no plate overview built; readout: {readout!r}")

    meta = win._reader.metadata
    got_regions = list(meta.get("regions") or [])
    if got_regions != want_regions:
        fails.append(f"regions {got_regions} != expected {want_regions}")

    n_pos = len(meta.get("fov_positions_um") or {})
    if n_pos != want_positions:
        fails.append(f"fov_positions_um has {n_pos} entries, expected {want_positions}")

    # The units contract: world space is micrometres and the key says so. A plate
    # spans tens of thousands of um, never tens - that is the 1000x tell.
    if n_pos:
        xs = [v[0] for v in meta["fov_positions_um"].values()]
        span = max(xs) - min(xs)
        if span < 1000:
            fails.append(f"x span {span:.1f} looks like mm, not um (units regression)")

    # An acquisition that opens must not also report that it cannot be opened.
    for bad in ("not supported", "not a well-plate", "cannot lay out",
                "not a readable", "no pixels"):
        if bad in readout.lower():
            fails.append(f"readout still reports failure: {readout!r}")
            break

    for ch_key in ("channels",):
        if not meta.get(ch_key):
            fails.append(f"metadata[{ch_key!r}] is empty")

    try:
        win.close()
    except Exception:
        pass
    return fails


def main():
    rc = 0
    for label, path, regions, positions in CASES:
        if not os.path.exists(path):
            print(f"SKIP  {label}\n      dataset not present: {path}")
            continue
        try:
            fails = check(label, path, regions, positions)
        except Exception:
            fails = ["harness error:\n" + traceback.format_exc()]
        if fails:
            rc = 1
            print(f"FAIL  {label}")
            for f in fails:
                print(f"      - {f}")
        else:
            print(f"PASS  {label}  ({len(regions)} regions, {positions} positions)")
    print("\nacceptance:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())

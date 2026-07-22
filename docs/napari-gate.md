# The napari gate — measured, then decided

The decision to move the viewer to napari was made on **cold-start** numbers. The two
measurements that could have **invalidated** it had never been run. They have now been run,
before any migration code was written. This file is the evidence.

Hardware: Apple M4, macOS 26.2. napari 0.8.0 / PyQt5 5.15.11 / zarr 3.2.1 in the scratch venv
for the timing rows; the embedding result was independently reproduced on the **napari 0.6.6**
installed in the working environment.

---

## Gate (a) — warm per-tile

The historical "ndviewer_light is 0.30 ms warm per tile" figure is a **data read with no canvas
in it**. Comparing a render cost against it is comparing different things, so it is reported
separately from the render.

**Data read (viewer-independent).** 512² uint16 chunks, each read exactly once so none is a
cache hit: **0.69–0.74 ms median**, p90 0.76 ms, n=144. This is the same order as the recorded
1.43 ms full-res chunk baseline; napari does not change it, because it is not napari's code.

**Render — the fair comparison.** Identical tiles, identical loop, forced repaint, both stacks
driven through the same probe. Fairness proof: **both runs printed checksum 154030901463.**

| stack | median ms | p90 | min | max | n |
|---|---|---|---|---|---|
| ndv 0.4.1 | 26.46 | 39.57 | 24.21 | 56.57 | 48 |
| **napari 0.8.0** | **16.74** | 38.24 | 12.30 | 44.67 | 48 |

**napari is 1.58x FASTER than ndv per warm tile.** The gate allowed 2x slower. It passes with
room, in the opposite direction from the risk.

## Gate (b) — clipped multiscale pan/zoom (napari issue #1942)

This is the failure mode our mosaics are the exact workload for: "large multiscale zarrs slow
on pan/zoom when clipped".

**A probe for this already existed and had never been executed. Running it as written would
have produced a misleading PASS**, for two reasons found on inspection:

1. It preloaded every pyramid level into numpy. napari then never touches the store during a
   pan, so it measures vispy upload, not #1942, which is about *fetching* while clipped.
2. Its fixture was 2084² — smaller than three canvas widths. Measured: `data_level` stayed **0
   for every zoom**, i.e. the multiscale code path never ran at all.

Rebuilt to remove both: a **16384² five-level pyramid, 512 chunks, backed by dask-over-zarr**
(the lazy path `napari-ome-zarr` actually gives you), canvas ~1000x800, and the probe now
**asserts that levels change** rather than assuming it.

- Levels actually exercised during the zoom sweep: **0, 2, 4** (`MULTISCALE_ACTUALLY_EXERCISED:
  true`). Clipped pan runs at **level 0**, showing well under 1% of a 16384² plane.

| measurement | ms |
|---|---|
| clipped pan, median | **22.59** |
| clipped pan, p90 | 29.80 |
| clipped pan, max | 37.81 |
| zoom step (level switches) | 22.0 – 28.6 |
| still repaint, no camera move (control) | 8.97 |

The control matters: a pan step costs ~13.6 ms *on top of* a plain redraw. That is the real
fetch+relevel cost, and it is bounded — it does not grow as you pan.

**The memory result is the decisive one.** Level 0 is **536.9 MB**. Across the whole clipped
pan, RSS went **345.8 MB -> 398.1 MB, i.e. +52 MB.** napari is fetching the clipped region, not
materialising the level. That is precisely the pathology #1942 describes, and **it does not
reproduce.**

`pan_async_wait_ms_max` was 0.0 — the probe blocks on `layer.loaded`, so these are not fast
numbers taken against an empty canvas.

### Verdict: **PASS**, on both criteria, by measurement rather than by assumption.

---

## Embedding — there is a PUBLIC path

Every previous demonstration (40+) drove napari through `viewer.window._qt_viewer`. That is
private. `Window.qt_viewer` is public but raises a `FutureWarning` calling itself an
"implementation detail" to be removed no earlier than v0.9.0. Neither is a foundation, and this
project already lost a day to a private binding that bound cleanly and did nothing.

The supported path needs no private access at all:

```python
from napari.components import ViewerModel   # in components.__all__
from napari.qt import QtViewer              # in napari.qt.__all__

model = ViewerModel()
qt_viewer = QtViewer(model)                 # a plain QWidget
```

`QtViewer.__init__` is annotated `viewer: ViewerModel`, so this is the intended construction.
Verified present, exported, and identically signed on **napari 0.6.6 and 0.8.0**.

Because no napari `Window` is constructed, the chrome is never built in the first place —
measured on the embedded widget: **0 menu items, 0 dock widgets, no layer-controls container**,
and it still paints real multiscale pixels (screenshot had 236 distinct values, not blank).
That is a structural answer to "watch out for feature bloat", not chrome hidden after the fact.

No `napari.run()`, no second event loop: the host QApplication drives it.

---

## The layer hierarchy — napari has NO groups

Julio's model is two levels deep:

```
PROCESSING LAYER   (raw | stitched | deconvolved | background-subtracted | ...)
  -> CHANNELS      (405, 488, 561, 638 ...)
     -> CONTRAST   per channel
```

**Confirmed: napari 0.8 has no layer groups.** `LayerGroup`/`GroupLayer` appear nowhere in the
package; `LayerList` is flat. The hierarchy is therefore built from three public pieces, and
the design was validated by probe *before* being built on:

| need | mechanism | public? |
|---|---|---|
| group identity | `layer.metadata["squidmip"] = {"op", "channel"}` | yes |
| before/after toggle | `layer.visible` flipped over one op group | yes |
| contrast shared per channel | `LayerList.link_layers(peers, ("contrast_limits",))` | yes |
| contrast notifications | `layer.events.contrast_limits` | yes |

Validated behaviour: contrast set on `raw/488` is present on `stitched/488` after the toggle
(123, 4321), other channels unaffected, exactly one op group visible.

**Identity lives in metadata, never parsed out of the layer name.** ian-stitcher recovers the
wavelength with `extractWavelength(layer.name)`; that bug class has already bitten this codebase
twice (petakit's reader emits channel names its own regex cannot parse; 3f1bf3f fixed
`Fluorescence_488_nm_Ex` failing a parser wanting `\s*nm`). The name is a label.

**One contrast value per channel, application-wide.** Because the peers are *linked*, a second
slider for the same channel cannot disagree with the first — they are the same property. That is
a structural answer to "I can still see the duplicated sliders", rather than another copy of the
contrast model.

---

## What this supersedes

- **IMA-261 / the ndv contrast tap.** `_ContrastTap` subclasses `ndv.views.bases.LutView` and
  hooks the private `_lut_controllers` dict — the most ndv-entangled design in the codebase, and
  not portable. `layer.events.contrast_limits` is the public replacement. `_napari_view`
  deliberately does **not** compute contrast windows: `_viewer._pct_window` keeps that rule,
  including its deliberate refusal to widen a degenerate window to `(lo, lo+1)` (which would
  render a blank channel as full white, i.e. as signal). No second contrast model was created.
- **IMA-255 3D volume.** Per-layer anisotropic scale is native (`layer.scale`), so the
  *capability* is free. The ~395 lines of ndv/vispy implementation and the 24 tests in
  `test_3d_volume_rendering.py` are a write-off — they assert against ndv module paths.
  Free in napari: voxel scale, anisotropy, 3D toggle. Needs porting: nothing conceptual.
  Lost: nothing, but the tests must be rewritten rather than migrated.
- **Our tiling survives untouched.** `_tiling.py` and `_tilesource.py` import only stdlib +
  numpy + squidmip internals — no viewer import of any kind. Both already speak stage
  micrometres, which is napari's world convention, so `bbox_um` maps onto `scale`/`translate`
  with only an axis flip (`(x0,y0,x1,y1)` -> `(row,col)`), pinned by a test. `DEFAULT_TILE_PX =
  512` already equals the OME-Zarr chunk size IMA-217 writes, so one tile read is one chunk
  read — keep those equal or read amplification comes back.

## Still open — honestly

- The pane is **not yet wired into the three-pane window**. `SQUIDMIP_VIEWER=napari` exists and
  defaults OFF; ndviewer_light remains the viewer until it is flipped.
- Fused-mosaic loading via `napari-ome-zarr` (ian-stitcher's proven path) is not implemented
  here; `add_mosaic` takes arrays/pyramids directly.
- Camera-settle coalescing is not implemented. Fetching per camera event is how #1942 happens;
  the measured numbers above are per settled move, not per event.
- The installer (`installer/ndviewer_light.spec`) is tuned to ndv+vispy. napari is a much
  heavier PyInstaller target. **No packaging experiment exists.** This is the largest unknown.
- True cold-disk numbers remain impossible (`purge` needs root); none are estimated.

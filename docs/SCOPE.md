# HCS Viewer — scope v1

Post-acquisition viewer + processor for on-disk Squid well-plate acquisitions. Reads finished data
(T/C/Z/FOV already on disk); **no live capture, no stage motion**. Grid-like plate, thousands of FOVs.

## Principles
- **Data-intensiveness above all** — bounded/streaming memory, flat in plate size; never OOM; fail
  loud, skip-and-continue on a bad well.
- **One engine** — every operator is a z-reduction Strategy streamed by `project_plate`; the GUI and
  the headless CLI share it. Adding an operator = one `_OPERATIONS` entry + one `build_tab`.
- **No slop / no hallucinated constants** — derive or expose tuning; no magic RAM formulas.

## Axes
- **Z is reduced, never "recorded":** MIP, or **Reference-plane** (Tenengrad autofocus; user can
  override to the plane they like in the viewer).
- **T is the video axis** (time-lapse). **C** is composited or per-channel.
- Video export axis auto-detected (T default; Z override is niche focus-QC).

## Operators (pluggable, engine-run)
1. **MIP** — max over z. ✅
2. **Reference-plane** — Tenengrad-sharpest z per well; user override. Pushed to the slider like MIP.
3. **Record → `.mp4`** — assemble T (default) at a chosen playback fps; **per-well** movie
   (whole-plate montage movie if cheap). In-viewer "play" = ndv playback of the same axis.

## Viewer model
- **Always push** each well's processed plane (1 z) to **ndv's growing FOV slider** = the plate
  navigator. Moving the FOV slider moves the **red box** on the plate. Double-click = "navigate here"
  (one meaning — no mode/subclass explosion). The z-slider appears only when inspecting one raw well.
- **All controls live inside ndv's ArrayViewer** — no external Qt controls bolted onto ndviewer_light;
  fps (190) / subset (191) / font (197) are **upstreamed to ndv**.
- **FOV-slider speed** = the real fix: make ndv **reuse** the ArrayViewer instead of rebuilding it per
  position (that rebuild is why FOV navigation is slow today).
- Zoom→full-res (level-of-detail off the written pyramids) is a v2 refinement.

## Multi-FOV (Nick = 4 FOV/well)
- **Now:** randomly pick 1 FOV/well + a GUI notice.
- **Target:** stitch → reduce (MIP/reference) per channel → one composite/well → push.
- Adapt `maragall/stitcher` into the high-throughput path (lower priority, big plus).

## Delivery
GitHub Action → **Linux AppImage + Windows + macOS** artifacts (mirroring `maragall/ndviewer_light`
and `maragall/stitcher` CI), with a **dependency-import smoke test**.

## Out of scope (v1)
Live capture, stage motion, "recording" spatial axes (already on disk), Minerva/Nautilus tabs.

## Order of work
- (a) ndv ArrayViewer-reuse speed fix + always-push model (kills the mode complexity)
- (b) Reference-plane operator (Tenengrad + viewer override)
- (c) Record → .mp4 (per-well; montage if cheap)
- (d) packaging CI (AppImage/Win/Mac + dep smoke test)
- (e) multi-FOV stitch (+ stitcher integration)

## Test datasets (symlinks, ~0 disk)
- `sim_1536wp_zt` — 1536wp, 20 z, 3 t, 4 ch (scale + z-scrub + t-playback)
- `sim_384wp_4fov` — 384wp, 4 FOV/well, 3 z, 2 t (multi-FOV path)

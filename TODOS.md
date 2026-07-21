# TODOS

Deferred work captured during plan-eng-reviews. Each item records the reasoning
so a future session doesn't rediscover it from zero.

## Writer cannot express "pixel size unknown" → future ticket
- **What:** `_output.py:173` does `p = float(pixel_size_um) if pixel_size_um else 1.0`, so a written plate records 1.0 for both "unknown" and "genuinely 1.0 µm/px". Give the writer a way to mark unknown (omit the scale, or record a sentinel/attribute), and teach readers to distinguish.
- **Why:** IMA-208's loupe draws a µm scale bar. On the computed-plate path the only pixel-size source is the multiscales `scale`, so the loupe cannot tell an unknown from a real 1.0 and must suppress the bar for BOTH. That is the honest behaviour but it silently degrades a legitimately-1.0µm acquisition.
- **Pros:** The loupe (and any future measurement UI) can show microns whenever they are actually known.
- **Cons:** Touches the writer, so already-written plates in the field stay ambiguous forever regardless; needs a migration story or a "legacy plates are ambiguous" acceptance.
- **Context:** Surfaced by both outside-voice passes during the IMA-208 eng review (2026-07-20) and verified. `_open_computed` (`_viewer.py:1683`) also builds `_meta` with no pixel size at all — IMA-208 adds the parse, but it can only read what the writer wrote.
- **Depends on / blocked by:** Nothing; independent of IMA-208, which works around it.

## `_open_computed` reuses well 0's FOV path for every well → future ticket
- **What:** `_viewer.py:1665` reads `fov0` from `wells_meta[0]` and `:1692` applies that same path to every well in `worker_wells`. A plate whose wells carry differing image ids silently renders the wrong image for those wells.
- **Why:** Latent silent-wrong-image bug in the computed-plate open path, independent of the loupe. IMA-208's D5 FOV helper covers the loupe's use of it, but the tile-loading path underneath stays wrong.
- **Pros:** Removes a whole class of silently-wrong renders; needed before multi-FOV plates are opened from disk.
- **Cons:** Requires per-well FOV resolution in `_open_computed` and a fixture with heterogeneous well image ids to test it.
- **Context:** Found by outside voice during the IMA-208 eng review (2026-07-20), verified by reading `_open_computed`. No current dataset triggers it — every well is written with the same fov id today — so it is latent rather than active.
- **Depends on / blocked by:** Overlaps viewer-side multi-FOV (IMA-187); worth doing together.

## Loupe neighbour prefetch on well crossing → fast-follow after IMA-208
- **What:** Prefetch the adjacent well's crop level while the loupe is held, so crossing a well boundary mid-hold is seamless.
- **Why:** IMA-208 deliberately scoped this out: a hold gesture usually stays within one well, and crop reads are already only a few MB, so the latency may never be noticeable.
- **Pros:** Removes the one visible hitch in the loupe's interaction.
- **Cons:** Speculative I/O and more cache surface for a case that may not matter; adds eviction pressure to a cache IMA-208 deliberately kept small.
- **Context:** IMA-208 D10 chose windowed crop reads (`_CHUNK_YX = 1024`, `_zarr_store.py:25`) over a whole-well cache. Revisit only if real use shows a hitch at well boundaries.
- **Depends on / blocked by:** IMA-208 landing and being used on a real plate.

## Scale-test fixture generator → IMA-188
- **What:** A generator that fans the 48 real hongquan FOVs across a 1536-well plate via **symlinks** (Squid layout), synthesizing 20 z (cycling the real 3) × 4 channels. On-disk ≈ source (~19 GB); logical read ≈ 1536×20×4×33 MB ≈ **4 TB** (served from OS cache — proves scale/parse/decode/memory, NOT raw disk bandwidth; that needs Nick's real storage).
- **Why:** It's the harness for the IMA-188 high-throughput scale test, not ingest. Building it in 189 bloats the keystone and risks CI breakage.
- **Pros:** Proves the reader + projection hold at plate scale with bounded memory, cheaply (symlinks, not 4 TB of real bytes).
- **Cons:** Breaks on Windows CI runners (no symlink checkout); ~120k inodes; slow to materialize.
- **Context:** The IMA-189 `SquidReader` reads one plane per call, so a symlink fan-out exercises the exact read path at scale. Keep 189's own tests on small real-shaped fixtures.
- **Depends on / blocked by:** IMA-189 reader (landed); belongs to **IMA-188 (this slot)**.

## Resume / checkpoint for long plate runs → fast-follow after IMA-184
- **What:** Skip wells whose complete output already exists; clean partial output files on rerun.
- **Why:** A full 1536wp run takes minutes-to-hours; a crash mid-run currently restarts from 0, and partial outputs can silently corrupt the plate.
- **Pros:** Turns a full-run loss into an incremental retry; mitigates the threads segfault residual (a rerun skips finished wells).
- **Cons:** Needs a per-well "complete output" definition + atomic write/rename or cleanup logic.
- **Context:** IMA-188 engine uses ThreadPoolExecutor; failure policy = per-well manifest. A C-level segfault in decode can still abort the whole process. Surfaced by /plan-eng-review outside-voice #7 (2026-07-04).
- **Depends on:** IMA-184 output layout (what a "complete well output" looks like).

## Brightfield / RGB channel ingest → future ticket
- **What:** Support Squid brightfield channels saved as `(H,W,3)` RGB (and per-LED `_B/_G/_R`) planes, with a defined reduction-to-2D (or explicit color) policy.
- **Why:** IMA-189 `read()` deliberately **raises** on non-2D planes (decision 5). Without this note the raise reads like a bug.
- **Pros:** Broadens input coverage to brightfield acquisitions.
- **Cons:** Requires an RGB→2D policy the MIP tool may never need; better decided when brightfield is actually in scope.
- **Context:** Linked to the `read()` non-2D assertion in `squidmip/reader.py`. tilefusion's `_to_grayscale_2d` is a reference implementation if reduction is chosen.
- **Depends on / blocked by:** IMA-189 reader.

## Multi-timepoint iteration / projection → low priority follow-up
- **What:** Iterate or project across timepoints (Nt>1) beyond the single `read(...,t=0)` hook.
- **Why:** No current dataset has Nt>1; 189 makes the API honest (`read(...,t=0)` + `metadata.n_t`) without building traversal.
- **Pros:** Ready for time-lapse acquisitions when they appear.
- **Cons:** Ahead of demand; the MIP tool projects over z, not t.
- **Context:** The `t=0` param + time-folder discovery already exist, so the extension is small.
- **Depends on / blocked by:** A real Nt>1 acquisition.

## Confirm IMA-193 navigator reads the pyramid + plate/well metadata → IMA-193
- **What:** Before/during IMA-193, verify its plate-view navigator actually reads multi-level pyramids and OME-NGFF plate/well group metadata — not just full-res array `0` the way ndviewer_light does.
- **Why:** IMA-184 writes a ≥2-level pyramid + spec plate/well metadata. ndviewer_light (today's only reader) ignores both — it directory-walks and reads only `field/0` + `omero`. So the pyramid is currently invisible; IMA-193 is the consumer that justifies it. If IMA-193 also reads only level 0, that extra output delivered nothing.
- **Pros:** Validates the load-bearing assumption behind IMA-184's canonical/multiscale scope before more work rides on it.
- **Cons:** Can't be closed until IMA-193 is designed; until then the pyramid is written on faith.
- **Context:** ndviewer_light discovers plates by directory walk and reads array `0` + `omero` only (`ndviewer_light/core.py:1149`, `:1070`). IMA-184's cross commit already proves the plate opens under strict `ome-zarr-py`, so the metadata is spec-valid regardless.
- **Depends on / blocked by:** IMA-193 design.

## Fix upstream squid2minerva/colors.py display_color nesting → external repo
- **What:** `squid2minerva/colors.py:load_yaml_colors` reads `channel["display_color"]`, but real `acquisition_channels.yaml` nests it under `channel.camera_settings.<cam>.display_color`. Its Minerva OME-TIFF exports only get right colors via the wavelength-fallback map — a custom yaml color is silently ignored.
- **Why:** Confirmed against a real dataset yaml. It's correct-by-luck today because the fallback palette matches the standard 4 channels; any non-default color drops silently.
- **Pros:** Fixes silently-wrong colors in a sibling tool's exports.
- **Cons:** Different repo, different owner; not on any SquidMIP critical path.
- **Context:** SquidMIP does **not** carry this bug — IMA-189's `squidmip/_channels.py` already resolves `display_color` correctly (top-level v1.0+ *and* nested `camera_settings`, mapped by name, raises on unresolved), and IMA-184 consumes `metadata.channels[].display_color` rather than re-parsing the yaml. This TODO is purely a flag for whoever owns `~/CEPHLA/projects/explorer/squid2minerva`.
- **Depends on / blocked by:** squid2minerva maintainer.

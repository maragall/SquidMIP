# TODOS

Deferred work captured during plan-eng-reviews. Each item records the reasoning
so a future session doesn't rediscover it from zero.

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

## Plane-op (nz=Nz) write + reopen path → IMA-223
- **What:** Let `write_plate` persist non-z-collapsed output and let the viewer reopen it: relax `_validate_image` (`_output.py:223` raises on `Z>1`), extend `_multiscales` beyond YX-only levels, and fix the reopen path (`_ComputedPlateWorker._read` reads `arr[0,:,0]`; `_open_computed` hardcodes `n_z: 1` and `np.uint16` at `_viewer.py:975/1683/1717`).
- **Why:** IMA-226 lands the `consumes` contract with a loud gate; the first plane-op (decon/bgsub/flatfield) cannot ship live+persisted until the writer and reopen layers accept `Nz>1`.
- **Pros:** Unblocks IMA-223/224/225 end-to-end; the gate in 226 flips to a real branch.
- **Cons:** Touches the IMA-184 writer contract, which ndviewer_light and ome-zarr-py conformance both depend on; needs its own review.
- **Context:** Found by the IMA-226 eng-review outside voice (2026-07-20): the viewer's five z-collapse sites were mapped, but the save path dies one layer down in `_output.py`. Also undefined: which z becomes the 88px plate thumbnail for a plane-op, and square-forced push downsampling (`_area_downsample(plane, 512, 512)`).
- **Depends on / blocked by:** IMA-226 (consumes contract); IMA-223 is the natural owner since decon is the first consumer.

## Fov-reducer engine path + geometry seam → IMA-222/IMA-210
- **What:** `select_fovs` "all FOVs" semantics for ragged counts (`projection.py:200` raises when `n_fovs > len(available)`), a group-by-region-then-reduce engine loop, a geometry-carrying operator seam (stage positions + overlap — a `project(planes)` list cannot express stitch), and a reworked memory contract (Nfov frames in flight per worker breaks the "~139 MB × workers" invariant and `test_engine.py:206`).
- **Why:** IMA-226's `{fov}` kind is gated with a named raise; stitch cannot be wired until these exist.
- **Pros:** The consumes taxonomy stops being frozen-on-faith; validated against the first real fov-reducer.
- **Cons:** The largest remaining chunk of IMA-210; memory feasibility must be measured, not asserted.
- **Context:** IMA-226 eng review (2026-07-20), outside voice #3/#6/#7. See also the `fov-axis-needs-geometry` learning: model FOVs as `list[{index,position,overlap}]`, not bare indices.
- **Depends on / blocked by:** IMA-226 (consumes contract); belongs to IMA-210/IMA-222.

## t-axis live streaming (t>0) → with first Nt>1 dataset
- **What:** `_on_well`/`_on_push` stream only `image[0,...]` / `register_array(0, ...)` — a time-lapse acquisition streams t=0 only.
- **Why:** Named limitation so "live for any operator" isn't read as "live for any axis"; no current dataset has Nt>1.
- **Pros:** Honest scope marker; pairs with the existing multi-timepoint TODO below.
- **Cons:** None now; ahead of demand.
- **Context:** IMA-226 eng review outside voice #8 (2026-07-20).
- **Depends on / blocked by:** A real Nt>1 acquisition (same trigger as the multi-timepoint projection TODO).

## Preview-mode slider eviction (save=False) → follow-up after IMA-226
- **What:** The IMA-226 slider-miss loader reads the written `plate.ome.zarr`; a preview run writes nothing, so evicted preview planes stay irrecoverable past ndviewer_light's 1024-plane LRU (`core.py:1666`).
- **Why:** Whole-plate previews on large plates would scrub back to blanks; small subset previews (the default, 4 wells) are unaffected.
- **Pros:** Closing it makes preview and saved runs behave identically at any scale.
- **Cons:** Needs either a spill-to-temp store or a bounded re-compute path; both are real designs.
- **Context:** IMA-226 eng review D9 + outside voice #5 (2026-07-20). Interim: limitation documented in the preview UI text.
- **Depends on / blocked by:** IMA-226's miss-loader hook landing first.

## Fix upstream squid2minerva/colors.py display_color nesting → external repo
- **What:** `squid2minerva/colors.py:load_yaml_colors` reads `channel["display_color"]`, but real `acquisition_channels.yaml` nests it under `channel.camera_settings.<cam>.display_color`. Its Minerva OME-TIFF exports only get right colors via the wavelength-fallback map — a custom yaml color is silently ignored.
- **Why:** Confirmed against a real dataset yaml. It's correct-by-luck today because the fallback palette matches the standard 4 channels; any non-default color drops silently.
- **Pros:** Fixes silently-wrong colors in a sibling tool's exports.
- **Cons:** Different repo, different owner; not on any SquidMIP critical path.
- **Context:** SquidMIP does **not** carry this bug — IMA-189's `squidmip/_channels.py` already resolves `display_color` correctly (top-level v1.0+ *and* nested `camera_settings`, mapped by name, raises on unresolved), and IMA-184 consumes `metadata.channels[].display_color` rather than re-parsing the yaml. This TODO is purely a flag for whoever owns `~/CEPHLA/projects/explorer/squid2minerva`.
- **Depends on / blocked by:** squid2minerva maintainer.

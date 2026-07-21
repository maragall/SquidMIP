# TODOS

Deferred work captured during plan-eng-reviews. Each item records the reasoning
so a future session doesn't rediscover it from zero.

## Gesture arbitration on PlateOverview (shift-select vs pan vs loupe) → IMA-221
- **What:** A single explicit gesture policy for `PlateOverview`'s mouse handlers, deciding between drag-to-pan (shipped), shift-drag marquee select (IMA-221), and press-and-hold loupe (IMA-208).
- **Why:** Three tickets independently add a gesture to the SAME handlers (`mousePressEvent:647`, `mouseMoveEvent:652`, `mouseReleaseEvent:670`). Whoever lands second inherits an undocumented conflict; the pan path already claims plain left-drag with a 3px threshold (:655).
- **Pros:** One place decides what a drag means; each later gesture ticket becomes additive instead of a rewrite of someone else's branch.
- **Cons:** Slightly more design up front than "add a modifier check"; needs agreement across three backlog tickets.
- **Context:** Today `mousePressEvent` unconditionally arms pan state on LeftButton. Shift-drag must branch BEFORE that. The loupe (IMA-208) wants press-and-hold, which competes with the same 3px pan threshold on the time axis rather than the modifier axis — so a modifier check alone won't settle it. `_sel` is currently a single `(ri,ci)` (:548) painted as one red box (:752-756); marquee select needs it to become a set with a multi-cell paint path.
- **Depends on / blocked by:** IMA-221 owns the selection gesture; coordinate with IMA-208 before either lands.

## Exploration-tab persistence across acquisitions → post-IMA-205
- **What:** Decide whether exploration tabs survive re-ingesting a different acquisition, and if so how their region sets are revalidated.
- **Why:** `ingest()` (:1493-1505) resets reader/`_fov_index`/`_overview` but never `_op_tabs`. The eng-review fix closes exploration tabs on ingest (the safe default), but the richer behavior — reopen the same selection on a re-ingest of the SAME acquisition — is a real workflow for anyone iterating on one plate.
- **Pros:** Users re-open the same plate constantly while tuning operators; losing their exploration set every time is friction.
- **Cons:** Requires acquisition identity in the tab key plus revalidation that every region still exists in the new `_fov_index`.
- **Context:** Surfaced by the /plan-eng-review outside voice (2026-07-20) and confirmed in code. The eng-review decision includes acquisition id in the content-addressed tab key specifically so this extension stays cheap.
- **Depends on / blocked by:** IMA-205 landing its ingest-teardown fix first.

## Partial `.hcs` cleanup after a stopped save run → fast-follow after IMA-205
- **What:** Clean up (or mark as partial) the `.hcs` output directory when a `save=True` operator run is stopped mid-flight.
- **Why:** IMA-205 makes stopping routine (closing an exploration tab stops its run). A stopped save leaves a partial plate that `resolve_plate_root` will later happily recognize as a real plate, so the user can re-open a half-written result as if it were complete.
- **Pros:** Prevents silently trusting a truncated plate; complements the resume/checkpoint TODO already filed against IMA-184.
- **Cons:** Needs a "complete output" definition — the same definition the resume/checkpoint item needs, so the two should be designed together.
- **Context:** Before IMA-205, stopping only happened on app close (`closeEvent:1970`) or re-ingest, both of which end the session anyway. Making close-tab stop a run turns a rare path into a common one. Surfaced by the /plan-eng-review outside voice (2026-07-20).
- **Depends on / blocked by:** Overlaps the existing "Resume / checkpoint for long plate runs" TODO — resolve as one design.

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

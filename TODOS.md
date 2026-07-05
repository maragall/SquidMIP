# TODOS

Deferred work captured across SquidMIP slots. Each item records the reasoning so a
future session doesn't rediscover it from zero.

## From IMA-189 plan-eng-review (2026-07-04)

### Scale-test fixture generator → IMA-188
- **What:** A generator that fans the 48 real hongquan FOVs across a 1536-well plate via **symlinks** (Squid layout), synthesizing 20 z (cycling the real 3) × 4 channels. On-disk ≈ source (~19 GB); logical read ≈ 1536×20×4×33 MB ≈ **4 TB** (served from OS cache — proves scale/parse/decode/memory, NOT raw disk bandwidth; that needs Nick's real storage).
- **Why:** It's the harness for the IMA-188 high-throughput scale test, not ingest. Building it in 189 bloats the keystone and risks CI breakage.
- **Pros:** Proves the reader + projection hold at plate scale with bounded memory, cheaply (symlinks, not 4 TB of real bytes).
- **Cons:** Breaks on Windows CI runners (no symlink checkout); ~120k inodes; slow to materialize.
- **Context:** The IMA-189 `SquidReader` reads one plane per call, so a symlink fan-out exercises the exact read path at scale. Keep 189's own tests on small real-shaped fixtures.
- **Depends on / blocked by:** IMA-189 reader must land first; belongs to **IMA-188**.

### Brightfield / RGB channel ingest → future ticket
- **What:** Support Squid brightfield channels saved as `(H,W,3)` RGB (and per-LED `_B/_G/_R`) planes, with a defined reduction-to-2D (or explicit color) policy.
- **Why:** IMA-189 `read()` deliberately **raises** on non-2D planes (decision 5). Without this note the raise reads like a bug.
- **Pros:** Broadens input coverage to brightfield acquisitions.
- **Cons:** Requires an RGB→2D policy the MIP tool may never need; better decided when brightfield is actually in scope.
- **Context:** Linked to the `read()` non-2D assertion in `squidmip/reader.py`. tilefusion's `_to_grayscale_2d` is a reference implementation if reduction is chosen.
- **Depends on / blocked by:** IMA-189 reader.

### Multi-timepoint iteration / projection → low priority follow-up
- **What:** Iterate or project across timepoints (Nt>1) beyond the single `read(...,t=0)` hook.
- **Why:** No current dataset has Nt>1; 189 makes the API honest (`read(...,t=0)` + `metadata.n_t`) without building traversal.
- **Pros:** Ready for time-lapse acquisitions when they appear.
- **Cons:** Ahead of demand; the MIP tool projects over z, not t.
- **Context:** The `t=0` param + time-folder discovery already exist, so the extension is small.
- **Depends on / blocked by:** A real Nt>1 acquisition.

## From IMA-184 eng review (2026-07-04, reconciled 2026-07-05)

### Confirm IMA-193 reads pyramid + plate metadata
- **What:** Before/during IMA-193, verify its plate-navigator actually reads multi-level pyramids and plate/well NGFF metadata (not just array `0` like ndviewer_light).
- **Why:** IMA-184's multiscale/canonical scope (pyramid + spec plate metadata) is justified largely by IMA-193. ndviewer_light ignores both, so if IMA-193 also reads only level 0, the pyramid work delivered no value.
- **Context:** ndviewer_light discovers plates by directory walking and reads only `field/0` + `omero` (see `ndviewer_light/core.py:1149`, `:1070`). The pyramid is invisible to it. Load-bearing assumption behind the IMA-184 output scope.
- **Depends on / blocked by:** IMA-193 design.

### Fix upstream squid2minerva/colors.py:53 nesting bug
- **What:** `load_yaml_colors` reads `channel["display_color"]`, but the real `acquisition_channels.yaml` nests it under `channel.camera_settings['1'].display_color`. (In SquidMIP, 189's reader already surfaces `channels[].display_color` correctly, so 184 does not re-parse — but the source repo still has the bug.)
- **Why:** squid2minerva's Minerva OME-TIFF exports only get correct colors via the wavelength-fallback map — any custom `display_color` in the yaml is silently ignored.
- **Context:** Confirmed against a real dataset yaml and `colors.py:45-55`. Correct-by-luck today because the fallback palette matches the standard 4-channel wavelengths.
- **Depends on / blocked by:** whoever owns `~/CEPHLA/projects/explorer/squid2minerva`.

### (Reconciled 2026-07-05) Write parallelism is IMA-188's, not IMA-184's
- Superseded by the build-order reconcile: IMA-188 is the parallel/streaming engine and may call 184's writer concurrently per `(well, fov)`. 184 does not add its own parallel layer; it must be **concurrency-safe** instead (guard shared `plate`/`well` group-metadata writes).

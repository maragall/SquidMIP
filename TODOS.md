# TODOS

Deferred work captured during plan-eng-reviews. Each item records the reasoning
so a future session doesn't rediscover it from zero.

## Dead `sim_*` fixtures — regenerate or delete → blocks IMA-183/188 baselines
- **What:** Every simulated dataset under `~/CEPHLA/Data/` is broken. `sim_1536wp`,
  `sim_384wp_4fov` and `sim_4wp_hongquan` are symlink fan-outs pointing at
  `~/Downloads/z_stack_2026-05-15_18-39-28.532906 hongquan/`, which no longer exists
  (500/500 sampled links dead, verified 2026-07-20). `sim_384wp_4fov` and
  `sim_4wp_hongquan` additionally have **no `coordinates.csv`**.
- **Why:** The integration suite is red, not skipped. `tests/test_performance.py:28`
  guards on `SIM_1536WP.is_dir()`, which passes because the directory still exists full
  of dead links, so the test proceeds and dies in tifffile:
  `FileNotFoundError: .../z_stack_2026-05-15…hongquan/0/B2_1_2_….tiff`. The IMA-183
  single-thread baseline is currently unrunnable, which means IMA-188's promised
  apples-to-apples comparison has no baseline to compare against.
- **Pros:** Restores the only scale fixture; makes the suite honest again.
- **Cons:** The fan-out is ~120k inodes and needs a real source acquisition to point at.
- **Context:** Two separate fixes, and the cheap one should land first. (1) Harden the
  guard to check that a *file* resolves, not just that the directory exists, so the
  suite skips instead of failing — that is a one-line change and is on IMA-233's
  critical path. (2) Regenerate the fan-out against a surviving acquisition, or delete
  the dead trees so nothing points at them. Note `sim_1536wp` is one FOV per well and
  cannot exercise a stitcher regardless.
- **Depends on / blocked by:** A surviving source acquisition for the regeneration half.

## Move `20x_scan` out of `~/Downloads` → prerequisite for IMA-233
- **What:** `~/Downloads/20x_scan_2025-09-05_17-57-50` (144 real TIFFs, 1.2 GB, region
  C5, 36 FOVs on a 0.705 mm step with per-FOV coordinate rows, 4 channels) is the only
  working overlapping-grid dataset on this machine. Move it under `~/CEPHLA/Data/`.
- **Why:** It is the sole viable dev fixture for any stitching or seam-quality work now
  that the `sim_*` trees are dead, and it currently lives in a directory people empty.
- **Pros:** Makes the one usable stitching fixture durable and discoverable.
- **Cons:** 1.2 GB; any hardcoded path referencing the Downloads location needs updating.
- **Context:** Discovered during the IMA-233 eng review while verifying that
  `sim_1536wp` could exercise a stitcher — it cannot (one FOV per well, no per-FOV
  coordinates). `20x_scan` is a genuine overlapping grid.
- **Depends on / blocked by:** Nothing.

## Record the lost `residual_benchmark` schema before the bytecode rots → stitcher repo
- **What:** `stitcher/benchmarks/residual_benchmark.py` and `compile_master.py` exist
  **only** as `.pyc`. Verified: `git log --all -- benchmarks/` empty,
  `git log --all -S"residual_benchmark"` empty, `git ls-files | grep -i bench` empty,
  and `.gitignore:5` shadows `__pycache__/`. Any `find -name __pycache__ -delete`, a
  Python minor-version bump, or a clean checkout destroys them permanently.
- **Why:** They encode a deliberate cross-machine merge contract — the module docstring
  calls itself *"the SINGLE source of the benchmark 'language'. Run the exact same file
  (same git sha) on every machine so the metrics and the CSV schema are identical and
  the rows merge into one master table"* — plus macOS/Linux peak-RSS unit normalization
  and dedup to the latest timestamp per `(host, dataset, git_sha)`.
- **Pros:** Zero-cost to write the schema down; preserves reasoning that is otherwise
  one `rm` from gone.
- **Cons:** Low value beyond the schema itself. The IMA-233 review deliberately declined
  to rebuild the master-table infrastructure, and the fact that the source was never
  committed suggests someone already decided it didn't belong in the repo. Do not spend
  an afternoon decompiling.
- **Context:** The 31 columns, recovered verbatim via `marshal` from the `.pyc`:
  `timestamp, host, platform, git_sha, dataset, path, n_tiles, tile_y, tile_x,
  pixel_size_um, n_channels, reg_channel, n_pairs_candidate, n_pairs_locked,
  n_pairs_rejected, ncc_mean, ncc_median, ncc_min, ncc_weak_frac, n_seams_fit,
  resid_before_median_px, resid_before_mean_px, resid_before_p90_px,
  resid_after_median_px, resid_after_mean_px, resid_after_p90_px, reduction_pct_median,
  t_register_s, t_optimize_s, t_distortion_s, peak_rss_mb`. Its metric definitions:
  `resid_before` = RMS over sub-blocks of phase-correlation shift magnitude at the
  optimized tile positions; `resid_after` = leave-one-out CV residual at the CV-chosen
  polynomial order. Both in full-resolution pixels. Writing this paragraph down *is* the
  deliverable.
- **Depends on / blocked by:** Nothing. Do it before the next cache clear.

## Verify a MATLAB / MCR license exists before promising a PetaKit5D adapter → IMA-211
- **What:** Confirm whether the team has a MATLAB or MATLAB Compiler Runtime license
  that permits running PetaKit5D headless in a benchmark.
- **Why:** IMA-211 commits to benchmarking PetaKit5D. If there is no license, that
  adapter is not hard — it is impossible, and the commitment should be withdrawn rather
  than discovered late.
- **Pros:** Removes an unpriced dependency from a ticket already contested on estimate.
- **Cons:** None; it is a question, not work.
- **Context:** Surfaced by the IMA-233 eng review outside voice. Nobody had checked. The
  IMA-233 plan starts with tilefusion + ASHLAR precisely so this is not on the critical
  path, but IMA-211's scope still assumes all four tools.
- **Depends on / blocked by:** Whoever owns tooling licenses.

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

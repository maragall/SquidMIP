# IMA-233 — Benchmark harness — eng review

_slug: benchmark-harness_ · reviewed 2026-07-20 via `/plan-eng-review`

> **Status after /plan-eng-review (2026-07-20): plan revised, estimate contested.**
> Linear says 3 points / Low / BLOCKED. This review concludes the ticket as written
> is not buildable and the estimate is off by roughly 5-10×. Three of the review's
> own recommendations were reversed by the outside voice and by direct verification.
> Read "Reversals" before implementing.

## Why

Nick wants live-stitching benchmarked against ASHLAR, MCmicro, BigStitcher and
PetaKit5D on speed, footprint and quality, to settle build-vs-adopt for the Squid
stitching path. IMA-233 is the harness half of IMA-211.

## The one-line problem

Speed and footprint are tool-agnostic (a stopwatch and a byte counter). **Quality is
not.** Each stitcher reports its own internal confidence number computed a different
way, so a table of self-reported numbers ranks nothing. The harness therefore needs
its own ruler — and the shape of that ruler is the entire technical content of this
ticket.

---

## Reversals (what this review got wrong, and why)

Three recommendations reached during the interactive review were overturned by the
outside voice and then confirmed by direct verification. They are recorded here
because the wrong versions are plausible and will be re-proposed otherwise.

### R1 — The quality metric cannot run on a fused mosaic

**Reversed:** "harness computes seam residual on each stitcher's fused output."

`_block_shifts(strip_i, strip_j, n_blocks, ncc_min)` at
`stitcher/src/tilefusion/distortion.py:95` takes **two aligned overlap strips** — two
*independent* views of the same physical region, one from each neighbouring source
tile — and phase-correlates them. `_fit_seam(tf, i, j, ...)` at `:140` obtains them by
reading tiles `i` and `j` back off the TileFusion object.

A fused mosaic has already blended that overlap into a single pixel set. The two
independent views no longer exist. You cannot phase-correlate an image against
itself.

```
WHAT THE METRIC NEEDS                    WHAT A FUSED MOSAIC HAS
  tile_i ──┐                               ┌──────────────────┐
           ├─ overlap region ─→ 2 strips   │  blended pixels  │ → 1 strip
  tile_j ──┘   (correlatable)              └──────────────────┘   (nothing to
                                                                   correlate)
```

**Correct formulation:** re-read the **input tiles**, place them at the **final tile
positions the stitcher solved for**, and correlate the overlaps at those positions.
That makes the real work *parsing each tool's position output*, not running a
correlation:

| Tool | Position output | Model | Parseable? |
|---|---|---|---|
| tilefusion | in-process, already solved | translation | yes (reference impl) |
| ASHLAR | positions output | translation | yes |
| MCmicro | ASHLAR under Nextflow | translation | via ASHLAR, inside container |
| BigStitcher | XML affines | **affine / non-rigid** | no single "shift" exists |
| PetaKit5D | `.mat` / json | translation | needs MATLAB or MCR |

BigStitcher's non-rigid model has no scalar shift to report, so the metric is not
merely harder there — it is **undefined**. Any plan that promises one number per tool
for all four tools is promising something that does not exist.

### R2 — Every simulated fixture on this machine is dead

**Reversed:** "build and validate against `sim_1536wp` today."

Verified 2026-07-20:

- `sim_1536wp` — every filename carries fov token `0` (`{region}_0_{z}_{channel}`), so
  it is **one FOV per well**. `coordinates.csv` is 1537 lines = header + 1536 wells,
  one XY per well, **no per-FOV rows**. There are no overlapping tiles: nothing to
  stitch, no seams to measure.
- **500 of 500 sampled symlinks are broken.** All point into
  `~/Downloads/z_stack_2026-05-15_18-39-28.532906 hongquan/`, which no longer exists.
- `sim_384wp_4fov` — has fov tokens `0..3`, but symlinks are broken **and it has no
  `coordinates.csv` at all**, so there is no geometry even if the pixels returned.
- `sim_4wp_hongquan` — one FOV, broken symlinks, no `coordinates.csv`.

**Consequence beyond this ticket — the test suite is red right now.** The fixture at
`tests/test_performance.py:28` guards only on `SIM_1536WP.is_dir()`, which *passes*
(the directory exists, full of dead links), so it never skips:

```
FAILED tests/test_performance.py::test_single_well_speed_baseline
FileNotFoundError: '/Users/julioamaragall/Downloads/z_stack_2026-05-15…hongquan/0/B2_1_2_….tiff'
```

The IMA-183 baseline is currently unrunnable. This is pre-existing and unrelated to
IMA-233, but it sits directly on this ticket's path.

**The only viable fixture is real and already on disk:**
`~/Downloads/20x_scan_2025-09-05_17-57-50` — 144 real (non-symlink) TIFFs, 1.2 GB,
single region `C5`, **36 FOVs on a 0.705 mm step with per-FOV coordinate rows**,
4 channels, single z. That is a genuine overlapping grid and the correct dev fixture.
It lives in `~/Downloads` and should be moved somewhere durable before anything
depends on it.

### R3 — The harness does not belong in `squidmip/`

**Reversed:** "create `squidmip/bench/` as the single home for measurement."

`pyproject.toml` states the boundary explicitly:

```
# No tilefusion dependency: IMA-189 ingest is self-contained (see docs/ima-189-eng-review.md).
# tilefusion's OME-zarr writer is vendored (squidmip/_zarr_store.py), not imported, at IMA-184:
# importing tilefusion runs its heavy __init__ (numba/GPU/basicpy).
```

Confirmed: the only `tilefusion` mention under `squidmip/` is the vendoring note in
`_zarr_store.py:1`. The R1 metric needs tilefusion. Putting it in `squidmip/` either
breaks that boundary or forces a second vendoring of the registration stack.

`squidmip` is also being packaged right now as an installable end-user tool for Nick
(the last five commits are the user guide, the PDF, and viewer fixes). Stitcher
benchmark machinery does not belong in that wheel.

**Correct placement:** the stitcher repo — which already has `benchmarks/` and the
prior art — or a standalone script. The one thing that *does* belong in SquidMIP is
fixing the red skip-guard.

---

## Scope

Build a benchmark runner that executes each stitcher as an isolated subprocess against
one dataset, measures it from outside, and emits one comparison row per (tool,
dataset, run).

```
                        ┌──────────────────────────────────────┐
  dataset  ──────────→  │  runner                              │
  (20x_scan C5,         │                                      │
   36 FOV grid)         │  for tool in tools:                  │
                        │    ├─ preflight: free space?         │
                        │    ├─ convert input → tool format    │  ← the real cost
                        │    ├─ spawn subprocess ──────────┐   │
                        │    │                             │   │
                        │    │   sampler thread ───────────┤   │
                        │    │     peak RSS (process tree) │   │
                        │    │     free space (low-water)  │   │
                        │    │       └─ breach → kill,     │   │
                        │    │          row=DISK_ABORT     │   │
                        │    │                             │   │
                        │    ├─ wall clock ────────────────┘   │
                        │    ├─ du(output_dir) → output_bytes  │
                        │    ├─ parse tool positions ──┐       │  ← undefined for
                        │    ├─ seam residual ─────────┘       │    BigStitcher
                        │    │    (input tiles @ solved pos)   │
                        │    ├─ append CSV row                 │
                        │    └─ delete output, next tool       │
                        └──────────────────────────────────────┘
                                        │
                                        ▼
                                  benchmark.csv
```

### Locked decisions

**D3 — measurement boundary (stands).** Every stitcher runs as a subprocess behind one
adapter interface, including Python-native ASHLAR, so one definition applies to all.
Measure externally: wall clock, peak RSS sampled across the process tree, `du` of the
output directory. Rationale: `tests/test_performance.py:57` uses `tracemalloc`, which
sees only allocations in *this* interpreter; against a JVM or MATLAB subprocess it
reports ≈0 and makes the heaviest tool look leanest.

Caveats the review initially missed, now in scope:
- **MCmicro runs in Docker cgroups**, not as descendants of our subprocess. Process-tree
  RSS does not see it; container stats are a separate mechanism.
- `du` of the output dir misses container volumes and scratch.
- **Wall clock will be dominated by JVM / Nextflow / MATLAB startup and image pull**,
  not by stitching. Report cold and warm separately or the number means nothing.
- `output_bytes` compares compressors and dtypes (pyramidal OME-TIFF vs zarr vs
  float32) unless compression, dtype and thread count are **pinned per tool**.
- Sampling interval must be stated; short spikes are missed by construction.

**D5 — storage guard (stands).** The harness owns disk safety around each subprocess:
pre-flight free-space check, mid-run polling on the same thread that samples peak RSS,
kill + `DISK_ABORT` row at a low-water mark, and delete each tool's output after
metrics are extracted before launching the next. Peak disk is then one mosaic, not
four.

`_check_disk()` at `squidmip/_viewer.py:1797` cannot serve here: it guards SquidMIP's
own in-process writes, is bypassed for subset runs (`:1756`), and fails open on
`OSError`. IMA-230's planned guard lives in the `_output.py` write path and
structurally cannot see subprocess writes. Independently confirmed: `grep -rniE
"disk_usage|statvfs|free_space"` returns **zero** hits across the stitcher and
squid-tools repos, and `stitcher/src/tilefusion/core.py:1161` loops region-after-region
with no free-space check between iterations.

**D4 — schema (reduced from the review's recommendation).** A flat CSV, one row per
run. Keep provenance (`timestamp, host, platform, dataset, path, n_tiles, tile_y,
tile_x, pixel_size_um, n_channels`), the tool-agnostic metrics (`t_wall_s`,
`t_wall_cold_s`, `peak_rss_mb`, `output_bytes`, `resid_median_px`, `resid_p90_px`),
and `status` (`OK | DISK_ABORT | TIMEOUT | CRASH | MISSING_TOOL | QUALITY_NA`).

Two corrections to the review's original D4:
- `git_sha` is meaningless for four *external* tools. Capture **per-tool version**
  instead: container digest, Fiji update-site version, PetaKit5D commit, ASHLAR
  `__version__`.
- Do **not** rebuild the master-table / cross-machine dedup infrastructure. That is
  regression-benchmark machinery for a one-shot build-vs-adopt question.

**D6 — sequencing (revised).** Build now, but against `20x_scan_2025-09-05_17-57-50`,
not `sim_1536wp` (R2). The tissue acquisition is needed only for the final run.
IMA-210 is not a blocker: subprocess adapters do not use the in-process
`consumes`-axis registry.

**D7 — placement (reversed, R3).** Harness lives in the stitcher repo or standalone,
not in the `squidmip` wheel.

### Start narrow

Per the outside voice, and because BigStitcher's quality metric is undefined and
PetaKit5D may be license-blocked:

1. **Ship two adapters first — tilefusion (incumbent) and ASHLAR.** Both are Python,
   both emit translation-only positions, both are parseable. Run on the `20x_scan` C5
   grid and produce the table.
2. Add MCmicro, BigStitcher, PetaKit5D **only if that first table doesn't answer the
   build-vs-adopt question**, and only after confirming a MATLAB/MCR license exists.

## Files

- **stitcher repo (or standalone):** new `bench/` — `adapters/`, `metrics.py`,
  `runner.py`, `report.py`.
- **SquidMIP:** `tests/test_performance.py` — fix the skip-guard only. No new module.

## Acceptance / oracle

1. `pytest tests/test_performance.py -m integration` either passes or **skips
   cleanly** — it must not fail on dead symlinks.
2. Running the harness on the `20x_scan` C5 grid produces a CSV with one populated row
   each for tilefusion and ASHLAR, carrying wall clock, peak RSS, output bytes and
   seam residual, with per-tool versions recorded.
3. A simulated low-disk condition yields a `DISK_ABORT` row and a killed subprocess
   rather than a full disk.
4. Every unsupported metric is explicitly `QUALITY_NA`, never a blank cell or a zero.

## Estimate

**Contested: 3 points is off by roughly 5-10×.** Four heterogeneous runtimes (Docker +
Nextflow, Fiji/Java, MATLAB) each need install, Squid→tool input conversion, output
reader and position parsing. The subprocess/RSS/`du` plumbing is the easy ~10%.
The narrow two-adapter version above is plausibly 5 points; all four is 13+ and
carries an unresolved MATLAB licensing risk.

## Depends on

- **Not** IMA-210 (subprocess adapters bypass the registry).
- IMA-211 for the final tissue run only.
- **New hard blocker:** a working overlapping-grid fixture. `20x_scan` covers dev;
  the `sim_*` fixtures need regenerating or deleting.
- **Unverified blocker:** MATLAB / MCR license for PetaKit5D.

## Open conflict for merge

IMA-211 requires the stitchers be "wrapped as plate operators using the operator
registry." This plan wraps them as subprocess adapters instead. Either IMA-233 adopts
the registry or IMA-211's scope shrinks — the two tickets currently disagree. Related:
the IMA-225 lock records IMA-210 as "deliberately not built" while the IMA-226 lock
records the "consumes registry half" as built. Reconcile before either lands.

## NOT in scope

- **Master-table / cross-machine CSV merge infrastructure** — regression-benchmark
  machinery for a one-shot question; the harness likely gets deleted after it answers.
- **Recovering `residual_benchmark.py` source from bytecode** — the 31-column schema
  and design intent are transcribed into TODOS.md, which captures the value. Note the
  file was never committed and `.gitignore:5` shadows it: someone already decided it
  doesn't belong in the repo.
- **BigStitcher / PetaKit5D / MCmicro adapters** — deferred to a second pass; see
  "Start narrow".
- **A quality metric for BigStitcher** — undefined under a non-rigid model, not merely
  unimplemented.
- **Regenerating the `sim_*` fixtures** — separate ticket; this plan only fixes the
  skip-guard so the suite stops lying.
- **IMA-230's `_output.py` storage guard** — orthogonal; it cannot see subprocess
  writes and is not a prerequisite.

## What already exists

| Need | Exists? | Where |
|---|---|---|
| Speed + RAM measurement | partial, **currently red** | `tests/test_performance.py:33` `benchmark_single_well` — tracemalloc-based, cannot see subprocesses |
| Seam-residual math | yes, but pre-fusion only | `stitcher/src/tilefusion/distortion.py:95` `_block_shifts`, `:140` `_fit_seam` |
| Disk guard | yes, wrong scope | `squidmip/_viewer.py:1797` `_check_disk` — in-process only, bypassed on subset (`:1756`), fails open |
| Staged profiler + CSV/plots | yes | `stitcher/profiling/` — `harness.py:19`, `stages.py:19`, `record.py:11` |
| Prior benchmark runner + schema | **bytecode only** | `stitcher/benchmarks/__pycache__/*.pyc`; sources never committed |
| Overlapping-grid fixture | yes, undurable | `~/Downloads/20x_scan_2025-09-05_17-57-50` |
| Disk guard in stitcher/squid-tools | **no** | zero `disk_usage`/`statvfs`/`free_space` hits |
| Per-region output cleanup | **no** | `stitcher/src/tilefusion/core.py:1161` accumulates |
| Operator registry (in-process) | yes, unused here | `squidmip/_engine.py:79` `add_projector`, `:129` `project_plate` |

## Failure modes

| Codepath | Realistic failure | Test? | Handled? | User sees |
|---|---|---|---|---|
| Fixture guard | dir exists, links dead | **no** | **no** | **red suite, misread as code bug — CRITICAL GAP** |
| Subprocess spawn | tool not installed | no | planned | `MISSING_TOOL` row |
| Subprocess run | hangs forever | no | planned | `TIMEOUT` row |
| Disk watchdog | fills between polls | no | planned | `DISK_ABORT`; a fast writer can still win the race |
| RSS sampler | Docker cgroup invisible | no | **no** | **silently ~0 RSS for MCmicro — CRITICAL GAP** |
| Position parser | affine model, no scalar shift | no | planned | `QUALITY_NA` |
| `du` accounting | container volume not counted | no | **no** | **understated `output_bytes` — CRITICAL GAP** |

Three critical gaps: each is currently silent and wrong rather than loud and absent.

## Test plan

```
CODE PATHS                                  USER FLOWS
[+] bench/runner.py                         [+] Run benchmark on 20x_scan
  ├── preflight_disk()                        ├── [GAP] two tools → 2 CSV rows
  │   ├── [GAP] enough space                  ├── [GAP] tool missing → MISSING_TOOL
  │   └── [GAP] insufficient → abort          └── [GAP] disk fills → DISK_ABORT + kill
  ├── run_adapter()
  │   ├── [GAP] exit 0                      [+] Regression (CRITICAL)
  │   ├── [GAP] non-zero → CRASH              └── [GAP] dead-symlink fixture SKIPS,
  │   └── [GAP] timeout → TIMEOUT                       does not FAIL
  ├── sample_tree()  [GAP] RSS, [GAP] cgroup N/A
[+] bench/metrics.py
  ├── seam_residual()
  │   ├── [GAP] translation → px
  │   ├── [GAP] affine → QUALITY_NA
  │   └── [GAP] no overlap → QUALITY_NA
[+] bench/adapters/{tilefusion,ashlar}.py
  └── parse_positions()  [GAP] valid, [GAP] malformed

COVERAGE: 0/17 (0%) — greenfield
CRITICAL: fixture skip-guard regression test (IRON RULE, no opt-out)
```

## Parallelization

| Lane | Work | Depends on |
|---|---|---|
| A | fix `test_performance.py` skip-guard (SquidMIP) | — |
| B | runner + sampler + disk watchdog + CSV | — |
| C | `metrics.py` seam residual from tiles + positions | — |
| D | tilefusion + ASHLAR adapters | C (position contract) |

Lanes A, B, C launch in parallel — A is a different repo, B and C share no modules.
D follows C. Lane A is the one to do first regardless: it is small and the suite is
red until it lands.

## Implementation Tasks
Synthesized from this review's findings. Each task derives from a specific finding
above. Run with Claude Code or Codex; checkbox as you ship.

- [ ] **T1 (P1, human: ~15min / CC: ~2min)** — tests — Fix the `sim_1536wp` skip-guard so the integration suite skips instead of failing
  - Surfaced by: R2 / Failure modes — guard at `tests/test_performance.py:28` checks `is_dir()` only; 500/500 symlinks dead; suite is RED today
  - Files: `tests/test_performance.py`
  - Verify: `pytest tests/test_performance.py -m integration` reports skipped, not failed
- [ ] **T2 (P1, human: ~2d / CC: ~2h)** — bench — Subprocess runner: spawn, process-tree RSS sampler, disk watchdog, `DISK_ABORT` kill, per-tool cleanup
  - Surfaced by: D3 + D5 — `tracemalloc` cannot see JVM/MATLAB/Docker; zero `disk_usage` hits in stitcher or squid-tools
  - Files: `bench/runner.py`
  - Verify: simulated low-disk yields a `DISK_ABORT` row and a killed subprocess
- [ ] **T3 (P1, human: ~3d / CC: ~3h)** — bench — Seam residual from input tiles at each tool's solved positions, not from the fused mosaic
  - Surfaced by: R1 — `_block_shifts` (`stitcher/src/tilefusion/distortion.py:95`) needs two independent overlap strips; fusion destroys them
  - Files: `bench/metrics.py`
  - Verify: residual on the `20x_scan` C5 grid matches tilefusion's own seam numbers within tolerance
- [ ] **T4 (P1, human: ~2d / CC: ~2h)** — bench — tilefusion + ASHLAR adapters including position parsing; other three tools deferred
  - Surfaced by: Start narrow — both are Python with translation-only positions; BigStitcher quality undefined, PetaKit5D license unverified
  - Files: `bench/adapters/tilefusion.py`, `bench/adapters/ashlar.py`
  - Verify: two populated CSV rows from one run on `20x_scan`
- [ ] **T5 (P2, human: ~4h / CC: ~30min)** — bench — CSV schema: provenance + per-tool version + status enum
  - Surfaced by: D4 reduced — `git_sha` is meaningless for external tools; no master-table infra for a one-shot question
  - Files: `bench/report.py`
  - Verify: every row carries a tool version; no blank cells (`QUALITY_NA` instead)
- [ ] **T6 (P2, human: ~4h / CC: ~30min)** — bench — Pin threads, compression and dtype per tool; report cold and warm wall clock separately
  - Surfaced by: Outside voice — `output_bytes` otherwise compares compressors not stitchers; wall clock dominated by JVM/Docker startup
  - Files: `bench/runner.py`
  - Verify: two timing columns present; compression settings recorded per row
- [ ] **T7 (P3, human: ~1h / CC: ~10min)** — ops — Move `20x_scan_2025-09-05_17-57-50` out of `~/Downloads` into `~/CEPHLA/Data`
  - Surfaced by: R2 — the only working overlapping-grid fixture on this machine, living in a directory people empty
  - Files: —
  - Verify: harness default fixture path resolves under `~/CEPHLA/Data`

## Implementation status (built 2026-07-21)

Shipped and verified on the real `20x_scan` acquisition. `bench/` sits at the repo
root, NOT inside `squidmip/` — `pyproject.toml` packages only `["squidmip"]`, so none
of this reaches Nick's wheel and the runtime MIP pipeline gains no dependency.

One improvement over the plan: the seam metric is implemented on numpy's FFT alone
(`bench/metrics.py`), so it needs neither scipy/skimage (not SquidMIP deps) nor an
import of `tilefusion`. That **removes** the R3 placement tension rather than working
around it — the metric is self-contained.

    bench/
      dataset.py    Squid acquisition -> tiles + stage positions (both CSV layouts)
      metrics.py    phase correlation, overlap strips, block gating, seam residual
      sampler.py    process-tree peak RSS + disk low-water watchdog + kill_tree
      runner.py     preflight -> spawn -> measure -> score -> reclaim -> row
      report.py     CSV schema (no git_sha; per-tool version) + markdown table
      adapters/     stage-baseline (control), tilefusion, ashlar
      drivers/      subprocess drivers emitting the shared positions.json contract

### Measured, on real data (region C5, 36 FOVs, 2084x2084, 0.3729 um/px)

| Tool | Status | Wall (s) | Peak RSS (MB) | Seam resid median (px) | p90 | Seams |
|---|---|---|---|---|---|---|
| stage-baseline | OK | 0.07 | 4.4 | 39.05 | 40.40 | 47 |
| tilefusion 0.1.0 | OK | 4.43 | 354.3 | 3.15 | 8.24 | 56 |
| ashlar | MISSING_TOOL | n/a | n/a | n/a | n/a | 0 |

The stage-baseline row is the no-registration control, added during implementation
because it turned out to be the number that makes the table interpretable: Squid's
stage alone leaves ~39 px (~14.6 um) of median seam misalignment, and tilefusion
removes 92% of it for 4.4 s and 354 MB. A sweep of the pixel-size conversion confirmed
the 39 px is genuine stage error, not a calibration artifact — the residual minimises
exactly at the nominal 0.37286 um/px.

The tilefusion driver registers only and never fuses (2.5 KB written, not a mosaic),
following the position contract at `tilefusion/core.py:1102`:
`solved_um = tile_positions + global_offsets * pixel_size`. Tile index is matched to
Squid FOV by stage geometry rather than assumed from ordering, so a reader reordering
surfaces as an error instead of silently mislabelled rows.

### Tests: 244 passing, 0 failing

Includes the two that make the metric trustworthy rather than merely green:
`test_injected_misalignment_is_recovered` (lie about a tile position by 5 px, assert
the metric reports 5 px) and `test_a_worse_stitcher_scores_worse` (the table can
actually rank). Every status branch is covered end-to-end against real subprocesses:
OK, MISSING_TOOL, CRASH, TIMEOUT, DISK_ABORT, QUALITY_NA.

**The pre-existing red suite is fixed.** Ten integration tests were failing with
`FileNotFoundError` because the `sim_*` guards checked `is_dir()` only, which passes on
a directory full of dangling symlinks. One shared helper — `require_usable_dataset` in
`tests/conftest.py` — now resolves a real plane before declaring a fixture usable, so
those tests skip with an actionable message. 10 failures -> 0.

### Still deferred

MCmicro, BigStitcher and PetaKit5D, per "Start narrow". ASHLAR's adapter and driver are
written but it is not installed here, so it reports `MISSING_TOOL` — install it and the
row populates with no code change.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES_OPEN | 6 issues, 3 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

**OUTSIDE VOICE:** Ran as a Claude subagent (Codex CLI not installed). It overturned
three of the five reviewer recommendations — D2 (quality on a fused mosaic is not
computable), D6 (`sim_1536wp` is single-FOV and its symlinks are dead), and D7
(`squidmip/` forbids a tilefusion dep). All three were independently verified against
source and disk before being folded in, and all three held.

**CROSS-MODEL:** No unresolved tension. Where the two reviewers disagreed, direct
verification settled it in the outside voice's favour every time, so the reversals are
recorded as fact in "Reversals" rather than left as open disagreements.

**VERDICT:** ENG REVIEW COMPLETE — plan revised, not cleared to implement as originally
scoped. The ticket is buildable only in the narrowed two-adapter form; the 3-point
estimate is contested and a pre-existing red test blocks the first task. IMA-211's
registry requirement conflicts with this plan's subprocess adapters and must be
reconciled before either lands.

NO UNRESOLVED DECISIONS

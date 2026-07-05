# TODOS

## Strategic: validate the tool isn't redundant (before build) — flag for /plan-ceo-review
- **What:** Confirm the SquidMIP epic isn't redundant and will be adopted.
- **Why:** The epic is largely orchestration glue over existing parts (tilefusion reader, squid2minerva writer, ndviewer_light, one `np.max`). The genuinely new code is a CLI + plate loop. A correct tool nobody adopts is wasted build.
- **Pros:** ~30-minute product check that could save 8 tickets of work.
- **Cons:** Adds a product gate that could pause the build; needs a short domain conversation.
- **Context:** Two open questions from the eng-review outside voice — (a) does the Squid acquisition/control software already do on-the-fly max projection, making an offline tool redundant? (b) Is "one command" a big enough win over an existing scripted FIJI macro to earn adoption? This is a product/strategy call (CEO review), not an engineering one.
- **Depends on / blocked by:** None; do before or in parallel with the packaging keystone (T1). Route through `/plan-ceo-review`.

## Make tilefusion's numba import lazy (upstream, tilefusion repo)
- **What:** Move numba's import (and the `NUMBA_THREADING_LAYER` env pin) behind the fusion codepath so consumers that only use `tilefusion.io` (readers) or the zarr writer don't trigger a numba import.
- **Why:** Importing any part of tilefusion currently pulls in numba and mutates process-wide threading config, even for read-only/zarr consumers like SquidMIP that never JIT anything. Slows CLI startup and adds a heavy transitive dep for no benefit.
- **Pros:** Faster startup and lighter footprint for every lightweight tilefusion consumer, not just SquidMIP.
- **Cons:** Cross-repo change into an actively-developed sibling repo; must not disturb the deterministic-prange threading-layer setup in `tilefusion/__init__.py`.
- **Context:** Surfaced in the IMA-176 eng review (Issue 3). `tilefusion/__init__.py` does `from .core import TileFusion` → `core` imports `fusion` → `fusion.py:11 from numba import njit, prange`. SquidMIP only needs `tilefusion.io.open_reader` + `tilefusion.io.zarr` writer + `ome_tiff_export`.
- **Depends on / blocked by:** None. Not on SquidMIP's critical path — SquidMIP pins a tilefusion SHA regardless.

## Whole-plate thumbnail montage PNG (fast-follow)
- **What:** Emit a single static plate-layout thumbnail grid PNG (one downsampled well per cell, A1 top-left).
- **Why:** Shareable at-a-glance "did the whole plate come out?" artifact for Slack/lab notebook without launching a viewer. The spec's own "easy to view" line names montage.
- **Pros:** Near-free — reuses the OME-zarr pyramid's downsampled level; the eng review already writes the pyramid.
- **Cons:** Adds an output artifact + plate-layout logic; overlaps with what ndviewer shows interactively.
- **Context:** Deferred in the CEO review (D5) — ndviewer covers interactive viewing for v1; montage is a cheap fast-follow once the pyramid exists. Priority P3.
- **Depends on / blocked by:** IMA-184 (OME-zarr pyramid) must exist first.

## Resumable / idempotent batch (skip already-written wells)
- **What:** On a re-run, skip wells whose output already exists so an interrupted 1536wp run resumes instead of restarting.
- **Why:** Long runs get interrupted (crash, Ctrl-C, machine sleep); without resume the user re-processes everything.
- **Pros:** Big time saver on large plates; pairs naturally with the per-well isolation already accepted.
- **Cons:** Needs a completeness check per well (does valid output exist?) and a `--force`/`--overwrite` escape hatch; more state to reason about.
- **Context:** Surfaced in the CEO review expansion scan; deferred as lower-priority (validate-first — don't build until real use demands it). Priority P3.
- **Depends on / blocked by:** Core write path (IMA-184) + per-well isolation.

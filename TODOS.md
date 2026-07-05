# TODOS

## Deferred

- [ ] **Resume / checkpoint for long plate runs** (fast-follow after IMA-184)
  - **What:** Skip wells whose complete output already exists; clean partial output files on rerun.
  - **Why:** A full 1536wp run takes hours; a crash at well 1400 currently restarts from 0, and partial TIFF/zarr chunks can silently corrupt the plate.
  - **Pros:** Turns a full-run loss into an incremental retry; mitigates the threads segfault residual (a rerun skips finished wells).
  - **Cons:** Needs a per-well "complete output" definition + atomic write/rename or cleanup logic.
  - **Context:** Engine uses ThreadPoolExecutor; failure policy = per-well manifest + non-zero exit; a C-level segfault in decode can still abort the whole process. Surfaced by /plan-eng-review outside-voice #7 (2026-07-04).
  - **Depends on:** IMA-184 output layout (what a "complete well output" looks like).

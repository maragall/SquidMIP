# TODOS

## Deferred

### Subprocess-per-acquisition isolation (hard isolation against OOM/segfault)
- **What:** Run each acquisition in its own subprocess during a batch, so an
  uncatchable failure (OS OOM-kill, C-extension segfault, SIGKILL) in one
  acquisition is recorded by the parent and the batch continues, instead of
  taking the whole process down.
- **Why:** IMA-192's failure isolation uses per-acquisition `try/except` +
  result records + summary. That catches Python exceptions (unreadable format,
  bad file) but NOT process-killing events. A single OOM on a large plate mid
  batch ends the run with no summary — silently defeating the isolation the
  ticket asks for.
- **Pros:** True hard isolation; overnight batches survive one violent failure.
- **Cons:** Adds a process boundary and result serialization back to the parent;
  more code and tests for a failure that the memory-bounded design (one well's
  z-stack resident per worker) makes rare. Disproportionate to a 2-pt Low ticket.
- **Context:** Raised by the outside voice during the IMA-192 plan-eng-review
  (2026-07-04). The core already uses a worker pool for wells within one
  acquisition; this TODO is about isolating at the acquisition granularity in
  batch mode. Revisit if OOMs actually occur in practice, or when plate sizes
  grow. Start point: wrap the per-acquisition call in `run_batch()` with a
  `concurrent.futures.ProcessPoolExecutor(max_workers=1)` submit-per-acquisition,
  or `multiprocessing.Process` + a result queue, and translate a non-zero child
  exit / crash into a failure record.
- **Depends on / blocked by:** IMA-186 (single-acquisition CLI) and IMA-192
  (batch layer) must exist first.

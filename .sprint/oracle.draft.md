# Oracle (DRAFT) — IMA-234

Review, then rename to `.sprint/oracle.md` to launch the AFK build (the armed ralph watcher fires on `oracle.md`).

## Correct means (measured):

The **harness** is the automated deliverable; the measured delta is a human step
afterwards (plan-eng-review Decision 1 — `odon` is not installed and IMA-212 is
still Backlog, so an unattended build cannot produce the number).

1. `pytest tests/test_odon_bench.py` passes with **no `odon` binary present**.
2. The critical concurrency test holds: two simultaneous requests under
   `--latency-ms 50` complete in well under 100 ms (proves `ThreadingHTTPServer`
   overlaps rather than serializes, so injected latency is not stacking).
3. `python scripts/odon_bench.py --zarr <fixture>/plate.ome.zarr --latency-ms 25`
   serves a real `write_plate` fixture, logs chunk requests, and prints a report
   containing: `ttfb_ms`, `first_view_ms`, `req_per_s`, `peak_concurrency`,
   `bytes_total`, per-status counts, and per-pyramid-level counts.
4. Zero-request and bad-path runs fail cleanly (no `ZeroDivisionError`, non-zero
   exit, readable message).

Also: `python -c "import squidmip"` succeeds and the existing test suite passes
(`pytest -m "not integration"`) — no regression, and `squidmip/` gains no new
module (the harness lives in `scripts/`, Decision 6).

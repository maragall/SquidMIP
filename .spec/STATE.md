# STATE — IMA-226

- **Ticket:** IMA-226
- **Branch:** juliomaragall/ima-226-live-any-operator
- **Spec:** .spec/open/ima-226.md
- **Phase:** PLAN LOCKED (eng review complete 2026-07-20; ready for Build)
- **Mode:** attended → user delegated finalize+push (2026-07-20)

## Now
Plan locked and pushed. Next session: Build T1 (compositor refactor commit).

## Next
T1 refactor `_compose_tile` → T2 engine `consumes` → T3 surface `reference` → T4 kind gate →
T5 tests (stub `register_array` + byte-identity guard) → T6 push-failure counter →
T7 ndviewer_light miss hook (parallel lane) → T8 operator-aware `_check_disk`.
Full task list with files/verify steps: spec "Implementation Tasks".

## Decisions
- Generalize live path on IMA-210's `consumes` — RESCOPED after outside voice: contract +
  loud gate this ticket; plane-op/fov wiring deferred to IMA-223/IMA-222 (why: `_output.py:223`
  hard-rejects nz>1 — full wiring unshippable without the writer layer; alternative rejected:
  ship it all now = rewrite IMA-184's writer inside a viewer ticket).
- Pull IMA-210's registry half (consumes frozenset) into this ticket; 210 keeps Stitch +
  group-by engine (alternative rejected: block on 210 = idle worktree; merge tickets = 8-pt diff).
- Acceptance proven by surfacing `reference` (already in `_PROJECTORS`, never in `_OPERATIONS`).
- D9 slider eviction: disk-backed miss loader, explicitly cross-repo (ndviewer_light has no
  miss hook, core.py:2569); preview-mode limitation documented, TODO'd.
- Refactor-first commit for the triplicated compositor (Beck: structural before behavioral).
- Cross-model tension resolved by reviewer under user delegation ("you don't need my input").

## Blockers
_(none)_

## Learnings
- pushReady→slider path was a silent no-op in ALL viewer tests (_StubDetail lacks
  register_array; _viewer.py:1838 hasattr guard) — logged to /learn.
- ndviewer_light register_array LRU (1024) evicts ~83% of a 1536wp×4ch MIP run today.
- _output.py:223 is the real z-collapse enforcement layer, not the viewer.

## Iterations
- 0 — plan-eng-review: 9 issues (3 arch, 3 quality, 1 test + regression guard, 2 perf),
  outside voice 8 findings (3 verified load-bearing, absorbed), plan locked, 0 unresolved.

# IMA-209 eng review — drag-out floating tab windows (ImageJ-style)

/plan-eng-review, 2026-07-20. Decisions D2–D6 locked interactively; outside-voice
findings absorbed under delegated authority (user: "you don't need my input anymore").

## Decisions

- **D2 — Generic detach, not exploration-specific.** IMA-205's exploration pane does
  not exist yet (zero code hits). Detach is a property of the tab container
  (`_left_tabs`), so ANY closable tab (index > 0) detaches; 205's tab inherits the
  behavior the day it lands, provided it opens through `_open_op_tab` (see TODOS.md).
  _Rejected: block on 205; absorb 205's scope here._
- **D3 — Custom `QTabBar` drag + `_detach_tab(index)` seam; no QDockWidget.**
  QDockWidget is the Qt built-in for float-out [Layer 1], but it requires QMainWindow
  dock areas — adopting it tears out the QSplitter three-pane layout and the
  scoped-Fusion dark-theme fix (`_dark_palette` docstring) for a primitive.
  All logic lives in `_detach_tab` (offscreen-testable); `_DetachTabBar` is a thin
  gesture shim. _Rejected: dock migration; button/menu-only detach (fails the
  "drag" oracle)._
- **D4 — `_floating: key -> _FloatWindow` second registry.** A key lives in exactly
  one of `_op_tabs` / `_floating`. `_open_op_tab` checks `_floating` first and
  focuses the float — otherwise the opener button either silently no-ops (stale
  `_op_tabs` entry: `setCurrentWidget` on a non-child) or rebuilds a duplicate
  (second live CLI shell). Regression-tested both ways (builder must not re-fire).
- **D5 — One teardown path: `_dispose_tab_widget`.** Registry pop, `_layers_tab`/
  `_layers_box` stale-ref clear, `shutdown()`, `deleteLater()` — called from tab
  close, float close, and app exit. `closeEvent` additionally drains `_floating`;
  before this, quitting with a floated CLI leaked the shell and kept the app alive
  (the review's one critical gap).
- **D6 — Re-dock button, not drag-back.** Re-dock reuses the detach machinery in
  reverse (same live widget — a CLI keeps its shell/history). ImageJ drag-back is
  the expensive, untestable half of custom tab dragging; deferred to TODOS.md until
  a user actually misses it.

## Outside voice (Claude subagent, fresh context — codex CLI not installed)

10 findings; absorbed into the build:

1. **Drag re-entrancy (accepted, fixed):** `removeTab` from inside the bar's own
   `mouseMoveEvent` mutates the bar mid-drag — detach is deferred via
   `QTimer.singleShot(0, …)`.
2. **`ingest()` with a float open (accepted as test + TODO):** floats deliberately
   follow docked-tab semantics across a plate swap (they persist; op-tab staleness
   on re-ingest pre-exists for docked tabs and is tracked in TODOS.md).
3. **Re-dock title (accepted):** `_FloatWindow` stores the tab title verbatim
   (`_tab_title`); never parsed back out of the window title.
4. **Positive floating-Layers test (accepted):** `_refresh_layers_tab` proven to
   keep writing into the detached widget; refs cleared on dispose only.
5. **Re-arguing D2/D3 (strategy, gesture-vs-button):** settled interactive
   decisions — stand as decided.

## Found during implementation (worth knowing)

**Python-owned `QStyle` + `deleteLater` segfault.** A per-widget Fusion style held
only as a Python attribute dies with the wrapper at GC; a `deleteLater`'d widget can
outlive it, and `~QWidget` then unpolishes a dangling style — segfault at the next
event drain, in a *different* test. `_FloatWindow` therefore uses palette +
stylesheet only (it has no tab strip; the Fusion hack exists solely for
`_left_tabs`' strip rendering). Caught by running the suite in order, not in
isolation.

## Verification

- `tests/test_viewer.py`: 8 new offscreen tests (detach/registry/same-object,
  home-tab refusal, focus-not-duplicate regression, float-close dispose,
  re-dock round trip, app-exit drain, floating-Layers refresh + dispose,
  float survives re-ingest). 19/19 pass.
- Full suite: 129 passed / 12 failed — the 12 are `FileNotFoundError` on a
  machine-local dataset (`~/Downloads/z_stack_… hongquan`) and fail identically
  on the pre-change baseline (verified by stash).
- Drag gesture itself (`_DetachTabBar.mouseMoveEvent`) is manual QA — offscreen Qt
  cannot synthesize the drag; everything downstream of the gesture is unit-tested.
  Manual pass needed on macOS AND Windows (QProcess CLI shutdown-on-float-close).

## NOT in scope

ImageJ drag-back re-attach (TODOS.md); IMA-205 pane; QDockWidget migration;
multi-tab float grouping; float geometry persistence; stale-op-tabs-on-reingest
(pre-existing, TODOS.md).

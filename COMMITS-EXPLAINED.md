# IMA-185 — Commits explained (longform)

Branch: `juliomaragall/ima-185-emit-navigable-output-multiscale-ome-zarr-plate-metadata`
Repo: `github.com/maragall/SquidMIP`
Written: 2026-07-05

---

## TL;DR

This session produced **exactly one commit** on this branch: `a5d9f45`, which adds
a single file, `TODOS.md`. There is **no IMA-185 implementation** in the diff. That is
deliberate: the engineering review concluded IMA-185's stated requirement is already
satisfied by IMA-184, so the ticket was put **on hold** pending a product decision.
The full plan and review live in `.spec/open/ima-185.md`, which is **gitignored** and
therefore does not appear in any commit.

---

## Branch state — two commits, one is mine

```
a5d9f45  IMA-185: add TODOS (deferred plate viewer + packaging coordination)   <- this session
b93b179  Scaffold SquidMIP repo (README, gitignore, design + UI prototypes)    <- pre-existing
```

- **b93b179** was already on the branch before this session. Not mine. It is the empty
  scaffold: `README.md`, `.gitignore`, and the two HTML prototypes under `docs/`.
- **a5d9f45** is the only commit I authored. Diff to `main` is `TODOS.md` only.

---

## Commit `a5d9f45` — full anatomy

### What changed
One new file, `TODOS.md` (previously did not exist). Nothing else. No source code,
no `pyproject.toml`, no tests. The repo still has zero Python.

### The commit message (verbatim)
```
IMA-185: add TODOS (deferred plate viewer + packaging coordination)

Eng review put IMA-185 on hold: the "opens in ndviewer_light" requirement
is already satisfied by IMA-184, so a static montage is only worth building
if stakeholders confirm they want a whole-plate contact-sheet artifact.
No implementation on this branch yet. Plan detail lives in .spec/ (gitignored).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

### What `TODOS.md` contains
Two tracked follow-up items, each with full context (what / why / pros / cons / depends-on):

1. **Interactive plate-grid viewer (deferred from IMA-185).** The clickable 2-D plate
   grid — pyramid-fed thumbnails, click-to-lazy-load full-res, global-per-channel
   contrast, region jump (E5 / AA5). This was the "bonus" in the ticket. It is a *new
   viewer*, not a wiring job, because `ndviewer_light` has no 2-D grid (it navigates
   wells via a 1-D FOV slider and reads only pyramid level 0). It is the foundation the
   future squid2minerva / Minerva path builds on.

2. **Package skeleton coordination (SquidMIP-wide).** SquidMIP has no `pyproject.toml`
   or source tree. IMA-183/184/185/186 all need the same skeleton. Whoever builds first
   should own it, or two tickets scaffold in parallel and collide. Flagged so the
   sequencing is explicit.

### Why the commit exists at all
The review created `TODOS.md` to preserve the reasoning behind two decisions that would
otherwise be lost: (a) the interactive viewer was cut on purpose, not forgotten, and
(b) the packaging gap is shared across tickets and needs coordination. A TODO without
context is worse than none — these carry the "why."

---

## Why the diff is *only* `TODOS.md` (the decision trail)

The commit is small because the review's conclusion was "don't build yet." Here is the
chain that got there:

1. **Scope challenge (Step 0).** IMA-185 says "emit navigable output." I checked what
   already exists. `ndviewer_light` already opens the `plate.ome.zarr` that IMA-184
   emits (`detect_format` at core.py:854 recognizes the HCS plate; `discover_zarr_v3_fovs`
   at core.py:1149 walks `{row}/{col}/{field}`), and it navigates wells with a slider.

2. **Scope locked to montage-only.** The interactive plate viewer was deferred as
   "bonus" (your own ticket said so). The only net-new deliverable was a per-plate
   montage — a static contact sheet the slider viewer can't produce.

3. **Design locked.** Five decisions: read the zarr low-res pyramid level for tiles;
   build a synthetic `plate.ome.zarr` fixture as the IMA-184 contract; add a minimal
   package skeleton; output a composite RGB PNG with global-per-channel contrast;
   keep memory flat with a streaming two-pass (histogram) contrast pass.

4. **Outside voice caught the real problem.** An independent review pointed out: the
   ticket's acceptance is "montage **OR** opens in ndviewer_light" — and the
   opens-in-ndviewer half is *already* IMA-184's acceptance criterion #1. So the
   requirement is met before IMA-185 writes a line. The montage is only worth building
   if a *static, shareable, whole-plate* artifact is specifically wanted.

5. **You agreed — ticket on hold.** IMA-185 is paused pending a stakeholder answer:

   > IMA-184 already lets you open the plate and scrub wells in ndviewer_light. Do you
   > also want a single static contact-sheet image of the whole plate (glance /
   > screenshot / Slack / print)? Or is navigating in ndviewer_light enough?

   - Yes → build the montage per the locked design.
   - No → close IMA-185 as satisfied by IMA-184; zero code.

Because of step 5, committing implementation would have been premature. `TODOS.md` is
the only durable artifact worth committing right now.

---

## What is intentionally NOT in this branch

- **No montage code, no CLI, no tests, no `pyproject.toml`.** IMA-185 is on hold.
- **The plan itself.** `.spec/open/ima-185.md` holds the full spec, ASCII data-flow
  diagram, design table, test map, failure modes, and the GSTACK review report. It is
  gitignored (`.gitignore:1` ignores `.spec/`), so it never enters a commit. It lives
  locally in the worktree.
- **This file (`COMMITS-EXPLAINED.md`)** is untracked and not committed unless you ask.

---

## Open items (unresolved on purpose)

1. **Stakeholder confirmation (blocks everything):** static montage wanted, or is
   ndviewer_light enough?
2. **Tile source (if resumed):** read low-res pyramid (DRY, but makes the montage the
   sole validator of IMA-184's pyramid) vs read level 0 + downsample (decoupled, heavier
   I/O).
3. **Fixture drift (if resumed):** generate the test fixture from tilefusion's real
   NGFF writer (the one IMA-184 will use) rather than hand-rolling, so it can't diverge.

---

## One honest caveat

If your mental model was that this branch contains working IMA-185 code, it does not.
The value delivered this session is the review and the on-hold decision, not shipped
software. The one commit is a documentation artifact. That is the faithful state.

# STATE — IMA-213

- **Ticket:** IMA-213
- **Branch:** juliomaragall/ima-213-rename-squidhcs
- **Spec:** .spec/open/ima-213.md
- **Phase:** THINK
- **Mode:** attended

## Now
_(the single item currently in flight)_

## Next
_(ordered queue, from the spec's decomposition)_

## Decisions
_(choice — why — alternative rejected; AFK defaults land here)_

Eng review 2026-07-20 (`/plan-eng-review`), full record in `.spec/open/ima-213.md`:

- **D2 land now, not last** — GitHub's repo redirect makes the original deferral
  premise moot; conflict surface today is one markdown file and grows with every
  landed ticket. Rejected: keep "lands last".
- **D3 migrate in `Setup-Windows.ps1`, no compat shim** — fixes the break in the
  file that causes it. Rejected: `squidmip` shim package. *Contested by outside
  voice; unresolved.*
- **D4 clean cut on console scripts, no aliases** — the ticket's goal is that the
  old name stops existing. Rejected: aliases, deprecation shims.
- **D5 fix PyInstaller target AND make `freeze` able to fail** — a build step that
  cannot fail is not a check. Rejected: string-only fix.
- **D6 preserve historical docs, annotate** — truthful record beats clean grep.
  Rejected: blanket rewrite, archive move.
- **D7 product stays "MIP tool"** — internal rename shouldn't reach Nick's desktop;
  `MIP` is domain vocabulary. Rejected: rename product. *Contested; unresolved.*
- **D8 run the installer on the Windows CI runner** — migration is new code on the
  highest-stakes path. Rejected: manual checklist, ship-and-see.
- **D9 drop `_video` + `imageio` + `imageio_ffmpeg` from `smoke_import.py`** —
  none are declared deps; they block `smoke`, and `freeze` needs `smoke`. (AFK
  default; widened from one module to three by the outside voice.)
- **D10 prerequisite: get `main` green first** — `main` CI red 11+ days and
  `freeze` has never executed, so 3 of 7 acceptance criteria were unmeasurable.
  (AFK default.)
- **D11 delete `scripts/mip-tool.bat`** — conda launcher contradicting the
  documented "NO conda" venv path. (AFK default.)

Correction on record: the first review pass claimed all 40 branches sat at 0
commits ahead of `main`. That was measured against *local* `main`; 11 branches are
1 commit ahead of `origin/main`. Conclusion unchanged, quantification corrected.

## Blockers
_(spec-vs-reality contradictions; a non-empty entry halts an AFK run)_

## Learnings
_(distilled in Reflect -> /learn)_

## Iterations
_(one line per Build iteration: n — what landed — verify result)_

# STATE — IMA-213

- **Ticket:** IMA-213
- **Branch:** juliomaragall/ima-213-rename-squidhcs
- **Spec:** .spec/open/ima-213.md
- **Phase:** BUILD (implemented 2026-07-21; T0-T8, T10 landed — T9 is a GitHub settings action)
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

1 — T0/T7 unblock CI: dropped `_video`, `imageio`, `imageio_ffmpeg` from
    `smoke_import.py` (all undeclared); gated `ome-zarr-models` behind
    `python_version >= '3.11'` (verified against PyPI: it declares `<3.14,>=3.11`,
    so the 3.10 CI leg could never resolve `.[test]`); `ngff_check.py` now
    `importorskip`s. — `smoke_import.py` prints "all imports OK" for the first
    time; freeze is unblocked.
2 — T1 rename: `git mv squidmip squidhcs` + rewrote 263 occurrences across 44
    files, exact tokens only (never case-insensitive `mip`, which is domain
    vocabulary). Git recorded every module as `R` so `--follow` survives. —
    clean-room `pip install .[test]` succeeds; `import squidhcs` OK.
3 — T2 migration in `Setup-Windows.ps1`: detects `%LOCALAPPDATA%\squidmip\venv`,
    builds the new venv, repoints the Desktop shortcut at `-m squidhcs._viewer`,
    removes stale old-name icons, reports the orphaned old venv. Idempotent. —
    verified by the T6 CI job, not locally (no Windows host).
4 — T3/T5 freeze: removed the contradictory `--windowed` + `-c` pair (both set
    the same PyInstaller option; the trailing `-c` silently won), dropped the
    undeclared `--collect-all imageio_ffmpeg`, `if-no-files-found: warn` ->
    `error`, added a non-empty-binary assert. — freeze run locally end-to-end:
    27 MB binary at `dist/hcs-viewer/hcs-viewer`, exit 0.
5 — T4/T5 regression tests: `test_rename_guard.py` (frozen allowlist + counts,
    anti-tautology) and `test_entry_points.py` (subprocess all 3 invocation
    forms + assert old commands are gone). The guard immediately caught 3 files
    I had missed. — 136 passed, 2 skipped, 0 failed with `.[gui,test]`.
6 — T6 `windows-migration` CI job; T8 historical-doc headers; T10 deleted
    `scripts/mip-tool.bat` (conda launcher contradicting the documented
    "NO conda" venv path). — both workflow files parse as valid YAML.

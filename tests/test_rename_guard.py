"""Guard the SquidMIP -> SquidHCS rename (IMA-213).

The rest of the suite imports the package directly, so it passes even if every *non-Python*
reference still points at the dead name. Nothing else in this repo reads ``.bat``, ``.ps1``,
``.yml``, ``.html`` or the docs, which is exactly where a rename rots:

    tests/test_*.py  --import-->  squidhcs/   ......... covered by the rest of the suite
    environment.yml, *.ps1, *.yml, README, docs/*.html   <-- covered ONLY by this file

So this walks every tracked file and asserts the old name is gone.

Anti-tautology (the reason the counts below are hardcoded): a guard whose allowlist the
implementer may edit to make their own test pass proves nothing. The allowlist is a frozen
literal with exact per-file occurrence counts. Adding a file, or letting an allowlisted file
grow more old-name references, fails the test. Shrinking is fine (the count is an upper bound),
so deleting historical docs later does not break CI.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Exact tokens. Deliberately NOT case-insensitive on "mip": the tool really does perform
# maximum intensity projection, and "MIP tool" / "MIP" is domain vocabulary that must survive.
OLD_NAME = re.compile(r"squidmip|SquidMIP")

# Frozen allowlist. file -> maximum old-name occurrences permitted.
#
#   docs/ima-*-eng-review.md, DESIGN-STATUS.md, COMMITS-EXPLAINED.md
#       Historical record (IMA-213 decision D6). These describe decisions as they were made,
#       when the package genuinely was squidmip. Annotated with a header rather than rewritten.
#   scripts/Setup-Windows.ps1
#       The rename migration itself. It has to know the OLD path (%LOCALAPPDATA%\squidmip\venv)
#       to find and repair a pre-rename install, so these references are load-bearing.
#   .github/workflows/build.yml
#       The windows-migration job fabricates a pre-rename install to prove the migration works.
#       It has to name the old venv and the old module, or it would be testing nothing.
ALLOWLIST: dict[str, int] = {
    "docs/ima-183-eng-review.md": 14,
    "docs/ima-184-eng-review.md": 11,
    "docs/ima-188-eng-review.md": 17,
    "docs/ima-189-eng-review.md": 17,
    "docs/DESIGN-STATUS.md": 15,
    "COMMITS-EXPLAINED.md": 6,
    "scripts/Setup-Windows.ps1": 9,
    ".github/workflows/build.yml": 7,
}

# Planning artifacts. `.spec/` holds the ticket spec and STATE record, which necessarily discuss
# the rename in both directions ("squidmip -> squidhcs"). They document the change rather than
# depending on it, so the whole directory is out of scope for the guard.
SKIP_PREFIXES = (".spec/", ".sprint/")

# Binary/vendored suffixes we never scan.
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".pdf", ".mp4", ".tif", ".tiff"}


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split("\0") if p]


def _count_old_name(path: Path) -> int:
    try:
        return len(OLD_NAME.findall(path.read_text(encoding="utf-8", errors="ignore")))
    except OSError:
        return 0


def test_no_old_name_outside_allowlist():
    """Every tracked file except the frozen allowlist is free of the pre-rename name."""
    offenders: dict[str, int] = {}
    for rel in _tracked_files():
        if (
            rel in ALLOWLIST
            or rel.startswith(SKIP_PREFIXES)
            or Path(rel).suffix.lower() in SKIP_SUFFIXES
        ):
            continue
        n = _count_old_name(REPO / rel)
        if n:
            offenders[rel] = n

    assert not offenders, (
        "Pre-rename name still present in "
        f"{len(offenders)} file(s): {offenders}. "
        "Rename them to squidhcs/SquidHCS, or — only if the reference is genuinely "
        "historical or load-bearing — add the file to ALLOWLIST with a justification."
    )


def test_allowlisted_files_do_not_accumulate_more():
    """An allowlisted file may shrink but never grow. Stops the allowlist absorbing new debt."""
    grown = {
        rel: (actual, cap)
        for rel, cap in ALLOWLIST.items()
        if (actual := _count_old_name(REPO / rel)) > cap
    }
    assert not grown, (
        f"Allowlisted file(s) gained old-name references: {grown} (actual, allowed). "
        "New code must use squidhcs; the allowlist covers existing historical text only."
    )


def test_allowlist_has_not_grown():
    """The allowlist itself is frozen. Adding an entry is a deliberate, reviewed act."""
    assert set(ALLOWLIST) == {
        "docs/ima-183-eng-review.md",
        "docs/ima-184-eng-review.md",
        "docs/ima-188-eng-review.md",
        "docs/ima-189-eng-review.md",
        "docs/DESIGN-STATUS.md",
        "COMMITS-EXPLAINED.md",
        "scripts/Setup-Windows.ps1",
        ".github/workflows/build.yml",
    }, "ALLOWLIST changed. If that is intentional, update this assertion in the same commit."


@pytest.mark.parametrize("rel", sorted(ALLOWLIST))
def test_allowlisted_files_still_exist(rel):
    """A stale allowlist entry would silently weaken the guard."""
    assert (REPO / rel).exists(), f"Allowlisted {rel} is gone; drop it from ALLOWLIST."


def test_package_directory_is_renamed():
    assert (REPO / "squidhcs" / "__init__.py").exists(), "squidhcs package missing"
    assert not (REPO / "squidmip").exists(), "pre-rename package directory still present"

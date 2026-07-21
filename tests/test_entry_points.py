"""The installed console scripts actually resolve and run (IMA-213).

Every other test in this suite imports the package directly::

    from squidhcs._cli import ProcessParameters, run

which proves nothing about what ``pip install`` puts on PATH. IMA-213 decision D4 was a clean
cut with no aliases, so ``squidhcs`` / ``squidhcs-view`` are now the only names that exist and
they are load-bearing: the README, the quickstart and Setup-Windows.ps1 all point at them. A
typo in ``[project.scripts]`` would ship green without these tests.

Three invocation forms, matching the three ways the tool is actually launched:

    squidhcs --help          console script     (docs, CLI users)
    squidhcs-view --help     console script     (Desktop shortcut target's sibling)
    python -m squidhcs       __main__.py        (README command-line section)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

TIMEOUT = 120  # generous: importing PyQt5 + tensorstore is slow on cold CI runners


def _env():
    """Qt must never try to open a display in CI."""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=TIMEOUT, env=_env()
    )


def test_module_invocation_runs():
    """`python -m squidhcs --help` — the form the README documents."""
    r = _run([sys.executable, "-m", "squidhcs", "--help"])
    assert r.returncode == 0, f"`python -m squidhcs --help` failed:\n{r.stdout}\n{r.stderr}"
    assert r.stdout.strip(), "expected help output on stdout"


def test_cli_console_script_resolves():
    """The `squidhcs` console script from [project.scripts]."""
    exe = shutil.which("squidhcs")
    if exe is None:
        pytest.skip("squidhcs console script not on PATH (package not installed in this env)")
    r = _run([exe, "--help"])
    assert r.returncode == 0, f"`squidhcs --help` failed:\n{r.stdout}\n{r.stderr}"


def test_viewer_console_script_resolves():
    """The `squidhcs-view` console script. Needs the [gui] extra; skipped when absent."""
    exe = shutil.which("squidhcs-view")
    if exe is None:
        pytest.skip("squidhcs-view console script not on PATH (gui extra not installed)")
    pytest.importorskip("PyQt5", reason="squidhcs-view requires the [gui] extra")
    r = _run([exe, "--help"])
    assert r.returncode == 0, f"`squidhcs-view --help` failed:\n{r.stdout}\n{r.stderr}"


def test_old_console_scripts_are_gone():
    """D4 was a clean cut: the pre-rename commands must NOT be installed."""
    for stale in ("squidmip", "squidmip-view"):
        assert shutil.which(stale) is None, (
            f"`{stale}` is still on PATH. IMA-213 D4 removed the old command names with no "
            "aliases; a lingering entry point means the rename is incomplete."
        )

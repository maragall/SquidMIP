# Setup-Windows.ps1 - set up the MIP tool on Windows with a plain Python venv (NO conda) and put a
# "MIP tool" shortcut on the Desktop.
#
# Run once, from the repo root, in Windows PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1
#
# Requires Python 3.10+ and git. The venv is self-contained, so the shortcut runs the venv's own
# pythonw directly - no activation, nothing global touched.

# NOTE: not using -ErrorAction Stop globally, because native tools (py, pip) legitimately write to
# stderr and that would otherwise abort the script. We check exit codes explicitly and Die on failure.
$ErrorActionPreference = "Continue"
$AppName = "MIP tool"
$Module  = "squidhcs._viewer"
$repo = Split-Path $PSScriptRoot -Parent

function Die($msg) { Write-Host ""; Write-Host ("ERROR: " + $msg) -ForegroundColor Red; exit 1 }

# 1. Pick a known-good Python from what's INSTALLED (parse 'py --list'; avoid launching missing ones,
#    and prefer 3.11/3.10/3.12 over a brand-new default that may lack wheels).
$pyExe = $null; $pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $listing = (cmd /c "py --list 2>&1" | Out-String)
    $avail = @()
    foreach ($m in [regex]::Matches($listing, '3\.(1[0-9])')) { $avail += [int]$m.Groups[1].Value }
    $avail = $avail | Sort-Object -Unique
    $pick = @(11, 10, 12, 13) | Where-Object { $avail -contains $_ } | Select-Object -First 1
    if (-not $pick -and $avail.Count -gt 0) { $pick = ($avail | Sort-Object | Select-Object -First 1) }
    if ($pick) { $pyExe = "py"; $pyArgs = @("-3.$pick") }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pyExe = (Get-Command python).Source
}
if (-not $pyExe) {
    Die "No Python 3.10+ found. Install Python 3.11 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run."
}
$ver = (& $pyExe @pyArgs --version 2>&1 | Out-String).Trim()
Write-Host ("Using " + $ver)

# 2. Create the venv (once).
#
# IMA-213 rename migration. Before the rename this venv lived at ...\squidmip\venv and the
# Desktop shortcut ran `-m squidmip._viewer` against an EDITABLE install. So a plain `git pull`
# (which the README documents as the update path) renames the package on disk and leaves the
# shortcut pointing at a module that no longer exists:
#
#   Desktop\MIP tool.lnk -> ...\squidmip\venv\Scripts\python.exe -m squidmip._viewer
#                                 |                                    |
#                                 |  editable -> repo\squidhcs\        v
#                                 +----------------------------> ModuleNotFoundError
#
# Re-running this script repairs that: it builds the venv under the new name, repoints the
# shortcut, and removes the stale one. Idempotent - safe to run any number of times.
$venv    = Join-Path $env:LOCALAPPDATA "squidhcs\venv"
$oldVenv = Join-Path $env:LOCALAPPDATA "squidmip\venv"
$vpy  = Join-Path $venv "Scripts\python.exe"
$vpyw = Join-Path $venv "Scripts\pythonw.exe"

$migrating = (Test-Path (Join-Path $oldVenv "Scripts\python.exe")) -and (-not (Test-Path $vpy))
if ($migrating) {
    Write-Host ""
    Write-Host "Found a pre-rename install (squidmip). Migrating it to squidhcs ..." -ForegroundColor Yellow
}

if (-not (Test-Path $vpy)) {
    Write-Host ("Creating virtual environment at " + $venv + " ...")
    & $pyExe @pyArgs -m venv $venv
    if (-not (Test-Path $vpy)) { Die "Could not create the virtual environment." }
}

# 3. Install the app + GUI deps. First run downloads a few packages.
Write-Host "Installing the MIP tool and its dependencies (first time takes a few minutes) ..."
& $vpy -m pip install --upgrade pip
# EDITABLE install of the app: after this, a `git pull` in the repo takes effect on the next launch
# with no reinstall (only the pinned deps below are a fixed snapshot).
& $vpy -m pip install -e ($repo + "[gui]")
if ($LASTEXITCODE -ne 0) {
    Die "pip install failed (see the errors above). This usually means a package has no wheel for this Python version; tell Julio the error and we'll pin a version."
}

# 4. Desktop shortcut -> venv python.exe -m module. Uses python.exe (NOT pythonw) ON PURPOSE so a
#    console window opens alongside the app, showing logs/errors + the [footprint] lines. Close that
#    window to quit the app.
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop ($AppName + ".lnk")
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $vpy
$sc.Arguments = "-m " + $Module
$sc.WorkingDirectory = $env:USERPROFILE
$sc.IconLocation = $vpy + ",0"
$sc.Description = $AppName
$sc.Save()

# 5. IMA-213 migration cleanup. The shortcut above is rewritten in place, so a same-named stale
#    icon is already handled. Remove any old-name variants so the Desktop never shows two icons,
#    and tell the user about the orphaned pre-rename venv (several hundred MB of PyQt5 +
#    tensorstore + ndviewer_light) rather than deleting it for them.
foreach ($stale in @("MIP tool (squidmip).lnk", "SquidMIP.lnk", "squidmip.lnk")) {
    $stalePath = Join-Path $desktop $stale
    if (Test-Path $stalePath) {
        Remove-Item $stalePath -Force -ErrorAction SilentlyContinue
        Write-Host ("Removed stale shortcut: " + $stale)
    }
}

if (Test-Path $oldVenv) {
    Write-Host ""
    Write-Host "The old pre-rename environment is still on disk and is no longer used:" -ForegroundColor Yellow
    Write-Host ("    " + $oldVenv)
    Write-Host "It is safe to delete if you want the disk space back."
}

Write-Host ""
Write-Host ("Done. '" + $AppName + "' is on your Desktop - double-click it, then drop an acquisition folder.")
if ($migrating) {
    Write-Host ""
    Write-Host "Migration complete. Your Desktop icon now points at the renamed package." -ForegroundColor Green
}

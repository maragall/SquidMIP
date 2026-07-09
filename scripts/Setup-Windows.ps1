# Setup-Windows.ps1 - set up the MIP tool on Windows with a plain Python venv (NO conda needed)
# and put a "MIP tool" shortcut on the Desktop.
#
# Run once, from the repo root, in Windows PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1
#
# Requires Python 3.10+ (from https://www.python.org/downloads/, "Add python.exe to PATH") and git.
# A venv is self-contained, so the Desktop shortcut runs the venv's pythonw directly - no activation,
# no environment-variable changes, nothing global touched.

$ErrorActionPreference = "Stop"
$AppName = "MIP tool"
$Module  = "squidmip._viewer"
$repo = Split-Path $PSScriptRoot -Parent

# 1. Find Python. Prefer a KNOWN-GOOD version (3.11 validated, 3.10/3.12 fully supported) over the
#    newest default, since brand-new Python releases often lack wheels for PyQt5/tensorstore. Uses the
#    'py' launcher when present, else 'python' on PATH.
$pyExe = $null; $pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    # Read the installed list (merge stderr so a stderr-only listing does not trip -ErrorAction Stop),
    # then pick a known-good minor from what is ACTUALLY installed (no launching missing versions).
    $listing = (& py --list 2>&1 | Out-String)
    $avail = @()
    foreach ($m in [regex]::Matches($listing, '3\.(1[0-9])')) { $avail += [int]$m.Groups[1].Value }
    $avail = $avail | Sort-Object -Unique
    $pick = @(11, 10, 12, 13) | Where-Object { $avail -contains $_ } | Select-Object -First 1
    if (-not $pick -and $avail.Count -gt 0) { $pick = ($avail | Sort-Object | Select-Object -First 1) }
    if ($pick) { $pyExe = "py"; $pyArgs = @("-3.$pick") } else { $pyExe = "py"; $pyArgs = @("-3") }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pyExe = (Get-Command python).Source
}
if (-not $pyExe) {
    Write-Error "No Python found. Install Python 3.11 from https://www.python.org/downloads/ (tick 'Add python.exe to PATH'), then re-run."
}
$ver = (& $pyExe @pyArgs -c "import sys;print('%d.%d'%sys.version_info[:2])").Trim()
if ([version]$ver -lt [version]"3.10") {
    Write-Error ("Found Python " + $ver + " but 3.10+ is required. Install Python 3.11 from python.org and re-run.")
}
Write-Host ("Using Python " + $ver)

# 2. Create the venv (once).
$venv = Join-Path $env:LOCALAPPDATA "squidmip\venv"
$vpy  = Join-Path $venv "Scripts\python.exe"
$vpyw = Join-Path $venv "Scripts\pythonw.exe"
if (-not (Test-Path $vpy)) {
    Write-Host ("Creating virtual environment at " + $venv + " ...")
    & $pyExe @pyArgs -m venv $venv
}

# 3. Install the app + GUI deps (PyQt5 + ndviewer_light). First run downloads a few packages.
Write-Host "Installing the MIP tool and its dependencies (first time takes a few minutes) ..."
& $vpy -m pip install --upgrade pip
& $vpy -m pip install ($repo + "[gui]")

# 4. Desktop shortcut -> venv pythonw -m module (self-contained; no console, no activation).
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop ($AppName + ".lnk")
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $vpyw
$sc.Arguments = "-m " + $Module
$sc.WorkingDirectory = $env:USERPROFILE
$sc.IconLocation = $vpyw + ",0"
$sc.Description = $AppName
$sc.Save()

Write-Host ""
Write-Host ("Done. '" + $AppName + "' is on your Desktop - double-click it, then drop an acquisition folder.")

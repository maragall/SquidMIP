# Setup-Windows.ps1 - create the conda env and a Desktop shortcut for the MIP tool.
#
# Run once, from the repo root, in Windows PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1
#
# Finds conda.exe directly (works even if 'conda' is not on PowerShell's PATH / you never ran
# 'conda init powershell'), creates the "squidmip" env if missing, and drops a "MIP tool" Desktop
# shortcut that launches the app with NO console window and with the env ACTIVATED (activation is
# required on Windows or Qt fails to load).

$ErrorActionPreference = "Stop"
$EnvName = "squidmip"
$Module  = "squidmip._viewer"
$AppName = "MIP tool"

# 1. Locate conda.exe without relying on PATH.
$conda = $env:CONDA_EXE
if (-not $conda -or -not (Test-Path $conda)) {
    $cands = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe"
    )
    $conda = $cands | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $conda) {
    Write-Error "Could not find conda.exe. Install Miniconda, or open the 'Anaconda PowerShell Prompt' and re-run."
}
$condaBase = Split-Path (Split-Path $conda)
$repo = Split-Path $PSScriptRoot -Parent

# 2. Create the env if it is not there.
$envDir = Join-Path $condaBase ("envs\" + $EnvName)
if (-not (Test-Path $envDir)) {
    Write-Host ("Creating conda env '" + $EnvName + "' (first time, may take a few minutes)...")
    & $conda env create -f (Join-Path $repo "environment.yml")
} else {
    Write-Host ("conda env '" + $EnvName + "' already exists.")
}

# 3. Write a hidden launcher: activate the env in cmd (so Qt DLLs load), then pythonw (no console).
$appDir = Join-Path $env:LOCALAPPDATA $EnvName
New-Item -ItemType Directory -Force -Path $appDir | Out-Null
$cmd = Join-Path $appDir "launch.cmd"
$activate = Join-Path $condaBase "Scripts\activate.bat"
$cmdLines = @(
    "@echo off",
    ('call "' + $activate + '" ' + $EnvName),
    ("pythonw -m " + $Module + " %*")
)
Set-Content -Path $cmd -Value $cmdLines -Encoding ASCII
$vbs = Join-Path $appDir "launch.vbs"
$vbsLine = 'CreateObject("WScript.Shell").Run """' + $cmd + '""", 0, False'
Set-Content -Path $vbs -Value $vbsLine -Encoding ASCII

# 4. Desktop shortcut -> wscript runs the VBS hidden (no console window at all).
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop ($AppName + ".lnk")
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = (Join-Path $env:SystemRoot "System32\wscript.exe")
$sc.Arguments = '"' + $vbs + '"'
$sc.WorkingDirectory = $env:USERPROFILE
$sc.IconLocation = (Join-Path $envDir "python.exe") + ",0"
$sc.Description = $AppName
$sc.Save()

Write-Host ""
Write-Host ("Done. '" + $AppName + "' is on your Desktop - double-click it, then drop an acquisition folder.")
Write-Host ("Launcher: " + $cmd)

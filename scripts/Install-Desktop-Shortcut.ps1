# Creates a "MIP tool" shortcut on the Windows Desktop that launches the app from the
# "squidmip" conda environment. No console window (uses pythonw).
#
# Run it (from anywhere) after the env exists:
#     powershell -ExecutionPolicy Bypass -File scripts\Install-Desktop-Shortcut.ps1
#
# (If the squidmip env is already active in this shell it is used directly; otherwise the script
#  locates it via `conda run -n squidmip`.)

$ErrorActionPreference = "Stop"

# 1. Find the squidmip env's python.
$py = $null
if ($env:CONDA_PREFIX -and (Split-Path $env:CONDA_PREFIX -Leaf) -eq "squidmip") {
    $py = Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    try { $py = (conda run -n squidmip python -c "import sys; print(sys.executable)").Trim() } catch { }
}
if (-not $py -or -not (Test-Path $py)) {
    Write-Error "Could not find the 'squidmip' conda environment. From the SquidMIP folder run:  conda env create -f environment.yml"
}

# 2. Prefer pythonw.exe (windowed, no console flash).
$envdir = Split-Path $py
$pyw = Join-Path $envdir "pythonw.exe"
if (-not (Test-Path $pyw)) { $pyw = $py }

# 3. Create the Desktop shortcut.
$desktop = [Environment]::GetFolderPath("Desktop")
$lnk = Join-Path $desktop "MIP tool.lnk"
$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnk)
$sc.TargetPath = $pyw
$sc.Arguments = "-m squidmip._viewer"
$sc.WorkingDirectory = $env:USERPROFILE
$sc.IconLocation = "$pyw,0"
$sc.Description = "MIP tool - open a Squid 1536-well acquisition"
$sc.Save()

Write-Host "Created Desktop shortcut: $lnk"
Write-Host "  launches: $pyw -m squidmip._viewer"
Write-Host "Double-click 'MIP tool' on your Desktop, then drag an acquisition folder onto the window."

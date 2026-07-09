@echo off
REM Launch the MIP tool from the "squidmip" conda environment.
REM Double-click this file, or pass an acquisition folder:  mip-tool.bat "C:\path\to\acquisition"

call conda activate squidmip 2>nul
if errorlevel 1 (
    echo Could not activate the 'squidmip' conda environment.
    echo First-time setup, from the SquidMIP folder:
    echo     conda env create -f environment.yml
    pause
    exit /b 1
)

squidmip-view %*

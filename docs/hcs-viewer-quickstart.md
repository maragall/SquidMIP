# MIP tool, quick start

What it does
- Opens a finished Squid well plate acquisition.
- Flattens each well's z stack into one max intensity projection (MIP), across the whole plate.
- Saves the result you can open again here, in napari, or in FIJI.
- Read only. It never changes your acquisition and never runs the microscope.
- A second small app, TIFFs to MP4, turns a folder of images into a movie.

Setup (one time, Windows)
- Install Python 3.11 from python.org. In the installer, tick "Add python.exe to PATH".
- Open PowerShell in the tool folder and run the setup line:
  - MIP tool: powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1
  - Video tool: powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1
- Each one puts an icon on your Desktop: "MIP tool" and "TIFFs to MP4".
- Nothing else on your machine is touched.
- To update later: git pull in the folder, then open the icon again.

Open an acquisition
- Double click "MIP tool". A small black console opens next to it. That is normal, it shows progress. Closing it quits the app.
- Use the menu: File, then Open acquisition folder.
- Pick the acquisition folder (the one holding the 0 folder and/or the ome_tiff folder).
- It reads both Squid formats (individual TIFFs and OME-TIFF), on 384 and 1536 plates.

The window
- Left: the buttons (run MIP, open CLI, layers).
- Bottom left: the plate. Grey dots are empty wells, so you always see the full plate shape. Scanned wells show their image.
- Right: the well in view, with focus (z), time (t), and FOV sliders.
- Double click a well to open it on the right. The red box marks the well in view.

Run MIP
- Click "Maximum Intensity Projection".
- Preview first (nothing saved): set "First N wells", click Preview. It computes just those wells and shows them. Good for a quick look before doing the whole plate.
- Whole plate: choose an output folder, click "Run on the whole plate". It writes the result.
- "Focus reference plane" (under the right viewer) jumps the z slider to the sharpest plane of the well in view.
- "Return to raw view" goes back to the unprocessed plate.

The output
- Running MIP writes a folder: <acquisition name>.hcs, with plate.ome.zarr inside.
- That is the result: a navigable plate you can reopen here (File, Open a computed MIP), or in napari or FIJI.
- To also get plain TIFFs for FIJI, use the command line with --tiff (see below). It adds a tiff folder next to the zarr.

Command line (optional, for batch or FIJI TIFFs)
- Click "Open CLI" for a terminal inside the app, or use your own PowerShell.
- Common command (first 8 wells, save TIFFs to Downloads):
  - python -m squidmip "C:\path\to\acquisition" --limit 8 --tiff --output-folder ~/Downloads
- Drop "--limit 8" to do the whole plate.
- python -m squidmip --help lists all options.

Make a movie (TIFFs to MP4)
- Double click "TIFFs to MP4".
- Drop a folder of TIFFs on it (or Browse), set the frames per second, click "Make MP4".
- One TIFF is one frame, in file name order. Set "first N" to try a few frames first.

Good to know
- Wells with more than one FOV: for now it uses the first FOV per well (no stitching yet). Full multi FOV is available on request.
- It never writes into your acquisition folder. Results go only where you point them.
- Memory stays low even on a 1536 plate: it only holds the well in view plus a small cache.

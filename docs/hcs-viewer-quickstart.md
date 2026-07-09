# HCS viewer — quick start

A desktop app for reviewing a **finished** Squid well-plate acquisition and processing it — MIP,
best-focus reference plane, or an .mp4 movie — **without** hand-tracking which files came from which
well or round-tripping through FIJI. It reads your data **read-only** and writes results to a folder
*you* choose.

> Post-acquisition only: it opens data already saved to disk. It never controls the microscope.

---

## 1. Install (one time)

Python 3.10+. In a terminal:

```bash
pip install "git+https://github.com/maragall/ndviewer_light"   # the embedded z-stack/array viewer
pip install "squidmip[gui]"                                    # the HCS viewer (this tool)
```

*(Given a folder instead of a package: `cd` into it and `pip install ".[gui]"`.)*
A frozen desktop build (Linux AppImage / Windows / macOS) is produced by CI — no Python needed then.

## 2. Launch

```bash
hcs-viewer                     # then drag an acquisition onto the window
hcs-viewer /path/to/acquisition
```

## 3. Use it — step by step

1. **Drop your acquisition folder** (the one with the numbered timepoint folder `0/` inside). The
   plate map draws immediately — one dot per acquired well, laid out A,B,C… down and 1,2,3… across.
   Grey = not processed. Hover shows the well; **double-click** a well to open its raw z-stack on the
   right.
2. **Process wells** (top-left console):
   - **Maximum Intensity Projection** — collapse each well's z-stack to one max image.
   - **Reference plane** — pick each well's sharpest z (Tenengrad autofocus).
   - **Record video (.mp4)** — one movie per well (time-lapse if there's a time series, else a focus
     sweep), at a playback fps you choose.
   Pick an operator, choose an output folder, run. Wells turn **amber** (working) → **blue** (done),
   filling in as they go. A well that can't be read is marked red-✕ and **skipped** — one bad file
   never aborts the run.
3. **Output.** MIP / Reference write a **navigable multiscale `plate.ome.zarr`** you can re-open here
   or in any OME-Zarr tool. Record writes `<well>.mp4`.

## 4. Same thing headless (CLI)

```bash
squidmip /path/to/acquisition                        # MIP every well -> <acq>.hcs/plate.ome.zarr
squidmip /path/to/acquisition --projector reference  # sharpest-plane per well
squidmip /path/to/acquisition --workers 8 --tiff     # tune threads + also export per-plane TIFFs
```

## 5. Notes

- **Nothing is written into your acquisition folder** — outputs go to the folder you pick.
- **One FOV per well** is the current scope (a well = a condition). Wells with multiple FOVs are
  reported and **one FOV is sampled** until high-throughput stitching lands.
- Large plates (384/1536) are supported; the plate view is a navigation map, memory is bounded
  (it streams from disk — a 1536-well plate uses the same RAM as a 4-well one).

## 6. If something looks wrong

- *"not a readable Squid acquisition"* — drop the top-level folder (the one with `0/` and
  `acquisition.yaml`/`acquisition parameters.json` inside).
- *"ndviewer_light unavailable"* — re-run the first `pip install` line.
- *"MIP would persist ~N GB … only M GB free"* — point the output at a disk with room (the full-res
  multiscale plate is large; the estimate is deliberately conservative).

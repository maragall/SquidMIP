# HCS viewer — quick start

A desktop app for browsing a finished Squid well‑plate acquisition and computing a Maximum
Intensity Projection (MIP) per well — without hand‑tracking which files came from which well or
round‑tripping through FIJI. Drop an acquisition, click a well to see its z‑stack, and hit
**Process well‑plates → Maximum Intensity Projection** to MIP the whole plate.

> Runs on acquisitions that are already saved to disk (post‑acquisition). It reads your data
> read‑only and never writes into your acquisition folder.

---

## 1. Install (one time)

You need Python 3.10+ . In a terminal:

```bash
# 1. get the two packages
pip install "git+https://github.com/maragall/ndviewer_light"     # the per‑FOV z‑stack viewer
pip install "squidmip[gui]"                                       # the HCS viewer (this tool)
```

That's it — `squidmip[gui]` pulls in the plate viewer and PyQt; `ndviewer_light` is the embedded
z‑stack detail view.

*(If you were given a folder instead of a package: `cd` into it and run
`pip install ".[gui]"`.)*

## 2. Launch

```bash
hcs-viewer
```

A dark window opens with a drop zone on the left.

## 3. Use it — step by step

1. **Drop your acquisition folder** onto the left panel (the folder that contains the numbered
   timepoint folder `0/` and `acquisition parameters.json`). The plate map draws immediately, one
   circle per acquired well, laid out A, B, C… down and 1, 2, 3… across. Grey = not processed yet.
2. **Double‑click any well** → its **raw z‑stack** opens in the right panel. Use the **Z** and
   **channel** sliders to scrub through focus and channels. This is the fast way to check "what did
   well B7 actually look like?" — the plate map tells you exactly which well you're viewing (red box).
3. **Compute MIPs** → menu bar **Process well‑plates → Maximum Intensity Projection**. Every well
   turns **amber** (working), then **blue** as its MIP finishes and a thumbnail appears in the cell.
   A well that fails is marked with a red ✕. The whole plate fills in as it goes — you don't wait
   for the end.
4. **Read the plate at a glance.** Colours are colour‑blind‑safe: grey → amber → blue = not‑started
   → processing → done. Hover a well to see its ID in the header.

## 4. What the colours mean

| Colour | Meaning |
|---|---|
| Grey dot | Well acquired, not yet processed |
| Amber dot | MIP is computing now |
| Blue ring + thumbnail | MIP done |
| Red ✕ | MIP failed for that well |

## 5. Notes

- **Nothing is copied or written into your data folder.** The z‑stack view reads your original
  TIFFs in place; the plate thumbnails live only in memory while the app is open.
- Large plates (384/1536) are fully supported — the plate view always fits the window; it is a
  navigation map, not a full‑resolution image.
- More operations beyond MIP (e.g. extended depth of focus) will appear under the same
  **Process well‑plates** menu as they're added.

## 6. If something looks wrong

- *"not a readable Squid acquisition"* — you dropped the wrong folder. Drop the top‑level
  acquisition folder (the one with `acquisition parameters.json` and a `0/` folder inside).
- *"ndviewer_light unavailable"* — the z‑stack viewer isn't installed; re‑run the first
  `pip install` line above.

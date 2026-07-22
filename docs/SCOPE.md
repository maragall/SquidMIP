# SquidHCS — the scope that must not fall out of scope

Julio: "We can't let this fall out of scope. Like these two are core."

This file exists because requirements were slipping between working sessions. It is the durable
copy: if it is not written down here, it is lost. Status is recorded HONESTLY — "done" means seen
working, not merged.

Supersedes scope v1 (the ndv-era list: "all controls live inside ndv's ArrayViewer", FOV sliders,
one FOV per well). That document described a different product and every line of it is now wrong.

Three sources:

- **Core i** — direction from Spencer Schwarz (CSO), 2026-07-22.
- **Core ii** — the SquidMIP -> SquidHCS requirements list.
- **Core iii** — external projects we are told to build on rather than reinvent.

---

## Core i — the CSO's direction

> "the basics of navigating and showing data (**HCS + 3D**) are what we need first."

Spencer will **demo live with this tool**, which is why responsiveness is a requirement and not a
preference. His machine has 96 GB of RAM — generous, but the constraint that matters in a live
demo is attention, not memory: what must never happen is a pane that looks frozen.

| # | Requirement | Status |
|---|---|---|
| i.1 | **3D rendering** of a region over the acquisition's z axis, using napari's own renderer | **PARTIAL.** napari's ndisplay button is on screen (it was always there — below the fold at y=752 in a 900 px window). The volume renders EDGE-ON: z spans 13.5 µm against ~57,000 µm in x/y, about 4000:1, and napari's default camera angle (0,0,90) looks straight down the thin axis. That is the "1D array". The fix is the CAMERA, plus rendering a SUBSET rather than a full 5731x4793 mosaic. Not done. |
| i.2 | **MIP** as the demo operator ("Maybe just MIP for now") | **DONE.** Plate-wide, persists a navigable OME-Zarr. |
| i.3 | **Spot detection for nuclei** — simple, traditional, to *test the interface* | **IN FLIGHT.** Explicitly an interface test: "I expect we'll be able to adapt additional, more complex, segmentation later as operators." |
| i.4 | **Responsiveness — "buttery"** | **PARTIAL.** The multiscale pyramid landed: peak RSS 1503-1932 MB -> 480-672 MB (~2.8x), z revisit 756-993 ms -> 32-54 ms (~20x). A fresh z step is decode-bound and is a wash — reported as such, not dressed up. Region *load* is still seconds. |
| i.5 | **"An indicator when it's working"** | **PARTIAL.** `squidmip/_activity.py` is the single registry of in-flight work, unit-tested. NOT yet wired to a widget, so nothing has changed on screen yet. |
| i.6 | **Cellpose-style "iterating operators"** — both parties named Cellpose | **NOT STARTED.** The seam is being shaped so a real segmenter (Cellpose, StarDist) is a sibling registry entry rather than a rewrite. Both return label masks, so the result contract must be: a label image, optional centroids, and a count. |
| i.7 | **Fractal** as prior art | Surveyed. Its `input_types`/`output_types` and its `compound` task type are the closest published model for what multi-plane recording needs. |
| i.8 | **Minerva Author** | **DESCOPED by the CSO.** "Let's skip Minerva for now... it was just a suggestion, not a concrete requirement." Storytelling and post-demo user stories, not the immediate demo. Spend no time here. |
| i.9 | **3D storytelling via Vitessce or Allen Cell Explorer** instead of Minerva | **NOT STARTED.** "If Vitessce makes prettier 3D stacks, that's the way to go." |

---

## Core ii — SquidMIP -> SquidHCS requirements

| # | Requirement | Status |
|---|---|---|
| ii.1 | **Multi-FOV well/slide support** — show each region as a fused multi-FOV **mosaic** | **DONE.** The navigation unit is the region; a region is a mosaic of FOVs, never a single field. |
| ii.2 | **Live stitching** | **PARTIAL.** The stitch operator runs (tilefusion) and the stitcher's own controls are in pane 1. Registration now always runs on the registration channel — that was a real soundness bug: selecting a channel subset silently moved registration to channel 0, so one region stitched to DIFFERENT offsets depending on the selection. Not yet proven end-to-end on a real tissue mosaic. |
| ii.3 | **Migrate ndv -> napari** | **DONE.** napari is the default and is EMBEDDED in our window. The ndviewer_light fallback still exists and is meant to be deleted. |
| ii.4 | **One layer per operation** — e.g. the stitching before -> after toggle | **DONE.** Processing layer -> channels, group identity in `layer.metadata` (never parsed out of the layer name), and the toggle is a visibility flip over a group. napari has no layer groups (`LayerList` is flat; upstream #2229 open since 2021), so the tree is synthesised. |
| ii.5 | **napari over odon**, so gallery view and fractal analysis can follow | Recorded. Gallery view is the one that does not fit the current operator model: it needs a "result" type the pipeline has no concept of — the same missing abstraction that makes background subtraction invisible in napari. |
| ii.6 | **Exploration pane** — a third vertical pane to view and process FOV subsets in **tabs** | **DONE.** Each tab is a real viewer on its subset, built by pane 2's own constructor (never a second viewer implementation), with a region slider under it. |
| ii.7 | Open **minerva-author** with the selected FOVs | **DESCOPED** — see i.8. |
| ii.8 | **Shift/ctrl** to open an exploration tab with the selected FOV subset | **DONE.** |
| ii.9 | **Per-channel plate preview** — toggle and contrast adjustment swap the plate composite | **DONE, and now correct.** The plate has NO controls of its own: napari owns contrast, channel visibility and colormap, and the plate is a pure sink of all three. Per-region contrast is deleted — it resolved with `follow=False`, i.e. it deliberately ignored napari's window, which is why contrast would not sync however often the sink was repaired. |
| ii.10 | **Benchmark live stitching** against ASHLAR, MCmicro, BigStitcher | **PARTIAL.** `squidmip/_bench_stitchers.py` names all three with citations and what each needs (BigStitcher needs a headless Fiji). `squidmip/_oracle.py` is the acceptance criterion: cut a known image into tiles at known positions, grade how far off a stitcher puts them back. Not yet run against the three. |
| ii.11 | **Drag a tab out** into a free-floating exploration window | **DONE.** |
| ii.12 | **Press-and-hold loupe** on the plate grid | **DONE.** |

---

## Core iii — external work we build on, not around

Julio: "other core resources that we can't let fall out of scope". The standing rule is that we
orchestrate well-known libraries and add the interface; these are the named ones.

| Project | Why it is on this list | Status |
|---|---|---|
| **[vitessce](https://github.com/vitessce/vitessce)** | The candidate for 3D exploration and storytelling *instead of* Minerva Author. "I'm sure it'd look prettier for customers to view a volume in Vitessce rather than in napari's 3D renderer." Web-native and OME-Zarr-first, which also lines up with the eventual web host + cloud compute. | **NOT STARTED.** Nothing read, nothing prototyped. |
| **[napari-ome-zarr-navigator](https://github.com/fractal-napari-plugins-collection/napari-ome-zarr-navigator)** | Part of the Fractal plugin collection. It is a napari plugin that navigates an OME-Zarr HCS PLATE — well selection, region loading, label layers. This is the closest existing implementation of what pane 1 + pane 2 do together, and it is the "OME-Zarr interactive editor" Spencer posted. | **NOT STARTED.** Must be read before more plate-navigation UI is written. |
| **[gallery-view](https://github.com/jsschwrz/gallery-view)** | The CSO's own gallery view. Gallery view is ii.5's open problem — the operator model has no "result" type for it. Read his implementation before designing ours. | **NOT STARTED.** |
| **Cellpose** (and StarDist) | The model for "iterating operators" (i.6), named independently by both parties. | **NOT STARTED.** |
| **Fractal** (fractal-analytics-platform) | The task/operator model we are closest to and should not reinvent (i.7). | Surveyed only. |

---

## The architecture, as settled by the owner — do not relitigate

> "our GUI, napari, is nested into our GUI. It's not that our GUI is nested into napari."

```
PlateWindow (ours, the HOST)
├── PANE 1  plate view + the OPERATOR INTERFACES     (no view controls: it is a sink)
├── PANE 2  the embedded napari window               (OWNS contrast, visibility, colormap, z/t)
└── PANE 3  exploration pane: the same viewer on a SUBSET, in tabs
```

The single rule everything serves: **every fact has exactly ONE owner; every other widget showing
it SUBSCRIBES.** Two objects holding one fact and hand-syncing it is this project's dominant defect
shape, with more than four confirmed instances.

## Standing constraints

- Datasets are **READ ONLY**. Never copy or convert them — a copy once filled the machine to 0 bytes.
- Orchestrate well-known libraries; do not invent algorithms. The interface is the value we add.
- Ground designs in prior art before writing them.
- No silent failures. A refusal must name itself.

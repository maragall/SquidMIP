"""IMA-185 navigable output: a static whole-plate montage from the canonical OME-zarr.

Consumes the OME-NGFF HCS plate that IMA-184's ``write_plate`` produced and renders one
static, shareable **thumbnail mosaic** of the whole plate — the artifact ndviewer_light
cannot give (it navigates wells one at a time via a slider). "Opens in ndviewer_light" is
already satisfied by the IMA-184 plate; this is the *navigable overview* on top.

Self-contained from the written plate (no reader, no raw acquisition): the plate is
self-describing, so the montage renders exactly the canonical output a viewer would see.

  plate.ome.zarr/                         zarr.json .ome.plate  -> rows / columns / wells (grid)
    {row}/{col}/                          zarr.json .ome.well   -> images[].path (raw fov ids)
      {fov}/                              zarr.json .ome.omero  -> per-channel label + hex color
        0/                                array (T, C, 1, Y, X) -> the projected pixels

Flow (single streaming pass — peak memory is the montage canvas + ONE well, never the plate)::

    read plate metadata ─► grid = (sorted rows) x (sorted columns), each well at (rowIdx,colIdx)
    per well (streamed):
        read array 0 at t=0 ─► (C, Y, X)          # one well resident, ~one field in flight
        area-downsample each channel ─► (C, cell, cell)
        write tile into canvas[c, y0:y1, x0:x1]   # canvas is montage-sized (downsampled), bounded
    after the pass:
        per channel: lo/hi = percentiles over FILLED cells ─► GLOBAL-per-channel window
                     (one window per channel across all wells, so wells stay comparable)
        window each channel to [0,1], composite additively via display_color ─► RGB uint8
        write plate_montage.png  +  plate_montage.json (region-jump: well id -> cell bbox)

Why global-per-channel contrast: a montage is for comparing wells at a glance; a per-well
window would make a dim well and a bright well look identical. Why downsample ``array 0``
(not a pyramid level): IMA-184 writes a single resolution level — the per-FOV pyramid is
deferred to IMA-193 — so the montage downsamples the full-res array itself.

Fail loud: a path that is not an HCS plate, a well whose field array is missing, or a
channel with no resolvable color is refused, never rendered as a silent blank/black.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorstore as ts

# Montage cell size (downsampled well thumbnail, px). 128 gives a legible 1536wp mosaic
# (32x48 wells -> ~4096x6144) while the per-channel canvas stays a few hundred MB.
_DEFAULT_CELL_PX = 128
# Contrast percentiles (per channel, across all wells): clip the darkest 1% and brightest
# 0.2% so a few hot pixels don't crush the window. Comparable to ndv's auto-scale.
_DEFAULT_PERCENTILES = (1.0, 99.8)


# --- plate metadata (read the self-describing zarr groups) ----------------------------------

def _read_group_ome(group_dir: Path) -> dict:
    """Return the ``attributes.ome`` dict of a zarr v3 group, or {} if absent."""
    doc = json.loads((group_dir / "zarr.json").read_text())
    return doc.get("attributes", {}).get("ome", {})


def _resolve_plate_dir(plate_path) -> Path:
    """Accept either the ``plate.ome.zarr`` itself or the dir ``write_plate`` wrote it into."""
    p = Path(plate_path)
    if (p / "zarr.json").exists() and "plate" in _read_group_ome(p):
        return p
    if (p / "plate.ome.zarr").is_dir():
        return p / "plate.ome.zarr"
    raise ValueError(
        f"{plate_path!s} is not an OME-NGFF HCS plate (no plate.ome.zarr / plate group metadata). "
        "Point build_montage at write_plate's output directory or its plate.ome.zarr."
    )


def _read_open_store(array_dir: Path) -> ts.TensorStore:
    return ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(array_dir)}}, open=True
    ).result()


class _PlateLayout:
    """The grid + per-well field paths + channels, parsed once from the plate's own metadata."""

    def __init__(self, plate_dir: Path):
        self.plate_dir = plate_dir
        plate = _read_group_ome(plate_dir).get("plate")
        if not plate:
            raise ValueError(f"{plate_dir!s} has no OME plate metadata (attributes.ome.plate).")
        self.rows = [r["name"] for r in plate["rows"]]
        self.cols = [c["name"] for c in plate["columns"]]
        # each well: (well_id, row_name, col_name, row_index, col_index, first_field_path)
        self.wells: list[tuple] = []
        for w in plate["wells"]:
            row_name, col_name = w["path"].split("/")
            well_dir = plate_dir / row_name / col_name
            images = _read_group_ome(well_dir).get("well", {}).get("images", [])
            if not images:
                raise ValueError(f"well {row_name}{col_name} has no images in its well metadata.")
            self.wells.append(
                (
                    row_name + col_name,
                    row_name,
                    col_name,
                    w["rowIndex"],
                    w["columnIndex"],
                    well_dir / str(images[0]["path"]),  # montage shows the first field per well
                )
            )
        if not self.wells:
            raise ValueError(f"{plate_dir!s} plate metadata lists no wells.")
        # Channels (label + color) come from the first field's omero — identical across fields.
        omero = _read_group_ome(self.wells[0][5]).get("omero")
        if not omero or not omero.get("channels"):
            raise ValueError(f"field {self.wells[0][5]!s} has no omero channel metadata.")
        self.channels = omero["channels"]


# --- pixel ops -------------------------------------------------------------------------------

def _area_downsample(plane: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Area-average *plane* (Y, X) down to (out_h, out_w) — anti-aliased, arbitrary sizes.

    Uses ``np.add.reduceat`` to sum contiguous row/column blocks (bin edges spread as evenly
    as integer division allows), then divides by the per-bin element count. Averaging (not
    striding) so a thumbnail reflects the whole cell, not one sampled pixel.
    """
    y, x = plane.shape
    if out_h >= y and out_w >= x:
        return plane.astype(np.float32, copy=False)
    row_edges = (np.arange(out_h) * y) // out_h
    col_edges = (np.arange(out_w) * x) // out_w
    row_counts = np.diff(np.append(row_edges, y))
    col_counts = np.diff(np.append(col_edges, x))
    summed = np.add.reduceat(plane.astype(np.float32), row_edges, axis=0)
    summed = np.add.reduceat(summed, col_edges, axis=1)
    return summed / (row_counts[:, None] * col_counts[None, :])


def _window(channel_plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear contrast window [lo, hi] -> [0, 1]; guards a degenerate (all-equal) channel."""
    span = hi - lo
    if span <= 0:  # empty / flat channel — nothing to stretch, avoid divide-by-zero
        return np.zeros_like(channel_plane, dtype=np.float32)
    return np.clip((channel_plane - lo) / span, 0.0, 1.0)


def _hex_to_rgb01(hex_color: str) -> np.ndarray:
    """'#20ADF8' / '20ADF8' -> float RGB in [0, 1]. Fail loud on a malformed color."""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        raise ValueError(f"channel display color {hex_color!r} is not a 6-digit hex RGB.")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


# --- public entry ----------------------------------------------------------------------------

def build_montage(
    plate_path,
    out_dir=None,
    *,
    cell_px: int = _DEFAULT_CELL_PX,
    percentiles: tuple[float, float] = _DEFAULT_PERCENTILES,
    t: int = 0,
) -> dict:
    """Render a static whole-plate montage (thumbnail mosaic) from an OME-zarr HCS plate.

    Parameters
    ----------
    plate_path:
        ``write_plate``'s output directory, or its ``plate.ome.zarr`` directly.
    out_dir:
        Where to write ``plate_montage.png`` + ``plate_montage.json`` (default: the directory
        containing ``plate.ome.zarr``).
    cell_px:
        Downsampled thumbnail size per well (square). Bounds the montage resolution and thus
        peak memory (the canvas is ``n_rows*cell_px x n_cols*cell_px``, not the full plate).
    percentiles:
        ``(low, high)`` percentile clip for the GLOBAL-per-channel contrast window.
    t:
        Timepoint to render (default 0). A montage is a single-timepoint overview.

    Returns
    -------
    dict
        Manifest: ``{"montage", "sidecar", "n_wells", "grid": (n_rows, n_cols), "cell_px"}``.

    Raises
    ------
    ValueError
        Not an HCS plate, a well missing its field array, or an unresolvable channel color.
    """
    from PIL import Image  # tried-and-true PNG encoder; imported lazily so import squidmip stays light

    if cell_px < 1:
        raise ValueError(f"cell_px must be >= 1, got {cell_px}")

    plate_dir = _resolve_plate_dir(plate_path)
    layout = _PlateLayout(plate_dir)
    out_dir = Path(out_dir) if out_dir is not None else plate_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols, n_ch = len(layout.rows), len(layout.cols), len(layout.channels)
    colors = np.stack([_hex_to_rgb01(c["color"]) for c in layout.channels])  # (C, 3), fail-loud

    # Per-channel downsampled mosaic canvas + a filled-cell mask. Canvas is bounded by the
    # montage resolution (n_rows*cell x n_cols*cell), NOT by the full-res plate.
    canvas = np.zeros((n_ch, n_rows * cell_px, n_cols * cell_px), dtype=np.float32)
    filled = np.zeros((n_rows * cell_px, n_cols * cell_px), dtype=bool)
    placements: list[dict] = []

    # --- single streaming pass: one well resident at a time -------------------------------
    for well_id, row_name, col_name, r_i, c_i, field_dir in layout.wells:
        store = _read_open_store(field_dir / "0")
        shape = store.shape  # (T, C, 1, Y, X)
        ti = min(int(t), shape[0] - 1)
        well = np.asarray(store[ti, :, 0].read().result())  # (C, Y, X) — this well only
        if well.shape[0] != n_ch:
            raise ValueError(
                f"well {well_id} field has C={well.shape[0]} but plate omero lists {n_ch} channels."
            )
        y0, x0 = r_i * cell_px, c_i * cell_px
        for ch in range(n_ch):
            canvas[ch, y0 : y0 + cell_px, x0 : x0 + cell_px] = _area_downsample(
                well[ch], cell_px, cell_px
            )
        filled[y0 : y0 + cell_px, x0 : x0 + cell_px] = True
        placements.append(
            {
                "well_id": well_id, "row": row_name, "col": col_name,
                "row_index": r_i, "col_index": c_i,
                "x0": int(x0), "y0": int(y0), "x1": int(x0 + cell_px), "y1": int(y0 + cell_px),
            }
        )
        del well  # release the full-res well before the next read (bounded memory)

    # --- global per-channel contrast, then composite to RGB -------------------------------
    rgb = np.zeros((n_rows * cell_px, n_cols * cell_px, 3), dtype=np.float32)
    windows = []
    for ch in range(n_ch):
        vals = canvas[ch][filled]  # only real well pixels drive the window (blanks would skew it)
        if vals.size:
            lo, hi = np.percentile(vals, percentiles)
        else:
            lo, hi = 0.0, 1.0
        windows.append((float(lo), float(hi)))
        scaled = _window(canvas[ch], lo, hi)  # (H, W) in [0, 1]
        rgb += scaled[:, :, None] * colors[ch][None, None, :]  # additive composite
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    # Blank (never-filled) cells stay pure black — a viewer reads them as "no well here".

    montage_path = out_dir / "plate_montage.png"
    Image.fromarray(rgb, mode="RGB").save(montage_path)

    sidecar_path = out_dir / "plate_montage.json"
    sidecar = {
        "montage": montage_path.name,
        "cell_px": int(cell_px),
        "timepoint": int(t),
        "grid": {"n_rows": n_rows, "n_cols": n_cols, "rows": layout.rows, "columns": layout.cols},
        "channels": [
            {"label": c.get("label"), "color": str(c["color"]).lstrip("#"),
             "window": {"low": windows[i][0], "high": windows[i][1]}}
            for i, c in enumerate(layout.channels)
        ],
        "wells": placements,  # region-jump: map a montage pixel/click back to a well id
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    return {
        "montage": str(montage_path),
        "sidecar": str(sidecar_path),
        "n_wells": len(layout.wells),
        "grid": (n_rows, n_cols),
        "cell_px": int(cell_px),
    }

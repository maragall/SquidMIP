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
                                 +  plate_montage.html (zero-dep viewer: hover a cell -> well id)

Why global-per-channel contrast: a montage is for comparing wells at a glance; a per-well
window would make a dim well and a bright well look identical. The montage downsamples
``array 0`` (full-res) directly; the writer now also emits per-FOV pyramid levels, but the
montage reads level 0 to stay independent of the pyramid's level choices.

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


# --- hover viewer (self-contained HTML over the montage + region-jump sidecar) ----------------

# The montage sits under a top bar; row (A..) + column (1..) labels frame it; black grid lines
# (the background color) separate the wells into tiles. Hovering a well draws a thin RED box on
# that cell and shows the region id in LARGE text in the bar ABOVE the montage (never over the
# wells). A cursor is mapped to a well purely from the sidecar geometry, so it needs no server.
# Full-res-on-click (detail) remains the Plate View navigator ticket.
_VIEWER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
  :root{--bg:#070a0f;--border:#232b3a;--ink:#e6edf3;--muted:#8b98ad;--faint:#5b6675;--accent:#58a6ff;--hdr:46px;--colh:30px;--grid:#000}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);overflow:hidden;
    font:13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
  header{display:flex;align-items:center;gap:18px;padding:8px 18px;border-bottom:1px solid var(--border);height:58px}
  h1{font-size:13px;font-weight:700;margin:0;color:var(--muted);letter-spacing:.02em;white-space:nowrap}
  /* the region readout: LARGE text, in the bar ABOVE the montage (never overlaps the wells) */
  #readout{font-size:clamp(24px,3vw,40px);font-weight:800;letter-spacing:.01em;color:var(--ink);
    font-variant-numeric:tabular-nums;min-width:5ch}
  #readout .empty{color:var(--faint)}
  #readout .idle{color:var(--faint);font-size:15px;font-weight:600}
  #readout small{font-size:.42em;font-weight:600;color:var(--faint);margin-left:10px;text-transform:uppercase;letter-spacing:.08em}
  .right{display:flex;align-items:center;gap:20px;margin-left:auto}
  .legend{display:flex;gap:13px;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .sw{width:10px;height:10px;border-radius:50%}
  .zoom{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:11.5px}.zoom input{width:130px}
  #plate{position:absolute;top:57px;left:0;right:0;bottom:0;overflow:auto;background:var(--bg)}
  #grid{display:grid;grid-template-columns:var(--hdr) max-content;grid-template-rows:var(--colh) max-content;width:max-content}
  .corner{position:sticky;top:0;left:0;z-index:6;background:var(--bg);border-right:1px solid var(--border);border-bottom:1px solid var(--border)}
  #colruler{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--border)}
  #rowruler{position:sticky;left:0;z-index:5;background:var(--bg);border-right:1px solid var(--border)}
  .lab{position:absolute;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:600;color:var(--muted);overflow:hidden}
  .lab.on{color:var(--accent);font-weight:800}
  #stage{position:relative;line-height:0}
  #montage{display:block}
  #lines{position:absolute;inset:0;pointer-events:none;z-index:1}   /* black grid lines between wells */
  #box{position:absolute;display:none;border:2px solid #ff2d2d;pointer-events:none;z-index:3}
</style></head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div id="readout"><span class="idle">hover a well</span></div>
  <div class="right">
    <div class="legend" id="legend"></div>
    <label class="zoom">Zoom <input type="range" id="zoom" min="8" max="140"/></label>
  </div>
</header>
<div id="plate">
  <div id="grid">
    <div class="corner"></div>
    <div id="colruler"></div>
    <div id="rowruler"></div>
    <div id="stage">
      <img id="montage" src="__PNG__" alt="plate montage"/>
      <div id="lines"></div>
      <div id="box"></div>
    </div>
  </div>
</div>
<script>
const D = __DATA__;
const NR = D.grid.n_rows, NC = D.grid.n_cols, byRC = {};
for (const w of D.wells) byRC[w.row_index + "," + w.col_index] = w;
const stage = document.getElementById("stage"), img = document.getElementById("montage"),
      colr = document.getElementById("colruler"), rowr = document.getElementById("rowruler"),
      lines = document.getElementById("lines"), box = document.getElementById("box"),
      readout = document.getElementById("readout"), zoom = document.getElementById("zoom");

// compact channel legend: color dot + wavelength (parsed from the channel label when present)
document.getElementById("legend").innerHTML = (D.channels || []).map(c => {
  const m = (c.label || "").match(/(\\d{3,4})/); const t = m ? m[1] : (c.label || "");
  return '<span><i class="sw" style="background:#' + c.color + '"></i>' + t + "</span>";
}).join("");

const colLabs = [], rowLabs = [];
for (let c = 0; c < NC; c++){ const el = document.createElement("div"); el.className = "lab"; el.textContent = D.grid.columns[c];
  colr.appendChild(el); colLabs.push(el); }
for (let r = 0; r < NR; r++){ const el = document.createElement("div"); el.className = "lab"; el.textContent = D.grid.rows[r];
  rowr.appendChild(el); rowLabs.push(el); }

let Dc = 20;  // displayed px per well
function layout(){
  const W = NC*Dc, H = NR*Dc;
  img.style.width = W+"px"; img.style.height = H+"px"; stage.style.width = W+"px"; stage.style.height = H+"px";
  colr.style.width = W+"px"; rowr.style.height = H+"px";
  // black grid lines every Dc px (1px lines, the background color) so each well reads as a tile
  // 3px black gutters between wells; a well will hold a multi-FOV grid later (IMA-187)
  lines.style.backgroundImage = "linear-gradient(to right,var(--grid) 3px,transparent 3px),linear-gradient(to bottom,var(--grid) 3px,transparent 3px)";
  lines.style.backgroundSize = Dc+"px "+Dc+"px";
  for (let c=0;c<NC;c++){ const e=colLabs[c]; e.style.left=(c*Dc)+"px"; e.style.top="0"; e.style.width=Dc+"px"; e.style.height="var(--colh)"; }
  for (let r=0;r<NR;r++){ const e=rowLabs[r]; e.style.top=(r*Dc)+"px"; e.style.left="0"; e.style.height=Dc+"px"; e.style.width="var(--hdr)"; }
}
function fitZoom(){ const a = document.getElementById("plate").clientWidth - 44; return Math.max(8, Math.min(140, Math.floor(a/NC))); }

let on = {c:-1,r:-1};
function clearLabs(){ if(on.c>=0) colLabs[on.c].classList.remove("on"); if(on.r>=0) rowLabs[on.r].classList.remove("on"); on={c:-1,r:-1}; }
function hide(){ box.style.display="none"; clearLabs(); readout.innerHTML = '<span class="idle">hover a well</span>'; }
stage.addEventListener("mousemove", e => {
  const r = stage.getBoundingClientRect();
  const ci = Math.floor((e.clientX-r.left)/Dc), ri = Math.floor((e.clientY-r.top)/Dc);
  if (ci<0||ri<0||ci>=NC||ri>=NR){ hide(); return; }
  box.style.display="block"; box.style.left=(ci*Dc)+"px"; box.style.top=(ri*Dc)+"px"; box.style.width=Dc+"px"; box.style.height=Dc+"px";
  clearLabs(); colLabs[ci].classList.add("on"); rowLabs[ri].classList.add("on"); on={c:ci,r:ri};
  const w = byRC[ri+","+ci];  // well id already encodes row+col, so don't repeat it
  readout.innerHTML = w ? (w.well_id)
                        : ('<span class="empty">'+D.grid.rows[ri]+D.grid.columns[ci]+'</span><small>empty</small>');
});
stage.addEventListener("mouseleave", hide);
zoom.addEventListener("input", () => { Dc = +zoom.value; layout(); hide(); });
function init(){ Dc = fitZoom(); zoom.value = Dc; layout(); }
if (img.complete) init(); else img.addEventListener("load", init);
</script>
</body></html>
"""


def _write_viewer_html(out_dir: Path, png_name: str, sidecar: dict, title: str) -> Path:
    """Emit the self-contained hover viewer next to the montage PNG."""
    html = (
        _VIEWER_HTML.replace("__TITLE__", title)
        .replace("__PNG__", png_name)
        .replace("__DATA__", json.dumps(sidecar))
    )
    path = out_dir / "plate_montage.html"
    path.write_text(html)
    return path


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
    from PIL import Image  # tried-and-true PNG encoder; imported lazily so import squidhcs stays light

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
            tile = _area_downsample(well[ch], cell_px, cell_px)   # never upsamples: a field smaller
            th, tw = tile.shape                                   # than cell_px stays its own size
            canvas[ch, y0 : y0 + th, x0 : x0 + tw] = tile         # corner-place by ACTUAL shape (no
            #                                    broadcast crash / no divide-by-zero on small fields)
        filled[y0 : y0 + th, x0 : x0 + tw] = True
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

    # self-contained hover-indicator viewer (uses the sidecar geometry; no server, no deps)
    viewer_path = _write_viewer_html(out_dir, montage_path.name, sidecar, title="SquidHCS plate montage")

    return {
        "montage": str(montage_path),
        "sidecar": str(sidecar_path),
        "viewer": str(viewer_path),
        "n_wells": len(layout.wells),
        "grid": (n_rows, n_cols),
        "cell_px": int(cell_px),
    }

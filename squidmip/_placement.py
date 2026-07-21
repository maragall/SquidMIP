"""Coordinate placement: stage micrometres -> pixel offsets (IMA-187).

The pure, GUI-free half of the multi-FOV mosaic. Everything here is arithmetic on
``fov_positions_um`` (from the reader) plus one scalar, ``pixel_size_um`` — no Qt, no I/O, no
numpy dependency on image content — so placement correctness is asserted numerically in
tests rather than eyeballed on a rendered plate.

Why that matters: placement bugs do not raise. They draw a plausible-but-wrong picture.
The three that actually happen::

    scale error      every offset off by a constant factor -> mosaic uniformly too tight/loose
    Y-axis flip      stage y mapped to decreasing row      -> mosaic mirrored vertically
    wrong origin     origin taken plate-wide, not per-region -> every tile shifted by a constant

A test that counts tiles catches none of them; a test that compares integer pixel offsets
catches all three. Hence this module.

Units: input positions are stage MICROMETRES (``metadata["fov_positions_um"]``; the reader
converts from the mm that coordinates.csv records). This module used to carry the ``* 1000``
mm->µm conversion itself, which meant an unsuffixed millimetre value travelled through the
metadata dict — the exact silent-1000x hazard the ``_um`` naming rule exists to prevent.

Geometry::

    stage (um), origin = per-region min          image (px), origin = mosaic top-left
    ┌─────────────────────────►  +x              ┌─────────────────────────►  +col
    │   (x0,y0)   (x1,y0)                        │   fov0      fov1
    │      ▪         ▪                           │    ▪         ▪
    │   (x0,y1)   (x1,y1)                        │   fov6      fov7
    │      ▪         ▪                           │    ▪         ▪
    ▼  +y                                        ▼  +row

    col_px = (x_um - min_x_um) / pixel_size_um
    row_px = (y_um - min_y_um) / pixel_size_um * _Y_SIGN

``_Y_SIGN`` is +1: Squid rasters +x/+y and image rows increase downward, so increasing stage
y maps to increasing row. It is a named module constant rather than an inline sign so the
convention is greppable and a future stage with the opposite handedness is a one-line change
with an obvious test to flip.

The origin is **per region**, not plate-wide: each well's mosaic is laid out in its own local
frame, and the well's position on the plate comes from the row/column grid the plate view
already draws. Mixing the two coordinate systems (stage-absolute FOVs inside a grid-placed
cell) is exactly the "wrong origin" bug above.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

# Stage +y maps to image +row (downward). See the module docstring.
_Y_SIGN = 1


def _require_pixel_size(pixel_size_um: Optional[float]) -> float:
    """Validate the mm->px conversion factor, failing loud on the values that silently ruin a mosaic.

    ``pixel_size_um`` is ``Optional`` throughout the metadata layer (``_acquisition.py`` returns
    ``objective.get("pixel_size_um")``, which is ``None`` on a dataset without it). A ``None`` here
    would make every offset ``None`` or zero and collapse the mosaic into a single stacked pile —
    with no error. Refuse instead.
    """
    if pixel_size_um is None:
        raise ValueError(
            "pixel_size_um is required to place FOVs by stage coordinate, but the acquisition "
            "metadata has none. Without it, micrometres cannot be converted to pixels and every "
            "FOV would be drawn at the same spot. Add objective.pixel_size_um to acquisition.yaml."
        )
    p = float(pixel_size_um)
    if not p > 0:
        raise ValueError(f"pixel_size_um must be > 0, got {pixel_size_um!r}.")
    return p


def fov_offsets_px(
    positions_um: Mapping[tuple, tuple],
    region: str,
    fovs: Iterable[int],
    pixel_size_um: Optional[float],
) -> dict[int, tuple[int, int]]:
    """Pixel offset of each FOV's top-left corner, relative to the region's own mosaic origin.

    Parameters
    ----------
    positions_um:
        ``{(region, fov): (x_um, y_um)}`` — ``reader.metadata["fov_positions_um"]``, stage
        MICROMETRES. Passing millimetres here silently shrinks the mosaic 1000x.
    region:
        The well / region being laid out.
    fovs:
        The FOVs of *region* to place (typically ``metadata["fovs_per_region"][region]``).
    pixel_size_um:
        Object-space pixel size. Required; see :func:`_require_pixel_size`.

    Returns
    -------
    dict[int, tuple[int, int]]
        ``{fov: (row_px, col_px)}``, both >= 0, with the top-left-most FOV at ``(0, 0)``.

    Raises
    ------
    KeyError
        If a requested FOV has no recorded position — a silent skip would leave a hole in the
        mosaic that looks like a failed acquisition.
    ValueError
        If *fovs* is empty, or *pixel_size_um* is unusable.
    """
    p = _require_pixel_size(pixel_size_um)
    fovs = list(fovs)
    if not fovs:
        raise ValueError(f"region {region!r}: no FOVs to place.")

    missing = [f for f in fovs if (region, f) not in positions_um]
    if missing:
        raise KeyError(
            f"region {region!r}: no stage position for FOV(s) {missing[:8]} "
            f"(have {sum(1 for k in positions_um if k[0] == region)} of {len(fovs)}). "
            "coordinates.csv and the image filenames disagree; refusing to draw a mosaic with holes."
        )

    xs = {f: float(positions_um[(region, f)][0]) for f in fovs}
    ys = {f: float(positions_um[(region, f)][1]) for f in fovs}
    x0, y0 = min(xs.values()), min(ys.values())

    out: dict[int, tuple[int, int]] = {}
    for f in fovs:
        col = (xs[f] - x0) / p
        row = (ys[f] - y0) / p * _Y_SIGN
        out[f] = (int(round(row)), int(round(col)))
    return out


def mosaic_extent_px(
    offsets: Mapping[int, tuple[int, int]],
    frame_shape: tuple[int, int],
) -> tuple[int, int]:
    """Full-resolution ``(height, width)`` of the mosaic that *offsets* + *frame_shape* describe.

    The extent is the bounding box of every placed frame, so it accounts for the real overlap
    between neighbours rather than assuming a dense grid.
    """
    if not offsets:
        raise ValueError("no offsets: nothing to size a mosaic from.")
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    h = max(r for r, _ in offsets.values()) + fh
    w = max(c for _, c in offsets.values()) + fw
    return int(h), int(w)


def cell_boxes(
    offsets: Mapping[int, tuple[int, int]],
    frame_shape: tuple[int, int],
    cell_px: int,
) -> dict[int, tuple[int, int, int, int]]:
    """Scale full-res offsets into a ``cell_px`` x ``cell_px`` thumbnail cell.

    Returns ``{fov: (top, left, height, width)}`` in cell pixels. This is what the plate view
    consumes: the mosaic is composited at THUMBNAIL scale, never at full resolution, so a
    36-FOV well costs ~``cell_px``^2 rather than the ~1 GB a full-res composite would.

    The mosaic is fitted to the cell preserving aspect ratio and centred, so a non-square
    mosaic (a 6x4 acquisition, a freeform strip) is not stretched. Every box is clamped to at
    least 1x1 px, so a FOV can never silently vanish at small cell sizes.
    """
    if cell_px < 1:
        raise ValueError(f"cell_px must be >= 1, got {cell_px}")
    mh, mw = mosaic_extent_px(offsets, frame_shape)
    fh, fw = int(frame_shape[0]), int(frame_shape[1])

    s = min(cell_px / mh, cell_px / mw)          # uniform scale; no aspect distortion
    off_y = (cell_px - mh * s) / 2.0             # centre the mosaic in the cell
    off_x = (cell_px - mw * s) / 2.0

    boxes: dict[int, tuple[int, int, int, int]] = {}
    for fov, (row, col) in offsets.items():
        top = int(round(off_y + row * s))
        left = int(round(off_x + col * s))
        h = max(1, int(round(fh * s)))
        w = max(1, int(round(fw * s)))
        top = max(0, min(top, cell_px - 1))       # keep the box inside the cell
        left = max(0, min(left, cell_px - 1))
        h = min(h, cell_px - top)
        w = min(w, cell_px - left)
        boxes[fov] = (top, left, h, w)
    return boxes

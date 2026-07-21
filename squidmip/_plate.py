"""Plate geometry: format dimensions + carrier-background calibration.

This is the single source of truth for "what shape is this plate" and "where does its
carrier artwork sit". It is deliberately Qt-free and I/O-free so the placement maths can be
unit-tested without a QApplication or a paint device.

WHY THE STAGE-MM TERMS ARE ABSENT
---------------------------------
Squid's NavigationViewer (control/core/core.py:1641-1656) computes

    origin_x_pixel = a1_x_pixel - a1_x_mm / mm_per_pixel

because it maps a LIVE STAGE POSITION onto the carrier image. This viewer is
post-acquisition and addressed by (row, col) — it never asks "where is the stage?" — so the
a1_*_mm terms cancel out entirely and placement collapses to a pure ratio. One widget cell
is exactly one well pitch, which is why the existing edge-to-edge cell grid is already
physically correct and needs no layout rework.

    scale  = (cd / well_spacing_mm) * mm_per_pixel      # display px per PNG px
    dest_x = ax + cd/2 - a1_x_pixel * scale             # A1's PNG pixel -> centre of cell (0,0)
    dest_y = ay + cd/2 - a1_y_pixel * scale

    PNG pixel space                        widget space
     (0,0)                                  (ax,ay) = lattice origin
       +----------------------+
       |  skirt               |      cell (0,0) centre == A1 pixel, at every zoom
       |   +--------------+   |
       |   | A1 · · · ·   |   |      carrier extends LEFT of ax and ABOVE ay
       |   | ·            |   |      (a1_x_pixel * scale > cd/2), so fit/centring
       |   +--------------+   |      must measure the carrier rect, not the lattice
       +----------------------+

Values are vendored from Squid ``objective_and_sample_formats/sample_formats.csv``.
Independently verified by edge-detecting every well centre in the artwork against the
predicted lattice: max error ~1.0 px, and the error does NOT grow with well index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Squid uses one scale for every 1509x1010 plate image (core.py:1654). Slides differ
# (0.1453) but have no calibrated anchors, so they get no carrier — see carrier_for().
_MM_PER_PIXEL = 0.084665

# Full (rows, cols) per well-plate format. Absorbed from the viewer's old _PLATE_DIMS so
# grid shape and carrier placement can never disagree: two tables that drift would draw one
# shape and place the artwork for another, silently.
_PLATE_DIMS: dict[int, tuple[int, int]] = {
    4: (2, 2), 6: (2, 3), 12: (3, 4), 24: (4, 6), 96: (8, 12),
    384: (16, 24), 1536: (32, 48),
}


@dataclass(frozen=True)
class CarrierSpec:
    """Everything needed to place a carrier PNG behind the well lattice.

    png_w/png_h are carried rather than read off disk so that placement stays pure — no
    file I/O, no image decode, testable in isolation.
    """

    png: str
    a1_x_pixel: float
    a1_y_pixel: float
    well_spacing_mm: float
    rows: int
    cols: int
    png_w: int = 1509
    png_h: int = 1010
    mm_per_pixel: float = _MM_PER_PIXEL

    @property
    def cells_per_png_px(self) -> float:
        """One PNG pixel, expressed in well pitches."""
        return self.mm_per_pixel / self.well_spacing_mm

    def image_path(self) -> Path:
        """Absolute path to the vendored artwork (ships inside the wheel)."""
        return Path(__file__).parent / "images" / self.png


# Only the formats the tool actually accepts. _viewer.py's _SUPPORTED_PLATES gate rejects
# everything else before a PlateOverview is ever built, so vendoring the other four
# carriers would be unreachable assets and untestable code. The anchors for 6/12/24/96 are
# already verified upstream in sample_formats.csv; add them here alongside the PNGs when
# the scope guard widens.
_CARRIERS: dict[int, CarrierSpec] = {
    384: CarrierSpec("384 well plate_1509x1010.png", 143, 106, 4.5, rows=16, cols=24),
    1536: CarrierSpec("1536 well plate_1509x1010.png", 130, 93, 2.25, rows=32, cols=48),
}


def format_key(wellplate_format) -> Optional[int]:
    """First integer in a Squid format string, or None.

    NOTE this takes the FIRST number, so "4 glass slide" resolves to 4 — a slide carrier
    masquerading as a 4-well plate. That is pre-existing viewer behaviour (it drove the old
    _PLATE_DIMS lookup too); carrier_for()'s capability check is what keeps it harmless.
    """
    m = re.search(r"(\d+)", str(wellplate_format or ""))
    return int(m.group(1)) if m else None


def plate_dims(wellplate_format) -> Optional[tuple[int, int]]:
    """(rows, cols) for a known well-plate format, else None."""
    key = format_key(wellplate_format)
    return _PLATE_DIMS.get(key) if key is not None else None


def carrier_for(wellplate_format) -> Optional[CarrierSpec]:
    """The carrier artwork for a format, or None when it has no usable calibration.

    Returning None is the normal, supported outcome — the plate view simply renders as it
    always has. This is a capability check, not an error path, and it is load-bearing:

      * "glass slide" has an all-zero row in sample_formats.csv (well_spacing_mm == 0),
        which would be a ZeroDivisionError in the scale formula, not a missing image.
      * "4 glass slide" resolves to key 4 via format_key(); there is no 4-well carrier.
      * unknown / empty / None formats resolve to no key at all.
    """
    key = format_key(wellplate_format)
    if key is None:
        return None
    spec = _CARRIERS.get(key)
    if spec is None or spec.well_spacing_mm <= 0 or spec.mm_per_pixel <= 0:
        return None
    return spec


def carrier_placement(spec: CarrierSpec, cd: float, ax: float, ay: float
                      ) -> tuple[float, float, float, float, float]:
    """Place the artwork so its A1 pixel lands on the centre of cell (0,0).

    ``cd`` is displayed px per well, ``(ax, ay)`` the lattice top-left in widget space.
    Returns ``(scale, dest_x, dest_y, dest_w, dest_h)`` as floats — callers must keep them
    floats through to QPointF/QRectF. Truncating to int here lets the carrier drift up to a
    pixel from the grid and visibly jitter while panning, with every unit test still green.
    """
    scale = (cd / spec.well_spacing_mm) * spec.mm_per_pixel
    return (scale,
            ax + cd / 2.0 - spec.a1_x_pixel * scale,
            ay + cd / 2.0 - spec.a1_y_pixel * scale,
            spec.png_w * scale,
            spec.png_h * scale)


def carrier_extent_cells(spec: CarrierSpec, nr: int, nc: int
                         ) -> tuple[float, float, float, float]:
    """Union of the lattice and the carrier, in well-pitch units.

    Returns ``(min_x, min_y, width, height)`` relative to the lattice origin (the left edge
    of cell (0,0) is x=0). min_x/min_y are NEGATIVE for real plates because the artwork
    extends left of and above A1 — roughly -2.2 and -1.5 cells for 384. Fit and centring
    must use this rather than the bare (nc, nr) lattice, or the skirt is clipped by the
    label gutters, and plates are NOT centred on their well array (24wp has a 233px left
    margin against 136px right) so a symmetric formula would be wrong in principle.
    """
    p = spec.cells_per_png_px
    left, top = 0.5 - spec.a1_x_pixel * p, 0.5 - spec.a1_y_pixel * p
    min_x, min_y = min(0.0, left), min(0.0, top)
    max_x, max_y = max(float(nc), left + spec.png_w * p), max(float(nr), top + spec.png_h * p)
    return min_x, min_y, max_x - min_x, max_y - min_y

"""Plate geometry + carrier calibration — pure, no Qt.

These cover the load-bearing maths for the carrier background (IMA-220) without a
QApplication or a paint device, which is the whole reason the placement lives in
squidmip/_plate.py rather than inside the widget.
"""

import pytest

from squidmip._plate import (_CARRIERS, _PLATE_DIMS, carrier_extent_cells, carrier_for,
                             carrier_placement, format_key, plate_dims)


# --- capability check: which formats get a carrier at all ----------------------------------

def test_supported_formats_have_a_carrier():
    for fmt in ("384 well plate", "1536 well plate"):
        spec = carrier_for(fmt)
        assert spec is not None, fmt
        assert spec.well_spacing_mm > 0 and spec.mm_per_pixel > 0


def test_glass_slide_yields_no_carrier_instead_of_dividing_by_zero():
    """sample_formats.csv's glass-slide row is all zeros. Placement divides by
    well_spacing_mm, so an unguarded lookup is a ZeroDivisionError, not a missing image."""
    assert carrier_for("glass slide") is None


def test_four_glass_slide_resolves_to_key_4_and_still_yields_no_carrier():
    """format_key takes the FIRST integer, so '4 glass slide' looks like a 4-well plate.
    That is pre-existing viewer behaviour; the capability check is what makes it harmless."""
    assert format_key("4 glass slide") == 4
    assert carrier_for("4 glass slide") is None


@pytest.mark.parametrize("fmt", [None, "", "unknown", "96 well plate", "6 well plate"])
def test_uncalibrated_or_unshipped_formats_yield_no_carrier(fmt):
    """96/6 are real formats whose PNGs are deliberately not vendored (unreachable behind
    _SUPPORTED_PLATES), so they must degrade exactly like an unknown format."""
    assert carrier_for(fmt) is None


def test_every_shipped_spec_resolves_to_a_file_on_disk():
    """Guards the packaging path: an asset outside the wheel works only on a machine that
    happens to have a Squid checkout."""
    for key, spec in _CARRIERS.items():
        p = spec.image_path()
        assert p.is_file(), f"{key}: missing {p}"
        assert p.stat().st_size > 0


# --- plate dimensions: regression against the deleted _PLATE_DIMS --------------------------

def test_plate_dims_match_the_pre_ima220_baseline():
    """REGRESSION. _viewer._PLATE_DIMS was folded into _plate.py; the baseline is hardcoded
    here precisely because the dict it used to be compared against no longer exists."""
    baseline = {4: (2, 2), 6: (2, 3), 12: (3, 4), 24: (4, 6), 96: (8, 12),
                384: (16, 24), 1536: (32, 48)}
    assert _PLATE_DIMS == baseline
    assert plate_dims("96 well plate") == (8, 12)
    assert plate_dims("1536 well plate") == (32, 48)
    assert plate_dims("glass slide") is None
    assert plate_dims(None) is None


def test_carrier_rows_cols_agree_with_plate_dims():
    """The two must never describe different plates — that was the point of folding them."""
    for key, spec in _CARRIERS.items():
        assert (spec.rows, spec.cols) == _PLATE_DIMS[key]


# --- placement ----------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["384", "1536"])
@pytest.mark.parametrize("cd", [12.0, 37.5, 400.0])
def test_a1_pixel_lands_on_the_centre_of_cell_0_0(fmt, cd):
    spec = carrier_for(fmt)
    ax, ay = 130.0, 70.0
    scale, dx, dy, _, _ = carrier_placement(spec, cd, ax, ay)
    assert dx + spec.a1_x_pixel * scale == pytest.approx(ax + cd / 2)
    assert dy + spec.a1_y_pixel * scale == pytest.approx(ay + cd / 2)


@pytest.mark.parametrize("fmt", ["384", "1536"])
@pytest.mark.parametrize("cd", [12.0, 37.5, 400.0])
def test_last_well_lands_on_the_centre_of_the_last_cell(fmt, cd):
    """The far corner is where a wrong scale would show up as accumulated drift."""
    spec = carrier_for(fmt)
    ax, ay = 130.0, 70.0
    scale, dx, dy, _, _ = carrier_placement(spec, cd, ax, ay)
    pitch_px = spec.well_spacing_mm / spec.mm_per_pixel      # PNG px between well centres
    last_x = spec.a1_x_pixel + (spec.cols - 1) * pitch_px
    last_y = spec.a1_y_pixel + (spec.rows - 1) * pitch_px
    assert dx + last_x * scale == pytest.approx(ax + cd / 2 + (spec.cols - 1) * cd)
    assert dy + last_y * scale == pytest.approx(ay + cd / 2 + (spec.rows - 1) * cd)


def test_scale_is_linear_in_zoom():
    spec = carrier_for("384")
    s1 = carrier_placement(spec, 20.0, 0, 0)[0]
    s2 = carrier_placement(spec, 40.0, 0, 0)[0]
    assert s2 == pytest.approx(2 * s1)


def test_placement_returns_floats_not_ints():
    """Truncating here lets the carrier drift up to 1px from the grid and jitter while
    panning, with every other assertion in this file still passing."""
    out = carrier_placement(carrier_for("384"), 33.3, 10.5, 20.25)
    assert all(isinstance(v, float) for v in out)
    assert out[1] != int(out[1])


# --- extent (drives fit / zoom-out floor) --------------------------------------------------

@pytest.mark.parametrize("fmt", ["384", "1536"])
def test_extent_starts_left_of_and_above_the_lattice(fmt):
    """The skirt extends past A1, so the extent origin is negative. If fit ignored this the
    artwork would be clipped by the label gutters."""
    spec = carrier_for(fmt)
    min_x, min_y, w, h = carrier_extent_cells(spec, spec.rows, spec.cols)
    assert min_x < 0 and min_y < 0
    assert w > spec.cols and h > spec.rows


@pytest.mark.parametrize("fmt", ["384", "1536"])
def test_extent_fully_contains_both_lattice_and_artwork(fmt):
    spec = carrier_for(fmt)
    min_x, min_y, w, h = carrier_extent_cells(spec, spec.rows, spec.cols)
    p = spec.cells_per_png_px
    left, top = 0.5 - spec.a1_x_pixel * p, 0.5 - spec.a1_y_pixel * p
    assert min_x <= left and min_y <= top                       # artwork start
    assert min_x + w >= left + spec.png_w * p                   # artwork end
    assert min_x + w >= spec.cols and min_y + h >= spec.rows     # lattice end


def test_extent_is_about_119_percent_of_the_lattice():
    """Pins the accepted trade-off: honouring the carrier shrinks displayed data ~16% at
    default zoom. If this ratio moves, the zoom-out floor moved with it."""
    for fmt in ("384", "1536"):
        spec = carrier_for(fmt)
        _, _, w, h = carrier_extent_cells(spec, spec.rows, spec.cols)
        assert w / spec.cols == pytest.approx(1.183, abs=0.01)
        assert h / spec.rows == pytest.approx(1.1875, abs=0.01)

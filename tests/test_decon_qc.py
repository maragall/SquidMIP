"""IMA-252: the semi-convergence QC tool's measurement, not its plumbing.

The tool's job is to be BELIEVABLE about a turning point, so the tests are about the two
ways it could lie: reporting a minimum that is really just the end of the sweep, and
reporting a halo that is not the halo. Both are checked against constructed volumes where
the right answer is known by construction, so a test can fail for a reason.

No petakit and no real acquisition here - deconvolution itself is tested in
``test_decon.py``. These tests exercise the QC layer over synthetic volumes, which is why
they run in milliseconds.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "decon_qc", Path(__file__).resolve().parent.parent / "tools" / "decon_qc.py")
decon_qc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(decon_qc)


DXY, DZ = 0.752, 1.5
CORE_UM = 0.61 * 0.525 / 0.3          # the NA-0.3 Airy radius, 1.0675 um
WINDOW_UM = 6.0


def _volume(halo_level, shape=(11, 64, 64), core_level=1000.0):
    """A bright core at the centre plus a uniform halo at *halo_level*, on a zero floor."""
    volume = np.zeros(shape, dtype=np.float32)
    zc, yc, xc = shape[0] // 2, shape[1] // 2, shape[2] // 2
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    r = np.sqrt(((zz - zc) * DZ) ** 2 + ((yy - yc) * DXY) ** 2 + ((xx - xc) * DXY) ** 2)
    volume[r <= WINDOW_UM] = halo_level
    volume[r <= CORE_UM] = core_level
    return volume, (zc, yc, xc)


# --- the metric ------------------------------------------------------------------------
def test_ratio_is_halo_brightness_over_core_brightness():
    """The number is a plain brightness ratio, so it can be checked by construction."""
    volume, centre = _volume(halo_level=200.0, core_level=1000.0)
    got = decon_qc.halo_core_ratio(volume, centre, DXY, DZ, CORE_UM, WINDOW_UM)
    assert got == pytest.approx(0.2, abs=1e-6)


def test_a_brighter_halo_scores_higher():
    """'The disc is growing again' has to move the number UP, or the curve means nothing."""
    dim, centre = _volume(halo_level=100.0)
    bright, _ = _volume(halo_level=400.0)
    assert (decon_qc.halo_core_ratio(bright, centre, DXY, DZ, CORE_UM, WINDOW_UM)
            > decon_qc.halo_core_ratio(dim, centre, DXY, DZ, CORE_UM, WINDOW_UM))


def test_a_constant_camera_offset_does_not_change_the_answer():
    """The floor subtraction has to actually neutralise the sensor's offset.

    Without it the same optics would score differently on the same sample simply because
    the camera pedestal changed, and iteration counts would not compare across channels.
    """
    volume, centre = _volume(halo_level=200.0)
    plain = decon_qc.halo_core_ratio(volume, centre, DXY, DZ, CORE_UM, WINDOW_UM)
    offset = decon_qc.halo_core_ratio(volume + 500.0, centre, DXY, DZ, CORE_UM, WINDOW_UM)
    assert offset == pytest.approx(plain, abs=1e-6)


def test_a_dark_core_is_refused_rather_than_divided_by():
    volume, centre = _volume(halo_level=0.0, core_level=0.0)
    with pytest.raises(ValueError, match="core"):
        decon_qc.halo_core_ratio(volume, centre, DXY, DZ, CORE_UM, WINDOW_UM)


# --- the window has to fit -------------------------------------------------------------
def test_window_never_exceeds_what_the_stack_can_hold_axially():
    """A 10-plane stack at 1.5 um cannot hold the preferred 8-Airy-radius sphere.

    If it were used anyway the metric would silently measure a truncated cap whose shape
    depends on where in the stack the structure happened to sit.
    """
    assert decon_qc.qc_window_um(CORE_UM, nz=10, dz_um=1.5) == pytest.approx(6.0)
    # Deep enough, and the preferred size is used unchanged.
    assert decon_qc.qc_window_um(CORE_UM, nz=40, dz_um=1.5) == pytest.approx(8 * CORE_UM)


def test_the_structure_is_picked_away_from_the_edges():
    """On the real first FOV the raw argmax lands on z=0; the window would be cut in half."""
    stack = np.zeros((10, 64, 64), dtype=np.uint16)
    stack[0, 32, 32] = 60000          # brightest overall, but unusable: top plane
    stack[5, 30, 30] = 30000          # dimmer, but a window fits around it
    z, y, x = decon_qc.brightest_structure(stack, DXY, DZ, CORE_UM, z_margin=4, xy_margin=8)
    assert 4 <= z < 6
    assert (int(y), int(x)) == (30, 30)


# --- the recommendation ----------------------------------------------------------------
def test_an_interior_minimum_is_reported_as_a_real_turn():
    best, kind, message = decon_qc.recommend([1, 2, 3, 4, 5], [0.9, 0.7, 0.5, 0.6, 0.8])
    assert (best, kind) == (3, "turn")
    assert "RECOMMENDATION: 3" in message


def test_a_still_falling_curve_is_not_dressed_up_as_a_turning_point():
    """The failure this guards: argmin of a monotone curve is just where the sweep ended.

    This is the case the real tissue data actually produces at 1..8, so the tool must say
    'no turn' rather than confidently recommending 8.
    """
    best, kind, message = decon_qc.recommend([1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6])
    assert (best, kind) == (4, "still-falling")
    assert "NO TURN" in message and "RECOMMENDATION" not in message


def test_a_curve_that_only_rises_says_so():
    best, kind, message = decon_qc.recommend([1, 2, 3, 4], [0.6, 0.7, 0.8, 0.9])
    assert (best, kind) == (1, "rising")
    assert "NO TURN" in message


# --- the picture -----------------------------------------------------------------------
def test_orthogonal_slices_are_xz_and_yz_through_the_structure():
    volume = np.zeros((5, 20, 30), dtype=np.float32)
    volume[2, 7, 11] = 1.0
    xz, yz = decon_qc.orthogonal_slices(volume, (2, 7, 11))
    assert xz.shape == (5, 30) and yz.shape == (5, 20)
    assert xz[2, 11] == 1.0 and yz[2, 7] == 1.0


def test_the_montage_view_is_cropped_around_the_structure():
    volume = np.zeros((5, 60, 60), dtype=np.float32)
    volume[2, 30, 30] = 1.0
    xz, yz = decon_qc.orthogonal_slices(volume, (2, 30, 30), half=8)
    assert xz.shape == (5, 16) and yz.shape == (5, 16)


def test_display_puts_background_at_the_bottom_of_the_colormap():
    """Turbo's low end has to land on the background, or the halo is lost in green."""
    panel = np.full((10, 40), 500.0)
    panel[5, 20] = 5000.0
    shown = decon_qc._display(panel)
    assert shown.max() == pytest.approx(1.0)
    assert shown[0, 0] == pytest.approx(0.0)


def test_the_montage_has_one_row_per_iteration_and_two_columns(tmp_path):
    """End to end on the rendering path: rows = iterations, columns = [xz, yz]."""
    matplotlib = pytest.importorskip("matplotlib")
    volume, centre = _volume(halo_level=200.0)
    rows = [("raw", volume), ("1", volume), ("2", volume)]
    out = tmp_path / "montage.png"
    decon_qc.write_montage(out, rows, centre, DXY, DZ, "test", view_half=8)
    assert out.exists() and out.stat().st_size > 0

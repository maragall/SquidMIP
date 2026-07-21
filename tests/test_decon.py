"""Unit tests for IMA-223: Richardson-Lucy deconvolution as a per-plane pre-filter to the MIP.

Covers the a-priori design contracts:
  * richardson_lucy() — measurably sharpens a synthetic blur (Tenengrad up), shape- and
    dtype-preserving, uint16 clipped (never wrapped), loud on bad input.
  * deconvolve_planes() — lazy/streaming, so the projector's memory stays plane-bounded.
  * project_decon() — a plain Projector: decon-then-MIP, registered as "decon", same
    (shape, dtype) contract as project().
  * REGRESSION — plain MIP is untouched by decon's arrival (same projector object, same pixels).
  * The optional SciPy fast path and the NumPy fallback agree, and neither is imported by
    ``import squidmip``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from squidmip import _engine as engine
from squidmip import open_reader, project, project_plate
from squidmip._decon import (
    DEFAULT_ITERATIONS,
    DEFAULT_SIGMA_PX,
    _gaussian_blur,
    deconvolve_planes,
    make_decon_projector,
    project_decon,
    richardson_lucy,
)
from squidmip.projection import _tenengrad


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _dots(shape=(64, 64), dtype=np.uint16) -> np.ndarray:
    """A sharp synthetic 'object': a sparse grid of bright point sources on a dim background."""
    img = np.full(shape, 100.0, dtype=np.float32)
    img[8::16, 8::16] = 8000.0
    return img.astype(dtype)


def _blurred(shape=(64, 64), sigma=2.0, dtype=np.uint16) -> np.ndarray:
    """The same object seen through a PSF — what the microscope actually records."""
    return _gaussian_blur(_dots(shape, np.float32), sigma).astype(dtype)


# --------------------------------------------------------------------------------------
# richardson_lucy — the primitive
# --------------------------------------------------------------------------------------
def test_deconvolution_sharpens_a_blurred_image():
    blurry = _blurred(sigma=2.0)
    sharpened = richardson_lucy(blurry, sigma_px=2.0, iterations=25)
    assert _tenengrad(sharpened) > _tenengrad(blurry)


def test_deconvolution_raises_peak_contrast_toward_the_original():
    """Sharper is not just 'more gradient': the point sources recover height they lost to the blur."""
    truth, blurry = _dots(), _blurred(sigma=2.0)
    sharpened = richardson_lucy(blurry, sigma_px=2.0, iterations=25)
    assert blurry.max() < sharpened.max() <= truth.max() * 1.2


def test_shape_and_dtype_are_preserved():
    for dtype in (np.uint8, np.uint16, np.float32):
        plane = _blurred(shape=(32, 48), dtype=dtype)
        out = richardson_lucy(plane, iterations=3)
        assert out.shape == plane.shape
        assert out.dtype == plane.dtype


def test_uint16_saturation_clips_instead_of_wrapping():
    """RL can push bright pixels past the dtype ceiling; that must clip, never wrap to zero."""
    plane = np.full((16, 16), 65000, dtype=np.uint16)
    plane[8, 8] = 65535
    out = richardson_lucy(plane, sigma_px=2.0, iterations=30)
    assert out.dtype == np.uint16
    assert out.max() <= 65535
    assert out.min() >= 60000          # a wrap would have produced near-zero pixels


def test_input_plane_is_not_mutated():
    plane = _blurred()
    before = plane.copy()
    richardson_lucy(plane, iterations=3)
    assert np.array_equal(plane, before)


def test_output_is_non_negative_for_float_input():
    out = richardson_lucy(_blurred(dtype=np.float32), iterations=5)
    assert out.min() >= 0.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"sigma_px": 0}, "sigma_px must be > 0"),
        ({"sigma_px": -1.0}, "sigma_px must be > 0"),
        ({"iterations": 0}, "iterations must be >= 1"),
    ],
)
def test_bad_parameters_raise_named(kwargs, match):
    with pytest.raises(ValueError, match=match):
        richardson_lucy(_blurred(), **kwargs)


def test_non_2d_input_raises():
    with pytest.raises(ValueError, match="2-D plane"):
        richardson_lucy(np.zeros((2, 8, 8), dtype=np.uint16))


# --------------------------------------------------------------------------------------
# deconvolve_planes / the projector composition
# --------------------------------------------------------------------------------------
def test_deconvolve_planes_is_lazy():
    """A generator, not a list: the projector must never materialise the whole z-stack."""
    consumed = []

    def source():
        for i in range(3):
            consumed.append(i)
            yield _blurred(shape=(16, 16))

    gen = deconvolve_planes(source(), iterations=2)
    assert consumed == []              # nothing pulled yet
    next(gen)
    assert consumed == [0]             # exactly one plane pulled per output plane


def test_project_decon_matches_the_projector_contract():
    planes = [_blurred(shape=(32, 32)) for _ in range(3)]
    out = project_decon(iter(planes))
    assert out.shape == planes[0].shape
    assert out.dtype == planes[0].dtype


def test_decon_mip_is_sharper_than_plain_mip():
    """The acceptance oracle: decon -> MIP beats MIP on the metric project_reference already uses."""
    planes = [_blurred(sigma=s) for s in (1.5, 2.0, 2.5)]
    plain = project(iter(planes))
    decon = project_decon(iter(planes))
    assert _tenengrad(decon) > _tenengrad(plain)


def test_project_decon_requires_at_least_one_plane():
    with pytest.raises(ValueError, match="at least one plane"):
        project_decon(iter([]))


def test_make_decon_projector_binds_parameters_and_reduce():
    planes = [_blurred(shape=(32, 32)) for _ in range(2)]
    weak = make_decon_projector(sigma_px=DEFAULT_SIGMA_PX, iterations=1)(iter(planes))
    strong = make_decon_projector(sigma_px=2.0, iterations=DEFAULT_ITERATIONS)(iter(planes))
    assert _tenengrad(strong) > _tenengrad(weak)

    first = make_decon_projector(iterations=2, reduce=lambda ps: next(iter(ps)))(iter(planes))
    assert first.shape == planes[0].shape          # the reduce= seam is honoured


def test_decon_is_registered_in_the_projector_table():
    assert "decon" in engine.available_projectors()
    assert engine._resolve_projector("decon") is project_decon


def test_decon_runs_end_to_end_through_project_plate(squid_dataset):
    """Selectable by name through the existing engine seam, with the plate's own (T,C,1,Y,X)."""
    root, _ = squid_dataset
    reader = open_reader(str(root))
    decon = {(r, f): img for r, f, img in project_plate(reader, projector="decon", workers=2)}
    mip = {(r, f): img for r, f, img in project_plate(reader, projector="mip", workers=2)}
    assert set(decon) == set(mip)
    for key, img in decon.items():
        assert img.shape == mip[key].shape     # z stays size-1: the writer needs no special-casing
        assert img.dtype == mip[key].dtype


# --------------------------------------------------------------------------------------
# regression guard — MIP must not have moved
# --------------------------------------------------------------------------------------
def test_mip_projector_is_unchanged_by_decon_registration():
    assert engine._resolve_projector("mip") is project


def test_plain_mip_pixels_are_unchanged():
    planes = [_blurred(sigma=s) for s in (1.5, 2.0, 2.5)]
    expected = np.maximum.reduce(planes)
    got = project(iter(planes))
    assert np.array_equal(got, expected)           # byte-identical, dtype included
    assert got.dtype == planes[0].dtype


# --------------------------------------------------------------------------------------
# optional SciPy: fast path, fallback, and the lazy-import contract
# --------------------------------------------------------------------------------------
def test_numpy_fallback_matches_scipy(monkeypatch):
    """With SciPy hidden, the self-contained NumPy blur must agree with it (same result, slower)."""
    pytest.importorskip("scipy")
    plane = _blurred(dtype=np.float32)
    with_scipy = richardson_lucy(plane, sigma_px=2.0, iterations=5)

    monkeypatch.setitem(sys.modules, "scipy.ndimage", None)   # None -> ImportError on import
    without_scipy = richardson_lucy(plane, sigma_px=2.0, iterations=5)
    assert np.allclose(with_scipy, without_scipy, rtol=1e-3, atol=1e-2)


def test_importing_squidmip_does_not_import_scipy():
    """Prior learning 'tilefusion-init-heavy-import': never drag an optional stack in at import."""
    code = "import squidmip, sys; assert 'scipy' not in sys.modules, sorted(sys.modules)[:0]"
    subprocess.run([sys.executable, "-c", code], check=True)

"""Richardson-Lucy deconvolution as a per-plane pre-filter to the z-reduction (IMA-223).

Widefield fluorescence images are the true object convolved with the microscope's point
spread function (PSF); deconvolution inverts that blur. For SquidMIP the deliverable is not
a deconvolved stack — it is a **sharper MIP**: deconvolve each plane on the way past, then
max-project, so out-of-focus haze contributes less to the projected plate.

This lands *additively*, inside the existing ``Projector`` contract (an iterable of planes in,
one plane out) — no engine surgery, no new registry shape::

    planes ──► deconvolve_planes(...) ──► reduce(...) ──► one plane   (project_decon)
                (per plane, streamed)      (MIP today)

Design contracts:
  * **Streaming, bounded memory.** Like :func:`squidmip.project`, ``project_decon`` holds one
    plane plus the running accumulator — the deconvolution happens plane-by-plane as they are
    pulled, so the per-worker footprint stays flat and the engine's window math is unchanged.
  * **Shape- and dtype-preserving.** Integer input round-trips through float32 and is rounded +
    clipped back to its own dtype range, so uint16 can neither overflow nor silently upcast.
  * **No new hard dependency.** SciPy is used for the Gaussian blur when it is installed (it is
    the fast path) and imported LAZILY inside the call, never at module import; without it a
    small self-contained NumPy separable-convolution fallback runs instead. ``import squidmip``
    therefore still imports nothing heavy.

The PSF: an isotropic 2-D Gaussian of width ``sigma_px``, the standard widefield stand-in for a
measured PSF. A Gaussian is symmetric, so its mirror equals itself and the Richardson-Lucy
update needs a single blur primitive rather than a blur + a flipped-PSF blur.

Documented tradeoff — this is a **lateral (2-D, per-plane)** deconvolution. A full 3-D
Richardson-Lucy with an axial PSF removes out-of-focus light better, but it must materialise the
whole z-stack (multi-GB per worker at 3000x3000x20) and needs a filter-then-reduce pipeline that
the current one-callable ``reduce=`` seam cannot express — that is IMA-210's ``filters=[]``
contract, and 3-D/PSF-model decon (petakit) rides on top of it. The 2-D form is the part that
composes with today's engine, costs one plane of memory, and is what ships here.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator

import numpy as np

from squidmip.projection import project

# Defaults tuned for Squid widefield frames: sigma ~1.5 px is a typical diffraction-limited spot
# radius at Squid's 0.325 um/px sampling, and RL converges usefully in ~10 iterations while staying
# short of the noise amplification that unbounded RL eventually produces. Both are adjustable.
DEFAULT_SIGMA_PX = 1.5
DEFAULT_ITERATIONS = 10

# Guard for the RL ratio step: the observed image is divided by the current re-blurred estimate, so
# an all-zero background region would divide by zero. Small next to one intensity count.
_EPS = 1e-6


def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Normalised 1-D Gaussian, truncated at 4 sigma (SciPy's own default radius)."""
    radius = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def _convolve1d(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable-convolution fallback along one axis with reflected edges (matches SciPy's mode).

    Used only when SciPy is absent. Sliding-window view + a single matmul, so it is vectorised
    rather than a Python loop, but it is still materially slower and more allocation-heavy than
    ``scipy.ndimage.gaussian_filter`` on full-size frames — SciPy is the recommended install.

    Edge handling: NumPy's ``"symmetric"`` (``d c b a | a b c d``) IS SciPy's ``mode="reflect"``.
    NumPy's own ``"reflect"`` is SciPy's ``"mirror"`` — using it here would make the two paths
    disagree at the border, which is exactly the kind of silent 2% drift a fallback must not have.
    """
    pad = kernel.size // 2
    moved = np.moveaxis(a, axis, -1)
    widths = [(0, 0)] * (moved.ndim - 1) + [(pad, pad)]
    padded = np.pad(moved, widths, mode="symmetric")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=-1)
    return np.moveaxis(windows @ kernel, -1, axis)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur *image* (float32, 2-D) with an isotropic Gaussian — the forward PSF model.

    SciPy is imported here, not at module scope, so ``import squidmip`` never pulls it in
    (and the package keeps no hard SciPy dependency); the NumPy path is the fallback.
    """
    try:
        from scipy.ndimage import gaussian_filter  # lazy: optional, and heavy-ish to import
    except ImportError:
        kernel = _gaussian_kernel(sigma)
        return _convolve1d(_convolve1d(image, kernel, axis=0), kernel, axis=1)
    return gaussian_filter(image, sigma, mode="reflect")


def richardson_lucy(
    plane: np.ndarray,
    sigma_px: float = DEFAULT_SIGMA_PX,
    iterations: int = DEFAULT_ITERATIONS,
) -> np.ndarray:
    """Richardson-Lucy deconvolve one plane against a Gaussian PSF. Same shape, same dtype.

    Richardson-Lucy is the maximum-likelihood estimator for Poisson (photon-counting) noise: it
    iterates ``est *= PSF ⊛ (observed / (PSF ⊛ est))``, which is non-negative by construction and
    conserves total intensity — the two properties that make it the standard choice for
    fluorescence. With a Gaussian PSF the mirrored PSF equals the PSF, so each iteration is two
    blurs and no explicit kernel array.

    Parameters
    ----------
    plane:
        A 2-D image (typically one z-plane of one channel), any integer or float dtype.
    sigma_px:
        Gaussian PSF standard deviation in **pixels** (default 1.5). Larger = assumes a blurrier
        microscope = more aggressive sharpening. It should approximate the real spot radius;
        overshooting it amplifies noise and can ring at high-contrast edges.
    iterations:
        RL iterations (default 10). RL converges slowly and is *not* monotonic in image quality —
        past convergence it amplifies noise into speckle — so this is deliberately a small number.

    Returns
    -------
    np.ndarray
        The deconvolved plane, same shape and dtype as *plane*. Integer dtypes are rounded and
        clipped to their own range on the way back, so uint16 can neither overflow nor wrap.

    Raises
    ------
    ValueError
        If *plane* is not 2-D, *sigma_px* <= 0, or *iterations* < 1 (each named loud rather than
        silently degrading to a no-op).
    """
    if plane.ndim != 2:
        raise ValueError(f"richardson_lucy expects a 2-D plane, got shape {plane.shape}")
    if not sigma_px > 0:
        raise ValueError(f"sigma_px must be > 0, got {sigma_px}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    dtype = plane.dtype
    observed = plane.astype(np.float32, copy=False)
    estimate = np.array(observed, dtype=np.float32, copy=True)  # own buffer; never touch the input
    for _ in range(iterations):
        blurred = _gaussian_blur(estimate, sigma_px)
        np.add(blurred, _EPS, out=blurred)          # never divide by an empty background
        estimate *= _gaussian_blur(observed / blurred, sigma_px)

    if np.issubdtype(dtype, np.integer):            # round + clip INSIDE float32, then cast: no wrap
        info = np.iinfo(dtype)
        return np.clip(np.rint(estimate), info.min, info.max).astype(dtype)
    return estimate.astype(dtype, copy=False)


def deconvolve_planes(
    planes: Iterable[np.ndarray],
    sigma_px: float = DEFAULT_SIGMA_PX,
    iterations: int = DEFAULT_ITERATIONS,
) -> Iterator[np.ndarray]:
    """Lazily deconvolve every plane of an iterable — the pre-filter stage, one plane at a time.

    A generator on purpose: it preserves the streaming contract of the projection primitives, so
    wrapping a projector in decon costs one extra plane of memory, not a whole z-stack.
    """
    for plane in planes:
        yield richardson_lucy(plane, sigma_px=sigma_px, iterations=iterations)


def make_decon_projector(
    sigma_px: float = DEFAULT_SIGMA_PX,
    iterations: int = DEFAULT_ITERATIONS,
    reduce: Callable[[Iterable[np.ndarray]], np.ndarray] = project,
) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Build a projector that deconvolves each plane and then reduces z with *reduce*.

    This is the composition seam: PSF width and iteration count are bound here, and the result is
    an ordinary ``Projector`` that ``add_projector`` accepts. A tuned variant is one line::

        add_projector("decon-strong", make_decon_projector(sigma_px=2.5, iterations=25))

    *reduce* defaults to :func:`squidmip.project` (MIP) but takes any z-reduction, so
    decon → reference-plane is available without a new wrapper.
    """
    def decon_projector(planes: Iterable[np.ndarray]) -> np.ndarray:
        return reduce(deconvolve_planes(planes, sigma_px=sigma_px, iterations=iterations))

    decon_projector.__name__ = "decon_projector"
    decon_projector.__doc__ = (
        f"Richardson-Lucy (sigma_px={sigma_px}, iterations={iterations}) per plane, then "
        f"{getattr(reduce, '__name__', reduce)!r}."
    )
    return decon_projector


# Bound once at import (cheap — it closes over two numbers and imports nothing) so every call reuses
# the same configured projector rather than rebuilding the closure per well.
_DEFAULT_DECON = make_decon_projector()


def project_decon(planes: Iterable[np.ndarray]) -> np.ndarray:
    """Deconvolve each z-plane (RL, Gaussian PSF, defaults) and max-project the result.

    The registered ``"decon"`` projector — decon-then-MIP at the documented defaults. Same
    contract as :func:`squidmip.project`: equal-shape planes in, one same-shape, same-dtype plane
    out, streamed. Costs ``2 x iterations`` Gaussian blurs per plane over plain MIP.
    """
    return _DEFAULT_DECON(planes)

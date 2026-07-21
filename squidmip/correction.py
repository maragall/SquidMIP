"""Flatfield / illumination correction as a decoration of the z-reduce seam (IMA-225).

Illumination shading — corners dimmer than centres — is multiplicative and systematic. Left
uncorrected it biases every intensity measurement across a plate. The Cephla stitcher already
computes these correction fields and saves them as a pickled ``.npy`` dict; this module lets the
MIP tool consume one (and, failing that, estimate a crude one from the data itself).

**Flatfield is not an alternative to MIP.** It reduces nothing, so it cannot be a peer of ``mip``
in ``_engine._PROJECTORS``: ``projection.py`` pins the output Z axis to size 1 and the writer, the
montage and the viewer all depend on that. A z-reduction that reduced nothing would either break
the writer or silently drop the stack. So the correction **decorates** the ``reduce=`` seam that
``project_well`` already exposes::

      flatfield .npy ──► load_flatfield ──► prepare_field ──► Field   ONCE, at run start
                                                               │      (single-threaded)
      (no file?)     ──► estimate_flatfield ─────────────────► │      immutable, shared read-only
                                                               ▼
              with_correction(reduce, field, c_index, side)  ──►  a plain reduce callable
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        │ side=BEFORE (default, always valid)  side=AFTER (opt-in)│
        │   planes ─► correct each ─► reduce   planes ─► reduce ─► correct
        │   (Nz corrections, Nz roundings)     (1 correction, 1 rounding)
        └────────────────────────────────────────────────────────┘
                                    ▼
              project_well(..., reduce=<composed>) ─► (T, C, 1, Y, X), native dtype

Nothing downstream of ``reduce`` changes. Z stays size-1. Memory stays bounded.

Why ``AFTER`` is legal for MIP
------------------------------
Per pixel the correction is ``f(x) = (x - D) / F`` with ``F > 0`` — a **non-decreasing** function
of ``x``. MIP is a per-pixel ``max``, and for any non-decreasing ``f``, ``max(f(a), f(b)) =
f(max(a, b))``, ties included. Clipping to the dtype range and truncating to the native integer
dtype are themselves non-decreasing, so the composition stays monotone and the identity survives
the integer round-trip: correcting **after** the reduction is bit-identical to correcting every
plane, at ``1/Nz`` the correction work.

This does NOT generalize. ``project_reference`` picks the sharpest plane by Tenengrad focus score,
and that pick is not licensed by monotonicity. So ``BEFORE`` is the default and ``AFTER`` is opt-in,
declared per reducer (``add_projector(..., commutes_with_scaling=True)``); today only ``mip``
declares it.

Numerics, stated
----------------
  * The profile is normalized to **mean 1.0 per channel** before use, so correction preserves the
    overall intensity scale (a stitcher-produced field is already mean-1, so this is a no-op there).
  * Non-finite (NaN/inf) and near-zero profile pixels (``<= 1e-6``) are replaced by **1.0** — those
    pixels pass through uncorrected rather than exploding to saturation. This is the same guard the
    stitcher applies, precomputed once here instead of per frame.
  * Arithmetic is float32; the result is clipped to the dtype range and truncated back, so an
    unsigned ``(I - D)`` underflow clips to 0 and an overflow clips to the max — never a wraparound.
  * **uint8/uint16 only.** ``np.clip(x, 0, iinfo.max)`` in float32 is exact for those (65535 is
    representable) but not for uint32, where ``iinfo.max`` rounds up to 2**32, the clip fails to
    bound and the cast is undefined. The reader only produces uint8/uint16 today, so
    :func:`prepare_field` asserts it rather than inheriting a silent assumption.
  * Shape validation is **exact**: ``(C, Y, X) == (n_channels, *frame_shape)``. This deliberately
    rejects binned acquisitions, cropped ROIs and fields estimated at a different sensor crop — a
    resampled field is a scientific claim we are not making. The error names both shapes.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

# Profile pixels at or below this gain are treated as "no information" and forced to 1.0 (pass
# through uncorrected). Mirrors tilefusion.flatfield.apply_flatfield's guard.
_MIN_GAIN = 1e-6

# The dtypes whose full integer range is exactly representable in float32 (see module docstring).
_SUPPORTED_DTYPES = (np.uint8, np.uint16)

BEFORE = "before"   # correct every plane, then reduce — always valid
AFTER = "after"     # reduce, then correct once — only for reducers that declare commutation
_SIDES = (BEFORE, AFTER)


@dataclass(frozen=True)
class Field:
    """A prepared, immutable illumination-correction field — built once, shared read-only.

    ``divisor``/``offset`` are already normalized, guarded and channel-ordered, so applying the
    correction is one subtract + one divide with no per-frame allocation of the guard. Building it
    once avoids ~6000 redundant full-frame ``np.where`` allocations on a 1536-well plate.

    Attributes
    ----------
    divisor:
        ``(C, Y, X)`` float32, strictly positive, mean ~1.0 per channel. The denominator.
    offset:
        ``(C, Y, X)`` float32 darkfield, or ``None`` when the profile carries no darkfield.
    dtype:
        The acquisition dtype this field was prepared for (uint8 or uint16).
    mapping:
        ``mapping[acquisition_channel_index] = profile_channel_index``. Identity unless the caller
        supplied one. The saved ``.npy`` stores a channel *count*, not names, so a shape check
        cannot see a reorder — the mapping is the only place a reorder can be declared.
    source:
        Human-readable provenance ("the .npy path" or "estimated from N wells").
    sha256:
        Digest of the guarded divisor (+ offset), so a written plate can be tied to its field.
    """

    divisor: np.ndarray
    offset: Optional[np.ndarray]
    dtype: np.dtype
    mapping: tuple[int, ...]
    source: str
    sha256: str

    @property
    def n_channels(self) -> int:
        return len(self.mapping)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (int(self.divisor.shape[1]), int(self.divisor.shape[2]))


# --- loading a stitcher-produced .npy ---------------------------------------------------------

def _install_numpy_core_compat() -> None:
    """Alias numpy<2's ``numpy.core.*`` module paths onto numpy 2.x's ``numpy._core.*``.

    Flatfield ``.npy`` files pickled under numpy 1.x (older Squid/stitcher runs) reference
    ``numpy.core.multiarray``; numpy 2 renamed the package to ``numpy._core`` and ships a thin
    ``numpy.core`` compat stub — which a frozen (PyInstaller) build often does NOT bundle, so the
    load fails with "No module named 'numpy.core.multiarray'". Registering the aliases from
    ``numpy._core`` (always present) makes the load work either way. Idempotent, best-effort.

    Vendored from ``tilefusion.flatfield`` rather than imported: importing tilefusion runs its
    heavy ``__init__`` (numba/GPU/basicpy) and this tool ships standalone (same standing decision
    as the vendored zarr store in ``_zarr_store.py``).
    """
    import sys

    try:
        import numpy._core  # noqa: F401
    except Exception:
        return  # numpy < 2 (no _core) — the old paths are already the real ones
    for sub in ("", ".multiarray", ".umath", "._multiarray_umath", ".numeric", ".numerictypes"):
        old, new = "numpy.core" + sub, "numpy._core" + sub
        if old in sys.modules:
            continue
        try:
            sys.modules[old] = __import__(new, fromlist=["_"])
        except Exception:
            pass


def load_flatfield(path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load ``(flatfield, darkfield)`` from a stitcher-produced ``.npy``.

    The file is a pickled dict ``{"flatfield": (C,Y,X) float32, "darkfield": ... | None,
    "channels": C, "shape": (Y, X)}`` — the format ``tilefusion.flatfield.save_flatfield`` writes.

    Parameters
    ----------
    path:
        Path to the ``.npy``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        ``(flat, dark)``; *dark* is ``None`` when the file carries no darkfield.

    Raises
    ------
    OSError
        The file cannot be read (missing, unreadable).
    ValueError
        The file is not a pickled dict with a ``flatfield`` entry, or cannot be unpickled at all
        (e.g. saved by an incompatible numpy).
    """
    path = Path(path)
    _install_numpy_core_compat()
    try:
        loaded = np.load(path, allow_pickle=True)
    except OSError as exc:
        raise OSError(f"cannot read flatfield file {str(path)!r}: {exc}") from exc
    except (ModuleNotFoundError, ImportError, pickle.UnpicklingError) as exc:
        raise ValueError(
            f"cannot unpickle flatfield file {str(path)!r} (it may have been saved with an "
            f"incompatible numpy version): {exc}"
        ) from exc

    try:
        data = loaded.item()
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"invalid flatfield file format at {str(path)!r}: expected a .npy holding the "
            "dictionary tilefusion's save_flatfield writes (keys 'flatfield', 'darkfield')."
        ) from exc
    if not isinstance(data, dict) or "flatfield" not in data:
        raise ValueError(
            f"invalid flatfield file format at {str(path)!r}: expected a dictionary with at "
            "least a 'flatfield' entry."
        )
    flat = np.asarray(data["flatfield"])
    dark = data.get("darkfield")
    return flat, (None if dark is None else np.asarray(dark))


def save_flatfield(path, flat: np.ndarray, dark: Optional[np.ndarray] = None) -> Path:
    """Write ``(flat, dark)`` in the stitcher's ``.npy`` format (round-trips :func:`load_flatfield`).

    Exists so an estimated profile can be persisted and re-used — and so the loader has something
    to be tested against without a stitcher install.
    """
    path = Path(path)
    flat = np.asarray(flat, dtype=np.float32)
    np.save(path, {
        "flatfield": flat,
        "darkfield": None if dark is None else np.asarray(dark, dtype=np.float32),
        "channels": int(flat.shape[0]),
        "shape": tuple(int(s) for s in flat.shape[1:]),
    }, allow_pickle=True)
    return path


# --- validation + preparation ------------------------------------------------------------------

def validate_field(flat: np.ndarray, dark: Optional[np.ndarray], n_channels: int,
                   frame_shape: Sequence[int]) -> None:
    """Check a raw profile against the acquisition, raising a **named, diagnostic** error.

    "My .npy won't load" will be the most common support question, so every failure names both
    shapes and the likely cause. Shape matching is EXACT on purpose (see the module docstring).

    Raises
    ------
    ValueError
        Wrong rank, channel-count mismatch, frame-shape mismatch, darkfield-shape mismatch, or a
        profile that is not finite/positive anywhere.
    """
    want = (int(n_channels), int(frame_shape[0]), int(frame_shape[1]))
    if flat.ndim != 3:
        raise ValueError(
            f"flatfield must be 3-D (C, Y, X); got shape {tuple(flat.shape)}. A single-channel "
            "field still needs a leading axis of length 1."
        )
    if tuple(int(s) for s in flat.shape) != want:
        raise ValueError(
            f"flatfield shape {tuple(int(s) for s in flat.shape)} does not match this acquisition "
            f"{want} (channels, height, width). SquidMIP requires an EXACT match — it will not "
            "resample a field. Likely causes: the field was estimated at a different camera "
            "binning or sensor crop, or it belongs to another acquisition."
        )
    if dark is not None and tuple(int(s) for s in dark.shape) != want:
        raise ValueError(
            f"darkfield shape {tuple(int(s) for s in dark.shape)} does not match the flatfield "
            f"{want}; the two must describe the same frame."
        )
    finite = np.isfinite(flat)
    if not finite.any() or not (flat[finite] > _MIN_GAIN).any():
        raise ValueError(
            "flatfield contains no usable gain (every pixel is non-finite or <= 1e-6); "
            "correcting with it would be a no-op at best."
        )


def _normalize_mapping(mapping: Optional[Sequence[int]], n_channels: int) -> tuple[int, ...]:
    """Validate the acquisition-channel -> profile-channel mapping; default is the identity."""
    if mapping is None:
        return tuple(range(n_channels))
    m = tuple(int(i) for i in mapping)
    if len(m) != n_channels:
        raise ValueError(
            f"channel mapping has {len(m)} entr(ies) but the acquisition has {n_channels} "
            "channel(s); give exactly one profile-channel index per acquisition channel."
        )
    bad = [i for i in m if not 0 <= i < n_channels]
    if bad:
        raise ValueError(
            f"channel mapping index/indices {bad} out of range for a {n_channels}-channel "
            "flatfield (valid: 0..{}).".format(n_channels - 1)
        )
    if len(set(m)) != len(m):
        raise ValueError(
            f"channel mapping {m} reuses a profile channel; two acquisition channels sharing one "
            "illumination profile is almost always a mis-entered mapping, so it is refused."
        )
    return m


def prepare_field(flat: np.ndarray, dark: Optional[np.ndarray], *, dtype, frame_shape: Sequence[int],
                  n_channels: int, mapping: Optional[Sequence[int]] = None,
                  source: str = "") -> Field:
    """Validate a raw profile and precompute the immutable guarded divisor ONCE.

    Per channel: normalize the profile to mean 1.0 (so correction preserves the intensity scale),
    then replace every non-finite or ``<= 1e-6`` pixel with 1.0 (those pixels pass through
    uncorrected rather than exploding). The result is a strictly positive float32 divisor that the
    hot path divides by directly — true division, so the ``BEFORE``/``AFTER`` bit-identity is exact
    equality rather than an approximation.

    Parameters
    ----------
    flat, dark:
        The raw ``(C, Y, X)`` profile and optional darkfield, e.g. from :func:`load_flatfield`.
    dtype:
        The acquisition dtype. **Must be uint8 or uint16** (see the module docstring's clip note).
    frame_shape, n_channels:
        The acquisition's ``metadata["frame_shape"]`` and channel count; matched exactly.
    mapping:
        ``mapping[acquisition_channel] = profile_channel``. Default identity.
    source:
        Provenance string recorded in the sidecar.

    Raises
    ------
    ValueError
        Unsupported dtype, or any :func:`validate_field` / mapping failure.
    """
    dtype = np.dtype(dtype)
    if dtype.type not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported acquisition dtype {dtype!s} for flatfield correction; only uint8 and "
            "uint16 are supported (their full range is exactly representable in float32, so the "
            "clip that prevents integer wraparound is exact). Wider dtypes need a wider "
            "accumulator, which is a deliberate decision, not a default."
        )
    validate_field(flat, dark, n_channels, frame_shape)
    m = _normalize_mapping(mapping, n_channels)

    prof = np.asarray(flat, dtype=np.float32)
    divisor = np.empty_like(prof)
    for c in range(prof.shape[0]):
        plane = prof[c]
        finite = np.isfinite(plane) & (plane > _MIN_GAIN)
        scale = float(plane[finite].mean()) if finite.any() else 1.0
        if scale <= _MIN_GAIN:
            scale = 1.0
        norm = plane / scale
        # Guard AFTER normalizing: a NaN/near-zero pixel carries no gain information, so the honest
        # answer is "leave this pixel alone" (divide by 1.0), not "amplify it to saturation".
        divisor[c] = np.where(np.isfinite(norm) & (norm > _MIN_GAIN), norm, 1.0)
    divisor.setflags(write=False)   # shared read-only across every engine worker

    offset = None
    if dark is not None:
        offset = np.nan_to_num(np.asarray(dark, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        offset.setflags(write=False)

    digest = hashlib.sha256(divisor.tobytes())
    if offset is not None:
        digest.update(offset.tobytes())
    return Field(divisor=divisor, offset=offset, dtype=dtype, mapping=m, source=source,
                 sha256=digest.hexdigest())


# --- the correction itself ----------------------------------------------------------------------

def apply_correction(a: np.ndarray, field: Field, c_index: int) -> np.ndarray:
    """Return *a* illumination-corrected: ``(a - darkfield) / normalized_profile``.

    Dtype- and shape-preserving, and it never mutates *a*. Arithmetic runs in float32; the result is
    clipped to the dtype's range and truncated back, so an unsigned underflow clips to 0 and an
    overflow clips to the max — no wraparound in either direction. Truncation (not rounding) matches
    the stitcher's ``apply_flatfield``, so a plate corrected here is bit-identical to one corrected
    there; both are non-decreasing, which is what the AFTER fast path needs.
    """
    if a.dtype != field.dtype:
        raise ValueError(f"plane dtype {a.dtype} != the dtype this field was prepared for "
                         f"({field.dtype}); prepare_field(dtype=...) must match the acquisition.")
    if a.shape != field.frame_shape:
        raise ValueError(f"plane shape {a.shape} != the field's frame shape {field.frame_shape}")
    c = field.mapping[c_index]
    out = a.astype(np.float32)
    if field.offset is not None:
        out -= field.offset[c]
    out /= field.divisor[c]
    info = np.iinfo(field.dtype)
    np.clip(out, info.min, info.max, out=out)   # exact for uint8/uint16 (asserted in prepare_field)
    return out.astype(field.dtype)


def with_correction(reduce: Callable[[Iterable[np.ndarray]], np.ndarray], field: Optional[Field],
                    c_index: int, side: str = BEFORE) -> Callable[[Iterable[np.ndarray]], np.ndarray]:
    """Wrap a z-reduction so it corrects illumination, returning a plain ``reduce`` callable.

    This is the composition seam: the result has exactly the ``reduce=`` signature ``project_well``
    expects, so nothing downstream changes.

    Parameters
    ----------
    reduce:
        The z-reduction to decorate (e.g. :func:`squidmip.project`).
    field:
        A prepared :class:`Field`, or ``None`` — in which case *reduce* is returned **unchanged**
        (an uncorrected run is byte-identical to today's, by construction rather than by test).
    c_index:
        Which acquisition channel these planes belong to (``project_well`` knows it; a bare
        ``reduce`` callable does not).
    side:
        ``BEFORE`` (default) corrects every plane then reduces — always valid. ``AFTER`` reduces
        then corrects once; only legal for reducers that declare ``commutes_with_scaling``.

    Raises
    ------
    ValueError
        If *side* is not ``"before"`` or ``"after"``.
    """
    if field is None:
        return reduce
    if side not in _SIDES:
        raise ValueError(f"unknown correction side {side!r}; expected one of {list(_SIDES)}")
    if side == AFTER:
        def _after(planes: Iterable[np.ndarray]) -> np.ndarray:
            return apply_correction(reduce(planes), field, c_index)
        return _after

    def _before(planes: Iterable[np.ndarray]) -> np.ndarray:
        # Generator, not a list: the reduction stays streaming, so memory stays bounded.
        return reduce(apply_correction(p, field, c_index) for p in planes)
    return _before


# --- the computed fallback (no .npy supplied) ----------------------------------------------------

def _box_blur(plane: np.ndarray, radius: int) -> np.ndarray:
    """Separable moving-average blur via summed-area sums, edge-clamped. numpy only.

    Applied three times this approximates a Gaussian (central limit) closely enough for an
    illumination envelope, and avoids adding scipy — which is NOT a declared dependency of this
    package (see pyproject) and must not become one for a fallback estimator.
    """
    if radius < 1:
        return plane.astype(np.float32, copy=True)
    out = plane.astype(np.float32, copy=True)
    for axis in (0, 1):
        n = out.shape[axis]
        r = min(radius, max(n - 1, 0))
        if r < 1:
            continue
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="edge")
        cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
        zeros = np.zeros_like(np.take(cumulative, [0], axis=axis))
        cumulative = np.concatenate([zeros, cumulative], axis=axis)
        hi = np.take(cumulative, range(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(cumulative, range(0, n), axis=axis)
        out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def estimate_flatfield(samples: Iterable[np.ndarray], *, n_channels: int,
                       frame_shape: Sequence[int], smooth_frac: float = 0.25) -> np.ndarray:
    """Estimate an illumination profile from the data itself — the fallback when no ``.npy`` exists.

    **Method, stated plainly.** Accumulate a per-channel *mean* image over the supplied samples,
    blur it heavily (three box passes at ``smooth_frac`` of the frame's short side, which
    approximates a Gaussian), and normalize each channel to mean 1.0. The reasoning: illumination
    shading is the low-frequency component that is *common* to every field of view, while specimen
    content varies between fields, so averaging many fields and keeping only the low frequencies
    leaves the shading.

    **This is NOT BaSiC.** BaSiC solves a low-rank + sparse decomposition that separates flatfield,
    darkfield and per-image baseline; this is an averaging heuristic with no solver, no darkfield
    and no baseline term. Its honest limits:

      * It is biased by content. A mean (unlike BaSiC's robust fit) is pulled by bright objects, so
        a plate with confluent wells in one corner will have that bias baked into the "shading".
      * It needs **many** samples whose content is positionally decorrelated. With few fields, or
        fields that all share the same layout, the estimate is mostly specimen, not illumination.
      * It estimates gain only — no darkfield, so an additive camera offset is left in.
      * The blur radius is a guess at the illumination scale, not a fit to it.

    A profile measured from a uniform fluorescent slide, or one produced by the stitcher's BaSiC
    estimator and loaded with :func:`load_flatfield`, is strictly better. Use this to *see* whether
    a plate has shading worth correcting, not as the correction of record.

    Parameters
    ----------
    samples:
        Iterable of ``(C, Y, X)`` arrays (one per sampled FOV). Consumed once, streaming — only the
        float64 accumulator is retained, so memory is O(one frame), flat in the sample count.
    n_channels, frame_shape:
        The expected profile shape; every sample must match.
    smooth_frac:
        Blur radius as a fraction of the frame's short side (default 0.25 — deliberately heavy;
        the point is to keep only the illumination envelope).

    Returns
    -------
    np.ndarray
        ``(C, Y, X)`` float32, mean 1.0 per channel — feed it to :func:`prepare_field`.

    Raises
    ------
    ValueError
        No samples, a sample of the wrong shape, or an invalid *smooth_frac*.
    """
    if not 0 < smooth_frac < 1:
        raise ValueError(f"smooth_frac must be in (0, 1), got {smooth_frac}")
    want = (int(n_channels), int(frame_shape[0]), int(frame_shape[1]))
    acc = np.zeros(want, dtype=np.float64)
    n = 0
    for sample in samples:
        s = np.asarray(sample)
        if tuple(int(x) for x in s.shape) != want:
            raise ValueError(f"sample shape {tuple(s.shape)} != expected (C, Y, X) {want}")
        acc += s
        n += 1
    if n == 0:
        raise ValueError("estimate_flatfield needs at least one sample; got an empty iterable.")
    acc /= n

    radius = max(1, int(round(smooth_frac * min(want[1], want[2]))))
    out = np.empty(want, dtype=np.float32)
    for c in range(want[0]):
        plane = acc[c].astype(np.float32)
        for _ in range(3):                      # 3 box passes ~ Gaussian
            plane = _box_blur(plane, radius)
        mean = float(plane.mean())
        # A dead channel (all zeros) has no shading to estimate — leave it as a flat 1.0 rather
        # than dividing by ~0 and inventing structure.
        out[c] = plane / mean if mean > _MIN_GAIN else 1.0
    return out


def sample_planes(reader, *, max_wells: int = 24, t: int = 0) -> Iterator[np.ndarray]:
    """Yield ``(C, Y, X)`` samples from up to *max_wells* wells — the input :func:`estimate_flatfield`
    wants, taken straight off an IMA-189 reader.

    One plane per channel from the middle z of each well's first FOV: the middle z is the most
    likely to be in focus, and one plane per well keeps the estimator's I/O to ``max_wells`` frames
    rather than the whole plate. Yields lazily so the estimator's streaming accumulator holds only
    one frame at a time.
    """
    meta = reader.metadata
    channels = [c["name"] for c in meta["channels"]]
    z_levels = list(meta["z_levels"])
    z = z_levels[len(z_levels) // 2]
    fovs = meta["fovs_per_region"]
    for region in list(meta["regions"])[:max_wells]:
        fov = fovs[region][0]
        yield np.stack([reader.read(region, fov, ch, z, t) for ch in channels])


# --- provenance ----------------------------------------------------------------------------------

def write_provenance(out_dir, field: Field, extra: Optional[dict] = None) -> Path:
    """Write ``flatfield.json`` beside the plate: what was applied, to what, and a digest of it.

    A corrected plate is a *derived* scientific artifact — without a record of which field produced
    it, it cannot be audited or reproduced. The sidecar sits beside ``plate.ome.zarr`` (rather than
    inside its metadata) because today's only consumer dir-walks the plate and reads array 0 plus
    omero; NGFF processing metadata is its own ticket.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "correction": "flatfield",
        "source": field.source,
        "sha256": field.sha256,
        "channel_mapping": list(field.mapping),
        "frame_shape": list(field.frame_shape),
        "dtype": str(field.dtype),
        "darkfield": field.offset is not None,
        "formula": "corrected = clip((plane - darkfield) / normalized_profile) -> native dtype",
    }
    if extra:
        info.update(extra)
    path = out_dir / "flatfield.json"
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return path


def corrected_dir_name(name: str) -> str:
    """Output folder name for a corrected run — distinct so it can NEVER clobber a raw run.

    A corrected plate and a raw plate of the same acquisition must be able to coexist: they are
    different data and a user comparing them is exactly the moment an overwrite would hurt.
    """
    return f"{name}.flatfield"

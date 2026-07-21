"""IMA-225 unit tests — flatfield / illumination correction.

Clean-room (no ``integration`` mark, no data on disk): synthetic vignetted frames and an in-memory
fake reader, so the correction's own contract is exercised without the real 1536wp fixture.

What is gated here, and why each one is load-bearing:
  * a vignetted frame is measurably FLATTENED (the feature actually works, not just runs);
  * dtype/shape are preserved and uint16 never wraps in either direction (a wraparound turns a
    saturated pixel into a black one — silent, catastrophic, and invisible in a thumbnail);
  * zero / NaN / near-zero profile pixels are guarded (division by ~0 would saturate the frame);
  * a ``.npy`` profile round-trips through the stitcher's own format;
  * the computed fallback runs and flattens;
  * ``BEFORE`` and ``AFTER`` are BIT-IDENTICAL for MIP — the property that licenses the fast path;
  * MIP with no field is byte-identical to plain MIP (the regression guard).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from squidmip import project, project_well
from squidmip.correction import (
    AFTER,
    BEFORE,
    apply_correction,
    corrected_dir_name,
    estimate_flatfield,
    load_flatfield,
    prepare_field,
    sample_planes,
    save_flatfield,
    validate_field,
    with_correction,
    write_provenance,
)

SHAPE = (32, 40)


def vignette(shape=SHAPE, floor: float = 0.35) -> np.ndarray:
    """A radial illumination profile: 1.0 at the centre falling to ~*floor* at the corners."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    r2 = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2
    return (floor + (1.0 - floor) * np.exp(-r2)).astype(np.float32)


def corner_centre_ratio(a: np.ndarray) -> float:
    """Mean of the four 4x4 corners over the mean of the 4x4 centre — 1.0 is perfectly flat."""
    h, w = a.shape
    corners = np.concatenate([a[:4, :4].ravel(), a[:4, -4:].ravel(),
                              a[-4:, :4].ravel(), a[-4:, -4:].ravel()]).astype(np.float64)
    cy, cx = h // 2, w // 2
    centre = a[cy - 2:cy + 2, cx - 2:cx + 2].astype(np.float64)
    return float(corners.mean() / centre.mean())


def make_field(profile=None, dark=None, *, dtype=np.uint16, n_channels=1, mapping=None):
    prof = vignette() if profile is None else profile
    flat = np.stack([prof] * n_channels) if prof.ndim == 2 else prof
    return prepare_field(flat, dark, dtype=dtype, frame_shape=SHAPE, n_channels=n_channels,
                         mapping=mapping, source="test")


# ── the feature actually works ──────────────────────────────────────────────────────────

def test_vignetted_image_is_flattened():
    """A uniform specimen seen through a vignette comes back measurably flatter."""
    profile = vignette()
    raw = (10000 * profile).astype(np.uint16)          # uniform 10000 counts, shaded
    before = corner_centre_ratio(raw)
    out = apply_correction(raw, make_field(profile), 0)
    after = corner_centre_ratio(out)
    assert before < 0.6                                 # the synthetic really is vignetted
    assert after > 0.98 and abs(after - 1.0) < abs(before - 1.0)


def test_correction_preserves_dtype_and_shape_and_does_not_mutate_input():
    raw = (10000 * vignette()).astype(np.uint16)
    keep = raw.copy()
    out = apply_correction(raw, make_field(), 0)
    assert out.dtype == np.uint16 and out.shape == raw.shape
    np.testing.assert_array_equal(raw, keep)            # the caller's buffer is never touched


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_dtype_round_trip(dtype):
    raw = np.full(SHAPE, np.iinfo(dtype).max // 2, dtype=dtype)
    out = apply_correction(raw, make_field(dtype=dtype), 0)
    assert out.dtype == np.dtype(dtype)


def test_unsupported_dtype_is_refused_by_name():
    with pytest.raises(ValueError, match="unsupported acquisition dtype"):
        make_field(dtype=np.uint32)


# ── the integer edges: overflow and underflow must CLIP, never wrap ─────────────────────

def test_uint16_overflow_clips_and_never_wraps():
    """Dividing a bright frame by a dim profile overflows uint16 — it must saturate, not wrap."""
    profile = np.full(SHAPE, 1.0, np.float32)
    profile[2, 3] = 0.02                                # a deeply shaded pixel -> a ~50x gain
    raw = np.full(SHAPE, 60000, np.uint16)
    out = apply_correction(raw, make_field(profile), 0)
    assert out[2, 3] == 65535                           # saturated, NOT wrapped to ~2500
    assert out.min() >= 60000 // 2                      # nothing else wrapped round to near-zero


def test_darkfield_underflow_clips_to_zero_without_wraparound():
    """(I - D) is negative wherever the darkfield exceeds the signal; unsigned, that would wrap."""
    dark = np.full((1, *SHAPE), 500.0, np.float32)
    raw = np.full(SHAPE, 100, np.uint16)                # every pixel is BELOW the darkfield
    out = apply_correction(raw, make_field(np.ones(SHAPE, np.float32), dark), 0)
    assert out.max() == 0                               # clipped to 0, not wrapped to ~65000


def test_darkfield_is_subtracted_before_dividing():
    dark = np.full((1, *SHAPE), 100.0, np.float32)
    raw = np.full(SHAPE, 1100, np.uint16)
    out = apply_correction(raw, make_field(np.full(SHAPE, 2.0, np.float32), dark), 0)
    # profile normalises to 1.0 (it is flat), so this is just (1100 - 100) / 1.0
    assert out[0, 0] == 1000


# ── zeros / NaNs / near-zero gains are guarded ──────────────────────────────────────────

@pytest.mark.parametrize("bad", [0.0, 1e-12, np.nan, np.inf, -1.0])
def test_dead_profile_pixels_pass_through_uncorrected(bad):
    """A pixel with no gain information is divided by 1.0 — never by ~0, never by NaN."""
    profile = np.ones(SHAPE, np.float32)
    profile[3, 4] = bad
    field = make_field(profile)
    assert np.isfinite(field.divisor).all() and (field.divisor > 0).all()
    raw = np.full(SHAPE, 1234, np.uint16)
    out = apply_correction(raw, field, 0)
    assert out[3, 4] == 1234                            # untouched, not saturated


def test_profile_with_no_usable_gain_is_refused():
    with pytest.raises(ValueError, match="no usable gain"):
        make_field(np.zeros(SHAPE, np.float32))


def test_profile_is_normalized_to_mean_one():
    """A profile at an arbitrary scale must not rescale the image's intensities."""
    field = make_field(vignette() * 137.0)
    assert abs(float(field.divisor[0].mean()) - 1.0) < 1e-4


# ── validation: named, diagnostic errors ────────────────────────────────────────────────

def test_channel_count_mismatch_names_both_shapes():
    with pytest.raises(ValueError, match=r"does not match this acquisition"):
        validate_field(np.ones((3, *SHAPE), np.float32), None, 2, SHAPE)


def test_frame_shape_mismatch_explains_the_likely_cause():
    with pytest.raises(ValueError, match="binning or sensor crop"):
        validate_field(np.ones((1, 8, 8), np.float32), None, 1, SHAPE)


def test_two_dimensional_field_is_refused():
    with pytest.raises(ValueError, match="must be 3-D"):
        validate_field(np.ones(SHAPE, np.float32), None, 1, SHAPE)


def test_darkfield_shape_mismatch_is_named():
    with pytest.raises(ValueError, match="darkfield shape"):
        validate_field(np.ones((1, *SHAPE), np.float32), np.ones((1, 8, 8), np.float32), 1, SHAPE)


# ── channel mapping ─────────────────────────────────────────────────────────────────────

def test_identity_mapping_is_the_default():
    assert make_field(n_channels=2).mapping == (0, 1)


def test_permuted_mapping_changes_the_result():
    """Proves the mapping is actually wired — a swap must produce different pixels."""
    flat = np.stack([np.full(SHAPE, 1.0, np.float32), _ramp()])
    raw = np.full(SHAPE, 20000, np.uint16)
    straight = apply_correction(raw, make_field(flat, n_channels=2), 0)
    swapped = apply_correction(raw, make_field(flat, n_channels=2, mapping=(1, 0)), 0)
    assert not np.array_equal(straight, swapped)


def _ramp() -> np.ndarray:
    return np.linspace(0.5, 1.5, SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)


def test_mapping_out_of_range_is_refused():
    with pytest.raises(ValueError, match="out of range"):
        make_field(n_channels=2, mapping=(0, 5))


def test_mapping_with_a_duplicate_index_is_refused():
    with pytest.raises(ValueError, match="reuses a profile channel"):
        make_field(n_channels=2, mapping=(0, 0))


def test_mapping_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="exactly one profile-channel index"):
        make_field(n_channels=2, mapping=(0,))


# ── the .npy round trip (the stitcher's format) ─────────────────────────────────────────

def test_npy_profile_round_trips(tmp_path):
    flat = np.stack([vignette(), _ramp()])
    dark = np.full((2, *SHAPE), 7.0, np.float32)
    path = save_flatfield(tmp_path / "ff.npy", flat, dark)
    got_flat, got_dark = load_flatfield(path)
    np.testing.assert_allclose(got_flat, flat)
    np.testing.assert_allclose(got_dark, dark)
    prepare_field(got_flat, got_dark, dtype=np.uint16, frame_shape=SHAPE, n_channels=2)


def test_npy_without_a_darkfield_loads_as_none(tmp_path):
    path = save_flatfield(tmp_path / "ff.npy", np.stack([vignette()]))
    _flat, dark = load_flatfield(path)
    assert dark is None


def test_missing_npy_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        load_flatfield(tmp_path / "nope.npy")


def test_malformed_npy_is_refused_by_name(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.arange(10))                        # a bare array, not the expected dict
    with pytest.raises(ValueError, match="invalid flatfield file format"):
        load_flatfield(path)


def test_npy_dict_without_a_flatfield_key_is_refused(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, {"darkfield": None}, allow_pickle=True)
    with pytest.raises(ValueError, match="at least a 'flatfield' entry"):
        load_flatfield(path)


# ── the computed fallback ───────────────────────────────────────────────────────────────

def test_estimate_flatfield_recovers_the_shading():
    """Random specimen content behind a fixed vignette: the estimate should look like the vignette."""
    rng = np.random.default_rng(0)
    profile = vignette()
    samples = [(profile * rng.uniform(400, 1200, SHAPE)).astype(np.uint16)[None] for _ in range(48)]
    est = estimate_flatfield(samples, n_channels=1, frame_shape=SHAPE)
    assert est.shape == (1, *SHAPE) and est.dtype == np.float32
    assert abs(float(est[0].mean()) - 1.0) < 1e-4
    # It is a smoothed average, not a fit, so assert the SHAPE of the shading, not its exact values.
    assert corner_centre_ratio(est[0]) < 0.9


def test_estimate_flatfield_then_correct_flattens():
    rng = np.random.default_rng(1)
    profile = vignette()
    samples = [(profile * rng.uniform(400, 1200, SHAPE)).astype(np.uint16)[None] for _ in range(48)]
    est = estimate_flatfield(samples, n_channels=1, frame_shape=SHAPE)
    field = prepare_field(est, None, dtype=np.uint16, frame_shape=SHAPE, n_channels=1)
    raw = (10000 * profile).astype(np.uint16)
    assert abs(corner_centre_ratio(apply_correction(raw, field, 0)) - 1.0) < abs(
        corner_centre_ratio(raw) - 1.0)


def test_estimate_flatfield_needs_samples():
    with pytest.raises(ValueError, match="at least one sample"):
        estimate_flatfield([], n_channels=1, frame_shape=SHAPE)


def test_estimate_flatfield_rejects_a_mismatched_sample():
    with pytest.raises(ValueError, match="sample shape"):
        estimate_flatfield([np.ones((1, 4, 4))], n_channels=1, frame_shape=SHAPE)


def test_estimate_flatfield_leaves_a_dead_channel_flat():
    est = estimate_flatfield([np.zeros((1, *SHAPE))] * 3, n_channels=1, frame_shape=SHAPE)
    np.testing.assert_allclose(est, 1.0)


# ── composition: the BEFORE/AFTER seam ──────────────────────────────────────────────────

class _FakeReader:
    """The slice of the IMA-189 reader contract project_well touches, with vignetted pixels."""

    def __init__(self, n_z=4, n_t=1, channels=("c0", "c1"), dtype=np.uint16, seed=0):
        self._channels, self._dtype = list(channels), np.dtype(dtype)
        self._n_z, self._n_t = n_z, n_t
        rng = np.random.default_rng(seed)
        prof = vignette()
        self._planes = {
            (c, z, t): (prof * rng.uniform(200, 9000, SHAPE)).astype(self._dtype)
            for c in self._channels for z in range(n_z) for t in range(n_t)
        }

    @property
    def metadata(self):
        return {"regions": ["B2", "B3"], "fovs_per_region": {"B2": [0], "B3": [0]},
                "channels": [{"name": c} for c in self._channels],
                "z_levels": list(range(self._n_z)), "n_z": self._n_z, "n_t": self._n_t,
                "frame_shape": SHAPE, "dtype": self._dtype}

    def read(self, region, fov, channel, z, t=0):
        return self._planes[(channel, z, t)]


def test_no_field_returns_the_reduce_unchanged():
    assert with_correction(project, None, 0) is project


def test_unknown_side_raises():
    with pytest.raises(ValueError, match="unknown correction side"):
        with_correction(project, make_field(), 0, side="sideways")


@pytest.mark.parametrize("n_z", [1, 2, 5])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_before_and_after_are_bit_identical_for_mip(n_z, dtype, seed):
    """The property that licenses the AFTER fast path: MIP commutes with a monotone rescale.

    A seeded numpy grid rather than hypothesis — that is not a declared test dependency, and a
    fallback estimator is no reason to add one.
    """
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype)
    planes = [rng.integers(0, info.max + 1, SHAPE, dtype=dtype) for _ in range(n_z)]
    # A profile that includes near-zero gain (guarded) and gains that clip at BOTH ends.
    prof = vignette(floor=0.05).copy()
    prof[0, 0] = 0.0
    prof[1, 1] = 12.0
    field = prepare_field(prof[None], None, dtype=dtype, frame_shape=SHAPE, n_channels=1)
    before = with_correction(project, field, 0, BEFORE)(iter(planes))
    after = with_correction(project, field, 0, AFTER)(iter(planes))
    np.testing.assert_array_equal(before, after)


def test_project_well_with_a_field_differs_from_without_and_keeps_the_contract():
    reader = _FakeReader()
    field = make_field(n_channels=2)
    plain = project_well(reader, "B2", 0)
    corrected = project_well(reader, "B2", 0, field=field, correction_side=AFTER)
    assert corrected.shape == plain.shape and corrected.dtype == plain.dtype
    assert corrected.shape[2] == 1                       # Z stays size-1: the writer's contract
    assert not np.array_equal(corrected, plain)
    assert corner_centre_ratio(corrected[0, 0, 0]) > corner_centre_ratio(plain[0, 0, 0])


def test_project_well_before_equals_after_for_mip():
    reader = _FakeReader(n_z=3, n_t=2)
    field = make_field(n_channels=2)
    np.testing.assert_array_equal(
        project_well(reader, "B2", 0, field=field, correction_side=BEFORE),
        project_well(reader, "B2", 0, field=field, correction_side=AFTER),
    )


def test_project_well_without_a_field_is_unchanged():
    """REGRESSION GUARD: the uncorrected path must be byte-identical to plain MIP."""
    reader = _FakeReader()
    np.testing.assert_array_equal(project_well(reader, "B2", 0),
                                  project_well(reader, "B2", 0, field=None))


def test_sample_planes_yields_stacked_frames():
    reader = _FakeReader()
    samples = list(sample_planes(reader, max_wells=2))
    assert len(samples) == 2
    assert all(s.shape == (2, *SHAPE) for s in samples)


# ── provenance + the no-overwrite guard ─────────────────────────────────────────────────

def test_provenance_sidecar_records_what_was_applied(tmp_path):
    field = make_field(n_channels=2, mapping=(1, 0))
    path = write_provenance(tmp_path / "out", field, {"projector": "mip"})
    info = json.loads(path.read_text())
    assert path.name == "flatfield.json"
    assert info["correction"] == "flatfield" and info["projector"] == "mip"
    assert info["channel_mapping"] == [1, 0] and info["sha256"] == field.sha256
    assert info["frame_shape"] == list(SHAPE) and info["dtype"] == "uint16"


def test_field_digest_is_stable_and_content_addressed():
    assert make_field().sha256 == make_field().sha256
    assert make_field().sha256 != make_field(vignette(floor=0.2)).sha256


def test_corrected_output_folder_cannot_clobber_a_raw_run():
    assert corrected_dir_name("acq_2026") != "acq_2026"
    assert "acq_2026" in corrected_dir_name("acq_2026")


def test_the_prepared_divisor_is_immutable():
    """It is shared read-only across every engine worker — a write would be a data race."""
    field = make_field()
    with pytest.raises(ValueError):
        field.divisor[0, 0, 0] = 2.0

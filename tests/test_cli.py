"""CLI (IMA-186) tests: the declarative params model + the run() that drives write_plate.

Headless, no Qt. Uses the shared tiny `squid_dataset` fixture (a real 2-well acquisition on disk).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from squidmip._cli import ProcessParameters, run


def test_input_folder_validator_rejects_missing(tmp_path):
    with pytest.raises(ValueError):
        ProcessParameters(input_folder=str(tmp_path / "nope"))


def test_run_writes_navigable_plate(squid_dataset, tmp_path):
    root, _ = squid_dataset                       # tiny real acquisition (B2, B3)
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path), tiff=False)
    manifest = run(params)

    plate = Path(manifest["plate"])
    assert plate.name == "plate.ome.zarr"
    assert plate.parent.name.endswith(".hcs")     # <acq-name>.hcs sibling
    assert manifest["n_wells"] == 2
    assert manifest["tiff"] is None                # CLI default: no uncompressed TIFF duplicate
    # the plate group + both wells' fields are on disk (level 0 present)
    assert (plate / "zarr.json").exists()
    for row, col in (("B", "2"), ("B", "3")):
        assert (plate / row / col / "0" / "zarr.json").exists()


def test_run_skips_unreadable_well_instead_of_aborting(squid_dataset, tmp_path):
    # Resilience (IMA-186): one corrupt/missing plane must NOT abort a whole-plate run — the bad
    # well is SKIPPED (logged + reported), the good wells still write.
    root, _ = squid_dataset                       # B2, B3
    victim = sorted((Path(root) / "0").glob("B3_*"))[0]
    victim.unlink()                               # break B3 (a plane it needs is now gone)
    params = ProcessParameters(input_folder=str(root), output_folder=str(tmp_path))
    manifest = run(params)
    assert manifest["skipped"] == ["B3"]          # bad well skipped, not fatal
    assert manifest["n_fields_written"] == 1      # B2 still written
    plate = Path(manifest["plate"])
    assert (plate / "B" / "2" / "0" / "zarr.json").exists()
    assert not (plate / "B" / "3" / "0" / "0").exists()   # B3 field never written


def test_run_defaults_output_next_to_acquisition(squid_dataset):
    root, _ = squid_dataset
    params = ProcessParameters(input_folder=str(root))     # no output_folder -> sibling of the acq
    assert params.output_folder is None
    manifest = run(params)
    assert Path(manifest["plate"]).parent.parent == Path(root).parent


# ── IMA-225: --flatfield ─────────────────────────────────────────────────────────────────

def _profile_for(root, path):
    """Save a radial .npy profile matching the acquisition at *root* (the stitcher's format)."""
    import numpy as np

    from squidmip import open_reader
    from squidmip.correction import save_flatfield

    meta = open_reader(str(root)).metadata
    ny, nx = meta["frame_shape"]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    cy, cx = max((ny - 1) / 2.0, 1e-6), max((nx - 1) / 2.0, 1e-6)
    prof = 0.4 + 0.6 * np.exp(-(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2))
    return save_flatfield(path, np.stack([prof.astype(np.float32)] * len(meta["channels"])))


def test_flatfield_validator_rejects_a_missing_file(tmp_path, squid_dataset):
    # Up-front, like --projector: a bad profile must fail BEFORE any plate skeleton is written.
    root, _ = squid_dataset
    with pytest.raises(ValueError, match="not an existing file"):
        ProcessParameters(input_folder=str(root), flatfield=str(tmp_path / "nope.npy"))


def test_flatfield_estimate_sentinel_is_accepted(squid_dataset):
    root, _ = squid_dataset
    assert ProcessParameters(input_folder=str(root), flatfield="estimate").flatfield == "estimate"


def test_flatfield_run_writes_its_own_folder_with_provenance(squid_dataset, tmp_path):
    import json

    root, _ = squid_dataset
    profile = _profile_for(root, tmp_path / "ff.npy")
    raw = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    corrected = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path),
                                      flatfield=str(profile)))
    # A corrected run and a raw run of the same acquisition COEXIST — no silent overwrite.
    assert Path(raw["plate"]).parent != Path(corrected["plate"]).parent
    assert Path(corrected["plate"]).parent.name.endswith(".flatfield.hcs")
    sidecar = json.loads(Path(corrected["flatfield"]).read_text())
    assert sidecar["correction"] == "flatfield" and sidecar["projector"] == "mip"
    assert sidecar["source"] == str(profile)


def test_flatfield_estimate_runs_end_to_end(squid_dataset, tmp_path):
    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path),
                                     flatfield="estimate"))
    assert manifest["n_fields_written"] == 2
    assert Path(manifest["flatfield"]).exists()


def test_no_flatfield_writes_no_sidecar(squid_dataset, tmp_path):
    root, _ = squid_dataset
    manifest = run(ProcessParameters(input_folder=str(root), output_folder=str(tmp_path)))
    assert "flatfield" not in manifest
    assert not (Path(manifest["plate"]).parent / "flatfield.json").exists()

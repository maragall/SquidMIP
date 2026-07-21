"""IMA-187: coordinates.csv -> metadata["fov_positions_um"], on both reader classes.

Units: the CSV records millimetres, the metadata key is micrometres — the conversion is the
producer's job and the ``_um`` suffix is the contract (see ``test_units_invariant`` below).

The load-bearing property is the row-order mapping and its cross-check. coordinates.csv has no
``fov`` column, so the Nth distinct position of a region IS that region's Nth sorted FOV. If
that mapping is wrong nothing crashes — the mosaic just draws every tile in the wrong place.
So the count check has to be exact, and it has to survive multi-z acquisitions that repeat a
stage position once per z-level.
"""

from __future__ import annotations

import pytest

from squidmip.reader import load_fov_positions_um, open_reader


def _csv(rows, header="region,x (mm),y (mm),z (mm)"):
    return header + "\n" + "\n".join(rows) + "\n"


# --- the row-order mapping ------------------------------------------------------------------

def test_positions_map_row_order_to_sorted_fovs(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1, 2]})
    assert pos == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000), ("A1", 2): (2000, 2000)}


def test_mapping_follows_sorted_fov_ids_not_their_values(tmp_path):
    """Non-contiguous FOV ids (7, 9, 11) still map in sorted order to rows 1, 2, 3."""
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [7, 9, 11]})
    assert pos[("A1", 7)] == (1000, 2000)
    assert pos[("A1", 9)] == (1500, 2000)
    assert pos[("A1", 11)] == (2000, 2000)


def test_multiple_regions_are_grouped_independently(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "B2,50.0,60.0,", "A1,1.5,2.0,", "B2,50.5,60.0,",
    ]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1], "B2": [0, 1]})
    assert pos[("A1", 1)] == (1500, 2000)
    assert pos[("B2", 1)] == (50500, 60000)


# --- multi-z de-duplication (the check that would otherwise break every real z-stack) --------

def test_repeated_positions_per_z_level_are_deduplicated(tmp_path):
    """A 3-z acquisition writes each position 3x. That must still resolve to 2 FOVs, not fail."""
    rows = []
    for _z in range(3):
        rows += ["A1,1.0,2.0,", "A1,1.5,2.0,"]
    (tmp_path / "coordinates.csv").write_text(_csv(rows))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000)}


def test_dedup_preserves_first_seen_order(tmp_path):
    rows = ["A1,9.0,9.0,", "A1,1.0,1.0,", "A1,9.0,9.0,", "A1,1.0,1.0,"]
    (tmp_path / "coordinates.csv").write_text(_csv(rows))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert pos[("A1", 0)] == (9000, 9000)      # first seen wins, file order preserved
    assert pos[("A1", 1)] == (1000, 1000)


# --- the cross-check ------------------------------------------------------------------------

def test_too_few_positions_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,"]))
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions_um(tmp_path, {"A1": [0, 1, 2]})


def test_too_many_positions_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions_um(tmp_path, {"A1": [0]})


# --- degradation + malformed input ----------------------------------------------------------

def test_absent_csv_returns_empty_not_missing(tmp_path):
    """Empty-but-present: consumers use .get()/[] freely and degrade to single-FOV rendering."""
    assert load_fov_positions_um(tmp_path, {"A1": [0]}) == {}


def test_unknown_regions_in_csv_are_ignored(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "ZZ9,5.0,5.0,"]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0]})
    assert set(pos) == {("A1", 0)}


def test_blank_coordinate_rows_are_skipped(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,,,", "A1,1.5,2.0,"]))
    pos = load_fov_positions_um(tmp_path, {"A1": [0, 1]})
    assert len(pos) == 2


def test_non_numeric_coordinate_raises_with_line_number(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,oops,2.0,"]))
    with pytest.raises(ValueError, match="line 3.*non-numeric"):
        load_fov_positions_um(tmp_path, {"A1": [0, 1]})


def test_missing_xy_columns_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text("region,foo,bar\nA1,1,2\n")
    with pytest.raises(ValueError, match="no recognisable x/y millimetre columns"):
        load_fov_positions_um(tmp_path, {"A1": [0]})


def test_header_whitespace_and_case_tolerated(tmp_path):
    (tmp_path / "coordinates.csv").write_text("region,X (MM),Y (mm),z\nA1,1.0,2.0,\n")
    assert load_fov_positions_um(tmp_path, {"A1": [0]}) == {("A1", 0): (1000, 2000)}


# --- reader integration (both classes expose the key) ---------------------------------------

def test_squid_reader_exposes_fov_positions_um(squid_dataset):
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert "fov_positions_um" in meta
    # conftest writes 2 regions x 2 fovs, each repeated per z-level
    assert len(meta["fov_positions_um"]) == 4
    assert meta["fov_positions_um"][("B2", 0)] == (10000, 20000)
    assert meta["fov_positions_um"][("B2", 1)] == (10500, 20000)


def test_fov_positions_um_present_even_without_csv(squid_dataset):
    """The key must exist on every acquisition — a missing key is a KeyError landmine."""
    root, _ = squid_dataset
    (root / "coordinates.csv").unlink()
    meta = open_reader(root).metadata
    assert meta["fov_positions_um"] == {}


def _ome_acquisition(root):
    """A minimal 2-channel OME-TIFF acquisition (mirrors tests/test_reader.py's fixture)."""
    import numpy as np
    import tifffile

    ome = root / "ome_tiff"
    ome.mkdir(parents=True)
    tifffile.imwrite(ome / "A1_0.ome.tiff", np.zeros((2, 2, 2, 16, 16), np.uint16),
                     metadata={"axes": "TZCYX"})
    tifffile.imwrite(ome / "A1_1.ome.tiff", np.zeros((2, 2, 2, 16, 16), np.uint16),
                     metadata={"axes": "TZCYX"})
    (root / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 405 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#20ADF8'\n      exposure_time_ms: 1.0\n"
        "- name: Fluorescence 488 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#00FF00'\n      exposure_time_ms: 1.0\n")
    (root / "acquisition.yaml").write_text(
        "sample:\n  wellplate_format: 384 well plate\nz_stack:\n  nz: 2\n  delta_z_mm: 0.0\n"
        "time_series:\n  nt: 2\n")
    return root


def test_ome_reader_exposes_fov_positions_um_empty_without_csv(tmp_path):
    """SquidOMEReader shares the interface, so it must carry the same key (empty is fine)."""
    meta = open_reader(_ome_acquisition(tmp_path / "acq")).metadata
    assert meta["fov_positions_um"] == {}


def test_ome_reader_reads_a_sibling_coordinates_csv(tmp_path):
    """An OME acquisition with a coordinates.csv beside it gets real placement for free."""
    root = _ome_acquisition(tmp_path / "acq")
    (root / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,1.5,2.0,"]))
    meta = open_reader(root).metadata
    assert meta["fov_positions_um"] == {("A1", 0): (1000, 2000), ("A1", 1): (1500, 2000)}


# --- units invariant (world space is MICROMETRES, every key ends in _um) ---------------------

def test_metadata_key_is_um_suffixed_and_no_mm_key_survives(squid_dataset):
    """The contract ``_tiling.py`` declares: world space is µm and the key says so.

    The un-suffixed ``fov_positions`` key carried MILLIMETRES into code documented in µm. A
    lingering alias would let a consumer keep reading mm and re-introduce the 1000x error, so
    assert the old key is gone rather than merely that the new one exists.
    """
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert "fov_positions_um" in meta
    assert "fov_positions" not in meta
    for key, value in meta.items():
        if key.startswith("fov_positions"):
            assert key.endswith("_um"), f"world-space key {key!r} must end in _um"
            assert all(isinstance(v, tuple) and len(v) == 2 for v in value.values())


def test_positions_are_micrometres_not_millimetres(squid_dataset):
    """conftest writes FOVs 0.5 mm apart. In µm that is 500, not 0.5 — the whole bug."""
    root, _ = squid_dataset
    pos = open_reader(root).metadata["fov_positions_um"]
    dx = pos[("B2", 1)][0] - pos[("B2", 0)][0]
    assert dx == pytest.approx(500.0), f"0.5 mm pitch must read as 500 um, got {dx}"


def test_placement_consumes_um_without_rescaling(squid_dataset):
    """End-to-end: reader µm -> _placement px, with no second mm->µm multiply anywhere.

    0.5 mm = 500 µm at 0.5 µm/px is exactly 1000 px. If either side still converted, this
    would be 1 px or 1_000_000 px — both of which render as a plausible picture.
    """
    from squidmip._placement import fov_offsets_px

    root, _ = squid_dataset
    meta = open_reader(root).metadata
    off = fov_offsets_px(meta["fov_positions_um"], "B2", [0, 1], 0.5)
    assert off == {0: (0, 0), 1: (0, 1000)}


# --- graceful degradation: a truncated CSV must not sink the whole acquisition ---------------

def test_truncated_coordinates_csv_still_yields_channels_and_dtype(squid_dataset):
    """The bug: the cross-check raised out of the middle of the ``metadata`` dict literal, so
    regions/channels/dtype — all derived from FILENAMES and one decoded frame, none of which
    the CSV can invalidate — became unreachable and the viewer declared the acquisition
    unreadable. Placement may degrade; identity may not.
    """
    root, _ = squid_dataset
    lines = (root / "coordinates.csv").read_text().splitlines()
    (root / "coordinates.csv").write_text("\n".join(lines[:2]) + "\n")   # header + ONE row

    with pytest.warns(UserWarning, match="unusable"):
        meta = open_reader(root).metadata

    assert meta["channels"], "channels come from filenames + yaml; a short CSV cannot erase them"
    assert meta["dtype"] is not None
    assert meta["regions"] == ["B2", "B3"]
    assert meta["frame_shape"]
    assert meta["fov_positions_um"] == {}    # the only thing lost: placement


def test_malformed_coordinates_csv_header_still_yields_metadata(squid_dataset):
    """Same containment for the other CSV failure mode (no recognisable x/y columns)."""
    root, _ = squid_dataset
    (root / "coordinates.csv").write_text("region,foo,bar\nB2,1,2\n")
    with pytest.warns(UserWarning, match="unusable"):
        meta = open_reader(root).metadata
    assert [c["name"] for c in meta["channels"]]
    assert meta["dtype"] is not None
    assert meta["fov_positions_um"] == {}


def test_degradation_does_not_swallow_unexpected_errors(squid_dataset, monkeypatch):
    """Only the deliberate ValueErrors degrade; a genuine bug must still surface."""
    import squidmip.reader as reader_mod

    def boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(reader_mod, "load_fov_positions_um", boom)
    root, _ = squid_dataset
    with pytest.raises(RuntimeError, match="disk on fire"):
        open_reader(root).metadata

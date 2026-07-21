"""IMA-187: coordinates.csv -> metadata["fov_positions"], on both reader classes.

The load-bearing property is the row-order mapping and its cross-check. coordinates.csv has no
``fov`` column, so the Nth distinct position of a region IS that region's Nth sorted FOV. If
that mapping is wrong nothing crashes — the mosaic just draws every tile in the wrong place.
So the count check has to be exact, and it has to survive multi-z acquisitions that repeat a
stage position once per z-level.
"""

from __future__ import annotations

import pytest

from squidmip.reader import load_fov_positions, open_reader


def _csv(rows, header="region,x (mm),y (mm),z (mm)"):
    return header + "\n" + "\n".join(rows) + "\n"


# --- the row-order mapping ------------------------------------------------------------------

def test_positions_map_row_order_to_sorted_fovs(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    pos = load_fov_positions(tmp_path, {"A1": [0, 1, 2]})
    assert pos == {("A1", 0): (1.0, 2.0), ("A1", 1): (1.5, 2.0), ("A1", 2): (2.0, 2.0)}


def test_mapping_follows_sorted_fov_ids_not_their_values(tmp_path):
    """Non-contiguous FOV ids (7, 9, 11) still map in sorted order to rows 1, 2, 3."""
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    pos = load_fov_positions(tmp_path, {"A1": [7, 9, 11]})
    assert pos[("A1", 7)] == (1.0, 2.0)
    assert pos[("A1", 9)] == (1.5, 2.0)
    assert pos[("A1", 11)] == (2.0, 2.0)


def test_multiple_regions_are_grouped_independently(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "B2,50.0,60.0,", "A1,1.5,2.0,", "B2,50.5,60.0,",
    ]))
    pos = load_fov_positions(tmp_path, {"A1": [0, 1], "B2": [0, 1]})
    assert pos[("A1", 1)] == (1.5, 2.0)
    assert pos[("B2", 1)] == (50.5, 60.0)


# --- multi-z de-duplication (the check that would otherwise break every real z-stack) --------

def test_repeated_positions_per_z_level_are_deduplicated(tmp_path):
    """A 3-z acquisition writes each position 3x. That must still resolve to 2 FOVs, not fail."""
    rows = []
    for _z in range(3):
        rows += ["A1,1.0,2.0,", "A1,1.5,2.0,"]
    (tmp_path / "coordinates.csv").write_text(_csv(rows))
    pos = load_fov_positions(tmp_path, {"A1": [0, 1]})
    assert pos == {("A1", 0): (1.0, 2.0), ("A1", 1): (1.5, 2.0)}


def test_dedup_preserves_first_seen_order(tmp_path):
    rows = ["A1,9.0,9.0,", "A1,1.0,1.0,", "A1,9.0,9.0,", "A1,1.0,1.0,"]
    (tmp_path / "coordinates.csv").write_text(_csv(rows))
    pos = load_fov_positions(tmp_path, {"A1": [0, 1]})
    assert pos[("A1", 0)] == (9.0, 9.0)      # first seen wins, file order preserved
    assert pos[("A1", 1)] == (1.0, 1.0)


# --- the cross-check ------------------------------------------------------------------------

def test_too_few_positions_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,"]))
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions(tmp_path, {"A1": [0, 1, 2]})


def test_too_many_positions_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv([
        "A1,1.0,2.0,", "A1,1.5,2.0,", "A1,2.0,2.0,",
    ]))
    with pytest.raises(ValueError, match="distinct stage position"):
        load_fov_positions(tmp_path, {"A1": [0]})


# --- degradation + malformed input ----------------------------------------------------------

def test_absent_csv_returns_empty_not_missing(tmp_path):
    """Empty-but-present: consumers use .get()/[] freely and degrade to single-FOV rendering."""
    assert load_fov_positions(tmp_path, {"A1": [0]}) == {}


def test_unknown_regions_in_csv_are_ignored(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "ZZ9,5.0,5.0,"]))
    pos = load_fov_positions(tmp_path, {"A1": [0]})
    assert set(pos) == {("A1", 0)}


def test_blank_coordinate_rows_are_skipped(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,,,", "A1,1.5,2.0,"]))
    pos = load_fov_positions(tmp_path, {"A1": [0, 1]})
    assert len(pos) == 2


def test_non_numeric_coordinate_raises_with_line_number(tmp_path):
    (tmp_path / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,oops,2.0,"]))
    with pytest.raises(ValueError, match="line 3.*non-numeric"):
        load_fov_positions(tmp_path, {"A1": [0, 1]})


def test_missing_xy_columns_raises_named(tmp_path):
    (tmp_path / "coordinates.csv").write_text("region,foo,bar\nA1,1,2\n")
    with pytest.raises(ValueError, match="no recognisable x/y millimetre columns"):
        load_fov_positions(tmp_path, {"A1": [0]})


def test_header_whitespace_and_case_tolerated(tmp_path):
    (tmp_path / "coordinates.csv").write_text("region,X (MM),Y (mm),z\nA1,1.0,2.0,\n")
    assert load_fov_positions(tmp_path, {"A1": [0]}) == {("A1", 0): (1.0, 2.0)}


# --- reader integration (both classes expose the key) ---------------------------------------

def test_squid_reader_exposes_fov_positions(squid_dataset):
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert "fov_positions" in meta
    # conftest writes 2 regions x 2 fovs, each repeated per z-level
    assert len(meta["fov_positions"]) == 4
    assert meta["fov_positions"][("B2", 0)] == (10.0, 20.0)
    assert meta["fov_positions"][("B2", 1)] == (10.5, 20.0)


def test_fov_positions_present_even_without_csv(squid_dataset):
    """The key must exist on every acquisition — a missing key is a KeyError landmine."""
    root, _ = squid_dataset
    (root / "coordinates.csv").unlink()
    meta = open_reader(root).metadata
    assert meta["fov_positions"] == {}


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


def test_ome_reader_exposes_fov_positions_empty_without_csv(tmp_path):
    """SquidOMEReader shares the interface, so it must carry the same key (empty is fine)."""
    meta = open_reader(_ome_acquisition(tmp_path / "acq")).metadata
    assert meta["fov_positions"] == {}


def test_ome_reader_reads_a_sibling_coordinates_csv(tmp_path):
    """An OME acquisition with a coordinates.csv beside it gets real placement for free."""
    root = _ome_acquisition(tmp_path / "acq")
    (root / "coordinates.csv").write_text(_csv(["A1,1.0,2.0,", "A1,1.5,2.0,"]))
    meta = open_reader(root).metadata
    assert meta["fov_positions"] == {("A1", 0): (1.0, 2.0), ("A1", 1): (1.5, 2.0)}

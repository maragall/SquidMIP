"""IMA-184 output writer — clean-room unit tests (no reader, no data on disk).

Drives ``write_from_stream`` with a fabricated metadata dict + a hand-built
``(region, fov, image)`` stream, then reads the written store back with tensorstore + json
(the same v3 store ndviewer_light reads). The real-seam cross commit (``project_plate`` on
``sim_1536wp`` + hongquan) lives in tests/test_integration.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorstore as ts

from squidmip._output import (
    parse_well_id,
    plate_metadata,
    split_well,
    write_from_stream,
)

CH = [
    {"name": "Fluorescence_638_nm_-_Penta", "display_name": "Fluorescence 638 nm - Penta", "display_color": "#FF0000"},
    {"name": "Fluorescence_405_nm_-_Penta", "display_name": "Fluorescence 405 nm - Penta", "display_color": "#20ADF8"},
]
# B10 present so column sort is natural (2,3,10 not 10,2,3) and no zero-padding is exercised.
REGIONS = ["B2", "B3", "B10"]


def _meta():
    return {
        "regions": REGIONS,
        "fovs_per_region": {r: [0] for r in REGIONS},
        "channels": CH,
        "pixel_size_um": 0.325,
    }


def _image(seed: int, t=1, c=2, y=8, x=8, dtype=np.uint16):
    # deterministic, unique-ish per (seed, t, c) plane, kept small so it also fits uint8
    base = np.arange(y * x).reshape(y, x)
    out = np.empty((t, c, 1, y, x), dtype=dtype)
    for ti in range(t):
        for ci in range(c):
            out[ti, ci, 0] = ((base + seed * 20 + ti * 7 + ci * 3) % 200).astype(dtype)
    return out


def _stream(images: dict):
    # completion order is arbitrary in reality; yield out of plate order on purpose
    for region in ("B3", "B10", "B2"):
        yield region, 0, images[region]


def _read_array(path: Path) -> np.ndarray:
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}}, open=True
    ).result()
    return np.asarray(store[...].read().result())


# --- pure helpers ---------------------------------------------------------------------------

def test_parse_well_id_uppercases_no_padding_roundtrips():
    # vendored Squid semantics: uppercase, multi-letter rows, no zero-padding
    assert parse_well_id("B2") == ("B", "2")
    assert parse_well_id("aa3") == ("AA", "3")  # lowercase -> upper (Squid parse_well_id)
    assert split_well is parse_well_id  # back-compat alias
    for region in ("B2", "H12", "AA1", "AF48"):
        row, col = parse_well_id(region)
        assert row + col == region  # ndviewer reconstructs well_id = row + col


def test_parse_well_id_fails_loud_on_non_plate_region():
    import pytest

    for bad in ("region_1", "1A", "B2C"):  # not <letters><digits> -> refuse, don't mislabel
        with pytest.raises(ValueError):
            parse_well_id(bad)


def test_plate_metadata_natural_column_sort_and_well_paths():
    ome = plate_metadata(REGIONS, field_count=1)["plate"]
    assert [c["name"] for c in ome["columns"]] == ["2", "3", "10"]  # int sort, not lexicographic
    assert [c["name"] for c in ome["rows"]] == ["B"]
    paths = {w["path"] for w in ome["wells"]}
    assert paths == {"B/2", "B/3", "B/10"}  # no zero-padding


# --- full write via the stream --------------------------------------------------------------

def test_write_from_stream_layout_and_pixels(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=True)

    plate = Path(manifest["plate"])
    assert plate.name == "plate.ome.zarr"
    assert manifest["n_wells"] == 3 and manifest["n_fields_written"] == 3

    # plate group metadata
    plate_doc = json.loads((plate / "zarr.json").read_text())
    assert plate_doc["node_type"] == "group"
    assert plate_doc["attributes"]["ome"]["plate"]["field_count"] == 1

    # each well: group metadata + single-level field + omero + pixel-exact full-res array
    for region in REGIONS:
        row, col = parse_well_id(region)
        well_doc = json.loads((plate / row / col / "zarr.json").read_text())
        assert well_doc["attributes"]["ome"]["well"]["images"] == [{"path": "0"}]  # raw fov id 0

        field = plate / row / col / "0"
        field_doc = json.loads((field / "zarr.json").read_text())["attributes"]["ome"]
        ds_paths = [d["path"] for d in field_doc["multiscales"][0]["datasets"]]
        assert ds_paths == ["0"]  # single level, Squid canonical (no pyramid)
        colors = [c["color"] for c in field_doc["omero"]["channels"]]
        assert colors == ["FF0000", "20ADF8"]  # hex without '#', in channel order

        assert np.array_equal(_read_array(field / "0"), images[region])  # full-res pixel-exact
        assert not (field / "1").exists()  # no pyramid level written

    # every group validates against the official OME-NGFF v0.5 pydantic models
    from tests.ngff_check import assert_valid_ngff_plate

    assert_valid_ngff_plate(plate)


def test_large_field_writes_pyramid(tmp_path):
    # A field larger than the pyramid floor (256 px) gets downsample LEVELS: 600 -> 300 -> 150.
    # Level 0 stays full-res pixel-exact; coarser levels are half-size area-averages. Small fields
    # (the other tests, 8x8) collapse to level 0 alone — canonical single-level output unchanged.
    big = {r: _image(i, y=600, x=600) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(big), tmp_path, n_fovs=1, tiff=False)
    assert manifest["levels"] == 3

    field = Path(manifest["plate"]) / "B" / "2" / "0"
    ds_paths = [d["path"] for d in
                json.loads((field / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]["datasets"]]
    assert ds_paths == ["0", "1", "2"]
    assert np.array_equal(_read_array(field / "0"), big["B2"])          # level 0 pixel-exact
    assert _read_array(field / "1").shape == (1, 2, 1, 300, 300)        # half-size
    assert _read_array(field / "2").shape == (1, 2, 1, 150, 150)
    # coarse-level scale reflects the real downsample factor (2x, 4x) in Y,X
    scales = [d["coordinateTransformations"][0]["scale"] for d in
              json.loads((field / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]["datasets"]]
    assert scales[1][-2:] == [0.325 * 2, 0.325 * 2]
    assert scales[2][-2:] == [0.325 * 4, 0.325 * 4]


def test_write_from_stream_individual_tiffs(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=True)

    import tifffile

    tiff_dir = tmp_path / "tiff" / "0"
    for region in REGIONS:
        for c_i, ch in enumerate(CH):
            f = tiff_dir / f"{region}_0_0_{ch['name']}.tiff"
            assert f.exists(), f
            plane = tifffile.imread(f)
            assert plane.dtype == np.uint16  # native dtype preserved
            assert np.array_equal(plane, images[region][0, c_i, 0])  # pixel-exact, z collapsed


def test_tiff_disabled(tmp_path):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    manifest = write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=False)
    assert manifest["tiff"] is None
    assert not (tmp_path / "tiff").exists()


def test_uint8_dtype_preserved(tmp_path):
    images = {r: _image(i, dtype=np.uint8) for i, r in enumerate(REGIONS)}
    write_from_stream(_meta(), _stream(images), tmp_path, n_fovs=1, tiff=False)
    arr = _read_array(tmp_path / "plate.ome.zarr" / "B" / "2" / "0" / "0")
    assert arr.dtype == np.uint8
    assert np.array_equal(arr, images["B2"])


def test_fails_loud_on_wrong_shape(tmp_path):
    import pytest

    # z not collapsed (Z=3) -> a seam bug; refuse rather than write a mislabelled field
    bad = np.zeros((1, 2, 3, 8, 8), np.uint16)
    with pytest.raises(ValueError, match="T, C, 1, Y, X"):
        write_from_stream(_meta(), iter([("B2", 0, bad)]), tmp_path, n_fovs=1, tiff=False)


def test_fails_loud_on_channel_count_mismatch(tmp_path):
    import pytest

    # image says 3 channels, metadata lists 2 -> refuse (would mislabel omero)
    bad = np.zeros((1, 3, 1, 8, 8), np.uint16)
    with pytest.raises(ValueError, match="channels"):
        write_from_stream(_meta(), iter([("B2", 0, bad)]), tmp_path, n_fovs=1, tiff=False)


def test_writer_memory_is_bounded_in_well_count(tmp_path):
    """The writer streams: peak RSS is flat in the number of wells (it never holds the plate).

    Feeds a lazy generator of N wells and checks 4x the wells does NOT ~4x the peak — each
    (region, fov, image) is written and released before the next is pulled.
    """
    import tracemalloc

    def stream(n):
        for i in range(n):
            yield f"B{i + 2}", 0, _image(i, y=256, x=256)  # ~256 KB/well, built lazily

    def peak_for(n, dest):
        meta = {
            **_meta(),
            "regions": [f"B{i + 2}" for i in range(n)],
            "fovs_per_region": {f"B{i + 2}": [0] for i in range(n)},
        }
        tracemalloc.start()
        write_from_stream(meta, stream(n), dest, n_fovs=1, tiff=True)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    p4 = peak_for(4, tmp_path / "a")
    p16 = peak_for(16, tmp_path / "b")
    assert p16 < p4 * 2  # 4x wells, <2x peak -> bounded/streaming, not proportional to plate size


# --- IMA-231: FOV ROI table -----------------------------------------------------------------
#
# The table is a whole-image ROI per field (SquidMIP never fuses), written where
# fractal-tasks-core resolves tables by name: {field}/tables/FOV_ROI_table.

import struct

import pytest

from squidmip._output import _ROI_COLUMNS, _roi_row, roi_table_enabled
from squidmip._zarr_store import create_array, write_array, write_group, write_string_array

DZ = 1.5  # um between z planes; _meta() deliberately omits it (see the skip test below)


def _meta_z():
    return {**_meta(), "dz_um": DZ}


def _attrs(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text()).get("attributes", {})


def _read_string_array(path: Path) -> list[str]:
    """Decode the hand-written vlen-utf8 chunk (tensorstore cannot read this dtype)."""
    doc = json.loads((path / "zarr.json").read_text())
    assert doc["data_type"] == "string"
    assert [c["name"] for c in doc["codecs"]] == ["vlen-utf8"]
    raw = (path / "c" / "0").read_bytes()
    (n,) = struct.unpack_from("<I", raw, 0)
    out, off = [], 4
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", raw, off)
        off += 4
        out.append(raw[off:off + ln].decode("utf-8"))
        off += ln
    return out


def _table_dir(root: Path, region="B2", fov=0) -> Path:
    row, col = parse_well_id(region)
    return root / "plate.ome.zarr" / row / col / str(fov) / "tables" / "FOV_ROI_table"


def _write(tmp_path, meta=None, **kw):
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    write_from_stream(meta or _meta_z(), _stream(images), tmp_path, n_fovs=1, **kw)
    return tmp_path


# -- pure row math ---------------------------------------------------------------------------

def test_roi_row_is_whole_image_at_origin():
    # (T, C, Z, Y=8, X=6) at 0.325 um/px, dz 1.5 um
    row = _roi_row((1, 2, 1, 8, 6), 0.325, DZ)
    assert row[:3] == [0.0, 0.0, 0.0]                      # origin: no fusion -> image top-left
    assert row[3] == pytest.approx(6 * 0.325)              # len_x from X
    assert row[4] == pytest.approx(8 * 0.325)              # len_y from Y
    assert row[5] == pytest.approx(DZ)                     # ONE plane, not n_z * dz


def test_roi_columns_are_the_fractal_six():
    assert _ROI_COLUMNS == (
        "x_micrometer", "y_micrometer", "z_micrometer",
        "len_x_micrometer", "len_y_micrometer", "len_z_micrometer",
    )


# -- up-front honesty gate -------------------------------------------------------------------

def test_missing_pixel_size_raises_before_anything_is_written(tmp_path):
    meta = {**_meta_z(), "pixel_size_um": None}
    images = {r: _image(i) for i, r in enumerate(REGIONS)}
    with pytest.raises(ValueError, match="pixel_size_um"):
        write_from_stream(meta, _stream(images), tmp_path, n_fovs=1)
    # the whole point of validating up front: nothing half-written on disk
    assert not (tmp_path / "plate.ome.zarr").exists()


def test_missing_dz_skips_the_table_but_still_writes_the_plate(tmp_path):
    """A single-plane acquisition has no delta_z_mm -> no honest len_z, but the MIP still ships."""
    assert roi_table_enabled(_meta_z()) is True
    assert roi_table_enabled(_meta()) is False             # _meta() has no dz_um
    root = _write(tmp_path, meta=_meta())
    assert (root / "plate.ome.zarr" / "B" / "2" / "0" / "0").exists()   # pixels written
    assert not _table_dir(root).exists()                                # sidecar omitted


# -- written layout --------------------------------------------------------------------------

def test_tables_group_advertises_the_table(tmp_path):
    root = _write(tmp_path)
    tables = _table_dir(root).parent
    assert _attrs(tables)["tables"] == ["FOV_ROI_table"]


def test_table_group_carries_anndata_and_fractal_encoding(tmp_path):
    a = _attrs(_table_dir(_write(tmp_path)))
    assert a["encoding-type"] == "anndata"
    assert a["type"] == "roi_table"
    assert a["fractal_table_version"] == "1"


def test_x_holds_the_six_values(tmp_path):
    root = _write(tmp_path)
    x = _read_array(_table_dir(root) / "X")
    assert x.shape == (1, 6)
    assert x.dtype == np.float64
    # _image default is 8x8 at pixel_size 0.325
    assert list(x[0]) == pytest.approx([0.0, 0.0, 0.0, 8 * 0.325, 8 * 0.325, DZ])
    assert _attrs(_table_dir(root) / "X")["encoding-type"] == "array"


def test_var_index_is_the_column_names(tmp_path):
    var = _table_dir(_write(tmp_path)) / "var"
    assert _attrs(var)["_index"] == "_index"
    assert _attrs(var)["encoding-type"] == "dataframe"
    assert _read_string_array(var / "_index") == list(_ROI_COLUMNS)


def test_obs_index_is_the_raw_fov_id(tmp_path):
    """Non-contiguous fov ids stay faithful — same rule the well images[] paths follow."""
    meta = {**_meta_z(), "regions": ["B2"], "fovs_per_region": {"B2": [7]}}
    write_from_stream(meta, iter([("B2", 7, _image(0))]), tmp_path, n_fovs=1)
    obs = _table_dir(tmp_path, fov=7) / "obs"
    assert _attrs(obs)["_index"] == "FieldIndex"
    assert _read_string_array(obs / "FieldIndex") == ["7"]


def test_anndata_empty_slots_exist(tmp_path):
    table = _table_dir(_write(tmp_path))
    for slot in ("layers", "obsm", "obsp", "varm", "varp", "uns"):
        assert _attrs(table / slot)["encoding-type"] == "dict"


def test_every_written_field_gets_a_table(tmp_path):
    root = _write(tmp_path)
    for region in REGIONS:
        assert _table_dir(root, region).exists()


def test_rerun_overwrites_cleanly(tmp_path):
    _write(tmp_path)
    root = _write(tmp_path)                      # same destination, second pass
    x = _read_array(_table_dir(root) / "X")
    assert x.shape == (1, 6)                     # not appended/duplicated
    assert _read_string_array(_table_dir(root) / "var" / "_index") == list(_ROI_COLUMNS)


# -- CRITICAL regressions: _zarr_store is shared with the shipped pyramid path ----------------

def test_create_array_default_is_unchanged_5d(tmp_path):
    """The image path must be byte-identical after the dimension_names widening."""
    store = create_array(tmp_path / "img", (1, 2, 1, 4, 4), np.uint16)
    write_array(store, np.zeros((1, 2, 1, 4, 4), np.uint16))
    doc = json.loads((tmp_path / "img" / "zarr.json").read_text())
    assert doc["dimension_names"] == ["t", "c", "z", "y", "x"]
    assert doc["data_type"] == "uint16"
    assert [c["name"] for c in doc["codecs"]] == ["bytes", "blosc"]


def test_create_array_rank_mismatch_fails_loud(tmp_path):
    with pytest.raises(ValueError, match="rank"):
        create_array(tmp_path / "bad", (1, 6), np.float64)  # default names are 5-D


def test_create_array_accepts_matching_lower_rank(tmp_path):
    store = create_array(tmp_path / "t", (1, 6), np.float64, dimension_names=("obs", "var"))
    write_array(store, np.zeros((1, 6), np.float64))
    doc = json.loads((tmp_path / "t" / "zarr.json").read_text())
    assert doc["dimension_names"] == ["obs", "var"]
    assert doc["chunk_grid"]["configuration"]["chunk_shape"] == [1, 6]


def test_write_group_existing_shapes_unchanged(tmp_path):
    write_group(tmp_path / "bare")
    assert json.loads((tmp_path / "bare" / "zarr.json").read_text()) == {
        "zarr_format": 3, "node_type": "group", "attributes": {},
    }
    write_group(tmp_path / "o", {"version": "0.5"})
    assert _attrs(tmp_path / "o") == {"ome": {"version": "0.5"}}


def test_write_group_raw_attributes(tmp_path):
    write_group(tmp_path / "r", attributes={"tables": ["FOV_ROI_table"]})
    assert _attrs(tmp_path / "r") == {"tables": ["FOV_ROI_table"]}


def test_write_string_array_roundtrip(tmp_path):
    write_string_array(tmp_path / "s", ["a", "bb", "ccc"])
    assert _read_string_array(tmp_path / "s") == ["a", "bb", "ccc"]


def test_write_string_array_handles_non_ascii(tmp_path):
    write_string_array(tmp_path / "s", ["µm", "z"])
    assert _read_string_array(tmp_path / "s") == ["µm", "z"]


# -- ACCEPTANCE ORACLE: the real consumer reads it back ---------------------------------------
#
# The whole reason IMA-231 hand-writes the anndata encoding (instead of taking ngio as a runtime
# dependency) is that ngio needs Python 3.11+ and would break the 3.10 CI leg. That trade is only
# safe if the real library actually validates the output — so these tests ARE the correctness
# proof, not a nice-to-have. Installed via the [oracle] extra; skipped where it isn't available.

def test_anndata_reads_the_table(tmp_path):
    ad = pytest.importorskip("anndata", reason="pip install .[oracle]")
    root = _write(tmp_path)
    a = ad.read_zarr(_table_dir(root))
    assert list(a.var_names) == list(_ROI_COLUMNS)
    assert list(a.obs_names) == ["0"]
    assert list(a.X[0]) == pytest.approx([0.0, 0.0, 0.0, 8 * 0.325, 8 * 0.325, DZ])


def test_ngio_parses_it_as_a_roi_table(tmp_path):
    """The acceptance criterion: ngio resolves the table by name and yields a world-space ROI."""
    ngio = pytest.importorskip("ngio", reason="pip install .[oracle]")
    root = _write(tmp_path)
    row, col = parse_well_id("B2")
    container = ngio.open_ome_zarr_container(root / "plate.ome.zarr" / row / col / "0")

    assert container.list_tables() == ["FOV_ROI_table"]
    table = container.get_table("FOV_ROI_table")
    assert type(table).__name__ == "RoiTableV1"

    (roi,) = table.rois()
    assert roi.name == "0"                                   # the raw fov id
    # ngio models an ROI as world-space slices per axis; check start/end directly.
    extents = {ax: (roi.get(ax).start, roi.get(ax).end) for ax in ("x", "y", "z")}
    assert extents["x"] == pytest.approx((0.0, 8 * 0.325))
    assert extents["y"] == pytest.approx((0.0, 8 * 0.325))
    assert extents["z"] == pytest.approx((0.0, DZ))

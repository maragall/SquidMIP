"""Tests for open_reader + the three readers (AC1, AC4, AC5, AC6 + edge cases + decisions 4/5/6).

IMA-229 added the HCS zarr sections at the bottom, plus OME characterization tests written
before the shared-helper refactor.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import tifffile

from squidmip import open_reader
from tests.conftest import CH_IN_YAML, CH_NOT_IN_YAML, _write_timepoint


# --- AC1 / AC5: metadata discovery ------------------------------------------
def test_metadata_discovery(squid_dataset):
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert meta["regions"] == ["B2", "B3"]
    assert meta["fovs_per_region"] == {"B2": [0, 1], "B3": [0, 1]}
    assert meta["n_z"] == 2
    assert meta["z_levels"] == [0, 1]
    assert meta["frame_shape"] == (4, 4)
    assert meta["dtype"] == np.uint16
    assert meta["n_t"] == 1
    assert meta["dz_um"] == 1.5
    # 0.325 is the stored acquisition.yaml value, NOT the recomputed 3.76/20=0.188 -> proves
    # we read the authoritative pixel size rather than recomputing it.
    assert meta["pixel_size_um"] == 0.325
    assert meta["wellplate_format"] == "1536 well plate"


def test_metadata_no_dead_attributes(squid_dataset):
    # every metadata key must be present AND functionally derived (no dead/None scalars)
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert set(meta) == {
        "regions",
        "fovs_per_region",
        "channels",
        "n_z",
        "z_levels",
        "dz_um",
        "pixel_size_um",
        "wellplate_format",
        "frame_shape",
        "dtype",
        "n_t",
    }
    for key, value in meta.items():
        assert value is not None, f"metadata[{key!r}] is None — dead attribute"
    assert all(meta.values()) or meta["n_z"] >= 1  # no empty containers on a real dataset


def test_channel_count_independent_of_nz(squid_dataset):
    # AC5: 2 channels, NOT 2 * Nz(=2)
    root, _ = squid_dataset
    meta = open_reader(root).metadata
    assert len(meta["channels"]) == 2
    names = {c["name"] for c in meta["channels"]}
    assert names == {CH_IN_YAML, CH_NOT_IN_YAML}


def test_channel_colors_yaml_and_fallback(squid_dataset):
    # AC2: 638 from YAML nested camera_settings; 561 absent from YAML -> wavelength fallback
    root, _ = squid_dataset
    by_name = {c["name"]: c for c in open_reader(root).metadata["channels"]}
    assert by_name[CH_IN_YAML]["display_color"] == "#FF0000"
    assert by_name[CH_IN_YAML]["display_name"] == "Fluorescence 638 nm - Penta"
    assert by_name[CH_NOT_IN_YAML]["display_color"] == "#FFCF00"  # 561 from CHANNEL_COLORS_MAP


# --- AC4: exact-pixel read ---------------------------------------------------
def test_read_exact_pixels(squid_dataset):
    root, arrays = squid_dataset
    reader = open_reader(root)
    for key, expected in arrays.items():
        region, fov, z, ch = key
        got = reader.read(region, fov, ch, z)
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


def test_read_matches_tifffile_directly(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)
    got = reader.read("B3", 1, CH_IN_YAML, 0)
    direct = tifffile.imread(root / "0" / f"B3_1_0_{CH_IN_YAML}.tiff")
    np.testing.assert_array_equal(got, direct)


# --- AC6: laziness -----------------------------------------------------------
def test_read_is_lazy_one_file(squid_dataset, monkeypatch):
    root, _ = squid_dataset
    reader = open_reader(root)
    reader.metadata  # warm metadata first (its own single-frame read is separate)

    calls = {"n": 0}
    real = tifffile.imread

    def counting_imread(path, *a, **k):
        calls["n"] += 1
        return real(path, *a, **k)

    monkeypatch.setattr("squidmip.reader.tifffile.imread", counting_imread)
    reader.read("B2", 0, CH_IN_YAML, 0)
    assert calls["n"] == 1


# --- decision 5: non-2D refusal ---------------------------------------------
def test_read_rejects_non_2d(squid_dataset):
    root, _ = squid_dataset
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    tifffile.imwrite(root / "0" / f"B2_0_0_{CH_IN_YAML}.tiff", rgb)  # overwrite with RGB
    reader = open_reader(root)
    with pytest.raises(ValueError, match="not a 2D grayscale plane|not supported"):
        reader.read("B2", 0, CH_IN_YAML, 0)


# --- dtype contract: uint8/uint16 only (Squid's real grayscale set) ----------
def test_read_rejects_uint32(squid_dataset):
    root, _ = squid_dataset
    tifffile.imwrite(
        root / "0" / f"B2_0_0_{CH_IN_YAML}.tiff", np.arange(16, dtype=np.uint32).reshape(4, 4)
    )
    with pytest.raises(ValueError, match="dtype"):
        open_reader(root).read("B2", 0, CH_IN_YAML, 0)


def test_read_accepts_uint8_native(squid_dataset):
    # MONO8 is a valid (if contrast-poor) Squid format; accept it, preserve native dtype
    root, _ = squid_dataset
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    tifffile.imwrite(root / "0" / f"B2_0_0_{CH_IN_YAML}.tiff", arr)
    got = open_reader(root).read("B2", 0, CH_IN_YAML, 0)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, arr)


# --- decision 6: time dimension ---------------------------------------------
def test_multi_timepoint(squid_dataset):
    root, arrays = squid_dataset
    t1_arrays: dict = {}
    _write_timepoint(root / "1", t1_arrays, tag=1)
    # keep the dataset self-consistent (nt=2) so the Nt cross-check stays quiet
    (root / "acquisition.yaml").write_text(
        "z_stack:\n  nz: 2\n  delta_z_mm: 0.0015\ntime_series:\n  nt: 2\n"
    )
    reader = open_reader(root)
    assert reader.metadata["n_t"] == 2
    got = reader.read("B2", 0, CH_IN_YAML, 0, t=1)
    np.testing.assert_array_equal(got, t1_arrays[("B2", 0, 0, CH_IN_YAML)])
    # t=0 and t=1 differ (tag offset), proving t routes to the right folder
    assert not np.array_equal(got, arrays[("B2", 0, 0, CH_IN_YAML)])


def test_read_t_out_of_range(squid_dataset):
    root, _ = squid_dataset
    with pytest.raises(IndexError, match="out of range"):
        open_reader(root).read("B2", 0, CH_IN_YAML, 0, t=5)


# --- validation + edges ------------------------------------------------------
@pytest.mark.parametrize(
    "args",
    [
        ("ZZ", 0, CH_IN_YAML, 0),   # bad region
        ("B2", 99, CH_IN_YAML, 0),  # bad fov
        ("B2", 0, "Nope", 0),       # bad channel
        ("B2", 0, CH_IN_YAML, 9),   # bad z
    ],
)
def test_read_invalid_args_raise(squid_dataset, args):
    root, _ = squid_dataset
    with pytest.raises(KeyError):
        open_reader(root).read(*args)


def test_tif_suffix_fallback(squid_dataset):
    # a plane stored as .tif (not .tiff) is still discovered and read
    root, _ = squid_dataset
    arr = np.full((4, 4), 7, dtype=np.uint16)
    tifffile.imwrite(root / "0" / f"B2_0_5_{CH_IN_YAML}.tif", arr)
    reader = open_reader(root)
    got = reader.read("B2", 0, CH_IN_YAML, 5)
    np.testing.assert_array_equal(got, arr)


def test_nz_mismatch_warns(squid_dataset):
    # acquisition.yaml is authoritative: declare nz=5 while filenames only have z in {0,1}
    root, _ = squid_dataset
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\nz_stack:\n  nz: 5\n  delta_z_mm: 0.0015\n"
        "time_series:\n  nt: 1\n"
    )
    with pytest.warns(UserWarning, match="Nz"):
        open_reader(root).metadata


# --- format dispatch ---------------------------------------------------------
def test_open_reader_uses_ome_reader_when_ome_files_present(tmp_path):
    # ome_tiff/ that CONTAINS .ome.tiff files -> the OME-TIFF reader (5-D TZCYX per well-FOV).
    import numpy as np
    import tifffile

    from squidmip.reader import SquidOMEReader

    ome = tmp_path / "ome_tiff"
    ome.mkdir()
    tifffile.imwrite(ome / "A1_0.ome.tiff", np.zeros((2, 2, 2, 16, 16), np.uint16),   # T,Z,C,Y,X
                     metadata={"axes": "TZCYX"}, compression="lzw")
    (tmp_path / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 405 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#20ADF8'\n      exposure_time_ms: 1.0\n"
        "- name: Fluorescence 488 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#00FF00'\n      exposure_time_ms: 1.0\n")
    (tmp_path / "acquisition.yaml").write_text(
        "sample:\n  wellplate_format: 384 well plate\nz_stack:\n  nz: 2\n  delta_z_mm: 0.0\n"
        "time_series:\n  nt: 2\n")
    r = open_reader(tmp_path)
    assert isinstance(r, SquidOMEReader)
    assert r.metadata["regions"] == ["A1"]
    assert r.metadata["n_z"] == 2 and r.metadata["n_t"] == 2 and r.metadata["frame_shape"] == (16, 16)
    assert len(r.metadata["channels"]) == 2
    assert r.read("A1", 0, r.metadata["channels"][1]["name"], 1, 1).shape == (16, 16)


def test_open_reader_ignores_empty_ome_tiff_placeholder(tmp_path):
    # Squid leaves an EMPTY ome_tiff/ beside an individual-TIFF acquisition; it must NOT block the
    # individual-TIFF reader. With individual TIFFs present, open_reader should succeed.
    import numpy as np
    import tifffile

    (tmp_path / "ome_tiff").mkdir()                        # empty placeholder
    (tmp_path / "0").mkdir()
    tifffile.imwrite(tmp_path / "0" / "A1_0_0_Fluorescence_488_nm_-_Penta.tiff",
                     np.zeros((4, 4), np.uint16))
    (tmp_path / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n- name: Fluorescence 488 nm - Penta\n"
        "  camera_settings:\n    '1':\n      display_color: '#00FF00'\n      exposure_time_ms: 1.0\n")
    (tmp_path / "acquisition.yaml").write_text(
        "sample:\n  wellplate_format: 384 well plate\nz_stack:\n  nz: 1\n  delta_z_mm: 0.0\n"
        "time_series:\n  nt: 1\n")
    r = open_reader(tmp_path)                              # must NOT raise
    assert r.metadata["regions"] == ["A1"]


# --- OME reader characterization (IMA-229 T0) --------------------------------
# Written BEFORE the shared-helper refactor. Only ONE pre-existing test touched
# SquidOMEReader.metadata (single well, single FOV, matching yaml), so the refactor's
# "existing tests stay green" gate covered almost none of the code being moved. These pin the
# uncovered branches: the yaml/n_c mismatch ladder, _ome_channel_names, multi-region plate
# ordering, and the OME Nz warning.

def _write_ome(dirpath, name, shape=(1, 2, 2, 8, 8), axes="TZCYX", channel_names=None):
    """One {region}_{fov}.ome.tiff. channel_names -> OME-XML Channel Name= entries."""
    import tifffile

    md = {"axes": axes}
    if channel_names is not None:
        md["Channel"] = {"Name": list(channel_names)}
    tifffile.imwrite(dirpath / name, np.zeros(shape, np.uint16), metadata=md, compression="lzw")


def _ome_acq_yaml(tmp_path, nz=2, nt=1):
    (tmp_path / "acquisition.yaml").write_text(
        f"objective:\n  pixel_size_um: 0.5\nsample:\n  wellplate_format: 384 well plate\n"
        f"z_stack:\n  nz: {nz}\n  delta_z_mm: 0.001\ntime_series:\n  nt: {nt}\n")


def _ome_channels_yaml(tmp_path, names):
    body = "version: 1\nchannels:\n"
    for n in names:
        body += (f"- name: {n}\n  camera_settings:\n    '1':\n"
                 f"      display_color: '#00FF00'\n      exposure_time_ms: 1.0\n")
    (tmp_path / "acquisition_channels.yaml").write_text(body)


def test_ome_channel_names_fall_back_to_ome_xml_when_yaml_count_mismatches(tmp_path):
    # yaml declares ONE channel, the file has TWO -> the count guard fires and names come from
    # the OME-XML Channel Name= entries (normalized), not from the yaml.
    from squidmip.reader import SquidOMEReader

    ome = tmp_path / "ome_tiff"
    ome.mkdir()
    _write_ome(ome, "A1_0.ome.tiff", channel_names=["Fluorescence 405 nm - Penta",
                                                    "Fluorescence 488 nm - Penta"])
    _ome_channels_yaml(tmp_path, ["Fluorescence 405 nm - Penta"])   # only ONE
    _ome_acq_yaml(tmp_path)

    r = open_reader(tmp_path)
    assert isinstance(r, SquidOMEReader)
    names = [c["name"] for c in r.metadata["channels"]]
    assert len(names) == 2
    # normalized (spaces -> underscores), sourced from the OME-XML not the yaml
    assert names == ["Fluorescence_405_nm_-_Penta", "Fluorescence_488_nm_-_Penta"]


def test_ome_generic_label_fallback_is_unreachable_and_raises(tmp_path):
    """CHARACTERIZATION of a latent bug, pinned so the refactor cannot change it silently.

    reader.py's OME channel ladder ends in generic ``f"C{i}"`` labels when the yaml count is
    wrong AND the OME-XML carries no channel names. That last rung is DEAD: ``resolve_channels``
    refuses any channel whose color it cannot resolve (``_channels.py:145``), and "C0" matches
    neither the yaml nor CHANNEL_COLORS_MAP. So the intended graceful fallback actually raises.

    Pre-existing (IMA-189), NOT introduced or fixed by IMA-229 — see TODOS.md. This test pins
    today's real behavior so the T1 refactor is provably behavior-preserving.
    """
    ome = tmp_path / "ome_tiff"
    ome.mkdir()
    _write_ome(ome, "A1_0.ome.tiff")                       # no Channel names in the OME-XML
    _ome_channels_yaml(tmp_path, ["Fluorescence 405 nm - Penta"])   # 1 != 2
    _ome_acq_yaml(tmp_path)

    with pytest.raises(ValueError, match="Could not resolve a display color for channel 'C0'"):
        open_reader(tmp_path).metadata


def test_ome_multi_region_uses_plate_row_major_order(tmp_path):
    # regions must come back in TRUE plate row-major order (A..Z then AA..), not lexicographic,
    # and fovs_per_region must be grouped + sorted per well.
    ome = tmp_path / "ome_tiff"
    ome.mkdir()
    for stem in ["B10_1", "B10_0", "B2_0", "AA3_0", "A1_0"]:
        _write_ome(ome, f"{stem}.ome.tiff")
    _ome_channels_yaml(tmp_path, ["Fluorescence 405 nm - Penta", "Fluorescence 488 nm - Penta"])
    _ome_acq_yaml(tmp_path)

    meta = open_reader(tmp_path).metadata
    # single-letter rows before double-letter; column by integer, so B2 < B10 and B < AA
    assert meta["regions"] == ["A1", "B2", "B10", "AA3"]
    assert meta["fovs_per_region"]["B10"] == [0, 1]
    assert meta["fovs_per_region"]["A1"] == [0]


def test_ome_nz_mismatch_warns_and_file_wins(tmp_path):
    # acquisition.yaml says nz=99, the OME file says 2 -> warn, and the FILE value is used.
    ome = tmp_path / "ome_tiff"
    ome.mkdir()
    _write_ome(ome, "A1_0.ome.tiff")
    _ome_channels_yaml(tmp_path, ["Fluorescence 405 nm - Penta", "Fluorescence 488 nm - Penta"])
    _ome_acq_yaml(tmp_path, nz=99)

    with pytest.warns(UserWarning, match="Recorded Nz"):
        meta = open_reader(tmp_path).metadata
    assert meta["n_z"] == 2


def test_ome_unknown_region_raises_keyerror(tmp_path):
    ome = tmp_path / "ome_tiff"
    ome.mkdir()
    _write_ome(ome, "A1_0.ome.tiff")
    _ome_channels_yaml(tmp_path, ["Fluorescence 405 nm - Penta", "Fluorescence 488 nm - Penta"])
    _ome_acq_yaml(tmp_path)

    r = open_reader(tmp_path)
    ch = r.metadata["channels"][0]["name"]
    with pytest.raises(KeyError, match="No such well/FOV"):
        r.read("ZZ9", 0, ch, 0)


def test_open_reader_rejects_non_directory(tmp_path):
    f = tmp_path / "x.tiff"
    f.write_bytes(b"")
    with pytest.raises(NotImplementedError, match="not a directory"):
        open_reader(f)


def test_empty_dir_raises(tmp_path):
    (tmp_path / "0").mkdir()
    with pytest.raises(ValueError, match="No Squid individual-TIFF"):
        open_reader(tmp_path).metadata


# --- IMA-229: HCS zarr reader ------------------------------------------------
import json as _json

import tensorstore as _ts

from tests.conftest import (
    _build_zarr_plate,
    _write_zarr_sidecars,
    _zarr_pixel_value,
)


def _patch_field_attrs(root, region, fov, mutate):
    """Read {row}/{col}/{fov}/zarr.json, apply mutate(attributes), write it back."""
    p = root / "plate.ome.zarr" / region[0] / region[1:] / str(fov) / "zarr.json"
    doc = _json.loads(p.read_text())
    mutate(doc["attributes"])
    p.write_text(_json.dumps(doc, indent=2))


def test_zarr_metadata_discovery(squid_zarr_dataset):
    root, _ = squid_zarr_dataset
    from squidmip.reader import SquidZarrReader

    r = open_reader(root)
    assert isinstance(r, SquidZarrReader)
    meta = r.metadata
    assert meta["regions"] == ["B2", "B3"]
    assert meta["fovs_per_region"] == {"B2": [0, 1], "B3": [0, 1]}
    assert meta["n_z"] == 2 and meta["z_levels"] == [0, 1]
    assert meta["n_t"] == 1
    assert meta["frame_shape"] == (4, 4)
    assert meta["dtype"] == np.uint16
    # scalars come from acquisition.yaml, not from the _squid block
    assert meta["pixel_size_um"] == 0.325
    assert meta["dz_um"] == 1.5
    assert meta["wellplate_format"] == "1536 well plate"


def test_zarr_metadata_contract_matches_tiff_reader(squid_zarr_dataset, squid_dataset):
    """Both readers must return the SAME eleven keys — that is the whole point of the contract."""
    zroot, _ = squid_zarr_dataset
    troot, _ = squid_dataset
    assert set(open_reader(zroot).metadata) == set(open_reader(troot).metadata)


def test_zarr_read_exact_pixels(squid_zarr_dataset):
    root, arrays = squid_zarr_dataset
    r = open_reader(root)
    names = [c["name"] for c in r.metadata["channels"]]
    for (region, fov, z, c_i), expected in arrays.items():
        np.testing.assert_array_equal(r.read(region, fov, names[c_i], z), expected)


def test_zarr_read_matches_tensorstore_directly(squid_zarr_dataset):
    """AC3: a plane is bit-identical to a direct tensorstore read of the same slice."""
    root, _ = squid_zarr_dataset
    r = open_reader(root)
    ch = r.metadata["channels"][1]["name"]
    got = r.read("B3", 1, ch, 1)
    store = _ts.open({"driver": "zarr3", "kvstore": {
        "driver": "file", "path": str(root / "plate.ome.zarr" / "B" / "3" / "1" / "0")}},
        open=True).result()
    np.testing.assert_array_equal(got, np.asarray(store[0, 1, 1, :, :].read().result()))


def test_zarr_read_is_lazy_one_plane(squid_zarr_dataset):
    """read() must return one (Y, X) plane, not the whole 5-D field."""
    root, _ = squid_zarr_dataset
    r = open_reader(root)
    plane = r.read("B2", 0, r.metadata["channels"][0]["name"], 0)
    assert plane.shape == (4, 4) and plane.ndim == 2


def test_zarr_read_native_dtype_preserved_uint8(tmp_path):
    root = tmp_path / "acq8"
    _write_zarr_sidecars(root)
    _build_zarr_plate(root, regions=["B2"], fovs=[0], dtype=np.uint8)
    r = open_reader(root)
    assert r.metadata["dtype"] == np.uint8
    assert r.read("B2", 0, r.metadata["channels"][0]["name"], 0).dtype == np.uint8


@pytest.mark.parametrize("kwargs,exc", [
    (dict(region="ZZ9", fov=0, z=0), KeyError),      # unknown well
    (dict(region="B2", fov=99, z=0), KeyError),      # unknown fov
    (dict(region="B2", fov=0, z=99), IndexError),    # z out of range
])
def test_zarr_read_invalid_args_raise(squid_zarr_dataset, kwargs, exc):
    root, _ = squid_zarr_dataset
    r = open_reader(root)
    ch = r.metadata["channels"][0]["name"]
    with pytest.raises(exc):
        r.read(kwargs["region"], kwargs["fov"], ch, kwargs["z"])


def test_zarr_read_unknown_channel_and_t_raise(squid_zarr_dataset):
    root, _ = squid_zarr_dataset
    r = open_reader(root)
    with pytest.raises(KeyError, match="No such channel"):
        r.read("B2", 0, "Not_A_Channel", 0)
    with pytest.raises(IndexError, match="t=5 out of range"):
        r.read("B2", 0, r.metadata["channels"][0]["name"], 0, t=5)


# --- D4: omero labels are the channel ground truth ---------------------------
def test_zarr_channels_come_from_omero_in_c_axis_order(squid_zarr_dataset):
    root, _ = squid_zarr_dataset
    names = [c["name"] for c in open_reader(root).metadata["channels"]]
    assert names == [CH_IN_YAML, CH_NOT_IN_YAML]      # omero order, normalized


def test_zarr_omero_order_wins_over_yaml_and_warns(tmp_path):
    """If the yaml lists the SAME channels in a DIFFERENT order, omero wins and we warn.

    This is the silent-mislabel case: both lists have the right length, so a count guard alone
    would pass them straight through and every channel would be labeled with its neighbour's
    name. omero is written by iterating the C axis, so it is the authoritative order.
    """
    root = tmp_path / "acq_swapped"
    _write_zarr_sidecars(root)
    # A yaml carrying BOTH channels, in the opposite order to omero.
    (root / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n"
        "- name: Fluorescence 638 nm - Penta\n  camera_settings:\n    '1':\n"
        "      display_color: '#FF0000'\n      exposure_time_ms: 50.0\n"
        "- name: Fluorescence 561 nm - Penta\n  camera_settings:\n    '1':\n"
        "      display_color: '#00FF00'\n      exposure_time_ms: 50.0\n")
    _build_zarr_plate(root, regions=["B2"], fovs=[0],
                      channel_names=[CH_NOT_IN_YAML, CH_IN_YAML])   # omero reversed
    with pytest.warns(UserWarning, match="authoritative channel order"):
        names = [c["name"] for c in open_reader(root).metadata["channels"]]
    assert names == [CH_NOT_IN_YAML, CH_IN_YAML]


def test_zarr_matching_omero_and_yaml_order_does_not_warn(tmp_path):
    """The drift warning must not cry wolf when the two agree."""
    root = tmp_path / "acq_agree"
    _write_zarr_sidecars(root)
    (root / "acquisition_channels.yaml").write_text(
        "version: 1\nchannels:\n"
        "- name: Fluorescence 638 nm - Penta\n  camera_settings:\n    '1':\n"
        "      display_color: '#FF0000'\n      exposure_time_ms: 50.0\n"
        "- name: Fluorescence 561 nm - Penta\n  camera_settings:\n    '1':\n"
        "      display_color: '#00FF00'\n      exposure_time_ms: 50.0\n")
    _build_zarr_plate(root, regions=["B2"], fovs=[0],
                      channel_names=[CH_IN_YAML, CH_NOT_IN_YAML])
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # any warning fails the test
        names = [c["name"] for c in open_reader(root).metadata["channels"]]
    assert names == [CH_IN_YAML, CH_NOT_IN_YAML]


def test_zarr_refuses_when_channel_identity_is_undeterminable(tmp_path):
    root = tmp_path / "acq_nochan"
    _write_zarr_sidecars(root)
    _build_zarr_plate(root, regions=["B2"], fovs=[0])
    _patch_field_attrs(root, "B2", 0, lambda a: a["ome"]["omero"].__setitem__("channels", []))
    with pytest.raises(ValueError, match="cannot determine channel identity"):
        open_reader(root).metadata


# --- D5: acquisition.yaml wins, _squid is a cross-check ----------------------
def test_zarr_squid_pixel_size_drift_warns_but_yaml_wins(tmp_path):
    root = tmp_path / "acq_drift"
    _write_zarr_sidecars(root)
    _build_zarr_plate(root, regions=["B2"], fovs=[0], pixel_size_um=0.9)   # yaml says 0.325
    with pytest.warns(UserWarning, match="pixel_size_um disagrees"):
        meta = open_reader(root).metadata
    assert meta["pixel_size_um"] == 0.325


def test_zarr_nz_mismatch_warns(tmp_path):
    root = tmp_path / "acq_nz"
    _write_zarr_sidecars(root)
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.325\nsample:\n  wellplate_format: 1536 well plate\n"
        "z_stack:\n  nz: 99\n  delta_z_mm: 0.0015\ntime_series:\n  nt: 1\n")
    _build_zarr_plate(root, regions=["B2"], fovs=[0])
    with pytest.warns(UserWarning, match="Recorded Nz"):
        assert open_reader(root).metadata["n_z"] == 2


def test_zarr_requires_acquisition_yaml(tmp_path):
    root = tmp_path / "acq_noyaml"
    root.mkdir()
    _build_zarr_plate(root, regions=["B2"], fovs=[0])
    with pytest.raises(FileNotFoundError, match="acquisition.yaml not found"):
        open_reader(root).metadata


# --- D8: dtype refused at OPEN, not per-plane --------------------------------
def test_zarr_rejects_float_dtype_at_open(tmp_path):
    root = tmp_path / "acq_float"
    _write_zarr_sidecars(root)
    _build_zarr_plate(root, regions=["B2"], fovs=[0], dtype=np.float32)
    # must raise from metadata (open time), NOT only when a plane is read
    with pytest.raises(ValueError, match="dtype float32"):
        open_reader(root).metadata


# --- D3: completeness gate + plan-vs-disk intersection -----------------------
def test_zarr_incomplete_refused_by_default(squid_zarr_partial):
    with pytest.raises(ValueError, match="acquisition_complete"):
        open_reader(squid_zarr_partial).metadata


def test_zarr_incomplete_names_crash_vs_abort(tmp_path):
    root = tmp_path / "acq_aborted"
    _write_zarr_sidecars(root)
    _build_zarr_plate(root, regions=["B2"], fovs=[0], complete=False, aborted=True)
    with pytest.raises(ValueError, match="ABORTED by the user"):
        open_reader(root).metadata

    root2 = tmp_path / "acq_crashed"
    _write_zarr_sidecars(root2)
    _build_zarr_plate(root2, regions=["B2"], fovs=[0], complete=False, aborted=False)
    with pytest.raises(ValueError, match="crash or power loss"):
        open_reader(root2).metadata


def test_zarr_allow_incomplete_proceeds_and_warns(squid_zarr_partial):
    """The escape hatch must WORK on a partial plate — not die on planned-but-absent wells.

    This is the regression that matters most: Squid writes plate/well metadata up front from the
    acquisition PLAN, so the metadata here claims B4 and B5 (no directories) and fov 1 (no
    directory). Walking the metadata naively raises FileNotFoundError; intersecting it with the
    disk yields exactly what was acquired.
    """
    with pytest.warns(UserWarning):
        meta = open_reader(squid_zarr_partial, allow_incomplete=True).metadata
    assert meta["regions"] == ["B2", "B3"]                 # B4/B5 planned but never acquired
    assert meta["fovs_per_region"] == {"B2": [0], "B3": [0]}   # fov 1 planned, never acquired


def test_zarr_partial_plate_reads_the_wells_that_exist(squid_zarr_partial):
    with pytest.warns(UserWarning):
        r = open_reader(squid_zarr_partial, allow_incomplete=True)
        ch = r.metadata["channels"][0]["name"]
        plane = r.read("B2", 0, ch, 0)
    assert plane.shape == (4, 4)
    assert plane[0, 0] == _zarr_pixel_value(0, 0, 0, 0)


def test_zarr_discovery_falls_back_to_dirwalk_without_well_metadata(tmp_path):
    """A crash can leave a well directory with no zarr.json; its fields must still be found."""
    root = tmp_path / "acq_nowellmeta"
    _write_zarr_sidecars(root)
    _build_zarr_plate(root, regions=["B2"], fovs=[0, 1])
    (root / "plate.ome.zarr" / "B" / "2" / "zarr.json").unlink()
    assert open_reader(root).metadata["fovs_per_region"] == {"B2": [0, 1]}


def test_zarr_refuses_when_no_planned_well_exists(tmp_path):
    root = tmp_path / "acq_empty"
    _write_zarr_sidecars(root)
    _build_zarr_plate(root, regions=[], fovs=[], planned_regions=["B2", "B3"])
    with pytest.raises(ValueError, match="No acquired fields"):
        open_reader(root).metadata


# --- D2: the viewer seam -----------------------------------------------------
def test_zarr_exposes_fov_store_path_not_plane_ref(squid_zarr_dataset):
    root, _ = squid_zarr_dataset
    r = open_reader(root)
    assert r.supports_plane_ref is False
    assert not hasattr(r, "plane_ref")
    p = Path(r.fov_store_path("B3", 1))
    assert p.is_dir() and (p / "zarr.json").exists() and (p / "0").is_dir()
    with pytest.raises(KeyError):
        r.fov_store_path("ZZ9", 0)


def test_tiff_readers_still_support_plane_ref(squid_dataset):
    """The capability flag must not regress the TIFF path the viewer depends on."""
    root, _ = squid_dataset
    r = open_reader(root)
    assert getattr(r, "supports_plane_ref", True) is not False
    path, page = r.plane_ref("B2", 0, CH_IN_YAML, 0)
    assert page == 0 and Path(path).exists()


# --- D1 / D9: dispatch -------------------------------------------------------
def test_open_reader_refuses_per_fov_zarr_layout_by_name(tmp_path):
    (tmp_path / "zarr" / "B2" / "fov_0.ome.zarr").mkdir(parents=True)
    with pytest.raises(NotImplementedError, match="Per-FOV zarr layout"):
        open_reader(tmp_path)


def test_open_reader_refuses_6d_zarr_layout_by_name(tmp_path):
    (tmp_path / "zarr" / "B2" / "acquisition.zarr").mkdir(parents=True)
    with pytest.raises(NotImplementedError, match="6-D zarr layout"):
        open_reader(tmp_path)


def test_non_hcs_zarr_no_longer_reports_missing_tiffs(tmp_path):
    """REGRESSION: these layouts used to fall through to SquidReader and report missing TIFFs."""
    (tmp_path / "zarr" / "B2" / "fov_0.ome.zarr").mkdir(parents=True)
    with pytest.raises(NotImplementedError) as exc:
        open_reader(tmp_path)
    assert "individual-TIFF" not in str(exc.value)


def test_open_reader_refuses_squidmip_own_output(squid_dataset, tmp_path):
    """D9: write_plate output is structurally a Squid HCS plate; it must not be ingested."""
    from squidmip import write_plate

    root, _ = squid_dataset
    out = tmp_path / "out"
    write_plate(open_reader(root), out, tiff=False)
    assert (out / "plate.ome.zarr").is_dir()
    with pytest.raises(ValueError, match="looks like SquidMIP OUTPUT"):
        open_reader(out)


# --- integration: the real hongquan dataset (AC1 + AC4 on real data) ---------
@pytest.mark.integration
def test_real_dataset(real_dataset):
    reader = open_reader(real_dataset)
    meta = reader.metadata
    assert set(meta["regions"]) == {"B2", "B3", "B4"}
    total_fovs = sum(len(v) for v in meta["fovs_per_region"].values())
    assert total_fovs == 48
    assert len(meta["channels"]) == 4
    assert meta["n_z"] == 3
    assert meta["frame_shape"] == (4168, 4168)
    assert meta["dtype"] == np.uint16
    # AC4 exact read against tifffile
    got = reader.read("B3", 15, "Fluorescence_638_nm_-_Penta", 0)
    direct = tifffile.imread(real_dataset / "0" / "B3_15_0_Fluorescence_638_nm_-_Penta.tiff")
    np.testing.assert_array_equal(got, direct)

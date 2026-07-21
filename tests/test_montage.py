"""IMA-185 montage — clean-room unit tests (no reader, no acquisition on disk).

Fabricates a tiny real ``plate.ome.zarr`` with IMA-184's ``write_from_stream`` (a hand-built
metadata dict + stream), then drives ``build_montage`` over it and inspects the PNG + sidecar.
The real-seam cross commit (``write_plate`` on ``sim_1536wp`` + hongquan -> montage) lives in
tests/test_integration.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squidhcs import build_montage
from squidhcs._montage import _area_downsample, _hex_to_rgb01, _window
from squidhcs._output import write_from_stream

# Two channels: red (638) and blue (405), so composite color is unambiguous per channel.
CH = [
    {"name": "Fluorescence_638_nm_-_Penta", "display_name": "638", "display_color": "#FF0000"},
    {"name": "Fluorescence_405_nm_-_Penta", "display_name": "405", "display_color": "#20ADF8"},
]
Y = X = 8


def _meta(regions):
    return {
        "regions": regions,
        "fovs_per_region": {r: [0] for r in regions},
        "channels": CH,
        "pixel_size_um": 0.325,
    }


def _image(ch_levels, y=Y, x=X, dtype=np.uint16):
    """(1, C, 1, y, x): each channel filled with a flat constant from *ch_levels*."""
    out = np.zeros((1, len(ch_levels), 1, y, x), dtype=dtype)
    for c_i, level in enumerate(ch_levels):
        out[0, c_i, 0] = level
    return out


def _ramp(ch_levels, y=Y, x=X, dtype=np.uint16):
    """(1, C, 1, y, x): each channel a 0..level ramp, so the cell has real dynamic range."""
    grad = np.arange(y * x).reshape(y, x)
    out = np.zeros((1, len(ch_levels), 1, y, x), dtype=dtype)
    for c_i, level in enumerate(ch_levels):
        out[0, c_i, 0] = (grad * level // (y * x)).astype(dtype)
    return out


def _make_plate(tmp_path: Path, images: dict) -> Path:
    """Write a plate.ome.zarr from {region: image} and return the containing dir."""
    regions = list(images)
    stream = ((r, 0, images[r]) for r in regions)
    write_from_stream(_meta(regions), stream, tmp_path, n_fovs=1, tiff=False)
    return tmp_path


# --- pure helpers ---------------------------------------------------------------------------

def test_area_downsample_is_block_mean():
    # a 4x4 of four 2x2 quadrants (10, 20, 30, 40) -> 2x2 == the quadrant means
    plane = np.block([
        [np.full((2, 2), 10.0), np.full((2, 2), 20.0)],
        [np.full((2, 2), 30.0), np.full((2, 2), 40.0)],
    ])
    ds = _area_downsample(plane, 2, 2)
    np.testing.assert_allclose(ds, [[10, 20], [30, 40]])


def test_area_downsample_no_upsample():
    # asking for a larger output than the source returns the source (float), never invents pixels
    plane = np.arange(16, dtype=np.uint16).reshape(4, 4)
    out = _area_downsample(plane, 8, 8)
    assert out.shape == (4, 4)


def test_hex_to_rgb01_parses_and_fails_loud():
    np.testing.assert_allclose(_hex_to_rgb01("#FF0000"), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(_hex_to_rgb01("20ADF8"), [0x20 / 255, 0xAD / 255, 0xF8 / 255])
    with pytest.raises(ValueError):
        _hex_to_rgb01("#FFF")  # not 6 digits


def test_window_guards_flat_channel():
    # a flat channel (lo == hi) must not divide by zero — it maps to all-zero
    flat = np.full((4, 4), 500.0, dtype=np.float32)
    out = _window(flat, 500.0, 500.0)
    assert out.shape == (4, 4)
    assert np.all(out == 0.0)


# --- full montage over a fabricated plate ---------------------------------------------------

def test_montage_png_and_sidecar_shape(tmp_path):
    # 3 wells in row B, columns 2/3/10 -> grid 1x3 (natural column sort, no zero-pad)
    images = {"B2": _image([100, 0]), "B3": _image([0, 100]), "B10": _image([50, 50])}
    out = _make_plate(tmp_path, images)

    manifest = build_montage(out, cell_px=4)
    assert manifest["n_wells"] == 3
    assert manifest["grid"] == (1, 3)
    assert manifest["cell_px"] == 4

    from PIL import Image

    rgb = np.asarray(Image.open(manifest["montage"]))
    assert rgb.shape == (1 * 4, 3 * 4, 3)  # (n_rows*cell, n_cols*cell, RGB)

    side = json.loads(Path(manifest["sidecar"]).read_text())
    assert side["grid"] == {"n_rows": 1, "n_cols": 3, "rows": ["B"], "columns": ["2", "3", "10"]}
    assert {w["well_id"] for w in side["wells"]} == {"B2", "B3", "B10"}
    # every well's bbox is inside the canvas and cell-sized
    for w in side["wells"]:
        assert (w["x1"] - w["x0"], w["y1"] - w["y0"]) == (4, 4)
        assert 0 <= w["x0"] < w["x1"] <= 3 * 4 and 0 <= w["y0"] < w["y1"] <= 1 * 4


def test_montage_composite_colors_follow_display_color(tmp_path):
    # B2 lit only in the 638=red channel, B3 only in 405=blue. The montage cells must be
    # red-dominant and blue-dominant respectively — colors come from display_color.
    images = {"B2": _image([300, 0]), "B3": _image([0, 300])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)

    from PIL import Image

    rgb = np.asarray(Image.open(manifest["montage"])).astype(int)
    side = {w["well_id"]: w for w in json.loads(Path(manifest["sidecar"]).read_text())["wells"]}

    def cell_mean(w):
        return rgb[w["y0"] : w["y1"], w["x0"] : w["x1"]].reshape(-1, 3).mean(axis=0)

    r_cell = cell_mean(side["B2"])
    b_cell = cell_mean(side["B3"])
    assert r_cell[0] > r_cell[2]           # B2 red channel dominates blue
    assert b_cell[2] > b_cell[0]           # B3 blue dominates red (405 color is blue-ish)


def test_montage_global_contrast_preserves_relative_brightness(tmp_path):
    # Same channel, one bright well and one dim well. GLOBAL-per-channel contrast must keep the
    # bright well brighter — a per-well window would wrongly equalize them.
    images = {"B2": _image([1000, 0]), "B3": _image([100, 0])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)

    from PIL import Image

    rgb = np.asarray(Image.open(manifest["montage"])).astype(int)
    side = {w["well_id"]: w for w in json.loads(Path(manifest["sidecar"]).read_text())["wells"]}
    bright = rgb[:, side["B2"]["x0"] : side["B2"]["x1"]].max()
    dim = rgb[:, side["B3"]["x0"] : side["B3"]["x1"]].max()
    assert bright > dim


def test_montage_blank_cells_are_black(tmp_path):
    # Wells B2 and C3 -> grid rows [B,C] x cols [2,3]; only (B,2) and (C,3) filled.
    # The two empty intersections must render pure black (no well there). Ramps (not flat
    # constants) so the filled cells have real dynamic range and don't map to black.
    images = {"B2": _ramp([600, 600]), "C3": _ramp([600, 600])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)
    assert manifest["grid"] == (2, 2)

    from PIL import Image

    rgb = np.asarray(Image.open(manifest["montage"]))
    # (row B=0, col 3=1) and (row C=1, col 2=0) are the blanks
    assert rgb[0:4, 4:8].sum() == 0
    assert rgb[4:8, 0:4].sum() == 0
    # a filled cell is not all black
    assert rgb[0:4, 0:4].sum() > 0


def test_montage_emits_self_contained_hover_viewer(tmp_path):
    # build_montage also writes a zero-dependency HTML viewer that maps a hover to a well id
    # from the region-jump sidecar geometry. Assert it is emitted, self-contained, and carries
    # the well ids + the cursor-locate handler (no external fetch, so the data is inlined).
    images = {"B2": _ramp([300, 0]), "C3": _ramp([0, 300])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)

    viewer = Path(manifest["viewer"])
    assert viewer.name == "plate_montage.html" and viewer.exists()
    html = viewer.read_text()
    assert '<img id="montage" src="plate_montage.png"' in html  # points at the montage, same dir
    assert "mousemove" in html and "getBoundingClientRect" in html  # hover indicator wired
    assert '"well_id": "B2"' in html and '"well_id": "C3"' in html  # sidecar geometry inlined
    assert "http://" not in html and "https://" not in html  # self-contained, no external deps


def test_montage_sidecar_records_per_channel_window(tmp_path):
    images = {"B2": _image([200, 50])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)
    side = json.loads(Path(manifest["sidecar"]).read_text())
    assert [c["color"] for c in side["channels"]] == ["FF0000", "20ADF8"]
    for c in side["channels"]:
        assert c["window"]["high"] >= c["window"]["low"]  # a real, ordered window per channel


def test_montage_memory_is_bounded_not_whole_plate(tmp_path):
    """build_montage reads ONE well at a time: peak ~ canvas + one well, never all N wells.

    Feeds a plate of N sizable wells and checks the peak stays far below "all wells resident"
    (N x one-well bytes). Proves the montage never accumulates the full-res plate — it holds one
    well plus the downsampled montage canvas (which scales with montage resolution, not plate size).
    """
    import tracemalloc

    N, F = 20, 512  # 20 wells, 512x512 frames -> "all resident" would be obvious
    ch = [{"name": "c0", "display_name": "c0", "display_color": "#FF0000"}]
    regions = [f"A{i + 1}" for i in range(N)]
    meta = {"regions": regions, "fovs_per_region": {r: [0] for r in regions}, "channels": ch,
            "pixel_size_um": 0.325}

    def _img():
        a = np.zeros((1, 1, 1, F, F), np.uint16)
        a[0, 0, 0] = (np.arange(F * F).reshape(F, F) % 1000).astype(np.uint16)
        return a

    write_from_stream(meta, ((r, 0, _img()) for r in regions), tmp_path, n_fovs=1, tiff=False)
    well_bytes = F * F * 2

    tracemalloc.start()
    build_montage(tmp_path, cell_px=48)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # holding all N wells would be N*well_bytes; assert peak stays well under a quarter of that.
    assert peak < N * well_bytes * 0.25, f"montage peak {peak} not bounded (all wells = {N * well_bytes})"


# --- fail loud ------------------------------------------------------------------------------

def test_build_montage_rejects_non_plate(tmp_path):
    (tmp_path / "not_a_plate").mkdir()
    with pytest.raises(ValueError, match="HCS plate"):
        build_montage(tmp_path / "not_a_plate")


def test_build_montage_rejects_bad_cell_px(tmp_path):
    images = {"B2": _image([100, 100])}
    out = _make_plate(tmp_path, images)
    with pytest.raises(ValueError, match="cell_px"):
        build_montage(out, cell_px=0)


def test_montage_small_field_no_crash_no_nan(tmp_path):
    # Audit MED: a field SMALLER than cell_px (128) must corner-place, not broadcast-crash or
    # divide-by-zero into NaN (which would blacken a whole channel). Square-small + non-square.
    from PIL import Image
    imgs = {"B2": _ramp([1, 2], y=40, x=40), "B3": _ramp([1, 2], y=100, x=60)}
    manifest = build_montage(_make_plate(tmp_path, imgs))   # default cell_px=128 > field -> corner-place
    arr = np.asarray(Image.open(manifest["montage"]))
    assert arr.ndim == 3 and int(arr.max()) > 0             # rendered, not crashed / all-black NaN

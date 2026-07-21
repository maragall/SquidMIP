"""Shared test fixtures.

`squid_dataset` builds a tiny, real-shaped Squid individual-TIFF acquisition on disk
(2 regions x 2 fov x 2 z x 2 channels, 4x4 uint16 frames) with a legacy-schema
coordinates.csv and a pre-v1.0 (camera_settings-nested color) acquisition_channels.yaml,
plus the acquisition parameters.json scalars. Returns (root_path, {(region,fov,z,ch): array}).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

REGIONS = ["B2", "B3"]
FOVS = [0, 1]
NZ = 2
# One channel present in the YAML (color via nested camera_settings), one ABSENT from the
# YAML (exercises the CHANNEL_COLORS_MAP wavelength fallback). Both contain '_' and '-'.
CH_IN_YAML = "Fluorescence_638_nm_-_Penta"
CH_NOT_IN_YAML = "Fluorescence_561_nm_-_Penta"
CHANNELS = [CH_IN_YAML, CH_NOT_IN_YAML]

_YAML = """\
version: 1
objective: 20x
channels:
- name: Fluorescence 638 nm - Penta
  camera_settings:
    '1':
      display_color: '#FF0000'
      exposure_time_ms: 50.0
"""

# Legacy flat sidecar (fallback source). Note magnification/sensor -> recomputed px 0.188,
# deliberately DIFFERENT from acquisition.yaml's stored 0.325 so tests prove which is used.
_PARAMS = {
    "Nz": NZ,
    "Nt": 1,
    "dz(um)": 1.5,
    "objective": {"magnification": 20.0},
    "sensor_pixel_size_um": 3.76,
}

# Authoritative rich metadata. pixel_size_um is stored (binning-aware), not recomputed.
_ACQ_YAML = """\
objective:
  pixel_size_um: 0.325
  magnification: 20.0
  sensor_pixel_size_um: 3.76
sample:
  wellplate_format: 1536 well plate
z_stack:
  nz: 2
  delta_z_mm: 0.0015
time_series:
  nt: 1
"""


def _pixel_value(r_i, fov, z, c_i):
    # deterministic, unique per plane so exact-read comparisons are meaningful
    return r_i * 1000 + fov * 100 + z * 10 + c_i


def _write_timepoint(folder: Path, arrays: dict, tag: int = 0):
    folder.mkdir(parents=True, exist_ok=True)
    for r_i, region in enumerate(REGIONS):
        for fov in FOVS:
            for z in range(NZ):
                for c_i, ch in enumerate(CHANNELS):
                    base = _pixel_value(r_i, fov, z, c_i) + tag * 5000
                    arr = (np.arange(16, dtype=np.uint16).reshape(4, 4) + base).astype(np.uint16)
                    tifffile.imwrite(folder / f"{region}_{fov}_{z}_{ch}.tiff", arr)
                    arrays[(region, fov, z, ch)] = arr


@pytest.fixture
def squid_dataset(tmp_path):
    root = tmp_path / "acq"
    arrays: dict = {}
    _write_timepoint(root / "0", arrays, tag=0)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    return root, arrays


# --- IMA-229: Squid HCS Zarr v3 fixtures -------------------------------------------------
#
# Written BY HAND to match SQUID's writer, deliberately NOT generated with squidmip.write_plate.
# Our writer and Squid's differ in exactly the ways the reader must notice: Squid stamps a
# `_squid` attributes block and writes no `plate.field_count`; we write `field_count` and no
# `_squid`. A fixture produced by the writer we control would validate the reader against
# ourselves and hide the bugs this ticket exists to prevent.
#
# Shape mirrors zarr_writer.py: unpadded columns (B2 -> B/2), group metadata at {fov}/zarr.json
# (not on the array), omero.channels[] in C-axis order, datasets[0].path == "0".

ZARR_REGIONS = ["B2", "B3"]
ZARR_FOVS = [0, 1]
ZARR_NT, ZARR_NC, ZARR_NZ, ZARR_NY, ZARR_NX = 1, 2, 2, 4, 4


def _zarr_group(path: Path, attributes: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": attributes}, indent=2))


def _squid_field_attrs(*, channel_names, complete=True, aborted=False,
                       pixel_size_um=0.325, z_step_um=1.5, dtype="uint16"):
    """The attributes Squid writes on a field group: ome.multiscales + ome.omero + _squid."""
    squid = {
        "structure": "5D-TCZYX",
        "pixel_size_um": pixel_size_um,
        "z_step_um": z_step_um,
        "time_increment_s": None,
        "chunk_mode": "full_frame",
        "compression": "fast",
        "shape": [ZARR_NT, ZARR_NC, ZARR_NZ, ZARR_NY, ZARR_NX],
        "dtype": dtype,
        "is_hcs": True,
        "acquisition_complete": bool(complete),
    }
    if aborted:
        squid["aborted"] = True
    return {
        "ome": {
            "version": "0.5",
            "multiscales": [{
                "version": "0.5",
                "name": "0",
                "axes": [
                    {"name": "t", "type": "time", "unit": "second"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [{"path": "0", "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, z_step_um, pixel_size_um, pixel_size_um]}]}],
                "coordinateTransformations": [{"type": "identity"}],
            }],
            "omero": {
                "name": "0", "version": "0.5",
                # C-axis order — this list IS the channel order (zarr_writer builds it by
                # iterating the C axis).
                "channels": [{"label": n, "active": True, "color": "00FF00",
                              "window": {"start": 0, "end": 65535, "min": 0, "max": 65535}}
                             for n in channel_names],
            },
        },
        "_squid": squid,
    }


def _zarr_pixel_value(r_i, fov, z, c_i):
    return r_i * 1000 + fov * 100 + z * 10 + c_i


def _write_zarr_field(field_dir: Path, r_i, fov, *, dtype=np.uint16, arrays=None):
    """Write the {fov}/0 array with deterministic, per-plane-unique values."""
    from squidmip._zarr_store import create_array, write_array

    data = np.zeros((ZARR_NT, ZARR_NC, ZARR_NZ, ZARR_NY, ZARR_NX), dtype=dtype)
    for c_i in range(ZARR_NC):
        for z in range(ZARR_NZ):
            base = _zarr_pixel_value(r_i, fov, z, c_i)
            plane = (np.arange(ZARR_NY * ZARR_NX).reshape(ZARR_NY, ZARR_NX) + base).astype(dtype)
            data[0, c_i, z] = plane
            if arrays is not None:
                arrays[(f"{ZARR_REGIONS[r_i]}", fov, z, c_i)] = plane
    store = create_array(field_dir / "0", data.shape, dtype)
    write_array(store, data)


def _build_zarr_plate(root: Path, *, regions=None, fovs=None, planned_regions=None,
                      planned_fovs=None, complete=True, aborted=False, dtype=np.uint16,
                      channel_names=None, arrays=None, pixel_size_um=0.325):
    """Build <root>/plate.ome.zarr.

    ``planned_regions`` / ``planned_fovs`` default to the acquired ones. Passing MORE planned
    than acquired reproduces Squid's real partial-plate shape: metadata written up front from
    the plan, directories only for what was actually reached.
    """
    regions = regions if regions is not None else ZARR_REGIONS
    fovs = fovs if fovs is not None else ZARR_FOVS
    planned_regions = planned_regions if planned_regions is not None else regions
    planned_fovs = planned_fovs if planned_fovs is not None else fovs
    channel_names = channel_names if channel_names is not None else [CH_IN_YAML, CH_NOT_IN_YAML]

    plate = root / "plate.ome.zarr"
    rows = sorted({r[0] for r in planned_regions})
    cols = sorted({r[1:] for r in planned_regions}, key=int)
    _zarr_group(plate, {"ome": {"version": "0.5", "plate": {
        "version": "0.5", "name": "plate",
        "rows": [{"name": r} for r in rows],
        "columns": [{"name": c} for c in cols],
        # NOTE: no "field_count" — Squid does not write it. Our own writer does, which is
        # exactly how open_reader tells an acquisition from SquidMIP output.
        "wells": [{"path": f"{r[0]}/{r[1:]}", "rowIndex": rows.index(r[0]),
                   "columnIndex": cols.index(r[1:])} for r in planned_regions],
    }}})

    for region in planned_regions:
        row, col = region[0], region[1:]
        well_dir = plate / row / col
        if region in regions:
            _zarr_group(well_dir, {"ome": {"version": "0.5", "well": {
                "version": "0.5",
                # PLANNED fields — list(range(fov_count)) in Squid, regardless of what landed.
                "images": [{"path": str(f)} for f in planned_fovs],
            }}})
            r_i = ZARR_REGIONS.index(region) if region in ZARR_REGIONS else 0
            for fov in fovs:
                field_dir = well_dir / str(fov)
                _zarr_group(field_dir, _squid_field_attrs(
                    channel_names=[c.replace("_", " ") for c in channel_names],
                    complete=complete, aborted=aborted, dtype=np.dtype(dtype).name,
                    pixel_size_um=pixel_size_um))
                _write_zarr_field(field_dir, r_i, fov, dtype=dtype, arrays=arrays)
        # region planned but NOT acquired -> no directory at all (the real partial-plate shape)
    return plate


def _write_zarr_sidecars(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)


@pytest.fixture
def squid_zarr_dataset(tmp_path):
    """A complete 2-region x 2-FOV Squid HCS zarr acquisition. Returns (root, {key: plane})."""
    root = tmp_path / "zarr_acq"
    arrays: dict = {}
    _write_zarr_sidecars(root)
    _build_zarr_plate(root, arrays=arrays)
    return root, arrays


@pytest.fixture
def squid_zarr_partial(tmp_path):
    """A PARTIAL plate: metadata plans 4 wells x 2 FOVs, only B2/B3 x fov 0 exist on disk.

    This is the fixture that proves the discovery intersection. Squid writes plate and well
    metadata up front from the plan (job_processing.py), so a crashed run leaves metadata
    claiming wells and fields whose directories were never created.
    """
    root = tmp_path / "zarr_partial"
    _write_zarr_sidecars(root)
    _build_zarr_plate(
        root,
        regions=["B2", "B3"], fovs=[0],
        planned_regions=["B2", "B3", "B4", "B5"], planned_fovs=[0, 1],
        complete=False,
    )
    return root


@pytest.fixture
def real_dataset():
    """The real hongquan dataset if present locally; else skip (used by integration tests)."""
    path = Path.home() / "Downloads" / "z_stack_2026-05-15_18-39-28.532906 hongquan"
    if not path.is_dir():
        pytest.skip("real hongquan dataset not present")
    return path

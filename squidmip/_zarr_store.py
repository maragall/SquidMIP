"""Low-level zarr v3 store + NGFF group primitives (vendored from tilefusion io/zarr.py).

Vendored, NOT imported: importing ``tilefusion`` runs its heavy ``__init__`` (numba's
threading-layer pin, GPU/``cupy`` probes, ``basicpy``), which would make SquidMIP fail to
install/run on a machine without those. ``create_array`` here is a thin tensorstore-config
wrapper (the substantive reuse); the group writers are plain ``zarr.json`` JSON.

Two node kinds in an OME-NGFF v0.5 / zarr-v3 store:
  * arrays  — created by ``create_array`` (tensorstore writes the array ``zarr.json`` + chunks)
  * groups  — plain ``zarr.json`` with ``node_type: group`` + optional ``attributes.ome``,
              written by ``write_group`` (plate / well / row / field-image groups).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import tensorstore as ts

# Full-res arrays are chunked to this per-plane tile so a viewer reads a region without
# pulling the whole (Y, X) plane; downsample levels are smaller so they clamp to their shape.
_CHUNK_YX = 1024

# The canonical Squid image axes. Kept as the default so every image/pyramid call site is
# unchanged; sidecar arrays (IMA-231 ROI tables) override it.
_IMAGE_DIMS = ("t", "c", "z", "y", "x")


def create_array(
    path,
    shape: Sequence[int],
    dtype,
    *,
    chunk: Optional[Sequence[int]] = None,
    max_workers: int = 4,
    dimension_names: Optional[Sequence[str]] = _IMAGE_DIMS,
) -> ts.TensorStore:
    """Create a zarr v3 array at *path* (blosc-zstd) and return an open tensorstore handle.

    Shape defaults to 5-D ``(t, c, z, y, x)`` (Squid canonical order). ``chunk`` defaults to one
    ``(1, 1, 1, <=1024, <=1024)`` tile; every chunk dim is clamped into ``[1, shape_i]`` so
    tiny arrays (e.g. 4x4 test frames) and odd shapes are always valid. ``delete_existing``
    makes a rewrite idempotent (a rerun overwrites cleanly).

    ``dimension_names`` must have the same rank as *shape* — zarr v3 rejects a mismatch. It
    defaults to the 5-D image axes, so every existing image/pyramid call is unchanged. Sidecar
    arrays of a different rank (IMA-231's ROI table writes ``(1, 6)`` and 1-D index arrays) pass
    their own names, or ``None`` to omit the key entirely. Without this the store was image-only:
    ``create_array(p, (1, 6), float64)`` raised ``"dimension_names": Array has length 5 but
    should have length 2``.
    """
    shape = tuple(int(s) for s in shape)
    if chunk is None:
        if len(shape) < 2:
            chunk = shape or (1,)                      # 1-D/0-D sidecar: one chunk holds it all
        else:
            y, x = shape[-2], shape[-1]
            # Image default keeps its (1, 1, 1, y, x) tiling; a lower-rank array takes the
            # trailing part of that pattern so the leading 1s never pad a shorter shape.
            chunk = (1, 1, 1, min(y, _CHUNK_YX), min(x, _CHUNK_YX))[-len(shape):]
    chunk = tuple(max(1, min(int(c), int(s))) for c, s in zip(chunk, shape))
    dt = np.dtype(dtype)
    if dimension_names is not None and len(dimension_names) != len(shape):
        raise ValueError(
            f"dimension_names {list(dimension_names)} has rank {len(dimension_names)} but shape "
            f"{shape} has rank {len(shape)}; zarr v3 requires them to match. Pass matching names "
            "or dimension_names=None for an unnamed sidecar array."
        )

    config: dict[str, Any] = {
        "context": {
            "file_io_concurrency": {"limit": max_workers},
            "data_copy_concurrency": {"limit": max_workers},
        },
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {
            "shape": list(shape),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunk)}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {
                    "name": "blosc",
                    "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle"},
                },
            ],
            "data_type": dt.name,
        },
    }
    if dimension_names is not None:
        config["metadata"]["dimension_names"] = list(dimension_names)
    # create (overwriting any prior store so a rerun is idempotent). delete_existing may not be
    # combined with open=True, and create already returns an open, writable handle.
    return ts.open(config, create=True, delete_existing=True).result()


def write_array(store: ts.TensorStore, data: np.ndarray) -> None:
    """Write a whole array into an open store (contiguous copy so tensorstore is happy)."""
    store[...].write(np.ascontiguousarray(data)).result()


def write_group(path, ome: Optional[dict] = None, attributes: Optional[dict] = None) -> None:
    """Write a zarr v3 group ``zarr.json`` at *path*, with optional ``attributes.ome``.

    A bare group (``ome=None``) is a structural node (plate row); an ``ome`` payload carries
    the plate / well / multiscales+omero metadata that ndviewer and ome-zarr readers consume.

    ``attributes`` merges RAW top-level keys alongside ``ome``. OME-NGFF puts everything under
    ``ome``, but the table sidecars this store also has to write do not: the ngio/Fractal table
    spec wants ``tables`` on the tables group and anndata wants ``encoding-type`` at the top
    level of the table group. Both are omitted when None, so existing callers are byte-identical.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"zarr_format": 3, "node_type": "group", "attributes": {}}
    if ome is not None:
        doc["attributes"]["ome"] = ome
    if attributes:
        doc["attributes"].update(attributes)
    (path / "zarr.json").write_text(json.dumps(doc, indent=2))


def merge_array_attributes(path, attributes: dict) -> None:
    """Merge *attributes* into an already-written array's ``zarr.json``.

    tensorstore owns the array ``zarr.json`` and offers no hook for user attributes, so anndata's
    per-array ``encoding-type`` markers are merged in afterwards. Read-modify-write of a small
    JSON file the writer thread just created — no other writer touches it (wells write to
    disjoint directories).
    """
    p = Path(path) / "zarr.json"
    doc = json.loads(p.read_text())
    doc.setdefault("attributes", {}).update(attributes)
    p.write_text(json.dumps(doc, indent=2))


def write_string_array(path, values: Sequence[str], attributes: Optional[dict] = None) -> None:
    """Write a 1-D zarr v3 ``string`` array (vlen-utf8), by hand.

    tensorstore cannot do this: its zarr3 driver rejects ``data_type: "string"`` outright
    ("string data type is not one of the supported data types"), and these index arrays are
    mandatory in anndata's on-disk dataframe encoding. They are tiny (one row label per FOV, six
    column names), so a hand-rolled single-chunk writer is the right size of tool.

    Chunk payload is the numcodecs vlen-utf8 layout that zarr-python / anndata read back::

        uint32 little-endian  element count
        per element:  uint32 little-endian byte length, then the UTF-8 bytes
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    values = [str(v) for v in values]
    n = len(values)
    doc: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [n],
        "data_type": "string",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [max(1, n)]}},
        "chunk_key_encoding": {"name": "default"},
        "fill_value": "",
        "codecs": [{"name": "vlen-utf8"}],
        "attributes": dict(attributes or {}),
    }
    (path / "zarr.json").write_text(json.dumps(doc, indent=2))

    buf = bytearray(struct.pack("<I", n))
    for v in values:
        raw = v.encode("utf-8")
        buf += struct.pack("<I", len(raw)) + raw
    chunk_dir = path / "c"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "0").write_bytes(bytes(buf))

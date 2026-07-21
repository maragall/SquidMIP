"""Shared plate-shaped primitives used by every reader (IMA-229 T1).

Three readers now derive the same ``metadata`` contract from three very different on-disk
layouts. What they legitimately share is the *shape* of the answer, not how they find it:

  * ``plate_key``      — true plate row-major ordering for well ids
  * ``group_regions``  — {region: {fov, ...}} -> sorted regions + fovs_per_region
  * ``build_metadata`` — the one definition of the eleven-key metadata dict
  * ``cross_check_nz`` / ``cross_check_nt`` — the recorded-vs-observed warnings

Before this module the first three existed twice (``SquidReader.metadata`` and
``SquidOMEReader.metadata``) and a third copy was about to be written for zarr. The risk was
never ugliness: it was that a fix to the contract would land in one or two copies and the
readers would quietly start disagreeing.

Also home to :func:`read_group_ome`, the zarr-v3 group-metadata accessor shared by the
OME-zarr *writer's* consumers (``_montage``) and the OME-zarr *reader* (``reader``).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Iterable, Optional

_WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def plate_key(region: str):
    """Sort well ids in true plate ROW-MAJOR order: A,B,...,Z,AA,AB,... with the column by
    integer (so B2 < B3 < B10, and B < AA — single-letter rows before double-letter, not
    lexicographic where "AA" < "B"). Downstream consumers (projection engine, plate viewer)
    then process wells top-to-bottom, left-to-right. Non-well-plate region names fall back
    after the plate wells.
    """
    m = _WELL_RE.match(region)
    if not m:
        return (1, len(region), region, 0)          # non-plate ids: stable, after the wells
    return (0, len(m.group(1)), m.group(1).upper(), int(m.group(2)))


def group_regions(pairs: Iterable[tuple]) -> tuple[list, dict]:
    """``[(region, fov), ...]`` -> ``(regions_row_major, {region: [fov, ...]})``.

    Filesystem and metadata iteration order are not stable, so both levels are sorted here
    rather than at each call site.
    """
    fovs: dict[str, set] = {}
    for region, fov in pairs:
        fovs.setdefault(str(region), set()).add(int(fov))
    regions = sorted(fovs, key=plate_key)
    return regions, {r: sorted(fovs[r]) for r in regions}


def cross_check_nz(declared: Optional[int], observed: int, source: str) -> None:
    """Warn when the recorded Nz disagrees with what the data actually contains.

    The data is always ground truth; the recorded value is a cross-check. *source* names where
    the observed count came from, so the warning says which one won.
    """
    if declared is not None and int(declared) != int(observed):
        warnings.warn(
            f"Recorded Nz ({declared}) != {source} ({observed}); using the observed value."
        )


def cross_check_nt(declared: Optional[int], observed: int, source: str) -> None:
    """Warn when the recorded Nt disagrees with what the data actually contains."""
    if declared is not None and int(declared) != int(observed):
        warnings.warn(
            f"Recorded Nt ({declared}) != {source} ({observed}); using the observed value."
        )


def build_metadata(
    *,
    regions: list,
    fovs_per_region: dict,
    channels: list,
    z_levels: list,
    frame_shape: tuple,
    dtype,
    n_t: int,
    acq: dict,
) -> dict:
    """The single definition of the reader ``metadata`` contract.

    Every reader returns exactly these eleven keys, so the engine, CLI, writer and viewer
    consume any of them unchanged. ``acq`` is ``load_acquisition_metadata()``'s output — the
    authoritative source for the physical scalars (see ``_acquisition`` for why the yaml wins
    over anything recomputed or embedded).
    """
    z_levels = list(z_levels)
    return {
        "regions": list(regions),
        "fovs_per_region": dict(fovs_per_region),
        "channels": list(channels),
        "n_z": len(z_levels),
        "z_levels": z_levels,
        "dz_um": acq["dz_um"],
        "pixel_size_um": acq["pixel_size_um"],   # authoritative (acquisition.yaml), not recomputed
        "wellplate_format": acq["wellplate_format"],
        "frame_shape": tuple(frame_shape),
        "dtype": dtype,
        "n_t": int(n_t),
    }


def read_group_ome(group_dir) -> dict:
    """Return the ``attributes.ome`` dict of a zarr v3 group, or {} if absent/unreadable.

    A missing or malformed ``zarr.json`` is normal on a partial acquisition (the well was
    planned but never reached), so this returns {} rather than raising — callers decide whether
    an empty answer is fatal.
    """
    path = Path(group_dir) / "zarr.json"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return doc.get("attributes", {}).get("ome", {})


def read_group_attrs(group_dir) -> dict:
    """Return the whole ``attributes`` dict of a zarr v3 group (``ome`` plus vendor blocks such
    as Squid's ``_squid``), or {} if absent/unreadable."""
    path = Path(group_dir) / "zarr.json"
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return doc.get("attributes", {})

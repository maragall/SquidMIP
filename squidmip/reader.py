"""SquidMIP reader: format-aware ingest for Squid individual-TIFF acquisitions.

``open_reader(path)`` dispatches on the on-disk format and returns a reader. Only the
individual-TIFFs layout is implemented (IMA-189); the other Squid output formats
(multi-page TIFF, OME-TIFF, Zarr) are detected and rejected with a clear message, marking
the seam where future readers plug in.

Individual-TIFFs layout (one channel per file), verified against real data::

    <acq>/
    ├── acquisition parameters.json
    ├── acquisition_channels.yaml
    ├── coordinates.csv
    └── 0/                                    # timepoint folder (1/, 2/, … if Nt>1)
        └── {region}_{fov}_{z}_{channel}.tiff

Discovery flow::

    open_reader ──► detect format ──► SquidReader
                                          │
        glob timepoint folders (0/,1/…) ──┤─► n_t
        glob *.tiff in t0, parse stems ───┤─► regions, fovs_per_region, channels, z-levels
        read ONE frame ──────────────────┤─► frame_shape, dtype   (NOT hardcoded)
        coordinates.csv (dedup by x,y) ───┤─► fov_positions_um {(region,fov): (x_um, y_um)}
        acquisition.yaml (or JSON) ───────┴─► dz_um, pixel_size_um, wellplate_format, Nz/Nt cross-check

The (region, fov, z, channel) index is parsed from FILENAMES — the ground truth. Scalar
metadata comes from acquisition.yaml (authoritative pixel size etc.), the flat JSON as a
legacy fallback. read() constructs the path directly and returns exactly what tifffile
decodes (native dtype), refusing non-2D planes and dtypes outside {uint8, uint16}.

``coordinates.csv`` IS read (IMA-187), into ``metadata["fov_positions_um"]``, so multiple FOVs
per region can be placed at their true stage offsets. TWO schemas ship in real Squid output and
both parse (IMA-215), discriminated by the HEADER and never by row count::

    (a) region,fov,z_level,x (mm),y (mm),z (um),time     # fov id STATED
    (b) region,x (mm),y (mm),z (mm)                      # fov id is ROW ORDER

In (a) the ``fov`` column is the row -> image mapping, so row order is irrelevant; repeats (one
row per z-level) collapse, and a repeat that disagrees about x/y is a hard error.

In (b) there is NO ``fov`` column — the only link from a row to an image is position within the
region's rows, so the Nth row of a region maps to that region's Nth sorted FOV. Rows are
de-duplicated on (region, x, y) FIRST, because a multi-z / multi-timepoint acquisition can
repeat a stage position once per z-level; comparing raw row counts against FOV counts would
then fail on every genuine z-stack. The de-duplicated count IS cross-checked and fails loud
on a mismatch: a wrong mapping does not crash, it silently draws a scrambled mosaic. That
failure is CONTAINED to the coordinate half of the metadata (``_fov_positions_um_or_empty``):
a truncated CSV costs placement, not the whole acquisition.

Units: coordinates.csv records MILLIMETRES, but this package's world space is MICROMETRES and
every world-space key ends in ``_um`` (see ``squidmip/_tiling.py``). The conversion happens
here, at the producer, and the metadata key is ``fov_positions_um``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from squidmip._acquisition import load_acquisition_metadata
from squidmip._channels import load_channel_yaml, resolve_channels

# region has no underscore; fov and z are ints; channel is the remainder (may contain _ and -).
_STEM_RE = re.compile(r"^(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)$")
_TIFF_SUFFIXES = (".tiff", ".tif")

# Squid grayscale planes are MONO8 (uint8) or MONO12/MONO16 (uint16); see
# software/squid/camera/utils.py get_available_pixel_formats. It never writes uint32/float
# grayscale (RGB formats are color -> ndim>2, rejected separately). We preserve the native
# dtype but refuse anything outside this set so a non-raw stack can't be silently projected.
_SUPPORTED_DTYPES = (np.dtype("uint8"), np.dtype("uint16"))


def _validate_plane(arr, path: Path):
    """Guard a decoded plane: 2D grayscale, dtype uint8/uint16. Returns arr unchanged."""
    if arr.ndim != 2:
        raise ValueError(
            f"{path.name} is not a 2D grayscale plane (shape {arr.shape}); "
            "color/RGB (brightfield) channels are not supported (deferred)."
        )
    if arr.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"{path.name} has dtype {arr.dtype}; Squid writes uint8 (MONO8) or uint16 "
            "(MONO12/MONO16). An unexpected dtype (e.g. uint32/float) usually means the input "
            "is not a raw Squid capture; refused rather than silently projected."
        )
    return arr


_COORDS_NAME = "coordinates.csv"
# Header tolerance: Squid writes "x (mm)"/"y (mm)", but whitespace and case drift across
# generations. Match on the leading axis letter of a column that mentions mm.
_X_COL_RE = re.compile(r"^\s*x\b.*\(\s*mm\s*\)", re.I)
_Y_COL_RE = re.compile(r"^\s*y\b.*\(\s*mm\s*\)", re.I)


def _coord_columns(fieldnames) -> tuple[str, str]:
    """Locate the x/y millimetre columns in a coordinates.csv header, failing loud if absent."""
    names = list(fieldnames or [])
    x = next((n for n in names if n and _X_COL_RE.match(n)), None)
    y = next((n for n in names if n and _Y_COL_RE.match(n)), None)
    if x is None or y is None:
        raise ValueError(
            f"{_COORDS_NAME} has no recognisable x/y millimetre columns (header: {names}). "
            "Expected something like 'x (mm)' and 'y (mm)'; without them FOVs cannot be placed."
        )
    return x, y


# IMA-215: the second real on-disk schema carries an explicit ``fov`` column. Its presence in the
# HEADER is the whole format signal — see _fov_column.
_FOV_COL_RE = re.compile(r"^\s*fov\s*$", re.I)


def _fov_column(fieldnames):
    """The explicit ``fov`` column if this coordinates.csv has one, else ``None``.

    This single header lookup IS the format discriminator (IMA-215). Two schemas ship in real
    Squid output:

        (a) ``region,fov,z_level,x (mm),y (mm),z (um),time``   — the fov id is STATED
        (b) ``region,x (mm),y (mm),z (mm)``                    — the fov id is row ORDER

    Detection must be on the header and never on row counts. Row counts are data: a type-(a) file
    can happen to have exactly one row per FOV (a single-z acquisition), and a type-(b) file can
    happen to have a count that looks like a z-repeat pattern. Guessing from them would silently
    swap the two mappings, and a swapped mapping does not crash — it draws every tile in the wrong
    place. The header is the schema; the schema decides.
    """
    return next((n for n in list(fieldnames or []) if n and _FOV_COL_RE.match(n)), None)


_MM_TO_UM = 1000.0


def _parse_mm_pair(raw_x: str, raw_y: str, region: str, line_no: int):
    """``(x_mm, y_mm)`` floats, or ``None`` for a blank (position-less) row. Raises on garbage."""
    if not raw_x or not raw_y:
        return None                     # a blank position row carries no placement info
    try:
        return float(raw_x), float(raw_y)
    except ValueError:
        raise ValueError(
            f"{_COORDS_NAME} line {line_no}: region {region!r} has non-numeric "
            f"coordinates ({raw_x!r}, {raw_y!r}); refusing to guess a stage position."
        ) from None


def _positions_from_fov_column(reader, fovs_per_region: dict, fov_col, x_col, y_col) -> dict:
    """Parse the type-(a) ("monkey") schema, where each row STATES its FOV id (IMA-215).

    ``region,fov,z_level,x (mm),y (mm),z (um),time``. This is the strictly better of the two
    schemas: the row -> FOV mapping is declared, so row order is irrelevant and a shuffled or
    interleaved file still places correctly. The positional fallback in
    :func:`load_fov_positions_um` exists only because the type-(b) schema has nothing else to go on.

    Three properties are enforced, all for the same reason the positional path enforces its count:
    a wrong mapping renders a plausible-looking, wrong mosaic and nothing downstream can catch it.

    1. **Repeats of the same FOV are collapsed, not counted.** A z-stack writes one row per
       z-level per FOV (the real 10x dataset: 550 rows, 55 FOVs, Nz=10). The first row for a FOV
       wins.
    2. **A repeat that DISAGREES about x/y is a hard error**, not a silent first-wins. Two
       different stage positions filed under one FOV id means the file is corrupt or was
       concatenated from two runs; picking either one would be a guess.
    3. **The FOV id set must match the filename-derived set exactly.** An id in the CSV with no
       image (or an image with no row) leaves part of the plate unplaceable.

    Units: x/y are millimetres here exactly as in the type-(b) schema, so the same single
    ``_MM_TO_UM`` conversion applies. The ``z (um)`` column is ALREADY micrometres and is not
    read at all — there is no third unit crossing this boundary and no second scale factor.
    """
    by_region: dict[str, dict[int, tuple]] = {}
    for line_no, row in enumerate(reader, start=2):
        region = (row.get("region") or "").strip()
        if not region or region not in fovs_per_region:
            continue
        pair = _parse_mm_pair(
            (row.get(x_col) or "").strip(), (row.get(y_col) or "").strip(), region, line_no
        )
        if pair is None:
            continue
        raw_fov = (row.get(fov_col) or "").strip()
        try:
            fov = int(raw_fov)
        except ValueError:
            raise ValueError(
                f"{_COORDS_NAME} line {line_no}: region {region!r} has a non-integer fov id "
                f"({raw_fov!r}); the fov column is the row -> image mapping and cannot be guessed."
            ) from None
        x, y = pair
        key = (round(x, 6), round(y, 6))    # tolerate float-repr drift, as the positional path does
        seen = by_region.setdefault(region, {})
        if fov in seen:
            if seen[fov][0] != key:
                raise ValueError(
                    f"{_COORDS_NAME} line {line_no}: region {region!r} fov {fov} appears at two "
                    f"conflicting stage positions ({seen[fov][0]} and {key} mm). A repeated fov id "
                    "is normal (one row per z-level) only when the position is identical; differing "
                    "positions mean the file is corrupt or concatenated — refusing to pick one."
                )
            continue                        # same position repeated (one row per z / per t)
        seen[fov] = (key, (x * _MM_TO_UM, y * _MM_TO_UM))

    positions: dict = {}
    for region, seen in by_region.items():
        expected = set(fovs_per_region[region])
        if set(seen) != expected:
            missing = sorted(expected - set(seen))
            extra = sorted(set(seen) - expected)
            raise ValueError(
                f"{_COORDS_NAME}: region {region!r} lists {len(seen)} distinct stage position(s) "
                f"for fov ids that do not match the {len(expected)} FOV(s) found in the filenames "
                f"(missing from the CSV: {missing}; in the CSV but not on disk: {extra}). "
                "Refusing to place a partially-known plate at positions that would look plausible "
                "but be wrong."
            )
        for fov, (_key, xy) in seen.items():
            positions[(region, fov)] = xy
    return positions


def load_fov_positions_um(root, fovs_per_region: dict) -> dict:
    """Parse ``coordinates.csv`` into ``{(region, fov): (x_um, y_um)}`` — MICROMETRES.

    The file records millimetres; world space in this package is micrometres (``_tiling.py``),
    and the units invariant is that every world-space value is µm and every key carrying one
    ends in ``_um``. The mm -> µm conversion therefore happens HERE, at the single producer,
    rather than in each consumer — an unsuffixed mm value crossing into µm code is a silent
    1000x error that draws a plausible picture.

    Returns ``{}`` (present but empty — never a missing key) when the file is absent, so a
    consumer can degrade to single-FOV rendering instead of hitting a KeyError.

    The mapping is **row order within a region**: coordinates.csv carries no ``fov`` column, so
    the Nth distinct position recorded for a region is that region's Nth sorted FOV. Two
    safeguards keep that honest:

    1. **De-duplicate on (region, x, y) before counting.** A multi-z or multi-timepoint
       acquisition can write one row per z-level at the same stage position; raw row counts
       would then be a multiple of the FOV count and every z-stack would fail the check below.
       De-duplicating first makes the count mean "distinct stage positions", which is the thing
       that should equal the FOV count. First occurrence wins, preserving file order.
    2. **Cross-check the de-duplicated count against the filename-derived FOV count** and raise
       on a mismatch. This is the load-bearing guard: an off-by-one or a truncated CSV produces
       a mosaic that looks entirely reasonable while every tile sits in the wrong place, and
       nothing downstream can detect it.

    Rows whose region is not in *fovs_per_region* are ignored (a CSV may describe regions whose
    images were never written); a region with images but no rows is simply absent from the
    result, and placement for it raises later with a clear message.
    """
    import csv

    path = Path(root) / _COORDS_NAME
    if not path.exists():
        return {}

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        x_col, y_col = _coord_columns(reader.fieldnames)
        fov_col = _fov_column(reader.fieldnames)
        if fov_col is not None:
            return _positions_from_fov_column(reader, fovs_per_region, fov_col, x_col, y_col)
        ordered: dict[str, list] = {}
        seen: dict[str, set] = {}
        for line_no, row in enumerate(reader, start=2):
            region = (row.get("region") or "").strip()
            if not region or region not in fovs_per_region:
                continue
            pair = _parse_mm_pair(
                (row.get(x_col) or "").strip(), (row.get(y_col) or "").strip(), region, line_no
            )
            if pair is None:
                continue
            x, y = pair
            key = (round(x, 6), round(y, 6))   # tolerate float-repr drift when de-duplicating
            if key in seen.setdefault(region, set()):
                continue                    # same position repeated (one row per z / per t)
            seen[region].add(key)
            # De-duplication compares the raw mm values; only the stored value is converted.
            ordered.setdefault(region, []).append((x * _MM_TO_UM, y * _MM_TO_UM))

    positions: dict = {}
    for region, coords in ordered.items():
        fovs = list(fovs_per_region[region])
        if len(coords) != len(fovs):
            raise ValueError(
                f"{_COORDS_NAME}: region {region!r} lists {len(coords)} distinct stage "
                f"position(s) but {len(fovs)} FOV(s) were found in the filenames. "
                "Without a 'fov' column the Nth position must be the Nth FOV, so a count "
                "mismatch means the mapping is unknowable — refusing to place FOVs at "
                "positions that would look plausible but be wrong."
            )
        for fov, xy in zip(fovs, coords):
            positions[(region, fov)] = xy
    return positions


def _fov_positions_um_or_empty(root, fovs_per_region: dict) -> dict:
    """``load_fov_positions_um`` degraded to ``{}`` on an unusable coordinates.csv.

    ``metadata`` is the acquisition's whole identity: regions, channels, dtype, frame shape.
    Those come from the FILENAMES and one decoded frame and are readable whatever the CSV says.
    Before this, a truncated or malformed coordinates.csv raised out of the middle of the
    metadata dict literal, so every one of those fields became unreachable and the viewer
    reported "not a readable Squid acquisition" for an acquisition it could render perfectly
    well minus the multi-FOV mosaic (IMA-187).

    Degrading to ``{}`` is safe precisely because ``{}`` already means "no stage positions" —
    consumers fall back to single-tile rendering. It does NOT weaken the cross-check: an
    ambiguous CSV still never produces a scrambled mosaic, it produces no mosaic, loudly
    (``UserWarning``). Only :class:`ValueError` (the parse/cross-check failures this module
    raises deliberately) is absorbed; anything else still propagates.
    """
    try:
        return load_fov_positions_um(root, fovs_per_region)
    except ValueError as e:
        warnings.warn(
            f"{_COORDS_NAME} is unusable ({e}) — continuing WITHOUT stage positions: the "
            "acquisition still opens, but multi-FOV wells render as a single tile instead of "
            "a coordinate-placed mosaic."
        )
        return {}


def _plate_key(region: str):
    """Sort well ids in true plate ROW-MAJOR order: A,B,...,Z,AA,AB,... with the column by integer
    (so B2 < B3 < B10, and B < AA — single-letter rows before double-letter, not lexicographic
    where "AA" < "B"). Downstream consumers (projection engine, plate viewer) then process wells
    top-to-bottom, left-to-right. Non-well-plate region names fall back after the plate wells.

    Changed from a plain natural sort in IMA-189: the old key ordered "AA" before "B", so a 1536wp
    plate processed row A, then the AA-AF rows, then B..Z — filling the plate view out of visual
    order. Row-major here fixes fill/scrub order for every slot. (Owner: IMA-185; see eng review.)
    """
    m = re.match(r"^([A-Za-z]+)(\d+)$", region)
    if not m:
        return (1, len(region), region, 0)          # non-plate ids: stable, after the wells
    return (0, len(m.group(1)), m.group(1).upper(), int(m.group(2)))


def open_reader(path) -> "SquidReader":
    """Detect the acquisition format at *path* and return a reader.

    Raises NotImplementedError for formats other than individual TIFFs (the dispatch seam).
    """
    path = Path(path)
    if not path.is_dir():
        raise NotImplementedError(
            f"{path!s} is not a directory. Point open_reader at a Squid acquisition folder."
        )
    ome = path / "ome_tiff"
    # OME-TIFF only if ome_tiff/ actually CONTAINS .ome.tiff files. Squid often leaves an EMPTY
    # ome_tiff/ placeholder next to an individual-TIFF acquisition — that empty folder must NOT
    # shadow the individual-TIFF reader.
    if ome.is_dir() and any(ome.rglob("*.ome.tif*")):
        return SquidOMEReader(path)
    if (path / "zarr.json").exists() or any(path.glob("*.zarr")):
        raise NotImplementedError(
            "Zarr layout detected. Not implemented in IMA-189; format-dispatch seam."
        )
    return SquidReader(path)


class SquidReader:
    """Lazy reader over a Squid individual-TIFF acquisition folder."""

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._time_folders: Optional[list[Path]] = None
        self._index: Optional[dict] = None
        self._meta: Optional[dict] = None

    # -- timepoints -------------------------------------------------------
    def _discover_time_folders(self) -> list[Path]:
        if self._time_folders is None:
            numeric = [d for d in self._path.iterdir() if d.is_dir() and d.name.isdigit()]
            self._time_folders = (
                sorted(numeric, key=lambda d: int(d.name)) if numeric else [self._path]
            )
        return self._time_folders

    # -- index ------------------------------------------------------------
    def _build_index(self) -> dict:
        """Map {(region, fov, z, channel): file_suffix} from the first timepoint folder."""
        if self._index is not None:
            return self._index
        folder = self._discover_time_folders()[0]
        index: dict = {}
        for f in folder.iterdir():
            if f.suffix.lower() not in _TIFF_SUFFIXES:
                continue
            m = _STEM_RE.match(f.stem)
            if not m:
                continue  # e.g. {region}_{fov}_stack.tiff (multi-page) — not this reader's format
            key = (m["region"], int(m["fov"]), int(m["z"]), m["channel"])
            index[key] = f.suffix
        if not index:
            raise ValueError(
                "No Squid individual-TIFF files "
                "({region}_{fov}_{z}_{channel}.tiff) found in "
                f"{folder!s}"
            )
        self._index = index
        return index

    # -- metadata ---------------------------------------------------------
    @property
    def metadata(self) -> dict:
        if self._meta is not None:
            return self._meta
        index = self._build_index()
        time_folders = self._discover_time_folders()

        fovs: dict[str, set] = {}
        channels: set = set()
        z_levels: set = set()
        for (region, fov, z, channel) in index:
            fovs.setdefault(region, set()).add(fov)
            channels.add(channel)
            z_levels.add(z)
        # Deterministic, natural-sorted order (filesystem iteration order is not stable).
        regions = sorted(fovs, key=_plate_key)   # true plate row-major (A,B,...,Z,AA,...)

        z_sorted = sorted(z_levels)
        n_z = len(z_sorted)
        n_t = len(time_folders)

        # Filenames + timepoint folders are ground truth; the recorded Nz/Nt are cross-checks.
        acq = load_acquisition_metadata(self._path)
        if acq["n_z_declared"] is not None and acq["n_z_declared"] != n_z:
            warnings.warn(
                f"Recorded Nz ({acq['n_z_declared']}) != distinct z levels in filenames "
                f"({n_z}); using the filename-derived value."
            )
        if acq["n_t_declared"] is not None and acq["n_t_declared"] != n_t:
            warnings.warn(
                f"Recorded Nt ({acq['n_t_declared']}) != timepoint folders found ({n_t}); "
                "using the folder-derived value."
            )

        # frame shape + dtype come from a real frame — they vary with binning / pixel format.
        sample_key = next(iter(index))
        sample_path = self._resolve_file(time_folders[0], sample_key, index[sample_key])
        sample = _validate_plane(tifffile.imread(sample_path), sample_path)

        fovs_per_region = {r: sorted(fovs[r]) for r in regions}
        self._meta = {
            "regions": regions,
            "fovs_per_region": fovs_per_region,
            # {(region, fov): (x_um, y_um)} — MICROMETRES, per the package units invariant.
            # {} when coordinates.csv is absent OR unusable (never raises out of metadata).
            # Present on BOTH reader classes so consumers never have to ask which reader they
            # hold (IMA-187).
            "fov_positions_um": _fov_positions_um_or_empty(self._path, fovs_per_region),
            "channels": resolve_channels(sorted(channels), load_channel_yaml(self._path)),
            "n_z": n_z,
            "z_levels": z_sorted,
            "dz_um": acq["dz_um"],
            "pixel_size_um": acq["pixel_size_um"],  # authoritative (acquisition.yaml), not recomputed
            "wellplate_format": acq["wellplate_format"],
            "frame_shape": tuple(sample.shape),
            "dtype": sample.dtype,
            "n_t": n_t,
        }
        return self._meta

    # -- read -------------------------------------------------------------
    def read(self, region, fov, channel, z, t=0):
        """Return one plane as a 2D array in its native dtype. Lazy: reads exactly one file."""
        index = self._build_index()
        time_folders = self._discover_time_folders()
        key = (str(region), int(fov), int(z), str(channel))
        if key not in index:
            raise KeyError(
                f"No such plane region={region!r} fov={fov} channel={channel!r} z={z}. "
                f"Known regions={sorted({k[0] for k in index})}, "
                f"channels={sorted({k[3] for k in index})}."
            )
        t = int(t)
        if not 0 <= t < len(time_folders):
            raise IndexError(f"t={t} out of range (n_t={len(time_folders)}).")
        path = self._resolve_file(time_folders[t], key, index[key])
        return _validate_plane(tifffile.imread(path), path)

    def plane_path(self, region, fov, channel, z, t=0) -> Path:
        """Path to one raw plane's TIFF on disk (no decode). The HCS viewer points the embedded
        ndviewer at these raw files directly (register_image), so the detail view is the true
        z-stack with zero extra bytes copied — read-only, never written."""
        index = self._build_index()
        time_folders = self._discover_time_folders()
        key = (str(region), int(fov), int(z), str(channel))
        if key not in index:
            raise KeyError(f"No such plane region={region!r} fov={fov} channel={channel!r} z={z}.")
        t = int(t)
        if not 0 <= t < len(time_folders):
            raise IndexError(f"t={t} out of range (n_t={len(time_folders)}).")
        return self._resolve_file(time_folders[t], key, index[key])

    def plane_ref(self, region, fov, channel, z, t=0) -> tuple:
        """(filepath, page_index) for one plane — the viewer registers this into ndviewer. Individual
        TIFFs hold one plane per file, so the page index is always 0."""
        return str(self.plane_path(region, fov, channel, z, t)), 0

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _resolve_file(folder: Path, key, suffix: str) -> Path:
        """Build the plane's path, tolerating .tiff/.tif suffix drift across timepoints."""
        region, fov, z, channel = key
        candidate = folder / f"{region}_{fov}_{z}_{channel}{suffix}"
        if candidate.exists():
            return candidate
        for alt in _TIFF_SUFFIXES:
            other = folder / f"{region}_{fov}_{z}_{channel}{alt}"
            if other.exists():
                return other
        return candidate  # let tifffile raise a clear FileNotFoundError


# {region}_{fov} stem (region = well id, no trailing _<digits>; fov = trailing integer).
_OME_STEM_RE = re.compile(r"^(?P<region>.+)_(?P<fov>\d+)$")
_OME_SUFFIXES = (".ome.tiff", ".ome.tif", ".OME.TIFF", ".OME.TIF")


class SquidOMEReader:
    """Lazy reader over a Squid OME-TIFF acquisition.

    Layout (from Squid's utils_ome_tiff_writer): ``<acq>/ome_tiff/{region}_{fov}.ome.tiff`` — ONE
    file per well-FOV, each a 5-D ``TZCYX`` stack (dimension order written as TZCYX). Presents the
    SAME interface as :class:`SquidReader` (``metadata`` + ``read`` + ``plane_ref``), so the engine,
    CLI and viewer consume it unchanged. Reads one plane at a time (``TiffFile.pages[p]``) so memory
    stays bounded; the TiffFile handles are cached per file.
    """

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._ome = self._path / "ome_tiff"
        self._files: Optional[dict] = None      # {(region, fov): Path}
        self._meta: Optional[dict] = None
        self._axes: Optional[str] = None        # non-spatial axes order, e.g. "TZC"
        self._handles: dict = {}                # Path -> tifffile.TiffFile (cached)

    def _discover(self) -> dict:
        if self._files is not None:
            return self._files
        files: dict = {}
        for f in sorted(self._ome.iterdir() if self._ome.is_dir() else []):
            name = f.name
            stem = next((name[: -len(s)] for s in _OME_SUFFIXES if name.endswith(s)), None)
            if stem is None:
                continue
            m = _OME_STEM_RE.match(stem)
            if m:
                files[(m["region"], int(m["fov"]))] = f
        if not files:
            raise ValueError(f"No {{region}}_{{fov}}.ome.tiff files found in {self._ome!s}")
        self._files = files
        return files

    def _tif(self, path: Path):
        tif = self._handles.get(path)
        if tif is None:
            tif = tifffile.TiffFile(path)
            self._handles[path] = tif
        return tif

    @property
    def metadata(self) -> dict:
        if self._meta is not None:
            return self._meta
        files = self._discover()
        sample = self._tif(next(iter(files.values()))).series[0]
        dims = dict(zip(sample.axes, sample.shape))     # e.g. {'T':2,'Z':3,'C':2,'Y':64,'X':80}
        n_t, n_z, n_c = dims.get("T", 1), dims.get("Z", 1), dims.get("C", 1)
        self._axes = "".join(a for a in sample.axes if a in "TZC")   # non-spatial order for paging

        fovs: dict[str, set] = {}
        for (region, fov) in files:
            fovs.setdefault(region, set()).add(fov)
        regions = sorted(fovs, key=_plate_key)

        # Channels come from acquisition_channels.yaml, in file order (== the writer's C-axis order).
        yaml_map = load_channel_yaml(self._path)
        names = list(yaml_map.keys())
        if len(names) != n_c:
            # yaml disagrees with the file — fall back to the OME channel names, else generic labels.
            ome_names = _ome_channel_names(self._tif(next(iter(files.values()))))
            names = [_normalize_local(n) for n in ome_names] if len(ome_names) == n_c \
                else [f"C{i}" for i in range(n_c)]
        channels = resolve_channels(names, yaml_map)

        acq = load_acquisition_metadata(self._path)
        if acq["n_z_declared"] is not None and acq["n_z_declared"] != n_z:
            warnings.warn(f"Recorded Nz ({acq['n_z_declared']}) != OME Z ({n_z}); using {n_z}.")
        fovs_per_region = {r: sorted(fovs[r]) for r in regions}
        self._meta = {
            "regions": regions,
            "fovs_per_region": fovs_per_region,
            # Same key, same meaning as SquidReader — the shared-interface promise in this
            # class's docstring. An OME acquisition with a sibling coordinates.csv gets real
            # placement; without one this is {} and the mosaic degrades to a single field.
            # (Positions inside the OME-XML are not read: different parsing path, no dataset
            # on hand to validate it against. See .spec NOT-in-scope.)
            "fov_positions_um": _fov_positions_um_or_empty(self._path, fovs_per_region),
            "channels": channels,
            "n_z": n_z,
            "z_levels": list(range(n_z)),
            "dz_um": acq["dz_um"],
            "pixel_size_um": acq["pixel_size_um"],
            "wellplate_format": acq["wellplate_format"],
            "frame_shape": (int(dims.get("Y", sample.shape[-2])), int(dims.get("X", sample.shape[-1]))),
            "dtype": np.dtype(sample.dtype),
            "n_t": n_t,
        }
        return self._meta

    def _page_index(self, t: int, z: int, c: int) -> int:
        """Flat IFD page index for (t, z, c), honouring the file's non-spatial axis order."""
        meta = self.metadata
        sizes = {"T": meta["n_t"], "Z": meta["n_z"], "C": len(meta["channels"])}
        pos = {"T": t, "Z": z, "C": c}
        order = self._axes or "TZC"
        return int(np.ravel_multi_index([pos[a] for a in order], [sizes[a] for a in order]))

    def _channel_index(self, channel) -> int:
        names = [c["name"] for c in self.metadata["channels"]]
        return names.index(str(channel))

    def read(self, region, fov, channel, z, t=0):
        """Return one plane as a 2D native-dtype array (reads exactly one IFD page)."""
        files = self._discover()
        key = (str(region), int(fov))
        if key not in files:
            raise KeyError(f"No such well/FOV region={region!r} fov={fov}. Known: {sorted(files)[:8]}")
        p = self._page_index(int(t), int(z), self._channel_index(channel))
        tif = self._tif(files[key])
        return _validate_plane(np.asarray(tif.pages[p].asarray()), files[key])

    def plane_ref(self, region, fov, channel, z, t=0) -> tuple:
        """(filepath, page_index) for one plane — the viewer registers this (with the page) into
        ndviewer, so the raw z-stack displays straight from the .ome.tiff, zero bytes copied."""
        p = self._page_index(int(t), int(z), self._channel_index(channel))
        return str(self._discover()[(str(region), int(fov))]), p


def _normalize_local(name: str) -> str:
    from squidmip._channels import normalize
    return normalize(name)


def _ome_channel_names(tif) -> list:
    """Best-effort channel names from the OME-XML (Channel Name=...), else []."""
    try:
        xml = tif.ome_metadata or ""
        return re.findall(r'<Channel[^>]*\bName="([^"]*)"', xml)
    except Exception:
        return []


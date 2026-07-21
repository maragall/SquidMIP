"""SquidMIP reader: format-aware ingest for Squid acquisitions.

``open_reader(path)`` dispatches on the on-disk format and returns a reader. Three of Squid's
output formats are implemented; the rest are detected and refused BY NAME, so an unsupported
acquisition produces an accurate error instead of a misleading one.

Dispatch tree::

    open_reader(path)
       │
       ├── ome_tiff/ contains *.ome.tif*        ──► SquidOMEReader   (IMA-189)
       │
       ├── plate.ome.zarr/ exists               ──► SquidZarrReader  (IMA-229)
       │      └── unless it is SquidMIP's OWN write_plate output (plate.field_count and no
       │          _squid block) ──► ValueError: that is a RESULT, not an acquisition
       │
       ├── zarr/{region}/fov_*.ome.zarr         ──► NotImplementedError (per-FOV layout)
       ├── zarr/{region}/acquisition.zarr       ──► NotImplementedError (6-D layout)
       │
       └── otherwise                            ──► SquidReader      (IMA-189)

All three readers expose the SAME interface — ``metadata`` (one eleven-key contract, built by
``_plate.build_metadata``) and ``read(region, fov, channel, z, t)`` returning one native-dtype
2-D plane — so the engine, CLI, writer and viewer consume any of them unchanged. They differ in
one capability: the TIFF readers can hand the viewer a ``(file, page)`` via ``plane_ref``; the
zarr reader cannot (a plane is a slice of a chunked array) and offers ``fov_store_path``
instead, flagged by ``supports_plane_ref``.

Individual-TIFFs layout (one channel per file), verified against real data::

    <acq>/
    ├── acquisition.yaml                      # authoritative scalars (REQUIRED)
    ├── acquisition_channels.yaml
    ├── coordinates.csv                       # not read (see below)
    └── 0/                                    # timepoint folder (1/, 2/, … if Nt>1)
        └── {region}_{fov}_{z}_{channel}.tiff

    glob timepoint folders (0/,1/…) ──┐
    glob *.tiff in t0, parse stems ───┼─► regions, fovs_per_region, channels, z-levels, n_t
    read ONE frame ───────────────────┤─► frame_shape, dtype   (NOT hardcoded)
    acquisition.yaml ─────────────────┴─► dz_um, pixel_size_um, wellplate_format + Nz/Nt check

OME-TIFF layout: ``<acq>/ome_tiff/{region}_{fov}.ome.tiff``, one 5-D TZCYX stack per well-FOV;
the (t, z, c) index becomes a flat IFD page index.

HCS Zarr v3 layout: ``<acq>/plate.ome.zarr/{row}/{col}/{fov}/0``, one 5-D (T,C,Z,Y,X) array per
field; see :class:`SquidZarrReader` for the two non-obvious properties of Squid's zarr (group
metadata describes the acquisition PLAN, and unwritten chunks decode as zeros).

Common rules across readers: the on-disk data is ground truth and recorded scalars are a
cross-check that warns on disagreement; ``acquisition.yaml`` is the authoritative source for
physical scalars (pixel size is binning-aware and never recomputed); ``coordinates.csv`` is not
read — for one-FOV-per-well the plate layout comes from the well ID + wellplate_format, and
per-FOV stage positions are a deferred stitching concern; every returned plane must be 2-D with
dtype in {uint8, uint16} or it is refused rather than silently projected.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import tensorstore as ts
import tifffile

from squidmip._acquisition import load_acquisition_metadata
from squidmip._channels import load_channel_yaml, resolve_channels
from squidmip._plate import (
    build_metadata,
    cross_check_nt,
    cross_check_nz,
    group_regions,
    plate_key,
    read_group_attrs,
    read_group_ome,
)

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


# Plate row-major ordering. Lives in _plate now (all three readers need it); re-exported here
# because IMA-185/188 code and tests import it from this module.
#
# Changed from a plain natural sort in IMA-189: the old key ordered "AA" before "B", so a 1536wp
# plate processed row A, then the AA-AF rows, then B..Z — filling the plate view out of visual
# order. Row-major fixes fill/scrub order for every slot. (Owner: IMA-185; see eng review.)
_plate_key = plate_key


def _refuse_squidmip_output_if_ours(plate_dir: Path) -> None:
    """Refuse SquidMIP's OWN ``write_plate`` output, which is structurally a Squid HCS plate.

    ``write_plate`` writes ``<out>/plate.ome.zarr`` (``_output.py``) with the same
    row/col/fov/array shape Squid writes, so nothing in the layout distinguishes an ACQUISITION
    from a RESULT. Without this check, pointing the tool at its own output would build a reader
    and then die on the missing ``acquisition.yaml`` with an error about the wrong thing.

    The discriminator is free and exact: Squid stamps a ``_squid`` attributes block on every
    field group and never writes ``plate.field_count``; we write ``field_count`` and no
    ``_squid``.
    """
    plate = read_group_ome(plate_dir).get("plate")
    if not isinstance(plate, dict) or "field_count" not in plate:
        return                                   # no field_count -> not ours; let the reader try
    for well in plate.get("wells") or []:
        well_dir = plate_dir / str(well.get("path", ""))
        for image in read_group_ome(well_dir).get("well", {}).get("images", []):
            if "_squid" in read_group_attrs(well_dir / str(image.get("path", ""))):
                return                           # has _squid after all -> a real acquisition
    raise ValueError(
        f"{plate_dir!s} looks like SquidMIP OUTPUT, not a Squid acquisition: its plate metadata "
        "carries 'field_count' and no field has a '_squid' block. open_reader() ingests raw "
        "acquisitions; point it at the acquisition folder, or open this plate in the viewer."
    )


def open_reader(path, *, allow_incomplete: bool = False):
    """Detect the acquisition format at *path* and return a reader.

    Dispatch (see the module docstring for the full tree)::

        ome_tiff/ with *.ome.tif*      -> SquidOMEReader
        plate.ome.zarr/                -> SquidZarrReader   (or refused if it is our own output)
        zarr/{region}/fov_*.ome.zarr   -> NotImplementedError (per-FOV layout)
        zarr/{region}/acquisition.zarr -> NotImplementedError (6-D layout)
        otherwise                      -> SquidReader

    ``allow_incomplete`` is forwarded to the zarr reader only; the TIFF layouts cannot express a
    partially-written plane (a missing plane is a missing file), so it is meaningless there.
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

    plate_dir = path / "plate.ome.zarr"
    if plate_dir.is_dir():
        _refuse_squidmip_output_if_ours(plate_dir)
        return SquidZarrReader(path, allow_incomplete=allow_incomplete)

    # The two non-HCS zarr layouts live UNDER a directory literally named "zarr", so they match
    # neither `path/"zarr.json"` nor `path.glob("*.zarr")`. Before IMA-229 they fell through to
    # SquidReader and died with "No Squid individual-TIFF files found" — an error that sent users
    # hunting for TIFFs that were never supposed to exist. Refuse them BY NAME instead.
    #
    # Note (IMA-229): these layouts are NOT "the non-wellplate ones". Squid picks the layout from
    # the acquisition MODE (`is_hcs = is_select_wells or is_loaded_wells`,
    # multi_point_worker.py:268), so a genuine wellplate run in flexible/manual-region mode lands
    # here with well-shaped region ids. Deferred on effort, not because the data is unusable.
    zarr_dir = path / "zarr"
    if zarr_dir.is_dir():
        for region_dir in sorted(p for p in zarr_dir.iterdir() if p.is_dir()):
            if (region_dir / "acquisition.zarr").is_dir():
                raise NotImplementedError(
                    f"6-D zarr layout detected ({region_dir / 'acquisition.zarr'}): one "
                    "(FOV, T, C, Z, Y, X) array per region. Squid documents this as non-standard "
                    "OME-NGFF and most readers misinterpret it; SquidMIP does not read it. "
                    "Re-acquire with ZARR_USE_6D_FOV_DIMENSION=False, or use the HCS "
                    "(plate.ome.zarr) layout."
                )
            if any(region_dir.glob("fov_*.ome.zarr")):
                raise NotImplementedError(
                    f"Per-FOV zarr layout detected ({region_dir}): one fov_N.ome.zarr store per "
                    "FOV. SquidMIP currently reads only the HCS layout (plate.ome.zarr). Re-run "
                    "the acquisition in wellplate (select-wells) mode to get an HCS plate."
                )

    if (path / "zarr.json").exists() or any(path.glob("*.zarr")):
        raise NotImplementedError(
            f"A zarr store was found at {path!s}, but not in a layout SquidMIP reads. Supported: "
            "the HCS layout, i.e. an acquisition folder containing plate.ome.zarr/."
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

        channels: set = set()
        z_levels: set = set()
        for (region, fov, z, channel) in index:
            channels.add(channel)
            z_levels.add(z)
        regions, fovs_per_region = group_regions((r, f) for (r, f, _z, _c) in index)

        z_sorted = sorted(z_levels)
        n_t = len(time_folders)

        # Filenames + timepoint folders are ground truth; the recorded Nz/Nt are cross-checks.
        acq = load_acquisition_metadata(self._path)
        cross_check_nz(acq["n_z_declared"], len(z_sorted), "distinct z levels in filenames")
        cross_check_nt(acq["n_t_declared"], n_t, "timepoint folders found")

        # frame shape + dtype come from a real frame — they vary with binning / pixel format.
        sample_key = next(iter(index))
        sample_path = self._resolve_file(time_folders[0], sample_key, index[sample_key])
        sample = _validate_plane(tifffile.imread(sample_path), sample_path)

        self._meta = build_metadata(
            regions=regions,
            fovs_per_region=fovs_per_region,
            channels=resolve_channels(sorted(channels), load_channel_yaml(self._path)),
            z_levels=z_sorted,
            frame_shape=sample.shape,
            dtype=sample.dtype,
            n_t=n_t,
            acq=acq,
        )
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

        regions, fovs_per_region = group_regions(files)

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
        cross_check_nz(acq["n_z_declared"], n_z, "OME Z")
        self._meta = build_metadata(
            regions=regions,
            fovs_per_region=fovs_per_region,
            channels=channels,
            z_levels=range(n_z),
            frame_shape=(int(dims.get("Y", sample.shape[-2])), int(dims.get("X", sample.shape[-1]))),
            dtype=np.dtype(sample.dtype),
            n_t=n_t,
            acq=acq,
        )
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


class SquidZarrReader:
    """Lazy reader over a Squid HCS Zarr v3 acquisition (``<acq>/plate.ome.zarr``).

    Layout (Squid ``control/core/zarr_writer.py`` + ``control/utils.py:675``)::

        <acq>/
        ├── acquisition.yaml                REQUIRED — pixel size, wellplate_format, nz/nt
        ├── acquisition_channels.yaml       display colors
        └── plate.ome.zarr/
            ├── zarr.json                   ome.plate -> rows / columns / wells[].path
            └── {row}/{col}/
                ├── zarr.json               ome.well  -> images[].path  (fov ids)
                └── {fov}/
                    ├── zarr.json           ome.multiscales + ome.omero + _squid
                    └── 0/                  ARRAY (T, C, Z, Y, X), native dtype

    Presents the SAME interface as :class:`SquidReader` (``metadata`` + ``read``) so the engine,
    CLI and writer consume it unchanged — but **not** ``plane_ref``. A zarr plane is a slice of a
    chunked, compressed array, not a ``(file, page)``; ``supports_plane_ref`` is False and the
    viewer uses :meth:`fov_store_path` with ndviewer_light's native zarr API instead.

    Two things about Squid's zarr that drive this design and are NOT guessable from the layout:

    1. **Group metadata is the acquisition PLAN, not a record of what was acquired.**
       ``job_processing.py`` writes plate metadata (all planned wells) and well metadata
       (``fields = list(range(fov_count))``) up front, on the first frame, and never revises
       them. On a partial plate the metadata therefore OVER-claims: it lists wells and fields
       whose directories do not exist. Discovery INTERSECTS the plan with what is on disk.
    2. **Unwritten chunks decode as zeros, not as an error.** The array is allocated full-size
       at creation, so a crashed run projects to a plate that looks finished and is wrong. The
       individual-TIFF path has no such hole (a missing plane is a missing file -> KeyError).
       ``_squid.acquisition_complete`` gates this; see :meth:`_check_complete`.
    """

    supports_plane_ref = False

    def __init__(self, path, *, allow_incomplete: bool = False) -> None:
        self._path = Path(path)
        self._plate = self._path / "plate.ome.zarr"
        self._allow_incomplete = bool(allow_incomplete)
        self._fields: Optional[dict] = None     # {(region, fov): field group dir}
        self._meta: Optional[dict] = None
        self._stores: dict = {}                 # field dir -> open tensorstore handle
        self._checked_complete = False

    # -- discovery --------------------------------------------------------
    def _discover(self) -> dict:
        """{(region, fov): field_group_dir} — the acquisition PLAN intersected with the disk.

        Falls back to a directory walk when a group's metadata is missing (a crash can leave a
        well directory with no zarr.json), so discovery degrades in the safe direction: it never
        invents a well, and it never hides one that has real pixels.
        """
        if self._fields is not None:
            return self._fields
        if not self._plate.is_dir():
            raise ValueError(f"No plate.ome.zarr in {self._path!s}.")

        fields: dict = {}
        plate = read_group_ome(self._plate).get("plate") or {}
        planned = [str(w.get("path", "")) for w in (plate.get("wells") or [])]
        if not planned:                                    # missing/broken plate metadata -> walk
            planned = [f"{row.name}/{col.name}"
                       for row in sorted(p for p in self._plate.iterdir() if p.is_dir())
                       for col in sorted(p for p in row.iterdir() if p.is_dir())]

        for well_path in planned:
            well_dir = self._plate / well_path
            if not well_dir.is_dir():
                continue                                   # PLANNED but never acquired — skip
            row_col = well_path.split("/")
            if len(row_col) != 2:
                continue
            region = f"{row_col[0]}{row_col[1]}"            # Squid writes columns unpadded
            images = read_group_ome(well_dir).get("well", {}).get("images") or []
            fov_ids = [str(i.get("path", "")) for i in images]
            if not fov_ids:                                 # crashed before well metadata
                fov_ids = [d.name for d in well_dir.iterdir() if d.is_dir()]
            for fov_id in fov_ids:
                fov_dir = well_dir / fov_id
                if not fov_dir.is_dir() or not fov_id.isdigit():
                    continue                                # PLANNED but never acquired — skip
                fields[(region, int(fov_id))] = fov_dir

        if not fields:
            raise ValueError(
                f"No acquired fields found under {self._plate!s}. The plate metadata describes "
                "the acquisition plan; none of the wells it lists exist on disk."
            )
        self._fields = fields
        return fields

    # -- completeness -----------------------------------------------------
    def _check_complete(self, field_dir: Path) -> None:
        """Refuse a partial acquisition unless the caller opted in (checked once, lazily).

        Three end states, distinguished because the remedy differs:
          complete=True                 -> clean finish
          complete=False, aborted=True  -> the user stopped it
          complete=False, no aborted    -> hard crash / power loss (finalize never ran)
        """
        if self._checked_complete:
            return
        self._checked_complete = True
        squid = read_group_attrs(field_dir).get("_squid")
        if not isinstance(squid, dict) or squid.get("acquisition_complete") is not False:
            return                                    # complete, or not a Squid-stamped store
        aborted = bool(squid.get("aborted"))
        state = ("was ABORTED by the user" if aborted else
                 "did not finish (no abort flag: a crash or power loss, finalize never ran)")
        if not self._allow_incomplete:
            raise ValueError(
                f"This acquisition {state}: plate.ome.zarr is marked "
                "'_squid.acquisition_complete: false'. Planes that were never written decode as "
                "ZEROS rather than as an error, so projecting it would silently produce a "
                "complete-looking plate containing blank or dimmed wells. Pass "
                "allow_incomplete=True (CLI: --allow-incomplete) to project what was acquired."
            )
        warnings.warn(
            f"Acquisition {state}; projecting anyway (allow_incomplete=True). Wells that were "
            "never acquired are skipped, but a PARTIALLY written well projects its unwritten "
            "planes as zeros."
        )

    # -- stores -----------------------------------------------------------
    @staticmethod
    def _array_subdir(field_dir: Path) -> str:
        """Resolve the array directory from the field group's own multiscales metadata.

        Squid writes ``datasets[0].path == "0"`` for HCS, but reading it rather than hardcoding
        keeps us correct if that ever changes (the 6-D layout already uses ``"."``).
        """
        ms = read_group_ome(field_dir).get("multiscales") or []
        if ms:
            datasets = ms[0].get("datasets") or []
            if datasets:
                return str(datasets[0].get("path", "0"))
        return "0"

    def _store(self, field_dir: Path):
        store = self._stores.get(field_dir)
        if store is None:
            sub = self._array_subdir(field_dir)
            array_dir = field_dir if sub == "." else field_dir / sub
            store = ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(array_dir)}},
                open=True,
            ).result()
            self._stores[field_dir] = store
        return store

    # -- metadata ---------------------------------------------------------
    @property
    def metadata(self) -> dict:
        if self._meta is not None:
            return self._meta
        fields = self._discover()
        regions, fovs_per_region = group_regions(fields)

        sample_dir = fields[min(fields, key=lambda k: (plate_key(k[0]), k[1]))]
        self._check_complete(sample_dir)
        store = self._store(sample_dir)
        shape = tuple(int(s) for s in store.shape)
        if len(shape) != 5:
            raise ValueError(
                f"{sample_dir!s} holds a {len(shape)}-D array {shape}; the HCS layout is 5-D "
                "(T, C, Z, Y, X). A 6-D (FOV, T, C, Z, Y, X) store is the non-standard layout "
                "SquidMIP does not read."
            )
        n_t, n_c, n_z, height, width = shape

        dtype = np.dtype(store.dtype.numpy_dtype)
        # Fail fast: a zarr store DECLARES its dtype, unlike a TIFF which must be decoded to
        # learn it. Refusing here costs one metadata read; refusing per-plane would surface the
        # same error partway through a plate run the user had already committed to.
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"{sample_dir!s} has dtype {dtype}; Squid writes uint8 (MONO8) or uint16 "
                "(MONO12/MONO16). An unexpected dtype (e.g. uint32/float) usually means the "
                "input is not a raw Squid capture; refused rather than silently projected."
            )

        channels = self._resolve_channels(sample_dir, n_c)

        acq = load_acquisition_metadata(self._path)
        cross_check_nz(acq["n_z_declared"], n_z, "zarr Z")
        cross_check_nt(acq["n_t_declared"], n_t, "zarr T")
        self._cross_check_squid_scalars(sample_dir, acq)

        self._meta = build_metadata(
            regions=regions,
            fovs_per_region=fovs_per_region,
            channels=channels,
            z_levels=range(n_z),
            frame_shape=(height, width),
            dtype=dtype,
            n_t=n_t,
            acq=acq,
        )
        return self._meta

    def _resolve_channels(self, field_dir: Path, n_c: int) -> list:
        """Channel names + order from ``omero.channels[].label`` — the C-axis ground truth.

        Squid builds that list by iterating the C axis (``zarr_writer.py``), so its ORDER IS the
        array's channel order. ``acquisition_channels.yaml`` is a separate file that merely
        usually agrees; trusting its key order would silently mislabel every channel if it were
        ever edited or reordered. Same principle as IMA-189 trusting filenames over
        coordinates.csv: prefer the metadata co-located with the pixels.

        The yaml is still used for display colors, via the shared resolve_channels().
        """
        yaml_map = load_channel_yaml(self._path)
        omero = read_group_ome(field_dir).get("omero") or {}
        labels = [c.get("label") for c in (omero.get("channels") or []) if c.get("label")]
        if len(labels) == n_c:
            names = [_normalize_local(str(n)) for n in labels]
            yaml_names = list(yaml_map.keys())
            if len(yaml_names) == n_c and yaml_names != names:
                warnings.warn(
                    f"acquisition_channels.yaml lists channels {yaml_names} but the zarr omero "
                    f"metadata lists {names}. Using the omero order — it is written by iterating "
                    "the array's C axis, so it is the authoritative channel order."
                )
        else:
            # No usable omero labels: fall back to the yaml, which at least names real channels.
            names = list(yaml_map.keys())
            if len(names) != n_c:
                raise ValueError(
                    f"{field_dir!s}: cannot determine channel identity. The array has {n_c} "
                    f"channels, omero metadata lists {len(labels)} labels and "
                    f"acquisition_channels.yaml lists {len(names)}. Refusing rather than "
                    "guessing which slice is which channel."
                )
        return resolve_channels(names, yaml_map)

    @staticmethod
    def _cross_check_squid_scalars(field_dir: Path, acq: dict) -> None:
        """Warn when Squid's in-store ``_squid`` scalars disagree with acquisition.yaml.

        acquisition.yaml wins (see _acquisition: it is the authoritative, binning-aware source,
        and it is the ONLY place wellplate_format exists). This is a cross-check, like the
        existing Nz/Nt ones — not a second source of truth.
        """
        squid = read_group_attrs(field_dir).get("_squid")
        if not isinstance(squid, dict):
            return
        px, dz = squid.get("pixel_size_um"), squid.get("z_step_um")
        if px is not None and acq["pixel_size_um"] is not None and \
                not np.isclose(float(px), float(acq["pixel_size_um"])):
            warnings.warn(
                f"pixel_size_um disagrees: acquisition.yaml says {acq['pixel_size_um']}, the "
                f"zarr _squid block says {px}. Using acquisition.yaml (authoritative)."
            )
        if dz is not None and acq["dz_um"] is not None and \
                not np.isclose(float(dz), float(acq["dz_um"])):
            warnings.warn(
                f"z step disagrees: acquisition.yaml says {acq['dz_um']} um, the zarr _squid "
                f"block says {dz} um. Using acquisition.yaml (authoritative)."
            )

    # -- read -------------------------------------------------------------
    def _channel_index(self, channel) -> int:
        names = [c["name"] for c in self.metadata["channels"]]
        try:
            return names.index(str(channel))
        except ValueError:
            raise KeyError(
                f"No such channel {channel!r}. Known channels={names}."
            ) from None

    def read(self, region, fov, channel, z, t=0):
        """Return one plane as a 2D array in its native dtype (reads exactly one z/c slice)."""
        fields = self._discover()
        key = (str(region), int(fov))
        if key not in fields:
            raise KeyError(
                f"No such plane region={region!r} fov={fov} channel={channel!r} z={z}. "
                f"Known regions={sorted({k[0] for k in fields})}."
            )
        meta = self.metadata
        c = self._channel_index(channel)
        z, t = int(z), int(t)
        if not 0 <= t < meta["n_t"]:
            raise IndexError(f"t={t} out of range (n_t={meta['n_t']}).")
        if not 0 <= z < meta["n_z"]:
            raise IndexError(f"z={z} out of range (n_z={meta['n_z']}).")
        field_dir = fields[key]
        self._check_complete(field_dir)
        arr = np.asarray(self._store(field_dir)[t, c, z, :, :].read().result())
        # Cheap backstop: the store's declared dtype was validated at open, but a store whose
        # chunks disagree with its header must not slip through into a projection.
        return _validate_plane(arr, field_dir)

    # -- viewer seam ------------------------------------------------------
    def fov_store_path(self, region, fov) -> str:
        """Path to one field's zarr GROUP directory (the dir holding zarr.json + the array).

        This is the zarr counterpart of ``plane_ref``. ndviewer_light reads a zarr field itself
        (``start_zarr_acquisition``), so the viewer hands it these directories rather than
        registering planes one at a time — no bytes copied, and the detail view is the true
        z-stack straight from the acquisition. Read-only, never written.
        """
        fields = self._discover()
        key = (str(region), int(fov))
        if key not in fields:
            raise KeyError(f"No such field region={region!r} fov={fov}.")
        return str(fields[key])

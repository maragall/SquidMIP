# IMA-229 — Engineering Review (longform)

**Ticket:** IMA-229 — Zarr input
**Branch:** `juliomaragall/ima-229-zarr-read`
**Review:** `/plan-eng-review`, completed 2026-07-20.
**Status:** PLAN LOCKED — not implemented. `squidmip/reader.py:106` still raises.

> The working plan lives at `.spec/open/ima-229.md` (normally gitignored; force-added on this
> branch per the ima-176 / ima-216 precedent). This doc is the tracked, reviewable record of
> the **code recommendations** — what to change, where, and why.

---

## 1. Executive summary

Squid can save acquisitions as Zarr v3. Today `open_reader()` refuses them
(`reader.py:106-109`). This ticket implements the **HCS `plate.ome.zarr`** layout only; the
per-FOV and 6-D layouts are refused by name.

Genuinely new code is small — roughly one 120-line reader class. The review's value was not
sizing it but finding four things that would have shipped as bugs:

1. The detection branch misses two of the three real layouts, producing a misleading error.
2. The viewer would have crashed (or silently lost navigation) on zarr.
3. Zarr group metadata is the acquisition **plan**, not a record — so the obvious discovery
   implementation reads wells that do not exist.
4. `open_reader` would happily accept SquidMIP's own output as if it were an acquisition.

No new dependency: `tensorstore>=0.1.45` is already in `pyproject.toml`, and `_montage.py`
already opens zarr v3 stores with the exact spec this reader needs.

---

## 2. Ground truth (verified against source, not assumed)

Read from `~/CEPHLA/projects/Squid/software` and `~/CEPHLA/projects/ndviewer_light`.

| Fact | Source |
|---|---|
| Three Zarr layouts exist | `docs/zarr-v3-format.md:28-69`, `control/utils.py:672-702` |
| HCS array path is `{row}/{col}/{fov}/0`, 5-D `(T,C,Z,Y,X)` | `job_processing.py:410`, `zarr_writer.py:499-506` |
| Group metadata is at `{fov}/zarr.json`, not on the array | `zarr_writer.py:620-631` |
| `omero.channels[]` is built by iterating the C axis | `zarr_writer.py:543-548` |
| `_squid.acquisition_complete` starts `False`; `True` only on clean finalize; abort adds `aborted: True` | `zarr_writer.py:616`, `:765`, `:797-799` |
| Columns unpadded (`str(c)`) — `B2 → B/2` | `zarr_writer.py:301` |
| Squid writes **no** `field_count`; SquidMIP does | `zarr_writer.py:294-305` vs `_output.py:127` |
| Layout follows acquisition **mode**: `is_hcs = is_select_wells or is_loaded_wells` | `multi_point_worker.py:268` |
| Plate + well metadata written **up front from the plan** on the first frame | `job_processing.py:549-570` |
| `acquisition.yaml` written before format dispatch (format-independent) | `multi_point_controller.py:154`, `:882` |
| `wellplate_format` may be present-but-null | `multi_point_controller.py` |
| BALANCED/BEST shard is `(1, c_size, 1, Y, X)` — all channels per `(t,z)` | `zarr_writer.py:155-160` |
| ndviewer HCS-zarr seam: `start_zarr_acquisition(fov_paths, channels, num_z, fov_labels, height, width)` | `ndviewer_light/core.py:2788` |
| `go_to_well_fov` returns `False` when `_fov_labels` is empty | `ndviewer_light/core.py:2511` |

---

## 3. Code recommendations

### R1 — `open_reader()` dispatch (`reader.py:90-110`)

Current code recognises one layout and silently mis-routes two:

```python
if (path / "zarr.json").exists() or any(path.glob("*.zarr")):
    raise NotImplementedError("Zarr layout detected. Not implemented in IMA-189; ...")
```

`plate.ome.zarr` matches `*.zarr`. But per-FOV (`zarr/{region}/fov_N.ome.zarr`) and 6-D
(`zarr/{region}/acquisition.zarr`) live under a directory named `zarr` — which matches
neither clause. Both fall through to `SquidReader` and die with
`ValueError("No Squid individual-TIFF files ... found")`, sending the user hunting for TIFFs
that never existed.

Replace with an explicit tree:

```
open_reader(path)
   ├── ome_tiff/ contains *.ome.tif*   ──► SquidOMEReader
   ├── plate.ome.zarr/ exists
   │      ├── plate meta has field_count and no _squid ──► ValueError (R5: our own OUTPUT)
   │      └── otherwise                                ──► SquidZarrReader
   ├── zarr/{region}/fov_*.ome.zarr    ──► NotImplementedError("per-FOV zarr layout ...")
   ├── zarr/{region}/acquisition.zarr  ──► NotImplementedError("6-D zarr layout ...")
   └── otherwise                       ──► SquidReader
```

The refusal messages must **not** say "non-wellplate" — see R8.

### R2 — Discovery must intersect the plan with the disk (the load-bearing one)

`job_processing.py:549-570` writes plate metadata once, from
`info.get_hcs_structure()`, and each well's metadata from `fields = list(range(fov_count))`
— both derived from the acquisition **plan**, on the first frame. Nothing rewrites them as
wells complete.

So on any partial plate the metadata **over-claims**: it lists wells and fields whose
directories do not exist. The naive implementation (trust metadata, walk it) raises
`FileNotFoundError` on a planned-but-absent `{row}/{col}/{fov}/zarr.json`.

```
plate.ome.zarr/zarr.json ──► ome.plate.wells[].path      (PLANNED)
                                  └──► ∩ {row}/{col}/ dirs that exist  ──► regions
{row}/{col}/zarr.json    ──► ome.well.images[].path      (PLANNED)
                                  └──► ∩ {fov}/ dirs that exist        ──► fovs_per_region
```

This is not a nicety: without the intersection, `allow_incomplete=True` (R3) crashes on the
only case it exists for.

### R3 — Completeness gate, three states

A zarr array is allocated full-size up front; unwritten chunks decode as the fill value
(zeros). A crashed run therefore projects to a plate that looks finished and is wrong. The
individual-TIFF path has no equivalent hole — a missing plane is a missing file, so
`reader.py:217` raises `KeyError`.

Gate on `_squid.acquisition_complete`, and distinguish three end states, not two:

| State | `acquisition_complete` | `aborted` | Directories |
|---|---|---|---|
| Clean finish | `true` | absent | all present |
| User abort | `false` | `true` | up to abort point |
| Hard crash / power loss | `false` | absent | up to crash |

Default: refuse. Override: `allow_incomplete=True`, which warns and proceeds — and must be
reachable from **both** `_cli.py` (`--allow-incomplete`) and the `_viewer.py` open path. A
CLI-only override hard-blocks the GUI user, who is the one most likely to have a crashed plate.

### R4 — Viewer seam: `start_zarr_acquisition`, not `load_dataset`

`plane_ref()` promises `(filepath, page_index)` (`reader.py:242-245`), and
`ndviewer_light.register_image` documents `page_idx` as a TIFF page (`core.py:2294-2317`).
A zarr plane is a slice of a chunked array: no file, no page.

Two traps here.

**Trap 1 — raising is not safe.** `_viewer.py:1893` catches only
`(KeyError, IndexError, OSError, RuntimeError)` and `_viewer.py:1613` only
`(KeyError, IndexError, OSError)`. A `NotImplementedError` from `plane_ref` **crashes the GUI**.

**Trap 2 — `load_dataset()` is the wrong API.** It never sets `_fov_labels`, and
`go_to_well_fov` returns `False` immediately when `_fov_labels` is empty (`core.py:2511`). So
double-click navigation and the plate↔slider red-box link (`_viewer.py:1906` reads
`_detail._fov_labels`) both stop working — silently, with no error. It would also impose
ndviewer's own `discover_zarr_v3_fovs` ordering over SquidMIP's `_plate_key` row-major order,
desynchronising the slider index from `_fov_index[well]["idx"]`.

Use the seam Squid's own HCS GUI uses:

```python
class SquidZarrReader:
    supports_plane_ref = False
    def fov_store_path(self, region, fov) -> str: ...   # per-field group dir
```

```python
self._detail.start_zarr_acquisition(fov_paths, channels, num_z, fov_labels, height, width)
```

`fov_labels` is an explicit parameter, so SquidMIP keeps control of ordering. Note the reader
needs a **positive** capability (`fov_store_path`), not merely `supports_plane_ref = False`.

### R5 — Distinguish a Squid acquisition from SquidMIP's own output

`write_plate` writes `<out>/plate.ome.zarr` (`_output.py:301`) — structurally identical to
Squid's. After R1, pointing `open_reader` at our own output returns a `SquidZarrReader` that
then dies on a missing `acquisition.yaml`, complaining about the wrong thing entirely.

The discriminator is free: **Squid writes `_squid` and no `field_count`; we write
`field_count` and no `_squid`.** Refuse with a message that says so.

### R6 — Channels from `omero`, scalars from `acquisition.yaml`

`omero.channels[].label` is written by iterating the C axis (`zarr_writer.py:544`), so its
order **is** the array's C order — stronger evidence than `acquisition_channels.yaml` key
order, which is a separate file that merely usually agrees. Take names and order from omero;
use the yaml for display colors via the existing `resolve_channels()`; warn on disagreement.
Same principle as IMA-189 trusting filenames over `coordinates.csv`. This deliberately
diverges from `SquidOMEReader`'s yaml-first ladder (`reader.py:324-332`) — comment the
divergence in place.

Scalars go the other way. `pixel_size_um`, `dz_um` and `wellplate_format` come from
`load_acquisition_metadata()` as for every other reader (`_acquisition.py:3-5,62`);
`_squid.pixel_size_um` / `z_step_um` are read only as a cross-check that warns on drift,
mirroring `reader.py:179-188`. `acquisition.yaml` stays required — `wellplate_format` exists
nowhere else. Handle it being **present-but-null**.

### R7 — Validate dtype at open, keep the per-plane guard

A zarr array's dtype is in its `zarr.json`, so check it against `_SUPPORTED_DTYPES` once when
the store opens rather than on plane 1000 of a plate run. Squid explicitly contemplates float
stores (`zarr_writer.py:571-577`). Keep `_validate_plane` on each returned slice as a backstop
against a store whose chunks disagree with its header.

### R8 — Do not repeat the false rationale for deferring per-FOV

`multi_point_worker.py:268` selects the layout from the acquisition **mode**
(`is_hcs = is_select_wells or is_loaded_wells`), not from region naming. A genuine wellplate
run in flexible/manual-region mode emits the per-FOV layout with region ids like `B2` — which
`parse_well_id` accepts and `write_plate` would handle fine. The HCS-only scope stands on
effort grounds; the refusal message and the TODO must both say so honestly, because real plate
users will hit it.

### R9 — Refactor before feature, but test before refactor

`SquidReader.metadata` (`reader.py:163-208`) and `SquidOMEReader.metadata` (`:319-350`)
already duplicate the region/fov assembly, the eleven-key metadata dict, and the Nz
cross-check warning. A third copy is the drift risk. Extract shared helpers first, and lift
`_montage._PlateLayout` (`_montage.py:83-135`) — which already parses plate → well → field →
array from the same group metadata — into a shared module rather than re-deriving the walk.

**But the stated regression gate is thinner than it looks.** Exactly one of the 20 tests in
`test_reader.py` touches `SquidOMEReader.metadata` (`:197`), with a single well, single FOV
and matching yaml. Nothing covers the yaml/`n_c` mismatch fallback, `_ome_channel_names`,
multi-region `_plate_key` ordering, or the Nz warning on the OME path. Write OME
characterization tests **before** the refactor, or the gate is theater.

### R10 — Documentation is part of the change

`reader.py:1-6` still says OME-TIFF and Zarr are "detected and rejected" while
`SquidOMEReader` is implemented at `:267` — already false before this ticket. Replace the
single-path flow diagram (`:17-24`) with the dispatch tree from R1, and give
`SquidZarrReader` its own layout diagram.

### R11 — Memory claim to avoid in comments

Under NONE/FAST, chunks are `(1,1,1,y,x)` and one plane read decompresses one chunk. Under
BALANCED/BEST the shard is `(1, c_size, 1, Y, X)` — all channels for a `(t,z)` — so a
per-channel-per-z projection re-touches each shard once per channel. Not fatal, but do not
write "exactly one plane decompresses per call".

---

## 4. Test recommendations

42 new code paths, all currently uncovered. Full diagram in `.spec/open/ima-229.md`.

**Fixture strategy matters more than the count.** Build `squid_zarr_dataset` by hand-writing
the group `zarr.json` files to match **Squid's** shape (no `field_count`, unpadded columns,
`_squid` block), using `_zarr_store.create_array` only for pixel arrays. Do **not** generate it
with `write_plate` — that tests the reader against our own writer, and the two differ in
exactly the ways R5 depends on.

Ship a **partial-plate variant**: plate metadata listing 4 wells, only 2 present on disk. That
single fixture is what proves R2 and R3 correct.

---

## 5. Failure modes

| # | Failure | Test | Handling | User sees |
|---|---|---|---|---|
| 1 | Partial acquisition projected as zeros | R3 | refuses | Error naming the state |
| 2 | Metadata over-claims absent wells | R2 | disk intersection | Only real wells processed |
| 3 | `allow_incomplete` crashes on partial plate | R2/R3 | same intersection | Override actually works |
| 4 | Channel order swapped vs yaml | R6 | warns | Correctly labeled pixels |
| 5 | Per-FOV / 6-D layout opened | R1 | named refusal | Accurate message |
| 6 | Float dtype store | R7 | refused at open | Immediate, not mid-run |
| 7 | `plane_ref` on zarr in GUI | R4 | branch avoids it | Raw stack loads |
| 8 | Plate↔slider desync | R4 | explicit `fov_labels` | Navigation works |
| 9 | `open_reader` on our own output | R5 | named refusal | "output, not acquisition" |
| 10 | `wellplate_format` null | R6 | explicit `None` handling | Clear message |

**0 critical gaps** (no test AND no handling AND silent). Failures 1-3, 8 and 9 were critical
gaps before the review.

---

## 6. Outside-voice challenge and disposition

Codex unavailable; ran a Claude subagent with fresh context. It found 8 issues the interactive
review missed. I verified each against source before accepting; three overturned decisions
already locked in the interactive pass:

| Challenge | Verified against | Disposition |
|---|---|---|
| `load_dataset` is the wrong API | `core.py:2788`, `:2511`, `:3710` | **Accepted** — R4 rewritten |
| Metadata is a plan, not a record | `job_processing.py:549-570` | **Accepted** — R2 rewritten |
| Layout follows mode, not naming | `multi_point_worker.py:268` | **Accepted** — R8; scope unchanged |
| No output-vs-acquisition discriminator | `_output.py:301` | **Accepted** — R5 added |
| Regression gate is theater | `test_reader.py` (1 of 20) | **Accepted** — R9 amended |
| `allow_incomplete` is CLI-only | `_viewer.py` open path | **Accepted** — R3 |
| Two-state taxonomy misses hard crash | `zarr_writer.py:765`, `:797` | **Accepted** — R3 |
| Shard read amplification | `zarr_writer.py:155-160` | **Accepted** as a caveat — R11 |

It also confirmed D5 (`acquisition.yaml` is format-independent, `multi_point_controller.py:882`)
and the detection bug, so those were not re-litigated.

The lesson worth keeping: the three overturned items were all cases where the interactive
review reasoned from a plausible model of the writer instead of reading it. Every one was
caught by opening the file.

---

## 7. NOT in scope

- **Per-FOV zarr** — deferred on effort, **not** because it is non-wellplate (see R8). TODO
  with a concrete trigger.
- **6-D zarr** — Squid's own docs call it non-standard (`zarr-v3-format.md:61`).
- **Zarr v2** — Squid writes v3 only.
- **Multi-page TIFF** — unchanged (`reader.py:143`).
- **Live/streaming read** — `allow_incomplete` opens a snapshot, does not follow a run.
- **Brightfield / RGB zarr** — same deferral as the TIFF path.
- **Distribution** — no new artifact.

---

## 8. Review metadata

- Issues raised interactively: 8 (5 architecture, 3 code quality) — all resolved.
- Outside-voice findings: 8 — all verified, all absorbed.
- Test gaps mapped: 42. Critical failure gaps: 0.
- Implementation tasks: 12 (T0–T12), none started.
- Parallelization: effectively sequential; one lane through `reader.py`, a short viewer lane.
- New dependencies: none.

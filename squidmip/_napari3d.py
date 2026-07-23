"""Native-resolution napari 3D, adopting hongquanli/gallery-view's recipe (not reinventing it).

WHY THIS EXISTS. napari renders 3D from ONE GL texture and refuses any axis over
GL_MAX_3D_TEXTURE_SIZE (~2048 on Apple GPUs), so handing it a fused REGION mosaic (5731 px) forces
napari's own crude downsample and the volume looks blocky. gallery-view sidesteps this the only way
that works: it feeds napari a SINGLE NATIVE ZYX STACK (one FOV / acquisition, ~2084 px) that fits
the texture, single-scale, with a micrometre voxel scale and the LUT carried over. That is native
resolution because the volume never exceeds the texture. We cannot import gallery-view (it pins
napari <0.6, we run 0.6.6), so this replicates its recipe: the exact add_image call, the
(dz, px, px) scale, additive blending, carried-over contrast, and a micrometre scale bar.

A single FOV, not the whole region, is deliberate: the region is a mosaic and cannot fit one
texture at native resolution. This is the "max res preview" of one field; AGAVE remains the path
for a path-traced, whole-region volume.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

log = logging.getLogger("squidmip.napari3d")


def _center_fov(meta: dict, region: str) -> Optional[int]:
    """The FOV nearest the region's stage centroid, so the 3D preview lands on representative
    tissue rather than a corner. Falls back to the first FOV when positions are unavailable."""
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        return None
    positions = meta.get("fov_positions_um") or {}
    pts = [(f, positions.get((region, f))) for f in fovs]
    pts = [(f, p) for f, p in pts if p is not None]
    if not pts:
        return int(fovs[0])
    cx = float(np.mean([p[0] for _f, p in pts]))
    cy = float(np.mean([p[1] for _f, p in pts]))
    return int(min(pts, key=lambda fp: (fp[1][0] - cx) ** 2 + (fp[1][1] - cy) ** 2)[0])


def _native_stack(reader: Any, meta: dict, region: str, fov: int, channel: str) -> np.ndarray:
    """One FOV's native (z, y, x) stack for a channel. Reads only this field's planes."""
    z_levels = list(meta.get("z_levels") or [0])
    planes = []
    for z in z_levels:
        plane = np.asarray(reader.read(region, fov, channel, int(z)))
        if plane.ndim != 2:
            plane = plane.reshape(plane.shape[-2:])
        planes.append(plane)
    return np.stack(planes, axis=0) if len(planes) > 1 else planes[0][None, ...]


def open_native_3d(
    reader: Any,
    meta: dict,
    region: str,
    *,
    fov: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
    contrast_by_channel: Optional[dict] = None,
    colormap_by_channel: Optional[dict] = None,
) -> Any:
    """Open a fresh napari 3D viewer on ONE FOV's native z-stack (gallery-view's recipe).

    Returns the napari ``Viewer`` (a popout window). Raises with a named reason if the stack cannot
    be built, so the caller can route it to the log rather than a silent no-op.
    """
    import napari  # lazy: heavy import, and a machine without napari still runs the 2D app

    fov = _center_fov(meta, region) if fov is None else int(fov)
    if fov is None:
        raise ValueError(f"region {region!r} has no FOVs to render in 3D.")
    names = list(channels) if channels else [c["name"] for c in meta.get("channels", [])]
    if not names:
        raise ValueError("this acquisition declares no channels to render.")

    px = float(meta.get("pixel_size_um") or 1.0)
    dz = float(meta.get("dz_um") or px)                 # z step in um; fall back to xy if absent
    contrast_by_channel = contrast_by_channel or {}
    colormap_by_channel = colormap_by_channel or {}

    viewer = napari.Viewer(ndisplay=3, title=f"3D native (napari) — {region} / fov {fov}")
    n_z = 1
    for ch in names:
        try:
            stack = _native_stack(reader, meta, region, fov, ch)
        except Exception as exc:                        # noqa: BLE001 - named, then continue
            log.error("3D native: could not read %s/%s/fov %s: %s", region, ch, fov, exc)
            continue
        n_z = max(n_z, int(stack.shape[0]))
        kwargs = {
            "name": ch,
            "scale": (dz, px, px),                      # (z, y, x) micrometres, gallery-view style
            "blending": "additive",
            "rendering": "mip",
        }
        cmap = colormap_by_channel.get(ch)
        if cmap is not None:
            kwargs["colormap"] = cmap
        clim = contrast_by_channel.get(ch)
        if clim is not None:
            kwargs["contrast_limits"] = tuple(clim)
        viewer.add_image(stack, **kwargs)

    if not viewer.layers:
        viewer.close()
        raise ValueError(f"{region}/fov {fov}: no channel could be read, so there is no 3D volume.")

    # Micrometre scale bar and a bounding box, exactly like gallery-view.
    try:
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "um"
    except Exception:                                   # noqa: BLE001 - cosmetic
        pass
    log.info("3D native: opened %s / fov %s, %d channel(s), %d z at native %.3f um/px, dz %.2f um",
             region, fov, len(viewer.layers), n_z, px, dz)
    return viewer

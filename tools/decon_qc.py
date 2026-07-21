#!/usr/bin/env python3
"""IMA-252: see Richardson-Lucy SEMI-CONVERGENCE, and pick the iteration count by eye.

Why this tool exists
--------------------
RL does not converge to the truth and stop. The reconstruction error falls, reaches a
minimum, and then RISES again as the algorithm starts explaining the NOISE in the data.
On a point-like structure the visible tell is a halo: it tightens for a few iterations,
and then a disc around the core starts GROWING BACK, brighter and wider each iteration.
That is not more resolution, it is amplified noise wearing the shape of the PSF.

There is no universally correct iteration count - it depends on SNR, on the PSF and on
the sample - so this tool does not invent one. It runs RL for k = 1..N on ONE real FOV,
keeps every intermediate volume, and puts the two orthogonal sections through the
brightest structure (x-z and y-z, TURBO colormap, one row per iteration) in front of a
human. Turbo is used because it has a steep, high-contrast ramp through the low
intensities where a halo lives; on a grey ramp the halo is the part of the image the eye
is worst at.

Alongside the picture it emits a scalar so the turning point is not purely a matter of
taste: :func:`halo_core_ratio`, the brightness of the halo relative to the core it
surrounds. It falls while RL concentrates light and rises when the disc grows back. Read
that function's docstring before trusting the number - it includes a ground-truth control
showing how far the visible halo LAGS the true error minimum.

The deconvolution itself is NOT implemented here. It is ``squidmip._decon``, which is
Julio's petakit engine (PetaKit5D's RL port) with a VECTORIAL PSF computed from the
acquisition's own optics - NA 0.3 on this scope, not the NA 0.4 that once justified a
hardcoded Gaussian sigma of 1.5 px. The tool prints the PSF's measured lateral sigma on
every run precisely so that claim is checkable and not just asserted (the NA-0.3 PSF
comes out near 1.165 px; 1.5 px would be ~29% too wide).

Usage
-----
    python tools/decon_qc.py                       # defaults: tissue set, manual0/fov 0, 488
    python tools/decon_qc.py --iterations 12 --out /tmp/qc
    python tools/decon_qc.py --region manual1 --fov 3 --channel Fluorescence_405_nm_Ex

Outputs (into --out): ``decon_qc_montage.png``, ``decon_qc_curve.png``, ``decon_qc.csv``.
The datasets are opened READ ONLY and nothing is ever written back next to them.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# Run from anywhere: import the repo this file lives in, not whatever `squidmip` happens
# to be installed. The mac filesystem is case-insensitive, so an invoker sitting in
# .../CEPHLA/ instead of .../Cephla/ otherwise resolves a different tree.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TISSUE = ("/Users/julioamaragall/Downloads/"
          "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")

# k = 1..N. 8 is far past the current DEFAULT_ITERATIONS of 3, which is the point: the
# turn has to be INSIDE the sampled range or the recommendation is worthless.
DEFAULT_MAX_ITERATIONS = 8

# Lateral half-width of the region RL is actually run on, in pixels. A crop, not the
# whole 2084x2084 frame, because 8 separate RL runs over the full frame is minutes of
# compute for a picture of one structure. 128 px is ~96 um, ~90 Airy radii of margin
# around the structure at the centre, so the FFT edges are nowhere near it.
DEFAULT_CROP_HALF = 128

# Half-width of what the montage actually SHOWS, in pixels. 32 px is ~24 um across, ~22
# Airy radii: the structure and the halo around it fill the panel instead of being two
# pixels in the middle of a mostly empty strip.
DEFAULT_VIEW_HALF = 32


# --------------------------------------------------------------------------------------
# The QC metric
# --------------------------------------------------------------------------------------
def halo_core_ratio(volume, centre, dxy_um, dz_um, core_um, window_um):
    """Mean brightness of the HALO divided by the mean brightness of the CORE.

    WHAT IT MEASURES, exactly. Around the brightest structure, take two spheres measured
    in real micrometres (z scaled by dz_um, x/y by dxy_um - a sphere in the sample, not in
    voxels):

      * the CORE, radius *core_um* = the Airy radius 0.61*lambda/NA. That is the tightest
        spot this instrument could possibly form, so it is where deconvolution is trying
        to put the light.
      * the WINDOW, radius *window_um*. The shell between the two is the HALO - the light
        that is still smeared around the structure.

    The number reported is ``mean(halo) / mean(core)``: how bright the halo is relative to
    the core it surrounds. Means, not sums, so the answer does not simply reflect the fact
    that the shell has far more voxels than the core; ~1.0 means the halo is as bright as
    the thing it surrounds, ~0.1 means the structure is nearly all core.

    HOW TO READ IT. Concentrating light into the core is exactly RL's job, so the number
    FALLS while deconvolution is doing real work. "The disc looks like it is growing
    again" - the semi-convergence tell - is this number RISING: the core stops gaining and
    the surroundings start filling back in with amplified noise and ringing. The argmin is
    the last iteration that still bought concentration.

    WHY THIS ONE. It is the direct numerical reading of the thing being judged in the
    turbo x-z / y-z view, measured on the same structure and the same volume the montage
    shows, so the number and the picture cannot disagree. Sharpness metrics (gradient
    energy, variance-of-Laplacian, total variation) rise monotonically under RL - noise is
    sharp too - so they have no minimum and cannot see semi-convergence at all.

    WHAT IT IS NOT, measured rather than assumed. On a synthetic control where the truth
    IS known (four point sources at 3000 counts, blurred with this same NA-0.3 vectorial
    PSF, Poisson noise, RL run to 64 iterations), the true error against the truth bottoms
    out at k~5 (RMSE 15.65 -> 15.57 -> 15.63 -> 16.97) while this ratio keeps falling to
    k~32 before turning back up (0.397 -> 0.187 -> 0.236). So the visible halo is a LATE,
    CONSERVATIVE indicator: once you can see the disc growing you are certainly past the
    optimum, but not seeing it grow does NOT prove you are before it. That is a real limit
    of judging by eye and it is why this tool recommends rather than decides.

    (The residual-whiteness stopping rule was tried as an alternative and rejected on
    evidence: on the synthetic control it tracks the true optimum well, but on the real
    tissue crops its residual is dominated by model mismatch, not by noise - the measure
    lands around 3e4 instead of ~1 and rises monotonically from k=1, which would advise
    stopping before deconvolution has done anything. A metric that is only trustworthy on
    simulations is not a QC metric.)

    WHY A BACKGROUND FLOOR IS REMOVED. A constant camera offset - ~2500 counts on this
    sensor - sits in both terms and drags the ratio toward 1 regardless of what the optics
    did. The 10th percentile of the crop is subtracted (clipped at zero) as a background
    floor, estimated PER VOLUME because RL redistributes the floor as it runs.
    """
    volume = np.asarray(volume, dtype=np.float64)
    zc, yc, xc = centre
    zz, yy, xx = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
    r_um = np.sqrt(((zz - zc) * dz_um) ** 2
                   + ((yy - yc) * dxy_um) ** 2
                   + ((xx - xc) * dxy_um) ** 2)
    core = r_um <= core_um
    halo = (r_um <= window_um) & ~core
    if not core.any() or not halo.any():
        raise ValueError(
            f"core radius {core_um} um / window radius {window_um} um do not resolve into "
            f"voxels at dxy={dxy_um} um, dz={dz_um} um."
        )

    floor = float(np.percentile(volume, 10.0))
    signal = np.clip(volume - floor, 0.0, None)

    core_mean = float(signal[core].mean())
    if core_mean <= 0:
        raise ValueError(
            "the structure's core is at or below the background floor, so a halo/core "
            "ratio is undefined. Pick a different fov/channel."
        )
    return float(signal[halo].mean() / core_mean)


def recommend(ks, curve):
    """Turn a QC curve into ``(best_k, kind, message)``. ``kind`` is one of:

    ``"turn"``
        The minimum is INTERIOR to the sweep - the curve falls and then rises again. Only
        this case is a real semi-convergence minimum and only this case is a
        recommendation worth acting on.
    ``"still-falling"``
        The minimum is the LAST point sampled. The argmin is then an artefact of where the
        sweep stopped, not a property of the data. Saying "N iterations" here would be
        inventing a turning point, so the tool says so instead.
    ``"rising"``
        The minimum is the FIRST point sampled: RL already overshoots at k=1 on this
        structure.
    """
    index = int(np.argmin(curve))
    best = int(ks[index])
    if 0 < index < len(curve) - 1:
        return best, "turn", (
            f"RECOMMENDATION: {best} iterations - the curve falls and turns back up "
            f"INSIDE 1..{ks[-1]}, so this is a real semi-convergence minimum.")
    if index == len(curve) - 1:
        return best, "still-falling", (
            f"NO TURN in 1..{ks[-1]}: the halo is still shrinking at the last iteration "
            f"sampled. The minimum is {best} only because the sweep stopped there, so it "
            "is NOT a recommendation - re-run with a larger --iterations. Note the "
            "control result in halo_core_ratio(): the visible halo turns LATE, so "
            "'no visible turn yet' does not by itself mean more iterations are better.")
    return best, "rising", (
        f"NO TURN in 1..{ks[-1]}: the curve rises from the very first iteration, i.e. RL "
        "overshoots immediately on this structure. Use fewer iterations, or a fov with a "
        "better-isolated structure.")


def qc_window_um(core_um, nz, dz_um, preferred=8.0):
    """Window radius: *preferred* core radii, but never deeper than the stack can hold.

    The sphere has to fit AXIALLY or the metric silently measures a truncated cap whose
    shape depends on where in the stack the structure sits. 8 core radii is 8.5 um here
    but the stack is 10 planes at 1.5 um, so this caps it at 6 um - the largest sphere
    that can be centred on an interior plane with margin on both sides.
    """
    return float(min(preferred * core_um, max((nz // 2) - 1, 1) * dz_um))


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_stack(dataset, region, fov, channel):
    """Read ONE (region, fov, channel) z-stack as (Z, Y, X). Read-only, one file per z."""
    from squidmip import open_reader

    reader = open_reader(dataset)
    meta = reader.metadata
    if region is None:
        region = meta["regions"][0]
    if channel is None:
        channel = meta["channels"][0]["name"]
    planes = [reader.read(region, fov, channel, z) for z in meta["z_levels"]]
    return np.stack(planes), region, channel, meta


def brightest_structure(stack, dxy_um, dz_um, core_um, z_margin=0, xy_margin=0):
    """(z, y, x) of the brightest STRUCTURE, not the brightest pixel.

    A single hot pixel is the brightest voxel in most fluorescence frames and it is not a
    structure; centring the QC on one would measure the camera. Smoothing by the
    diffraction-limited core first means the maximum is the brightest thing the optics
    could actually have formed.

    Candidates within *z_margin* planes of the top or bottom of the stack, or within
    *xy_margin* pixels of the frame edge, are excluded: the QC window has to fit around
    the structure. On the first real FOV this matters - the raw argmax lands on z=0, where
    the measurement window would be sliced in half.
    """
    from scipy.ndimage import gaussian_filter

    sigma = (max(core_um / dz_um, 0.5), core_um / dxy_um, core_um / dxy_um)
    smoothed = gaussian_filter(stack.astype(np.float32), sigma)
    nz, ny, nx = smoothed.shape
    z0, z1 = min(z_margin, (nz - 1) // 2), max(nz - z_margin, (nz + 2) // 2)
    y1, x1 = max(ny - xy_margin, xy_margin + 1), max(nx - xy_margin, xy_margin + 1)
    allowed = np.zeros(smoothed.shape, dtype=bool)
    allowed[z0:z1, xy_margin:y1, xy_margin:x1] = True
    return np.unravel_index(
        int(np.argmax(np.where(allowed, smoothed, -np.inf))), smoothed.shape)


def crop_around(stack, centre, half):
    """Crop laterally to +-*half* px around *centre*, keeping every z plane.

    Returns (crop, centre_in_crop). Clamped to the frame, so the centre is not assumed to
    land in the middle of the crop.
    """
    _, ny, nx = stack.shape
    y0 = int(np.clip(centre[1] - half, 0, max(ny - 2 * half, 0)))
    x0 = int(np.clip(centre[2] - half, 0, max(nx - 2 * half, 0)))
    y1, x1 = min(y0 + 2 * half, ny), min(x0 + 2 * half, nx)
    return stack[:, y0:y1, x0:x1], (centre[0], centre[1] - y0, centre[2] - x0)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def orthogonal_slices(volume, centre, half=None):
    """The x-z and y-z sections through *centre*, both returned as (z, lateral).

    *half* crops each section to +-half pixels laterally around the structure. RL runs on
    a much wider crop so the FFT edges stay far away, but showing all of it would draw the
    structure two pixels wide; the halo being judged lives within a few micrometres.
    """
    zc, yc, xc = centre
    xz, yz = volume[:, yc, :], volume[:, :, xc]
    if half:
        x0, x1 = max(xc - half, 0), min(xc + half, xz.shape[1])
        y0, y1 = max(yc - half, 0), min(yc + half, yz.shape[1])
        xz, yz = xz[:, x0:x1], yz[:, y0:y1]
    return xz, yz


def _display(panel, gamma=0.5):
    """Background-subtract, normalise to the panel's own maximum, apply a display gamma.

    The same 10th-percentile background floor the metric uses is removed first, so the
    camera offset does not sit at mid-turbo and paint the whole field green.

    Per-panel normalisation is ON PURPOSE: RL raises the peak by a large factor as it
    concentrates light, and a shared scale would just make later rows look uniformly
    brighter. Scaling each row to its own core makes the HALO RELATIVE TO THE CORE - the
    exact thing being judged - comparable down the column. Gamma 0.5 lifts the low
    intensities where the halo lives; it is a display transform only and never touches
    the number in the QC curve.
    """
    panel = np.asarray(panel, dtype=np.float64)
    panel = np.clip(panel - np.percentile(panel, 10.0), 0.0, None)
    peak = panel.max()
    if peak <= 0:
        return np.zeros_like(panel)
    return (panel / peak) ** gamma


def write_montage(path, per_iteration, centre, dxy_um, dz_um, title, view_half=None):
    """rows = iterations, columns = [x-z, y-z], TURBO, iteration number labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(per_iteration)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 0.85 * n + 0.6), squeeze=False)
    aspect = dz_um / dxy_um          # z steps are 1.5 um, pixels 0.752 um: draw them square
    for row, (label, volume) in enumerate(per_iteration):
        xz, yz = orthogonal_slices(volume, centre, view_half)
        for col, (panel, name) in enumerate(((xz, "x-z"), (yz, "y-z"))):
            ax = axes[row][col]
            ax.imshow(_display(panel), cmap="turbo", vmin=0.0, vmax=1.0,
                      aspect=aspect, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_curve(path, iterations, values, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(iterations, values, marker="o", color="#1f77b4")
    if best is not None:
        ax.axvline(best, color="#d62728", linestyle="--",
                   label=f"argmin = {best} iterations")
        ax.legend(fontsize=8)
    ax.set_xlabel("RL iterations")
    ax.set_ylabel("energy outside the core / energy in window")
    ax.set_title("RL semi-convergence: down is sharpening, up is noise", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=TISSUE)
    p.add_argument("--region", default=None, help="default: the first region")
    p.add_argument("--fov", type=int, default=0)
    p.add_argument("--channel", default=None, help="default: the first channel")
    p.add_argument("--iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
                   help="run RL for k = 1..N (default 8)")
    p.add_argument("--crop-half", type=int, default=DEFAULT_CROP_HALF,
                   help="half-width in px of the region RL is run on")
    p.add_argument("--view-half", type=int, default=DEFAULT_VIEW_HALF,
                   help="half-width in px of the montage panels (the RL crop is wider on "
                        "purpose; this is what gets looked at)")
    p.add_argument("--out", default=".", help="directory for the montage, curve and csv")
    p.add_argument("--no-gpu", action="store_true",
                   help="force the CPU backend (same RL update, different backend)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from squidmip._decon import DEFAULT_ITERATIONS, METHOD, OpticsParams, _run, make_psf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stack, region, channel, meta = load_stack(
        args.dataset, args.region, args.fov, args.channel)
    optics = OpticsParams.from_acquisition(args.dataset, channel=channel)
    optics = OpticsParams(optics.na, optics.wavelength_um, optics.dxy_um, optics.dz_um,
                          int(stack.shape[0]), optics.ni)
    dxy_um, dz_um = optics.dxy_um, optics.dz_um

    # The Airy radius of THIS instrument: the smallest core the optics could form. It is
    # what separates "core" from "halo" below, and it comes from NA and wavelength, not
    # from a tuning constant.
    core_um = 0.61 * optics.wavelength_um / optics.na
    window_um = qc_window_um(core_um, stack.shape[0], dz_um)

    psf = make_psf(optics)
    print(f"dataset   : {args.dataset}")
    print(f"selection : region={region} fov={args.fov} channel={channel} "
          f"z={stack.shape[0]} planes, frame {stack.shape[1:]}")
    print(f"optics    : NA={optics.na} lambda_em={optics.wavelength_um} um "
          f"dxy={dxy_um} um dz={dz_um} um  (ni={optics.immersion_index})")
    print(f"psf       : vectorial, shape {psf.shape}, lateral sigma "
          f"{_lateral_sigma_px(psf):.3f} px  <- NA {optics.na}, not the old "
          f"hardcoded 1.5 px Gaussian")
    print(f"metric    : mean brightness of the halo ({core_um:.3f}..{window_um:.3f} um "
          f"shell) / mean brightness of the {core_um:.3f} um core")

    z_margin = int(np.ceil(window_um / dz_um))
    centre_full = brightest_structure(stack, dxy_um, dz_um, core_um,
                                      z_margin=z_margin, xy_margin=args.crop_half)
    crop, centre = crop_around(stack, centre_full, args.crop_half)
    print(f"structure : brightest at (z,y,x)={centre_full} in the frame; RL runs on a "
          f"{crop.shape} crop, structure at {centre}")
    print(f"engine    : petakit method={METHOD!r}, gpu={not args.no_gpu}\n")

    rows = [("raw", crop.astype(np.float32))]
    for k in range(1, args.iterations + 1):
        rows.append((f"{k}", _run(crop, psf, k, gpu=not args.no_gpu)))
        print(f"  ran RL k={k}", flush=True)

    values = [halo_core_ratio(v, centre, dxy_um, dz_um, core_um, window_um)
              for _, v in rows]
    raw_value, curve = values[0], values[1:]
    ks = list(range(1, args.iterations + 1))

    print("\niter  halo/core   delta vs previous")
    print(f" raw  {raw_value:.6f}")
    for k, value, prev in zip(ks, curve, [raw_value] + curve[:-1]):
        print(f"{k:>4}  {value:.6f}   {value - prev:+.6f}")

    best, kind, verdict = recommend(ks, curve)
    print()
    print(verdict)
    print(f"Measured on ONE FOV ({region}/{args.fov}/{channel}) - a recommendation for THIS "
          "sample at THIS exposure, never a global default; SNR and structure decide the "
          f"answer. The shipped default is DEFAULT_ITERATIONS={DEFAULT_ITERATIONS} and it "
          "stays until a human has looked at the montage and changed it deliberately.")

    montage = out / "decon_qc_montage.png"
    write_montage(montage, rows, centre, dxy_um, dz_um,
                  f"RL semi-convergence - {region}/{args.fov}/{channel} - turbo, "
                  f"per-row normalised", view_half=args.view_half)
    curve_png = out / "decon_qc_curve.png"
    write_curve(curve_png, ks, curve, best if kind == "turn" else None)
    csv_path = out / "decon_qc.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iterations", "halo_core_ratio"])
        w.writerow([0, f"{raw_value:.6f}"])
        for k, value in zip(ks, curve):
            w.writerow([k, f"{value:.6f}"])
    print(f"\nwrote {montage}\nwrote {curve_png}\nwrote {csv_path}")
    return 0


def _lateral_sigma_px(psf):
    """Second-moment-equivalent lateral sigma of the in-focus PSF plane, in pixels.

    Printed on every run so "the PSF really is the NA-0.3 one" is a measurement in the
    log rather than a claim in a docstring.
    """
    plane = np.asarray(psf[psf.shape[0] // 2], dtype=np.float64)
    total = plane.sum()
    if total <= 0:
        return float("nan")
    yy, xx = np.ogrid[:plane.shape[0], :plane.shape[1]]
    cy = float((plane * yy).sum() / total)
    cx = float((plane * xx).sum() / total)
    var = float((plane * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / total) / 2.0
    return float(np.sqrt(var))


if __name__ == "__main__":
    raise SystemExit(main())

"""Subprocess driver: register one region with tilefusion, emit positions.json.

Runs as its own process so the harness measures it from outside like every other tool,
and so tilefusion's heavy ``__init__`` (numba, GPU probe, basicpy) never enters the
harness process.

REGISTRATION ONLY -- deliberately never fuses. Two reasons:

  * The seam metric needs the SOLVED POSITIONS, not a mosaic. Fusing would burn
    minutes and gigabytes producing an artifact the metric cannot use anyway, because
    fusion blends the overlaps and destroys the two independent views it correlates.
  * It keeps the benchmark disk-safe, which is IMA-233's stated acceptance criterion.

The position contract, read out of ``tilefusion/core.py:1102``::

    solved_um[i] = tile_positions[i] + global_offsets[i] * pixel_size
                   \\_ stage (y,x) um   \\_ the least-squares solve, in pixels

Contract with the adapter -- writes into ``--out``:
    positions.json   {"positions_px": {"<fov>": [row_px, col_px]}, "tool", "version"}

Exit codes: 0 ok, 3 tilefusion not importable, 4 registration failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

# Stage positions must agree with coordinates.csv to within this, in um, for a tile
# index to be matched to a Squid FOV. Generous: it only has to beat the FOV pitch.
_MATCH_TOL_UM = 5.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="tilefusion benchmark driver")
    ap.add_argument("--input", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--channel", required=True)
    ap.add_argument("--z", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--method", default="ONE_ROUND")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import tilefusion
        from tilefusion import TileFusion
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 3

    tf = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from bench.dataset import load_acquisition

        acq = load_acquisition(args.input)
        ch_idx = acq.channels.index(args.channel) if args.channel in acq.channels else 0

        tf = TileFusion(
            tiff_path=args.input,
            output_path=str(out),
            region=args.region,
            channel_to_use=ch_idx,
            registration_z=args.z,
            max_workers=args.threads or None,
        )

        stage_um = np.asarray(tf.tile_positions, dtype=float)  # (n, 2) as (y, x)
        if stage_um.size == 0:
            print("tilefusion loaded no tiles", file=sys.stderr)
            return 4

        tf.refine_tile_positions_with_cross_correlation(ch_idx=0)
        tf.optimize_shifts(method=args.method)

        ps = np.asarray(tf.pixel_size, dtype=float)
        offsets = tf.global_offsets
        if offsets is None:
            print("optimize_shifts produced no global_offsets", file=sys.stderr)
            return 4
        solved_um = stage_um + np.asarray(offsets, dtype=float) * ps

        # Map tile index -> Squid fov by matching STAGE positions. Index order is an
        # implementation detail of tilefusion's reader; matching on geometry is
        # self-validating, so a reader reordering shows up as an error rather than as
        # silently mislabelled rows.
        fov_of = _match_tiles_to_fovs(stage_um, acq, args.region)
        if fov_of is None:
            print("could not match tilefusion tile order to Squid fovs", file=sys.stderr)
            return 4

        origin = solved_um.min(axis=0)
        positions_px = {
            int(fov_of[i]): (
                float((solved_um[i, 0] - origin[0]) / ps[0]),
                float((solved_um[i, 1] - origin[1]) / ps[1]),
            )
            for i in range(len(solved_um))
        }

        (out / "positions.json").write_text(
            json.dumps(
                {
                    "tool": "tilefusion",
                    "version": getattr(tilefusion, "__version__", "unknown"),
                    "region": args.region,
                    "channel": args.channel,
                    "method": args.method,
                    "positions_px": {str(k): [v[0], v[1]] for k, v in positions_px.items()},
                },
                indent=2,
            )
        )
        return 0
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 4
    finally:
        if tf is not None:
            try:
                tf.close()
            except Exception:
                pass


def _match_tiles_to_fovs(stage_um: np.ndarray, acq, region: str) -> dict[int, int] | None:
    """Match each tilefusion tile index to a Squid FOV by stage position.

    ``acq.positions_mm`` is ``(x, y)`` in mm; tilefusion's are ``(y, x)`` in um.
    """
    fovs = acq.fovs(region)
    if len(fovs) != len(stage_um):
        return None
    ref = {f: (acq.positions_mm[(region, f)][1] * 1000.0,
               acq.positions_mm[(region, f)][0] * 1000.0) for f in fovs}

    out: dict[int, int] = {}
    used: set[int] = set()
    for i, (y, x) in enumerate(stage_um):
        best, best_d = None, None
        for f, (ry, rx) in ref.items():
            if f in used:
                continue
            d = abs(ry - y) + abs(rx - x)
            if best_d is None or d < best_d:
                best, best_d = f, d
        if best is None or best_d > _MATCH_TOL_UM:
            return None
        out[i] = best
        used.add(best)
    return out


if __name__ == "__main__":
    raise SystemExit(main())

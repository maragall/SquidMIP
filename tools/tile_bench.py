"""IMA-256 evidence: how fast is tile assembly, and does registration solve PER WELL?

    python tools/tile_bench.py --dataset "/path/to/acquisition"
    python tools/tile_bench.py --dataset ... --repeats 3 --regions A1,A2
    python tools/tile_bench.py --dataset ... --no-scope-probe --json out.json

Two questions, both previously unanswered, both answered here with measurements:

Q1  SPEED. ``tools/benchmark.py`` times a whole ``stitch_plate`` run and prints ONE row
    per operator. That compares operators, but it cannot tell you what to optimise: it
    aggregates every well together and folds disk I/O into the ``project`` stage. This
    tool drives ``stitch_region`` one region at a time and reports ms/region, ms/tile,
    output Mpix/s, and the split across I/O / z-reduction / registration / blend.

Q2  REGISTRATION SCOPE. Solve each region alone, then solve every region in one
    ``stitch_plate`` call, and compare the offsets. Identical offsets prove nothing
    crosses a region boundary. See ``_benchmark.registration_scope_probe``.

Nothing is written unless you pass --json, and that file is a few KB. No mosaic is ever
persisted: each region's fused array is dropped as soon as it has been measured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squidmip._benchmark import (  # noqa: E402
    benchmark_regions,
    format_region_timings,
    format_scope_probe,
    registration_scope_probe,
)

DATASET = (
    "/Users/julioamaragall/Downloads/"
    "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--operator", default="stitch",
                    help="region operator to time (default stitch; 'coordinate' is the "
                         "unregistered control)")
    ap.add_argument("--regions", default=None, help="comma-separated well subset")
    ap.add_argument("--channels", default="1",
                    help="channel indices to fuse (default 1; a 4-channel 36-FOV mosaic is "
                         "~1 GB resident and would measure memory, not stitching)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the discarded first fuse. tilefusion's blend kernels are "
                         "numba-JIT'd, so without the warm-up the first region measured "
                         "carries several seconds of compilation that belong to no region.")
    ap.add_argument("--no-scope-probe", action="store_true",
                    help="skip Q2. The probe re-solves every region twice, so it roughly "
                         "doubles the run.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from squidmip import open_reader

    channels = [int(c) for c in args.channels.split(",") if c.strip() != ""] or None
    regions = [r.strip() for r in args.regions.split(",")] if args.regions else None

    reader = open_reader(args.dataset)
    print(f"dataset  : {args.dataset}")
    print(f"operator : {args.operator}   channels={channels}   repeats={args.repeats}")
    print("", flush=True)

    t0 = time.perf_counter()
    timings = benchmark_regions(
        reader, operator=args.operator, regions=regions, channels=channels,
        repeats=args.repeats, warmup=not args.no_warmup,
        on_region=lambda t: print(f"  {t.region}: {t.total_ms:.0f} ms "
                                  f"({t.tiles} tiles)", flush=True),
    )
    print("")
    print("Q1  tile assembly speed, per region:")
    print(format_region_timings(timings))

    probe = None
    if not args.no_scope_probe:
        print("")
        print("Q2  registration scope:", flush=True)
        probe = registration_scope_probe(reader, regions=regions, channels=channels)
        print(format_scope_probe(probe))

    print(f"\nsuite wall time: {time.perf_counter() - t0:.1f}s")

    if args.json:
        payload = {
            "dataset": args.dataset,
            "operator": args.operator,
            "channels": channels,
            "timings": [
                vars(t) | {"mosaic_megapixels": t.mosaic_megapixels,
                           "ms_per_tile": t.ms_per_tile,
                           "mpix_per_s": t.mpix_per_s,
                           "mip_ms": t.mip_ms,
                           "shares": t.shares()}
                for t in timings
            ],
            "scope_probe": probe,
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

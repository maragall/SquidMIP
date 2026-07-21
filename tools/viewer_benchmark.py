"""Odon vs ndviewer_light on ONE identical task — IMA-235.

    python tools/viewer_benchmark.py --hcs-dir /path/to/output.hcs
    python tools/viewer_benchmark.py --dataset /path/to/acquisition --clean

Sibling to ``tools/odon_benchmark.py``, which measures a different thing (Odon's own
local-vs-http transport). This one is a HEAD-TO-HEAD, and the task both tools perform is
defined in ``squidmip/_viewer_bench.py``'s docstring BEFORE any number is produced.

Disk: writes one well / one FOV (~23 MB) when given ``--dataset``, plus symlink plates of
~200 bytes per well for the scaling rows. ``--clean`` removes all of it. ``write_plate``'s
``check_disk_space`` gate stays ON.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squidmip._viewer_bench import format_report, run, write_json  # noqa: E402

DATASET = "/Users/julioamaragall/Downloads/synthetic_2x2_wellplate"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--hcs-dir", default=None, help="an EXISTING squidmip output dir")
    ap.add_argument("--field", default=None, help="an OME-Zarr image group, used directly")
    ap.add_argument("--out", default=None)
    ap.add_argument("--region", default=None, help="single well to write (default: first)")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--scale-wells", default="1,96,1536")
    ap.add_argument("--level", default=None, help="default: coarsest, as Odon picks")
    ap.add_argument("--odon-bin", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    written = None
    field_dir = args.field
    if field_dir is None:
        hcs_dir = args.hcs_dir
        if hcs_dir is None:
            from squidmip import open_reader, write_plate

            out = args.out or tempfile.mkdtemp(prefix="vbench-")
            written = out
            reader = open_reader(args.dataset)
            regions = [args.region] if args.region else [reader.metadata["regions"][0]]
            print(f"writing ONE well ({regions[0]}), ONE fov -> {out}", flush=True)
            m = write_plate(reader, out, regions=regions, n_fovs=1)
            print(f"  {m['n_fields_written']} field(s), {m['levels']} levels\n", flush=True)
            hcs_dir = out
        from squidmip._odon import _plate_dir, iter_fields

        fields = list(iter_fields(_plate_dir(hcs_dir)))
        if not fields:
            raise SystemExit(f"no complete field groups under {hcs_dir}")
        field_dir = fields[0][3]

    scratch = tempfile.mkdtemp(prefix="vbench-scale-")
    try:
        rep = run(field_dir, odon_bin=args.odon_bin, repeats=args.repeats,
                  scale_wells=tuple(int(x) for x in args.scale_wells.split(",")),
                  scratch=scratch, level=args.level)
        print(format_report(rep))
        if args.json:
            print(f"wrote {write_json(rep, args.json)}")
        # Exit 0 even when a side could not be measured: "it did not run, and here is why"
        # is a successful outcome for this harness; non-zero would claim the harness broke.
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if args.clean and written:
            shutil.rmtree(written, ignore_errors=True)
            print(f"cleaned {written}")


if __name__ == "__main__":
    sys.exit(main())

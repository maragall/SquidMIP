"""Subprocess driver: emit the raw stage positions, performing no registration.

The experimental control. Squid's stage already reports where it believes each FOV
sits; this driver writes those positions unchanged, so the table carries a row showing
what you get for zero compute. A stitcher that cannot beat this number is not earning
its runtime.

On the real 20x_scan grid the stage alone leaves ~39 px (~14.6 um) of median seam
misalignment, so the bar is real rather than rhetorical.

Exit codes: 0 ok, 4 the acquisition could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="stage-baseline benchmark driver")
    ap.add_argument("--input", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--channel", required=True)
    ap.add_argument("--z", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from bench.dataset import load_acquisition

        acq = load_acquisition(args.input)
        positions = acq.positions_px(args.region)
        if not positions:
            print(f"region {args.region} has no positions", file=sys.stderr)
            return 4
        (out / "positions.json").write_text(
            json.dumps(
                {
                    "tool": "stage-baseline",
                    "version": "n/a (no registration)",
                    "region": args.region,
                    "positions_px": {str(k): [v[0], v[1]] for k, v in positions.items()},
                },
                indent=2,
            )
        )
        return 0
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())

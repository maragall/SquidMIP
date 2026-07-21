"""The no-registration control: raw stage metadata, zero compute."""

from __future__ import annotations

import sys

from bench.adapters.base import StitcherAdapter, StitchRequest


class StageBaselineAdapter(StitcherAdapter):
    """Reports Squid's own stage positions without registering anything.

    Always available (it needs no third-party tool), so every benchmark run carries a
    control row. Its wall time and RSS are the harness's own overhead, which makes it
    double as a floor for interpreting the other rows.
    """

    name = "stage-baseline"

    def is_available(self) -> bool:
        return True

    def version(self) -> str:
        return "n/a (no registration)"

    def build_command(self, req: StitchRequest) -> list[str]:
        return [
            sys.executable,
            "-m",
            "bench.drivers.stage_baseline_driver",
            "--input", str(req.acquisition.root),
            "--region", req.region,
            "--channel", req.channel,
            "--z", str(req.z),
            "--out", str(req.out_dir),
        ]

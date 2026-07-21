"""Dependency-import smoke test (CI). Fails loudly if any runtime dep won't import on this OS —
the cheap guard the maintainer asked for before freezing artifacts."""
import importlib, sys

# Every entry must be an actual declared dependency or a real module in this package.
# IMA-213: `squidhcs._video`, `imageio` and `imageio_ffmpeg` were removed — `_video` lives
# only on the IMA-185/187/205 branches, and the two imageio packages were video-player deps
# dropped at IMA-185 that were never declared in pyproject.toml. All three failed here, and
# because `freeze` declares `needs: smoke`, that kept the PyInstaller job from ever running.
MODULES = [
    "squidhcs", "squidhcs.reader", "squidhcs._engine", "squidhcs.projection",
    "squidhcs._output", "squidhcs._zarr_store", "squidhcs._montage", "squidhcs._cli",
    "squidhcs._viewer",                              # _viewer needs PyQt5 (gui extra)
    "numpy", "tifffile", "tensorstore", "pydantic_settings",
    "PyQt5.QtWidgets", "ndviewer_light.core",
]
failed = []
for m in MODULES:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        failed.append((m, f"{type(e).__name__}: {e}"))
        print(f"FAIL  {m}: {type(e).__name__}: {e}")
    else:
        print(f"ok    {m}")
if failed:
    print(f"\n{len(failed)} import(s) failed", file=sys.stderr)
    sys.exit(1)
print("\nall imports OK")

"""AGAVE — the Allen Institute's path-traced volume renderer — as SquidMIP's 3D view.

Everything here is Qt-free and injectable, so the whole surface is testable on a machine with
no AGAVE installed. The Qt pane that drives it lives in ``_agave_pane``.

    pane 1 button
         |
         v
    AgaveServer.start()        agave --server --port=N   (spawned on demand, killed on close)
         |
         v
    write_region_volume(...)   region mosaic -> a SMALL OME-TIFF in VolumeCache
         |                     (AGAVE reads a PATH; it cannot take a numpy array over the wire)
         v
    AgaveView(connect(port))   load_data -> set_voxel_scale -> frame_scene -> orbit
         |
         v
    render_frame(...)          IMAGE BYTES over the websocket, not a file per frame

WHY A DERIVED FILE AT ALL. AGAVE's server loads OME-TIFF/OME-Zarr from a path it can see. A raw
Squid acquisition is thousands of per-plane TIFFs with the mosaic geometry living in
``coordinates.csv``, so there is no single path that means "this region's volume". Fusing one is
therefore not an optimisation, it is the only way in. The acquisition is never touched: the fused
volume is a GENERATED file in a capped cache under the OS temp dir, and the cache is cleared when
the tab closes.

WHY A COARSE LEVEL. MEASURED on the owner's 10x set (region ``manual0``, 4 channels, 10 z):
at the native 0.752 um pixel the region's volume is ~2.2 GB, which is minutes of write and
gigabytes of cache. Decimated to ``DEFAULT_MAX_PX`` = 1000 px on the long edge it is step 12,
10x4x956x799, **61 MB**, **5-7 s to fuse**, and it path-traces a 900x700 frame in **0.13-0.18 s
at 64 iterations** (0.26 s at 96). Returning to a region already fused is a cache hit at ~1.0 s.
The renderer is interactive at the coarse level and unusable at the fine one, so coarse is the
default and ``max_px`` is the dial.

WHY set_voxel_scale IS NOT OPTIONAL. Our z step is 1.5 um against 0.752 um pixels, and the
decimation makes xy coarser still (9.02 um at step 12). Handing AGAVE the array without the voxel
size renders the volume with the wrong aspect entirely. The related failure is turning it edge-on,
which is what made Julio's image "look like a 1D array" in napari twice; ``show_volume`` frames
the scene and orbits OFF AXIS, and ``AgaveView.orbit`` clamps so it can never be turned flat.

NO SILENT FAILURES. Every refusal in this module is a named exception:
``AgaveUnavailable`` (not installed / bad signature / server would not start),
``AgaveVolumeError`` (the volume could not be written, and why),
``AgaveRenderError`` (the socket gave us nothing). The pane prints them; it never shows an
empty frame instead.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

# stdlib logging, so everything here shows up in the app's log panel for free (_logpane sources
# it from the root logger). The in-pane message is still the primary channel — the log is the
# record, not a substitute for saying it where the user is looking.
log = logging.getLogger(__name__)


class AgaveUnavailable(RuntimeError):
    """AGAVE cannot be reached: not installed, not signed, or the server would not start."""


class AgaveVolumeError(RuntimeError):
    """The region's volume could not be produced for AGAVE to read."""


class AgaveRenderError(RuntimeError):
    """AGAVE was asked for a frame and did not deliver one."""


#: Environment override — point SquidMIP at a specific AGAVE build.
ENV_VAR = "SQUIDMIP_AGAVE"

#: The AGAVE server's own default port. The pane uses its own to avoid fighting a hand-started one.
DEFAULT_PORT = 1235

#: Default long-edge budget for the fused volume. 1000 px is the measured interactive point.
DEFAULT_MAX_PX = 1000

#: One region's volume may not exceed this. Level 0 of the 10x set is ~2.2 GB; that is the thing
#: this number exists to refuse, loudly, from geometry alone.
DEFAULT_VOLUME_BUDGET = 512 * 1024 ** 2

#: Total cache cap. Generated volumes only; pruned oldest-first, and cleared on tab close.
DEFAULT_CACHE_CAP = 1536 * 1024 ** 2


def _mac_candidates() -> list[str]:
    return ["/Applications/agave.app/Contents/MacOS/agave",
            os.path.expanduser("~/Applications/agave.app/Contents/MacOS/agave")]


def _candidates() -> list[str]:
    if sys.platform == "darwin":
        return _mac_candidates()
    if sys.platform == "win32":
        return [r"C:\Program Files\AGAVE\agave-install\agave.exe"]
    return ["/usr/local/bin/agave", "/usr/bin/agave", "/opt/agave/agave",
            os.path.expanduser("~/.local/bin/agave")]


def find_agave(env: Optional[dict] = None,
               exists: Optional[Callable[[str], bool]] = None,
               which: Optional[Callable[[str], Optional[str]]] = None) -> Optional[str]:
    """The AGAVE executable to use, or None. ``SQUIDMIP_AGAVE`` wins, then PATH, then the
    platform's standard install location. Pure lookup — no signature check, no launch."""
    env = os.environ if env is None else env
    exists = os.path.isfile if exists is None else exists
    override = (env.get(ENV_VAR) or "").strip()
    if override:
        return override if exists(override) else None
    if which is not None:
        on_path = which("agave")
        if on_path:
            return on_path
    for path in _candidates():
        if exists(path):
            return path
    return None


def _bundle_of(exe: str) -> str:
    """The .app bundle containing *exe*, or *exe* itself. ``codesign`` verifies bundles."""
    p = str(exe)
    marker = ".app/"
    i = p.find(marker)
    return p[: i + len(".app")] if i >= 0 else p


def signature_problem(exe: str, run: Optional[Callable] = None) -> Optional[str]:
    """A sentence describing why macOS will refuse to run *exe*, or None when it is fine.

    This exists because of a MEASURED failure: an AGAVE bundle whose signature was invalidated
    logs its init lines and then exits silently. That looks exactly like "AGAVE is broken", so
    the signature is checked BEFORE the launch and reported as itself.
    """
    if sys.platform != "darwin":
        return None
    run = subprocess.run if run is None else run
    try:
        res = run(["codesign", "-v", "--verify", _bundle_of(exe)],
                  capture_output=True, text=True, timeout=30)
    except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
        return f"could not run codesign on {exe}: {type(exc).__name__}: {exc}"
    if int(getattr(res, "returncode", 0) or 0) == 0:
        return None
    detail = (getattr(res, "stderr", "") or getattr(res, "stdout", "") or "").strip()
    return (f"the AGAVE bundle at {_bundle_of(exe)} has a broken code signature "
            f"({detail or 'codesign refused it'}). macOS kills it a moment after launch, which "
            f"looks like a crash but is not. Repair it with:\n"
            f"    codesign --force --deep --sign - {_bundle_of(exe)}")


def require_agave(env: Optional[dict] = None,
                  exists: Optional[Callable[[str], bool]] = None,
                  run: Optional[Callable] = None,
                  which: Optional[Callable] = None) -> str:
    """The usable AGAVE executable, or raise :class:`AgaveUnavailable` saying exactly why."""
    exe = find_agave(env=env, exists=exists, which=which)
    if not exe:
        looked = "\n".join(f"    {p}" for p in _candidates())
        raise AgaveUnavailable(
            "AGAVE is not installed on this machine, so the 3D view cannot open.\n"
            f"Looked in:\n{looked}\n"
            f"Install it from https://www.allencell.org/pathtrace-rendering.html, or set "
            f"{ENV_VAR}=/path/to/agave to point SquidMIP at a build you already have."
        )
    problem = signature_problem(exe, run=run)
    if problem:
        raise AgaveUnavailable(problem)
    return exe


# --- the server process -----------------------------------------------------------------------

#: Environment variables that must NOT be inherited by the AGAVE process. MEASURED: SquidMIP's
#: own test/headless setting ``QT_QPA_PLATFORM=offscreen`` makes AGAVE's BUNDLED Qt abort with
#: SIGABRT a moment after launch — which is indistinguishable, from the outside, from the broken
#: code signature. AGAVE already renders to its own offscreen GL surface in server mode, so it
#: needs none of the host app's Qt configuration, and inheriting SquidMIP's plugin/library paths
#: would point its Qt at OUR Qt's plugins.
_SCRUBBED_ENV_PREFIXES = ("QT_",)
_SCRUBBED_ENV = ("DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "LD_LIBRARY_PATH")


def server_env(base: Optional[dict] = None) -> dict:
    """The environment AGAVE is launched with: ours, minus everything that would break its Qt."""
    base = os.environ if base is None else base
    return {k: v for k, v in base.items()
            if not k.startswith(_SCRUBBED_ENV_PREFIXES) and k not in _SCRUBBED_ENV}


def port_in_use(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    """True when something is already listening on *port* — i.e. an AGAVE server is up."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


class AgaveServer:
    """AGAVE in ``--server`` mode: started on demand, killed on close, never orphaned.

    ADOPTION is deliberate. If something is already listening on the port we would use, that is
    an AGAVE the user (or a previous tab) started; we connect to it and record ``adopted``, and
    ``stop()`` then leaves it alone. Spawning a second server on a taken port would fail, and
    killing someone else's on the way out would be worse.
    """

    def __init__(self, exe: str, port: int = DEFAULT_PORT, *,
                 spawn: Optional[Callable] = None,
                 sleep: Optional[Callable[[float], None]] = None,
                 probe: Optional[Callable[[int], bool]] = None,
                 env: Optional[dict] = None) -> None:
        self.exe = str(exe)
        self.port = int(port)
        self.process = None
        self.adopted = False
        self._running = False
        self._spawn = subprocess.Popen if spawn is None else spawn
        self._sleep = time.sleep if sleep is None else sleep
        self._probe = port_in_use if probe is None else probe
        self.env = server_env(env)

    @property
    def running(self) -> bool:
        return self._running

    def start(self, timeout_s: float = 20.0) -> None:
        if self._running:
            return
        if self._probe(self.port):
            self.adopted = True
            self._running = True
            log.info("adopted an AGAVE server already listening on port %d", self.port)
            return
        try:
            self.process = self._spawn(
                [self.exe, "--server", f"--port={self.port}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=self.env)
        except OSError as exc:
            raise AgaveUnavailable(
                f"could not launch AGAVE at {self.exe}: {type(exc).__name__}: {exc}") from exc

        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            rc = self.process.poll()
            if rc is not None:
                self.process = None
                raise AgaveUnavailable(
                    f"AGAVE exited during startup (exit code {rc}) instead of serving on port "
                    f"{self.port}. On macOS this is almost always the code signature — run "
                    f"`codesign -v --verify {_bundle_of(self.exe)}` to confirm."
                )
            if self._probe(self.port):
                self._running = True
                log.info("AGAVE server listening on port %d (pid %s)", self.port,
                         getattr(self.process, "pid", "?"))
                return
            self._sleep(0.2)

        self._terminate()
        raise AgaveUnavailable(
            f"AGAVE did not start listening on port {self.port} within {timeout_s:.0f}s. "
            f"Another process may hold the port, or the bundle at {self.exe} may not run here."
        )

    def stop(self) -> None:
        """Idempotent. Tab close AND window close both call this; neither may double-kill."""
        self._running = False
        if self.adopted:
            self.adopted = False
            self.process = None
            return
        self._terminate()

    def _terminate(self) -> None:
        proc = self.process
        if proc is not None:
            log.info("stopping the AGAVE server on port %d", self.port)
        self.process = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:                 # noqa: BLE001 - fall through to kill
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception:                         # noqa: BLE001 - teardown must not raise
            pass


def connect(port: int = DEFAULT_PORT, url: Optional[str] = None):
    """An ``agave_pyclient.AgaveRenderer`` bound to an ALREADY-RUNNING server.

    ``auto_launch=False`` on purpose: :class:`AgaveServer` owns the process, and a client that
    launches its own would be a second lifecycle nobody kills.
    """
    try:
        from agave_pyclient import AgaveRenderer
    except Exception as exc:                      # noqa: BLE001 - named, never swallowed
        raise AgaveUnavailable(
            f"the agave-pyclient package is not importable ({type(exc).__name__}: {exc}); "
            "install it with `pip install agave-pyclient`."
        ) from exc
    url = url or f"ws://localhost:{int(port)}/"
    try:
        return AgaveRenderer(url, auto_launch=False)
    except Exception as exc:                      # noqa: BLE001
        raise AgaveUnavailable(
            f"could not connect to the AGAVE server at {url}: {type(exc).__name__}: {exc}"
        ) from exc


# --- the volume ---------------------------------------------------------------------------------

def _channel_names(meta: dict) -> list[str]:
    return [c["name"] for c in (meta.get("channels") or [])]


def write_region_volume(reader: Any, meta: dict, region: str, out_path,
                        *, channels: Optional[Sequence[str]] = None,
                        t: int = 0,
                        max_px: int = DEFAULT_MAX_PX,
                        budget_bytes: int = DEFAULT_VOLUME_BUDGET) -> dict:
    """Fuse one REGION's z-stack into a small OME-TIFF AGAVE can open, and describe it.

    The unit is a REGION (a mosaic of FOVs), never a single FOV — that is the unit of navigation
    in this app. Returns ``{path, shape, bytes, seconds, step, voxel_um, channel_names, nz}``.

    ``voxel_um`` is the whole reason this returns anything: the caller MUST pass it to
    ``set_voxel_scale`` or the volume renders as a squashed slab (see the module docstring).
    """
    from squidmip._mosaic_source import _planned_plane, fuse_region_mosaic

    out_path = Path(out_path)
    region = str(region)
    if region not in set(meta.get("regions") or []):
        raise AgaveVolumeError(
            f"region {region!r} is not in this acquisition "
            f"(known regions: {', '.join(map(str, meta.get('regions') or [])) or 'none'})."
        )
    names = list(channels) if channels else _channel_names(meta)
    if not names:
        raise AgaveVolumeError(f"{region}: the acquisition declares no channels to render.")

    planned = _planned_plane(meta, region, int(max_px))
    if planned is None:
        raise AgaveVolumeError(
            f"{region}: no stage positions / pixel size in this acquisition, so its FOVs cannot "
            "be placed into a mosaic. A volume without positions would be a wrong picture, not "
            "a rough one, so none is written."
        )
    h, w, step, dtype = planned
    nz = max(1, int(meta.get("n_z") or 1))
    nc = len(names)

    # Budget check BEFORE the allocation it guards — refusing after reading 2.2 GB is not refusing.
    need = int(h) * int(w) * nz * nc * int(dtype.itemsize)
    if need > int(budget_bytes):
        raise AgaveVolumeError(
            f"{region}: the fused volume would be {need / 1e9:.2f} GB "
            f"({nz}x{nc}x{h}x{w} {dtype}), over the {budget_bytes / 1e9:.2f} GB volume budget. "
            "Render a coarser level (lower max_px) rather than writing this to disk."
        )

    px = float(meta.get("pixel_size_um") or 1.0) * float(step)
    dz = float(meta.get("dz_um") or px)

    t0 = time.perf_counter()
    vol = np.zeros((nz, nc, int(h), int(w)), dtype=dtype)
    for ci, ch in enumerate(names):
        for z in range(nz):
            got = fuse_region_mosaic(reader, meta, region, ch, z=z, t=int(t), max_px=int(max_px))
            if got is None:
                raise AgaveVolumeError(
                    f"{region}/{ch}: the mosaic could not be fused at z={z}.")
            plane = np.asarray(got[0])
            hh, ww = min(plane.shape[0], int(h)), min(plane.shape[1], int(w))
            vol[z, ci, :hh, :ww] = plane[:hh, :ww]

    import tifffile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(out_path), vol, ome=True, photometric="minisblack",
        metadata={"axes": "ZCYX", "Channel": {"Name": names},
                  "PhysicalSizeX": px, "PhysicalSizeXUnit": "um",
                  "PhysicalSizeY": px, "PhysicalSizeYUnit": "um",
                  "PhysicalSizeZ": dz, "PhysicalSizeZUnit": "um"},
    )
    log.info("%s: fused a %s volume (%.0f MB, step %g) in %.1fs -> %s",
             region, (nz, nc, int(h), int(w)), out_path.stat().st_size / 1e6, step,
             time.perf_counter() - t0, out_path)
    return {
        "path": str(out_path),
        "shape": (nz, nc, int(h), int(w)),
        "bytes": out_path.stat().st_size,
        "seconds": time.perf_counter() - t0,
        "step": float(step),
        "voxel_um": (px, px, dz),
        "channel_names": names,
        "nz": nz,
    }


def volume_key(acq_path, region: str, t: int = 0, max_px: int = DEFAULT_MAX_PX) -> str:
    """A stable cache name for one (acquisition, region, timepoint, resolution) volume."""
    raw = f"{Path(str(acq_path)).resolve()}|{region}|t{int(t)}|p{int(max_px)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class VolumeCache:
    """Generated volumes on disk, capped and disposable.

    Disk on this machine has hit zero before, so the cap is enforced after every write and the
    whole directory is removed by ``clear()`` when the 3D tab closes. A write that raises leaves
    NOTHING behind: the writer is handed a temp path inside the cache and only renamed into
    place on success, so a half-written OME-TIFF can never be served as a cache hit.
    """

    def __init__(self, root=None, cap_bytes: int = DEFAULT_CACHE_CAP) -> None:
        self.root = Path(root) if root is not None else \
            Path(tempfile.gettempdir()) / "squidmip-agave-volumes"
        self.cap_bytes = int(cap_bytes)
        self._seq: dict[str, int] = {}
        self._n = 0

    def path(self, key: str) -> Path:
        return self.root / f"{key}.ome.tif"

    def ensure(self, key: str, write: Callable[[Path], Any]) -> Path:
        final = self.path(key)
        if final.exists():
            self._n += 1
            self._seq[final.name] = self._n
            return final
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f".{key}.partial"
        try:
            write(tmp)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(final)
        self._n += 1
        self._seq[final.name] = self._n
        self.prune()
        return final

    def total_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(f.stat().st_size for f in self.root.iterdir() if f.is_file())

    def prune(self) -> list[Path]:
        """Delete oldest-first until the cache is under its cap. Returns what it removed."""
        if not self.root.exists():
            return []
        files = [f for f in self.root.iterdir() if f.is_file()]
        files.sort(key=lambda f: (self._seq.get(f.name, 0), f.stat().st_mtime_ns))
        total = sum(f.stat().st_size for f in files)
        removed: list = []
        for f in files:
            if total <= self.cap_bytes:
                break
            total -= f.stat().st_size
            try:
                f.unlink()
                removed.append(f)
                log.info("pruned a cached AGAVE volume over the %d MB cap: %s",
                         self.cap_bytes // (1024 ** 2), f.name)
            except OSError:
                pass
        return removed

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self._seq.clear()


# --- rendering ------------------------------------------------------------------------------

def render_frame(client, *, width: int, height: int, iterations: int) -> bytes:
    """One path-traced frame, as ENCODED IMAGE BYTES.

    ``AgaveRenderer.redraw()`` decodes the frame with PIL and SAVES it to ``session_name``. For a
    live pane that is a disk round-trip per frame, so this reproduces redraw's protocol — flush
    the command buffer, wait for the binary websocket frame — and hands back the bytes. Same
    seam, one less filesystem.

    NOT necessarily PNG. Measured against AGAVE 1.10.0: the server streams **JPEG**
    (``ff d8 ff e0``). redraw() hides that behind PIL, so nothing upstream says so. Callers must
    let their decoder sniff the format — a hard-coded "PNG" hint silently fails to decode every
    frame, which is how this became a named test instead of a demo-day surprise.
    """
    client.set_resolution(int(width), int(height))
    client.render_iterations(int(iterations))
    client.cb.add_command("REDRAW")
    buf = client.cb.make_buffer()
    client.cb = type(client.cb)()
    client.ws.send(buf, True)
    got = client.ws.wait_for_image()
    if got is None:
        raise AgaveRenderError(
            "AGAVE returned no image for the frame — the render socket closed. The server "
            "process has probably died; close the 3D tab and reopen it."
        )
    data = got.getvalue() if hasattr(got, "getvalue") else bytes(got)
    if not data:
        raise AgaveRenderError("AGAVE returned an empty (zero-byte) frame.")
    return data


def _rgb(channel_name: str) -> tuple[float, float, float]:
    """Squid's own channel colour, 0..1. ``_channels`` owns the palette; this does not restate it."""
    try:
        from squidmip._channels import fallback_color

        hexc = fallback_color(channel_name)
    except Exception:                             # noqa: BLE001
        hexc = None
    if not hexc:
        return (1.0, 1.0, 1.0)
    h = hexc.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))   # type: ignore[return-value]


#: The VOLUME OPACITY transfer function AGAVE opens with, as (low, high) percentiles in 0..1.
#:
#: This is deliberately NOT napari's display window (``_viewer._PCT`` = 1%..99.8%). Those two
#: numbers look interchangeable and are not: napari's window maps intensity to BRIGHTNESS on one
#: 2-D plane, AGAVE's maps intensity to OPACITY through a whole volume. Seeding AGAVE with 1%
#: makes 99% of the voxels — background included — opaque, and the render comes out as a flat
#: yellow haze with no architecture in it. MEASURED on the owner's 10x set: mean luma 9.5 and no
#: visible tissue at (0.01, 0.998) versus a clean black background and legible cortex at
#: (0.5, 0.98). So napari still OWNS 2-D contrast and this does not compete with it; AGAVE owns
#: its own volume transfer function, which is a different quantity.
OPENING_THRESHOLD = (0.5, 0.98)

#: Opening density/exposure, chosen against the same renders. Brighter than AGAVE's defaults
#: because a 10-plane pancake has little material to accumulate through.
OPENING_DENSITY = 150.0
OPENING_EXPOSURE = 0.9

#: How far off axis the opening view sits. napari's 3D default is EDGE-ON, which is what made
#: Julio's pancake volumes "look like a 1D array" twice. AGAVE opens turned.
OPENING_ORBIT = (35.0, -25.0)

#: How far from the framed pose the user may turn the volume on either axis. See AgaveView.orbit
#: for the measured sweep this comes from.
ORBIT_LIMIT = 70.0


def _clamp(v: float, limit: float) -> float:
    return max(-limit, min(limit, float(v)))


class AgaveView:
    """The scene the pane drives: one loaded volume, a camera, and frames on demand.

    Deliberately thin. It owns the ORDER of the AGAVE calls (load, then scale, then colour, then
    frame+orbit) because getting that order wrong is what produces a squashed or edge-on volume,
    and it owns nothing else — no contrast rule of its own (see ``show_volume``), no threading,
    no widgets.
    """

    def __init__(self, client) -> None:
        self.client = client
        self.channels: list[str] = []
        self.path: Optional[str] = None
        self.voxel_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._theta, self._phi = 0.0, 0.0        # cumulative orbit from the framed pose

    def show_volume(self, path: str, voxel_um, channels: Sequence[str],
                    *, t: int = 0, density: float = OPENING_DENSITY,
                    exposure: float = OPENING_EXPOSURE):
        """Load *path* and set up a view of it that is worth looking at on the first frame."""
        self.path = str(path)
        self.channels = list(channels)
        self.voxel_um = tuple(float(v) for v in voxel_um)   # type: ignore[assignment]
        info = self.client.load_data_and_get_info(self.path, 0, 0, int(t), [], [])
        # AGAVE emits ONE image per command buffer. load_data_and_get_info flushed a buffer and
        # read only the JSON, so that buffer's image is still queued. Left there it desynchronises
        # the stream permanently: every later frame() returns the PREVIOUS render, so the pane
        # shows region B2 while the slider says B3. Consume it here. (Measured on AGAVE 1.10.0 —
        # ask for 160x120 and you get back the 900x700 frame you asked for one call ago.)
        self.client.ws.wait_for_image()

        vx, vy, vz = self.voxel_um
        self.client.set_voxel_scale(vx, vy, vz)   # REQUIRED: anisotropic z would render as a slab
        self.client.background_color(0.0, 0.0, 0.0)
        for i, name in enumerate(self.channels):
            r, g, b = _rgb(name)
            self.client.enable_channel(i, 1)
            self.client.mat_diffuse(i, r, g, b, 1.0)
            self.client.mat_opacity(i, 1.0)
            # AGAVE's own percentile threshold — the VOLUME opacity transfer function. See
            # OPENING_THRESHOLD for why this is not napari's display window.
            self.client.set_percentile_threshold(i, OPENING_THRESHOLD[0], OPENING_THRESHOLD[1])
        self.client.density(float(density))
        self.client.exposure(float(exposure))
        self.client.frame_scene()
        self._theta, self._phi = OPENING_ORBIT   # the framed pose is the origin of the clamp
        self.client.orbit_camera(*OPENING_ORBIT)
        return info

    def set_time(self, t: int) -> None:
        """The LAZY timepoint slider: no reload, no refuse — the volume already holds every t."""
        self.client.set_time(int(t))

    def orbit(self, theta: float, phi: float) -> None:
        """Turn the volume, CLAMPED so it can never be turned fully edge-on.

        Our volumes are pancakes: 10 planes of 1.5 um against a mosaic ~8600 um wide. MEASURED by
        rendering the sweep and counting lit pixels, the visible area peaks around 45 degrees off
        face-on and collapses to nothing by 90:

            theta   0    15    30    45    60    75    90
            lit    .105  .109  .131  .147  .132  .080  .007     (same curve on phi)

        At 90 degrees there is no picture left — which is exactly the failure Julio has reported
        twice about napari's edge-on 3D default ("my image ends up looking like a 1D array"). So
        the orbit is free inside +/-ORBIT_LIMIT of the framed pose and simply stops there, rather
        than letting a drag spin the tissue into a hairline.
        """
        t = _clamp(self._theta + float(theta), ORBIT_LIMIT)
        p = _clamp(self._phi + float(phi), ORBIT_LIMIT)
        dt, dp = t - self._theta, p - self._phi
        if dt == 0.0 and dp == 0.0:
            return                                # already at the stop: send nothing
        self._theta, self._phi = t, p
        self.client.orbit_camera(dt, dp)

    def frame(self, width: int, height: int, iterations: int = 64) -> bytes:
        return render_frame(self.client, width=width, height=height, iterations=iterations)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:                         # noqa: BLE001 - teardown must not raise
            pass


class AgaveEngine:
    """Server + volume cache + view as ONE object, so the Qt side owns no AGAVE state.

    Every collaborator is injected (``require`` / ``server_factory`` / ``connect``), which is why
    the whole thing is exercised on machines with no AGAVE installed. The pane's worker thread
    calls these methods and nothing else; that keeps the websocket, the AGAVE process and every
    blocking read on ONE thread that is joined at teardown.

    ``close()`` is the only teardown path and it is idempotent: the tab close, the window close
    and a failed open all route through it, and each must be able to run after the others.
    """

    def __init__(self, reader, meta: dict, acq_path, *,
                 cache: Optional[VolumeCache] = None,
                 port: int = DEFAULT_PORT,
                 max_px: int = DEFAULT_MAX_PX,
                 require: Optional[Callable[[], str]] = None,
                 server_factory: Optional[Callable[[str, int], Any]] = None,
                 connect: Optional[Callable[[int], Any]] = None) -> None:
        self.reader = reader
        self.meta = meta
        self.acq_path = str(acq_path)
        self.cache = cache if cache is not None else VolumeCache()
        self.port = int(port)
        self.max_px = int(max_px)
        self._require = require if require is not None else (lambda: require_agave())
        self._server_factory = server_factory if server_factory is not None else \
            (lambda exe, port: AgaveServer(exe, port=port))
        # `connect` the PARAMETER shadows `connect` the module function, so the default is bound
        # to the module-level name explicitly rather than through the shadowed one.
        _real_connect = globals()["connect"]
        self._connect = connect if connect is not None else (lambda port: _real_connect(port=port))
        self.server = None
        self.view: Optional[AgaveView] = None
        self.region: Optional[str] = None
        self.info: Optional[dict] = None

    # -- what the sliders are allowed to offer ---------------------------------------------
    @property
    def regions(self) -> list[str]:
        """The unit of navigation is the REGION (a mosaic of FOVs), never a single FOV."""
        return [str(r) for r in (self.meta.get("regions") or [])]

    @property
    def n_timepoints(self) -> int:
        return max(1, int(self.meta.get("n_t") or 1))

    # -- lifecycle -------------------------------------------------------------------------
    def open(self) -> None:
        """Locate AGAVE, start its server, connect. Raises :class:`AgaveUnavailable` by name."""
        if self.view is not None:
            return
        exe = self._require()                     # raises before anything is spawned
        self.server = self._server_factory(exe, self.port)
        self.server.start()
        try:
            self.view = AgaveView(self._connect(self.port))
        except Exception:
            self.server.stop()
            self.server = None
            raise

    def close(self) -> None:
        view, server = self.view, self.server
        self.view, self.server, self.info, self.region = None, None, None, None
        if view is not None:
            view.close()
        if server is not None:
            server.stop()
        self.cache.clear()                        # generated volumes do not outlive the tab

    # -- navigation ------------------------------------------------------------------------
    def show_region(self, region: str, t: int = 0) -> dict:
        """LAZY region navigation: fuse (or reuse) that region's volume and load it into AGAVE."""
        if self.view is None:
            raise AgaveRenderError("the 3D view is not open, so no region can be shown.")
        key = volume_key(self.acq_path, region, t=t, max_px=self.max_px)
        target = self.cache.path(key)
        cached = target.exists()
        holder: dict = {}

        def _write(tmp: Path):
            holder["info"] = write_region_volume(
                self.reader, self.meta, region, tmp, t=t, max_px=self.max_px)

        path = self.cache.ensure(key, _write)
        info = holder.get("info")
        if info is None:                          # cache hit: re-derive the geometry, read nothing
            info = self._describe(region, path)
        info = dict(info, path=str(path), cached=cached, region=region, t=int(t))
        self.view.show_volume(str(path), info["voxel_um"], info["channel_names"], t=int(t))
        self.region, self.info = region, info
        return info

    def _describe(self, region: str, path: Path) -> dict:
        """Geometry for a volume already on disk — the same numbers the writer would report."""
        from squidmip._mosaic_source import _planned_plane

        planned = _planned_plane(self.meta, region, self.max_px)
        if planned is None:
            raise AgaveVolumeError(f"{region}: cannot describe the cached volume (no positions).")
        h, w, step, dtype = planned
        px = float(self.meta.get("pixel_size_um") or 1.0) * float(step)
        dz = float(self.meta.get("dz_um") or px)
        names = _channel_names(self.meta)
        nz = max(1, int(self.meta.get("n_z") or 1))
        return {"path": str(path), "shape": (nz, len(names), int(h), int(w)),
                "bytes": path.stat().st_size, "seconds": 0.0, "step": float(step),
                "voxel_um": (px, px, dz), "channel_names": names, "nz": nz}

    def set_time(self, t: int) -> None:
        """The LAZY timepoint slider — AGAVE already holds every t in the loaded volume."""
        if self.view is None:
            raise AgaveRenderError("the 3D view is not open, so the timepoint cannot change.")
        self.view.set_time(int(t))

    def orbit(self, theta: float, phi: float) -> None:
        if self.view is not None:
            self.view.orbit(theta, phi)

    def frame(self, width: int, height: int, iterations: int = 64) -> bytes:
        if self.view is None or self.info is None:
            raise AgaveRenderError(
                "no volume is loaded yet, so there is no frame to render — pick a region first.")
        return self.view.frame(width, height, iterations)

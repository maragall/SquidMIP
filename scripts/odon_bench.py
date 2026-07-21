"""IMA-234 Odon remote benchmark — instrumented static server for a plate.ome.zarr.

WHAT THIS IS. A ``ThreadingHTTPServer`` that serves an OME-zarr plate and records every
request, so the cost of viewing a plate over HTTP can be measured from the *server* side.
Odon (``alexcoulton/odon``) is a compiled Rust binary with no FPS counter and no debug
log, so there is no client-side hook. But the two arms are asymmetric in a useful way:

    ARM A  odon <path>          local file reads   -> invisible without dtrace
    ARM B  odon http://host/... every chunk read   -> an HTTP request THIS server logs

That asymmetry is the whole design. The server is not just transport, it is the only
instrument available. ``first_view_ms`` and ``req_per_s`` below are server-side proxies
for first-paint and framerate; they are honest proxies, not the real thing, and the
report says so.

WHAT IT IS NOT. It cannot compute a "local -> remote delta": arm A produces no
server-side numbers at all. The deliverable is a **latency curve** across arms, plus a
stopwatch reading from arm A in its own units. See .spec/open/ima-234.md.

WHY ThreadingHTTPServer. The stdlib default handles one request at a time. Odon links
``tokio`` with ``rt-multi-thread`` and fetches chunks concurrently, so a single-threaded
server would queue them and the measured cost would be this harness's own serialization
-- worse the more parallel the viewer is, which is backwards. It also breaks
``--latency-ms``: on one thread, per-request sleeps stack instead of overlapping, so
25 ms becomes 25 ms x queue depth.

WHY --latency-ms. Loopback RTT is ~0.05 ms with effectively unlimited bandwidth; a real
object store is 20-100 ms away. Serving from localhost measures HTTP framing overhead
and nothing else, which would support the wrong conclusion (that remote is nearly free).
Sweeping 0/25/100 ms turns one flattering number into a sensitivity curve. It models
latency ONLY -- not TLS handshakes, keep-alive, connection pooling, or bandwidth-delay
product -- which is why the spec also calls for one run against a real bucket.

Usage::

    python scripts/odon_bench.py --zarr out/plate.ome.zarr --latency-ms 25
    # point Odon at the printed URL, drive the same pan/zoom sequence, then Ctrl-C
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Metadata filenames across zarr generations. SquidMIP writes zarr v3 (``zarr.json``);
# the v2 names are here so a v2-speaking client's probes are classified, not dumped in
# "other" where they would look like a parsing bug.
_METADATA_NAMES = {"zarr.json", ".zarray", ".zgroup", ".zattrs"}

DEFAULT_IDLE_GAP_MS = 500.0


@dataclass
class Entry:
    """One served request. Times are ``time.monotonic()`` seconds."""

    start: float
    end: float
    path: str
    status: int
    size: int
    level: Optional[int]
    kind: str  # "chunk" | "metadata" | "other"

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0


def classify_path(path: str) -> tuple[str, Optional[int]]:
    """Map a request path to ``(kind, pyramid_level)``.

    The store served at ``/`` is the plate, so paths look like::

        /A/1/0/0/c/0/0/0/0/0   chunk   -> row A, col 1, fov 0, LEVEL 0
        /A/1/0/1/c/0/0/0/0/0   chunk   -> level 1 (a pyramid read)
        /A/1/0/0/zarr.json     metadata for that field's level-0 array
        /zarr.json             plate-group metadata

    The level bucket is the point: it answers the open TODOS.md question about whether
    any real viewport-tile engine reads IMA-184's pyramid, or only ever level 0.

    Anything that doesn't fit is ``("other", None)`` -- never an exception. A malformed
    or unexpected path must not take the server down mid-measurement.
    """
    parts = [p for p in path.split("?")[0].split("/") if p]
    kind = "metadata" if parts and parts[-1] in _METADATA_NAMES else ("chunk" if parts else "other")
    # Layout is {row}/{col}/{fov}/{level}/...; the level is the 4th segment when numeric.
    level = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else None
    if kind == "chunk" and level is None:
        kind = "other"
    return kind, level


class RequestLog:
    """Thread-safe accumulator. ``ThreadingHTTPServer`` records from many threads at once.

    Tracks in-flight count so ``peak_concurrency`` can prove the server is actually
    overlapping requests -- if that value is 1 under concurrent load, the harness is
    serializing and every number it reports is an artifact.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entries: list[Entry] = []
        self._in_flight = 0
        self.peak_concurrency = 0

    def enter(self) -> float:
        with self._lock:
            self._in_flight += 1
            self.peak_concurrency = max(self.peak_concurrency, self._in_flight)
        return time.monotonic()

    def record(self, start: float, path: str, status: int, size: int) -> None:
        kind, level = classify_path(path)
        entry = Entry(start, time.monotonic(), path, status, size, level, kind)
        with self._lock:
            self._in_flight -= 1
            self.entries.append(entry)

    def __len__(self) -> int:
        with self._lock:
            return len(self.entries)


class ChunkLogHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler that times every GET, optionally delaying it, and logs the result.

    ``latency_ms`` is applied BEFORE the response so it behaves like network delay rather
    than slow disk. Status and byte count are captured by intercepting ``send_response``
    and the ``Content-Length`` header, because ``BaseHTTPRequestHandler.log_request``
    only ever receives the code (size arrives as ``'-'``).
    """

    latency_ms: float = 0.0
    request_log: Optional[RequestLog] = None
    protocol_version = "HTTP/1.1"  # keep-alive, so per-request cost isn't a new socket

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        log = self.request_log
        start = log.enter() if log is not None else time.monotonic()
        self._status = 200
        self._size = 0
        try:
            if self.latency_ms:
                time.sleep(self.latency_ms / 1000.0)
            super().do_GET()
        finally:
            if log is not None:
                log.record(start, self.path, self._status, self._size)

    def send_response(self, code, message=None):  # noqa: D102 - stdlib override
        self._status = code
        super().send_response(code, message)

    def send_header(self, keyword, value):  # noqa: D102 - stdlib override
        if keyword.lower() == "content-length":
            try:
                self._size = int(value)
            except (TypeError, ValueError):
                self._size = 0
        super().send_header(keyword, value)

    def log_message(self, fmt, *args):  # noqa: D102 - silence stderr spam
        return


class _Server(socketserver.ThreadingTCPServer):
    """``ThreadingHTTPServer`` plus address reuse and non-daemon-thread joins."""

    daemon_threads = True
    allow_reuse_address = True


@contextmanager
def serve(root, port: int = 0, latency_ms: float = 0.0):
    """Serve *root* over HTTP for the duration of the context.

    Yields ``(url, log)``. ``port=0`` picks a free port, which keeps concurrent test
    runs from colliding.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")

    log = RequestLog()
    # Per-serve subclass: latency and the log are class attrs the handler reads, and
    # concurrent serve() calls (tests) must not share them.
    bound = type(
        "BoundChunkLogHandler",
        (ChunkLogHandler,),
        {"latency_ms": latency_ms, "request_log": log},
    )
    server = _Server(("127.0.0.1", port), functools.partial(bound, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", log
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def summarize(log: RequestLog, idle_gap_ms: float = DEFAULT_IDLE_GAP_MS) -> dict:
    """Reduce a request log to the reported metrics.

    ``first_view_ms`` is the first-paint proxy: time from the first request until the
    first quiet period of at least *idle_gap_ms*. The reasoning is that a viewport-tile
    engine asks for chunks until the view is satisfied, then stops.

    CAVEAT, stated because the number invites over-reading: this conflates "painted"
    with "stopped fetching". Viewers prefetch beyond the viewport and trickle
    low-priority tiles after the visible image is already up, so this is an upper bound
    on first paint, not first paint.

    Returns zeroed fields rather than raising when nothing was served -- an operator who
    Ctrl-Cs before launching Odon should get a readable report, not a traceback.
    """
    entries = sorted(log.entries, key=lambda e: e.start)
    out: dict = {
        "n_requests": len(entries),
        "peak_concurrency": log.peak_concurrency,
        "bytes_total": sum(e.size for e in entries),
        "by_status": dict(Counter(e.status for e in entries)),
        # SUCCESSFUL reads only. A client probing for a level that isn't there gets a
        # 404 and reads nothing; counting it here would make "does the viewer use the
        # pyramid?" answer yes on the strength of a miss.
        "by_level": dict(
            Counter(e.level for e in entries if e.level is not None and e.status < 400)
        ),
        "by_kind": dict(Counter(e.kind for e in entries)),
        "n_missing": sum(1 for e in entries if e.status == 404),
        "ttfb_ms": None,
        "first_view_ms": None,
        "req_per_s": None,
        "wall_ms": None,
        "idle_gap_ms": idle_gap_ms,
    }
    if not entries:
        return out

    t0 = entries[0].start
    out["ttfb_ms"] = (entries[0].end - t0) * 1000.0
    out["wall_ms"] = (max(e.end for e in entries) - t0) * 1000.0

    # Walk the timeline; the first gap >= idle_gap closes the "first view" window.
    gap_s = idle_gap_ms / 1000.0
    frontier = entries[0].end
    first_view_end = None
    active_s = 0.0
    span_start = entries[0].start
    for entry in entries[1:]:
        if entry.start - frontier >= gap_s:
            if first_view_end is None:
                first_view_end = frontier
            active_s += frontier - span_start
            span_start = entry.start
        frontier = max(frontier, entry.end)
    active_s += frontier - span_start
    out["first_view_ms"] = ((first_view_end or frontier) - t0) * 1000.0
    out["req_per_s"] = len(entries) / active_s if active_s > 0 else None
    return out


def format_report(summary: dict, latency_ms: float = 0.0) -> str:
    """Render a summary as markdown. Safe on the zero-request summary."""

    def num(value, suffix="", digits=1):
        return "n/a" if value is None else f"{value:.{digits}f}{suffix}"

    lines = [
        "# Odon benchmark report",
        "",
        f"- injected latency: **{latency_ms:g} ms/request**",
        f"- requests served: **{summary['n_requests']}**",
        f"- peak concurrency: **{summary['peak_concurrency']}**",
        f"- bytes served: **{summary['bytes_total']}**",
        f"- time to first byte: **{num(summary['ttfb_ms'], ' ms')}**",
        f"- first view complete: **{num(summary['first_view_ms'], ' ms')}** "
        f"(idle gap {summary['idle_gap_ms']:g} ms)",
        f"- requests/sec (active): **{num(summary['req_per_s'], '', 2)}**",
        f"- wall time: **{num(summary['wall_ms'], ' ms')}**",
        "",
    ]
    if summary["n_requests"] == 0:
        lines += [
            "**No requests were served.** Nothing was measured -- the viewer never "
            "connected. Check the URL and that Odon opened the store.",
            "",
        ]
        return "\n".join(lines)

    lines += ["| status | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(summary["by_status"].items())]
    lines += ["", "| pyramid level | requests |", "|---|---|"]
    if summary["by_level"]:
        lines += [f"| {k} | {v} |" for k, v in sorted(summary["by_level"].items())]
    else:
        lines += ["| (none parsed) | 0 |"]
    lines += [
        "",
        "Successful reads only. Levels above 0 mean the client genuinely uses the "
        "pyramid; only level 0 means it pulls full resolution regardless of zoom (see "
        "the pyramid question in TODOS.md).",
        "",
        f"Missing-key probes (404): **{summary['n_missing']}**. Free on loopback, but "
        "each one costs a full round trip at 100 ms, so a probe-heavy client degrades "
        "non-linearly with distance.",
        "",
        "| request kind | count |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in sorted(summary["by_kind"].items())]
    lines += [
        "",
        "Caveats: `first_view_ms` measures when the client stopped fetching, which is an "
        "upper bound on first paint (viewers prefetch past the viewport). A high metadata "
        "count means time-to-first-byte is dominated by a `zarr.json` crawl of many tiny "
        "GETs -- a store-layout cost, not an image-serving cost.",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="odon_bench",
        description="Serve a plate.ome.zarr over HTTP and measure what the viewer fetches.",
    )
    parser.add_argument("--zarr", required=True, help="path to plate.ome.zarr")
    parser.add_argument("--port", type=int, default=8000, help="0 picks a free port")
    parser.add_argument("--latency-ms", type=float, default=0.0, help="delay per response")
    parser.add_argument("--idle-gap-ms", type=float, default=DEFAULT_IDLE_GAP_MS)
    args = parser.parse_args(argv)

    root = Path(args.zarr)
    if not root.is_dir():
        parser.error(f"--zarr is not a directory: {root}")

    with serve(root, port=args.port, latency_ms=args.latency_ms) as (url, log):
        print(f"serving {root} at {url}  (latency {args.latency_ms:g} ms/request)")
        print("point Odon at that URL, drive the pan/zoom sequence, then Ctrl-C")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print()
        summary = summarize(log, idle_gap_ms=args.idle_gap_ms)
    print(format_report(summary, latency_ms=args.latency_ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())

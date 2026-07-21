"""IMA-234 — tests for the Odon benchmark harness.

These pass with NO odon binary installed, which is the point: the harness is the
machine-buildable half of the ticket, and the measurement is a human step afterwards.

WHAT THESE TESTS DO NOT PROVE. The end-to-end test drives the server with a Python
fetch loop, so idle-gap detection is finding the shape of the test's own traffic. That
validates the log analyzer; it carries no information about how fast Odon renders. A
green suite here is not a benchmark result. The benchmark result requires the manual arm
in .spec/open/ima-234.md.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from odon_bench import (
    Entry,
    RequestLog,
    classify_path,
    format_report,
    main,
    serve,
    summarize,
)


# --- path classification --------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/A/1/0/0/c/0/0/0/0/0", ("chunk", 0)),  # full-res chunk
        ("/A/1/0/2/c/0/0/0/0/0", ("chunk", 2)),  # a real pyramid read
        ("/AA/12/3/1/c/0/0", ("chunk", 1)),  # 1536wp double-letter row
        ("/A/1/0/0/zarr.json", ("metadata", 0)),  # array metadata carries a level
        ("/zarr.json", ("metadata", None)),  # plate group metadata
        ("/A/1/0/0/.zarray", ("metadata", 0)),  # v2-speaking client probe
        ("/A/1/0/0/c/0/0?v=1", ("chunk", 0)),  # query string ignored
        ("/favicon.ico", ("other", None)),  # unparseable -> other, never raises
        ("/", ("other", None)),
        ("/A/1/x/y/c/0", ("other", None)),  # non-numeric level
    ],
)
def test_classify_path(path, expected):
    assert classify_path(path) == expected


# --- RequestLog -----------------------------------------------------------------------


def test_request_log_is_thread_safe():
    """ThreadingHTTPServer records from many threads; a lost update is a silent bug."""
    log = RequestLog()
    n_threads, per_thread = 8, 200

    def hammer():
        for _ in range(per_thread):
            start = log.enter()
            log.record(start, "/A/1/0/0/c/0/0", 200, 10)

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(log) == n_threads * per_thread
    assert log.peak_concurrency >= 1


def test_peak_concurrency_tracks_overlap():
    log = RequestLog()
    a = log.enter()
    b = log.enter()  # two in flight at once
    log.record(a, "/A/1/0/0/c/0", 200, 1)
    log.record(b, "/A/1/0/0/c/1", 200, 1)
    assert log.peak_concurrency == 2


# --- summarize ------------------------------------------------------------------------


def test_summarize_empty_log_does_not_divide_by_zero():
    """Operator Ctrl-Cs before launching the viewer: report, don't traceback."""
    summary = summarize(RequestLog())
    assert summary["n_requests"] == 0
    assert summary["req_per_s"] is None
    assert summary["first_view_ms"] is None
    assert summary["bytes_total"] == 0
    assert "No requests were served" in format_report(summary)


def _log_from(spans):
    """Build a log from (start, end) second pairs, bypassing the wall clock."""
    log = RequestLog()
    log.entries = [
        Entry(s, e, "/A/1/0/0/c/0/0", 200, 100, 0, "chunk") for s, e in spans
    ]
    return log


def test_summarize_first_view_ends_at_the_first_idle_gap():
    # Burst until t=0.30, then 1.0s of quiet, then a second burst (a pan).
    log = _log_from([(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (1.30, 1.40), (1.40, 1.50)])
    summary = summarize(log, idle_gap_ms=500)
    assert summary["first_view_ms"] == pytest.approx(300, abs=1)
    assert summary["ttfb_ms"] == pytest.approx(100, abs=1)
    # req/s excludes the idle stretch: 5 requests over 0.5s of active time.
    assert summary["req_per_s"] == pytest.approx(10.0, rel=0.05)


def test_summarize_without_any_gap_uses_the_full_span():
    log = _log_from([(0.0, 0.10), (0.10, 0.20), (0.20, 0.30)])
    summary = summarize(log, idle_gap_ms=500)
    assert summary["first_view_ms"] == pytest.approx(300, abs=1)
    assert summary["wall_ms"] == pytest.approx(300, abs=1)


def test_summarize_buckets_status_level_and_kind():
    log = RequestLog()
    log.entries = [
        Entry(0.0, 0.1, "/A/1/0/0/c/0/0", 200, 10, 0, "chunk"),
        Entry(0.1, 0.2, "/A/1/0/1/c/0/0", 200, 20, 1, "chunk"),
        Entry(0.2, 0.3, "/A/1/0/0/zarr.json", 200, 5, 0, "metadata"),
        Entry(0.3, 0.4, "/nope", 404, 0, None, "other"),
    ]
    summary = summarize(log)
    assert summary["by_status"] == {200: 3, 404: 1}
    assert summary["by_level"] == {0: 2, 1: 1}
    assert summary["by_kind"] == {"chunk": 2, "metadata": 1, "other": 1}
    assert summary["bytes_total"] == 35
    report = format_report(summary)
    assert "404" in report and "pyramid level" in report


def test_missed_level_probe_does_not_count_as_a_pyramid_read():
    """A 404 for level 9 means the client read NOTHING at level 9.

    Counting it would answer "does the viewer use the pyramid?" with yes on the
    strength of a miss -- the exact wrong conclusion for the question the by_level
    histogram exists to settle.
    """
    log = RequestLog()
    log.entries = [
        Entry(0.0, 0.1, "/A/1/0/0/c/0/0", 200, 10, 0, "chunk"),
        Entry(0.1, 0.2, "/A/1/0/9/c/0/0", 404, 0, 9, "chunk"),
    ]
    summary = summarize(log)
    assert summary["by_level"] == {0: 1}, "404 probe leaked into the pyramid histogram"
    assert summary["n_missing"] == 1


# --- serving --------------------------------------------------------------------------


@pytest.fixture
def plate(tmp_path, squid_dataset):
    """A real plate.ome.zarr written by the shipping writer, not a hand-made stand-in.

    Reuses tests/conftest.py::squid_dataset so the harness is exercised against the
    actual chunk layout and pyramid structure SquidMIP emits.
    """
    from squidmip import open_reader, write_plate

    root, _ = squid_dataset
    out = tmp_path / "out"
    write_plate(open_reader(root), out, n_fovs=1)
    plate_dir = out / "plate.ome.zarr"
    assert plate_dir.is_dir()
    return plate_dir


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.status, resp.read()


def test_serve_logs_real_chunk_requests(plate):
    with serve(plate, port=0) as (url, log):
        status, body = _get(f"{url}/zarr.json")
        assert status == 200 and body

    summary = summarize(log)
    assert summary["n_requests"] == 1
    assert summary["by_kind"]["metadata"] == 1
    assert summary["bytes_total"] > 0
    assert summary["ttfb_ms"] is not None


def test_serve_records_404_separately_and_keeps_serving(plate):
    """Viewers probe for optional keys. A miss must be counted, not fatal."""
    with serve(plate, port=0) as (url, log):
        with pytest.raises(urllib.error.HTTPError):
            _get(f"{url}/definitely/not/here.json")
        status, _ = _get(f"{url}/zarr.json")  # server still alive afterwards
        assert status == 200

    summary = summarize(log)
    assert summary["by_status"][404] == 1
    assert summary["by_status"][200] == 1


def test_serve_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        with serve("/no/such/plate.ome.zarr", port=0):
            pass


def test_latency_injection_delays_a_single_request(plate):
    with serve(plate, port=0, latency_ms=50) as (url, _):
        t0 = time.monotonic()
        _get(f"{url}/zarr.json")
        elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms >= 45  # allow a little clock slop below the 50ms floor


def test_concurrent_requests_overlap_rather_than_serialize(plate):
    """CRITICAL — guards Decisions 3 and 4 together.

    Two 50 ms-delayed requests issued at once must finish in well under 100 ms. If they
    take ~100 ms the server is serializing, injected latency is stacking instead of
    overlapping, and every number this harness produces is a harness artifact rather
    than a transport measurement.

    The bound is deliberately loose (85 ms, not 55 ms) per the repo's existing timing
    convention: the failure being caught DOUBLES the time, so a wide margin still
    catches it while surviving a loaded machine. A tight bound would just flake.
    """
    with serve(plate, port=0, latency_ms=50) as (url, log):
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_get, f"{url}/zarr.json") for _ in range(2)]
            for f in futures:
                assert f.result()[0] == 200
        elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < 85, f"requests serialized ({elapsed_ms:.0f} ms for 2x50 ms)"
    assert log.peak_concurrency == 2


# --- CLI ------------------------------------------------------------------------------


def test_main_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_main_requires_zarr():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_main_rejects_a_bad_zarr_path(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--zarr", str(tmp_path / "missing.ome.zarr")])
    assert exc.value.code != 0


def test_main_serves_then_reports_on_interrupt(plate, monkeypatch, capsys):
    """The operator flow: serve -> URL printed -> Ctrl-C -> report still prints."""
    sleeps = {"n": 0}

    def fake_sleep(_seconds):
        sleeps["n"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr("odon_bench.time.sleep", fake_sleep)
    assert main(["--zarr", str(plate), "--port", "0"]) == 0

    out = capsys.readouterr().out
    assert "serving" in out
    assert "Odon benchmark report" in out
    assert sleeps["n"] == 1

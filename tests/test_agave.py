"""AGAVE embedding — the NON-Qt half: executable discovery, server lifecycle, the volume
the renderer is fed, the disk cache that holds it, and the render seam.

Every one of these runs on a machine with NO AGAVE installed: the executable, the process and
the websocket client are all injected. The single test that talks to a REAL AGAVE server is
marked and skips with a named reason (see ``test_real_agave_*`` at the bottom).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from squidmip import _agave


# --- executable discovery: a MISSING agave must be a NAMED refusal, never a silent empty pane ---

def test_find_agave_prefers_the_explicit_env_override():
    seen = {}

    def exists(p):
        seen[p] = True
        return p == "/opt/custom/agave"

    got = _agave.find_agave(env={"SQUIDMIP_AGAVE": "/opt/custom/agave"}, exists=exists)
    assert got == "/opt/custom/agave"


def test_find_agave_falls_back_to_the_standard_mac_bundle():
    std = "/Applications/agave.app/Contents/MacOS/agave"
    got = _agave.find_agave(env={}, exists=lambda p: p == std)
    assert got == std


def test_find_agave_returns_none_when_nothing_is_installed():
    assert _agave.find_agave(env={}, exists=lambda p: False) is None


def test_require_agave_raises_agave_unavailable_naming_the_places_it_looked():
    with pytest.raises(_agave.AgaveUnavailable) as e:
        _agave.require_agave(env={}, exists=lambda p: False)
    msg = str(e.value)
    assert "AGAVE" in msg
    assert "/Applications/agave.app" in msg          # the place it looked, by name
    assert "SQUIDMIP_AGAVE" in msg                   # and how to point it elsewhere


def test_require_agave_refuses_a_broken_signature_by_name():
    """The measured failure mode: an unsigned/broken bundle logs init and then dies silently.
    That must NOT reach the user as 'AGAVE is broken' — it must name the signature."""
    exe = "/Applications/agave.app/Contents/MacOS/agave"

    def run(cmd, **kw):
        assert cmd[0] == "codesign"
        return _FakeProc(returncode=1, stderr="code object is not signed at all")

    with pytest.raises(_agave.AgaveUnavailable) as e:
        _agave.require_agave(env={}, exists=lambda p: p == exe, run=run)
    msg = str(e.value)
    assert "signature" in msg.lower()
    assert "codesign --force --deep --sign -" in msg     # the exact repair, spelled out


def test_require_agave_accepts_an_adhoc_signed_bundle():
    exe = "/Applications/agave.app/Contents/MacOS/agave"
    run = lambda cmd, **kw: _FakeProc(returncode=0, stderr="valid on disk")
    assert _agave.require_agave(env={}, exists=lambda p: p == exe, run=run) == exe


class _FakeProc:
    def __init__(self, returncode=0, stderr="", alive=True):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""
        self.terminated = 0
        self.killed = 0
        self._alive = alive
        self.args = None

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated += 1
        self._alive = False

    def kill(self):
        self.killed += 1
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return self.returncode


# --- server lifecycle: start on demand, kill on close, NEVER an orphan -------------------------

def _server(**kw):
    spawned = []

    def spawn(cmd, **kwargs):
        p = _FakeProc()
        p.args = list(cmd)
        spawned.append(p)
        return p

    srv = _agave.AgaveServer("/x/agave", port=kw.pop("port", 1235), spawn=spawn,
                             sleep=lambda s: None, **kw)
    return srv, spawned


def test_server_start_spawns_agave_in_server_mode_on_the_requested_port():
    srv, spawned = _server(port=1237, probe=_cycle([False, True]))
    srv.start()
    assert len(spawned) == 1
    assert spawned[0].args == ["/x/agave", "--server", "--port=1237"]
    assert srv.running is True
    assert srv.port == 1237


def test_server_reuses_a_server_already_listening_and_does_not_spawn_a_second():
    """Port 1235 already answering means a server is up — spawning another would fight it."""
    srv, spawned = _server(probe=lambda port: True)
    srv.start()
    assert spawned == []
    assert srv.running is True
    assert srv.adopted is True          # and we say so, so stop() does not kill someone else's


def test_server_stop_kills_only_a_process_we_started():
    srv, spawned = _server(probe=_cycle([False, True]))
    srv.start()
    srv.stop()
    assert spawned[0].terminated == 1
    assert srv.running is False


def test_server_stop_leaves_an_adopted_server_alone_and_forgets_the_adoption():
    """Stopping must also CLEAR `adopted`, or a later start() on a now-dead port believes it is
    still connected to someone else's server and never spawns one."""
    srv, spawned = _server(probe=lambda port: True)
    srv.start()
    srv.stop()
    assert spawned == []
    assert srv.running is False
    assert srv.adopted is False


def test_server_stop_is_idempotent_so_tab_close_plus_window_close_cannot_double_kill():
    srv, spawned = _server(probe=_cycle([False, True]))
    srv.start()
    srv.stop()
    srv.stop()
    assert spawned[0].terminated == 1


def test_server_that_never_listens_raises_naming_the_port_and_the_timeout():
    srv, spawned = _server(probe=lambda port: False)
    with pytest.raises(_agave.AgaveUnavailable) as e:
        srv.start(timeout_s=0.3)
    msg = str(e.value)
    assert "1235" in msg
    assert "did not start" in msg.lower() or "never began listening" in msg.lower()
    assert spawned[0].terminated == 1       # no orphan left behind by a failed start


def test_server_that_exits_during_startup_is_reported_with_its_exit_code():
    def spawn(cmd, **kwargs):
        return _FakeProc(returncode=133, alive=False)

    srv = _agave.AgaveServer("/x/agave", spawn=spawn, sleep=lambda s: None,
                             probe=lambda port: False)
    with pytest.raises(_agave.AgaveUnavailable) as e:
        srv.start(timeout_s=0.3)
    assert "133" in str(e.value)
    assert "exited" in str(e.value).lower()


def _cycle(values):
    it = iter(values)
    last = [values[-1]]

    def probe(port):
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    return probe


# --- the volume we hand AGAVE -----------------------------------------------------------------

class _StubReader:
    """A 2-region, 2-fov, 3-z, 2-channel acquisition with distinguishable planes."""

    def __init__(self):
        self.reads = 0

    def read(self, region, fov, channel, z, t=0):
        self.reads += 1
        base = {"B2": 10, "B3": 40}[region] + z + (0 if channel.endswith("488") else 100)
        return np.full((6, 8), base + fov, dtype=np.uint16)


def _stub_meta():
    return {
        "regions": ["B2", "B3"],
        "fovs_per_region": {"B2": [0, 1], "B3": [0, 1]},
        "fov_positions_um": {("B2", 0): (0.0, 0.0), ("B2", 1): (4.0, 0.0),
                             ("B3", 0): (0.0, 0.0), ("B3", 1): (4.0, 0.0)},
        "channels": [{"name": "Fluorescence_488_nm_Ex", "display_color": "#00FF00"},
                     {"name": "Fluorescence_561_nm_Ex", "display_color": "#FFFF00"}],
        "n_z": 3,
        "z_levels": [0, 1, 2],
        "dz_um": 1.5,
        "pixel_size_um": 0.5,
        "frame_shape": (6, 8),
        "dtype": np.dtype("uint16"),
        "n_t": 2,
    }


def test_write_region_volume_writes_a_zcyx_ome_tiff_agave_can_read(tmp_path):
    out = tmp_path / "vol.ome.tif"
    info = _agave.write_region_volume(_StubReader(), _stub_meta(), "B2", out)

    with tifffile.TiffFile(out) as tf:
        s = tf.series[0]
        assert s.axes == "ZCYX"                 # AGAVE's OME reader wants named axes
        assert s.shape[0] == 3                  # z
        assert s.shape[1] == 2                  # c
        assert s.dtype == np.uint16
        xml = tf.ome_metadata
    assert 'PhysicalSizeZ="1.5"' in xml
    assert "Fluorescence_488_nm_Ex" in xml      # channel identity survives the round trip
    assert info["shape"] == tuple(s.shape)
    assert info["bytes"] == out.stat().st_size


def test_write_region_volume_reports_the_anisotropic_voxel_scale_agave_must_be_told(tmp_path):
    """z 1.5 um against a decimated xy is a SQUASHED SLAB unless set_voxel_scale is called.
    The writer is what knows the decimation, so it is what reports the scale."""
    out_info = _agave.write_region_volume(
        _StubReader(), _stub_meta(), "B2", tmp_path / "v.ome.tif", max_px=4)
    vx, vy, vz = out_info["voxel_um"]
    assert vz == pytest.approx(1.5)
    assert vx == vy
    assert vx > 0.5                              # decimated: coarser than the native 0.5 um
    assert out_info["step"] > 1


def test_write_region_volume_at_a_later_timepoint_reads_that_timepoint(tmp_path):
    seen = []

    class R(_StubReader):
        def read(self, region, fov, channel, z, t=0):
            seen.append(t)
            return super().read(region, fov, channel, z, t)

    _agave.write_region_volume(R(), _stub_meta(), "B2", tmp_path / "vt.ome.tif", t=1)
    assert set(seen) == {1}


def test_write_region_volume_refuses_a_region_without_stage_positions_by_name(tmp_path):
    meta = _stub_meta()
    meta["fov_positions_um"] = {}
    with pytest.raises(_agave.AgaveVolumeError) as e:
        _agave.write_region_volume(_StubReader(), meta, "B2", tmp_path / "n.ome.tif")
    assert "B2" in str(e.value)
    assert "stage position" in str(e.value).lower()


def test_write_region_volume_refuses_an_unknown_region_by_name(tmp_path):
    """And says WHICH regions exist. Without its own guard this falls through to the
    'no stage positions' refusal, which blames the acquisition for a typo in the request."""
    with pytest.raises(_agave.AgaveVolumeError) as e:
        _agave.write_region_volume(_StubReader(), _stub_meta(), "Z9", tmp_path / "n.ome.tif")
    msg = str(e.value)
    assert "Z9" in msg
    assert "B2" in msg and "B3" in msg          # the regions that DO exist, listed
    assert "stage position" not in msg.lower()  # not blamed on missing coordinates


def test_write_region_volume_refuses_a_volume_over_the_byte_budget_before_allocating(tmp_path):
    """A 2.2 GB level-0 volume is the thing we must never write. Refuse from GEOMETRY."""
    meta = _stub_meta()
    r = _StubReader()
    with pytest.raises(_agave.AgaveVolumeError) as e:
        _agave.write_region_volume(r, meta, "B2", tmp_path / "big.ome.tif",
                                   max_px=100000, budget_bytes=100)
    assert "budget" in str(e.value).lower()
    assert r.reads == 0                          # refused before a single plane was read


# --- the disk cache: these are GENERATED files. Capped, and cleaned up. ------------------------

def test_cache_key_separates_region_timepoint_and_resolution():
    k = _agave.volume_key("/data/acq", "B2", t=0, max_px=1000)
    assert k != _agave.volume_key("/data/acq", "B3", t=0, max_px=1000)
    assert k != _agave.volume_key("/data/acq", "B2", t=1, max_px=1000)
    assert k != _agave.volume_key("/data/acq", "B2", t=0, max_px=2000)
    assert k == _agave.volume_key("/data/acq", "B2", t=0, max_px=1000)   # stable


def test_cache_reuses_a_volume_already_on_disk_instead_of_rewriting_it(tmp_path):
    cache = _agave.VolumeCache(tmp_path / "vols")
    calls = []

    def write(path):
        calls.append(path)
        path.write_bytes(b"x" * 100)

    p1 = cache.ensure("k1", write)
    p2 = cache.ensure("k1", write)
    assert p1 == p2
    assert len(calls) == 1                       # second call was a cache hit


def test_cache_prunes_oldest_volumes_once_over_the_cap(tmp_path):
    cache = _agave.VolumeCache(tmp_path / "vols", cap_bytes=250)
    for i in range(4):
        cache.ensure(f"k{i}", lambda p: p.write_bytes(b"x" * 100))
    total = sum(f.stat().st_size for f in cache.root.glob("*"))
    assert total <= 250
    assert cache.path("k3").exists()             # the newest survives


def test_cache_clear_removes_every_generated_volume(tmp_path):
    cache = _agave.VolumeCache(tmp_path / "vols")
    cache.ensure("a", lambda p: p.write_bytes(b"xy"))
    cache.ensure("b", lambda p: p.write_bytes(b"xy"))
    cache.clear()
    assert list((tmp_path / "vols").glob("*")) == []


def test_cache_does_not_keep_a_half_written_volume_when_the_writer_raises(tmp_path):
    cache = _agave.VolumeCache(tmp_path / "vols")

    def bad(p):
        p.write_bytes(b"partial")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        cache.ensure("k", bad)
    assert not cache.path("k").exists()
    assert list(cache.root.glob("*")) == []      # no temp litter either


# --- the render seam: bytes over the socket, NOT a PNG per frame on disk -----------------------

class _FakeBuffer:
    def __init__(self):
        self.commands = []

    def add_command(self, name, *args):
        self.commands.append((name, *args))

    def make_buffer(self):
        return b"BUF:" + repr(self.commands).encode()


class _FakeWs:
    """Models AGAVE's actual stream contract: ONE image per command buffer sent.

    That is not a detail — it is the bug this fake exists to catch. ``load_data_and_get_info``
    flushes a buffer and reads only the JSON reply, so the image that same buffer produced stays
    queued and every later frame comes back ONE BEHIND. Measured against AGAVE 1.10.0.
    """

    def __init__(self, image=b"\x89PNG-fake"):
        import collections

        self.sent = []
        self.default = image
        self.images: list = []            # optional script: one image per send, in order
        self._pending = collections.deque()
        self.closed = False
        self.json = {"x": 8, "y": 6, "z": 3, "c": 2, "t": 2,
                     "channel_names": ["a", "b"]}

    def send(self, buf, binary):
        self.sent.append(buf)
        self._pending.append(self.images.pop(0) if self.images else self.default)

    def wait_for_image(self):
        import io

        if not self._pending:
            return None
        return io.BytesIO(self._pending.popleft())

    def wait_for_json(self):
        return self.json

    def close(self):
        self.closed = True


class _FakeClient:
    """Stands in for agave_pyclient.AgaveRenderer — same duck type the real one exposes."""

    def __init__(self, image=b"\x89PNG-fake"):
        self.cb = _FakeBuffer()
        self.ws = _FakeWs(image)
        self.session_name = ""
        self.closed = False
        self.saved = []

    def __getattr__(self, name):
        # Every AgaveRenderer setter is `self.cb.add_command(NAME, *args)`.
        def call(*args):
            self.cb.add_command(name.upper(), *args)
        return call

    def load_data_and_get_info(self, *args):
        """Exactly what the real client does: flush the buffer, read the JSON, reset. The image
        that flush produced is left in the stream — which is the whole point."""
        self.cb.add_command("LOAD_DATA", *args)
        buf = self.cb.make_buffer()
        self.cb = _FakeBuffer()
        self.ws.send(buf, True)
        return self.ws.wait_for_json()

    def redraw(self):
        self.saved.append(self.session_name)     # the DISK round trip we must not take

    def close(self):
        self.closed = True


def test_render_returns_the_encoded_frame_bytes_and_never_writes_a_file_per_frame():
    c = _FakeClient(image=b"PNGBYTES")
    out = _agave.render_frame(c, width=320, height=240, iterations=8)
    assert out == b"PNGBYTES"
    assert c.saved == []                          # redraw()'s im.save path was NOT used
    names = [x[0] for x in c.cb.commands] if c.cb.commands else []
    assert names == []                            # buffer reset after the flush
    assert len(c.ws.sent) == 1


def test_render_asks_for_the_requested_resolution_and_iteration_count():
    c = _FakeClient()
    _agave.render_frame(c, width=640, height=480, iterations=96)
    body = c.ws.sent[0].decode()
    assert "SET_RESOLUTION', 640, 480" in body
    assert "RENDER_ITERATIONS', 96" in body
    assert "REDRAW'" in body


def test_render_reports_a_dead_socket_rather_than_returning_an_empty_frame():
    c = _FakeClient()
    c.ws.wait_for_image = lambda: None
    with pytest.raises(_agave.AgaveRenderError) as e:
        _agave.render_frame(c, width=10, height=10, iterations=1)
    assert "no image" in str(e.value).lower()


# --- AgaveView: the object the Qt pane drives --------------------------------------------------

def test_view_show_volume_sets_the_voxel_scale_so_the_pancake_is_not_a_squashed_slab():
    c = _FakeClient()
    v = _agave.AgaveView(c)
    v.show_volume("/tmp/v.ome.tif", voxel_um=(4.5, 4.5, 1.5), channels=["a", "b"])
    cmds = {x[0]: x[1:] for x in c.cb.commands}
    assert cmds["SET_VOXEL_SCALE"] == (4.5, 4.5, 1.5)


def test_view_show_volume_frames_the_scene_and_orbits_off_axis():
    """napari's 3D default is EDGE-ON and Julio has complained twice that his image 'looks like a
    1D array'. AGAVE must not repeat it."""
    c = _FakeClient()
    _agave.AgaveView(c).show_volume("/tmp/v.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    names = [x[0] for x in c.cb.commands]
    assert "FRAME_SCENE" in names
    assert "ORBIT_CAMERA" in names
    theta, phi = next(x[1:] for x in c.cb.commands if x[0] == "ORBIT_CAMERA")
    assert (theta, phi) != (0.0, 0.0)             # genuinely off-axis, not a no-op orbit


def test_view_show_volume_enables_every_channel_and_colours_it_from_squids_palette():
    c = _FakeClient()
    _agave.AgaveView(c).show_volume(
        "/tmp/v.ome.tif", voxel_um=(1, 1, 1),
        channels=["Fluorescence_488_nm_Ex", "Fluorescence_561_nm_Ex"])
    enabled = [x for x in c.cb.commands if x[0] == "ENABLE_CHANNEL"]
    assert enabled == [("ENABLE_CHANNEL", 0, 1), ("ENABLE_CHANNEL", 1, 1)]
    diff = {x[1]: x[2:5] for x in c.cb.commands if x[0] == "MAT_DIFFUSE"}
    assert diff[0][1] > 0.9 and diff[0][2] < 0.05                  # 488 -> #1FFF00, green
    assert diff[1][0] > 0.9 and diff[1][2] < 0.05                  # 561 -> #FFCF00, amber


def test_view_seeds_every_channel_with_agaves_own_volume_opacity_threshold():
    c = _FakeClient()
    _agave.AgaveView(c).show_volume("/tmp/v.ome.tif", voxel_um=(1, 1, 1), channels=["a", "b"])
    thr = [x for x in c.cb.commands if x[0] == "SET_PERCENTILE_THRESHOLD"]
    assert [x[1] for x in thr] == [0, 1]           # every channel, none left at a default
    for x in thr:
        assert (x[2], x[3]) == _agave.OPENING_THRESHOLD


def test_view_does_not_seed_agave_from_naparis_display_window():
    """MEASURED: napari's _PCT (1%..99.8%) is a 2-D BRIGHTNESS window; AGAVE's threshold is a
    volume OPACITY function. Feeding one to the other makes 99% of the voxels opaque and the
    render is a flat yellow haze — mean luma 9.5, no tissue architecture at all, against a clean
    black background and legible cortex at AGAVE's own threshold. Different quantities."""
    from squidmip._viewer import _PCT

    lo, hi = _agave.OPENING_THRESHOLD
    assert (lo, hi) != (_PCT[0] / 100.0, _PCT[1] / 100.0)
    assert lo >= 0.2, "a low cut near 0 makes the background opaque and the volume a haze"


def test_view_show_volume_consumes_the_image_the_load_produced_so_frames_are_not_one_behind():
    """MEASURED against AGAVE 1.10.0: every command buffer produces one image. load_data flushes
    a buffer and reads only the JSON, so its image stays queued and every subsequent frame comes
    back ONE BEHIND — you ask for region B3 and see B2. Asking for 160x120 returned the previous
    900x700 frame, three renders running. The load's image must be consumed at load time."""
    c = _FakeClient()
    c.ws.images = [b"IMAGE-FROM-THE-LOAD", b"IMAGE-FROM-THE-RENDER"]
    v = _agave.AgaveView(c)
    v.show_volume("/tmp/v.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    assert v.frame(10, 10, iterations=1) == b"IMAGE-FROM-THE-RENDER"


def test_view_set_time_is_lazy_and_does_not_reload_the_volume():
    c = _FakeClient()
    v = _agave.AgaveView(c)
    v.show_volume("/tmp/v.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    c.cb = _FakeBuffer()
    v.set_time(3)
    assert [x[0] for x in c.cb.commands] == ["SET_TIME"]
    assert c.cb.commands[0][1] == 3


def test_view_close_shuts_the_client_down():
    c = _FakeClient()
    _agave.AgaveView(c).close()
    assert c.closed is True


def test_view_frame_returns_the_png_bytes_for_the_current_scene():
    c = _FakeClient(image=b"FRAME1")
    v = _agave.AgaveView(c)
    v.show_volume("/tmp/v.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    assert v.frame(200, 150, iterations=4) == b"FRAME1"


def test_view_orbit_accumulates_so_dragging_keeps_turning_the_volume():
    c = _FakeClient()
    v = _agave.AgaveView(c)
    v.show_volume("/tmp/v.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    c.cb = _FakeBuffer()
    v.orbit(10.0, -5.0)
    assert c.cb.commands == [("ORBIT_CAMERA", 10.0, -5.0)]


def test_view_orbit_stops_before_the_volume_goes_edge_on():
    """A 10-plane pancake 8600 um wide has NO picture left at 90 degrees off face-on — that is
    Julio's 'my image ends up looking like a 1D array' complaint, and it must not be reachable."""
    c = _FakeClient()
    v = _agave.AgaveView(c)
    v.show_volume("/tmp/v.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    c.cb = _FakeBuffer()
    for _ in range(50):
        v.orbit(30.0, 30.0)
    total_t = sum(x[1] for x in c.cb.commands) + _agave.OPENING_ORBIT[0]
    total_p = sum(x[2] for x in c.cb.commands) + _agave.OPENING_ORBIT[1]
    assert total_t == pytest.approx(_agave.ORBIT_LIMIT)
    assert total_p == pytest.approx(_agave.ORBIT_LIMIT)
    assert _agave.ORBIT_LIMIT < 90.0


def test_view_orbit_at_the_stop_sends_nothing_rather_than_a_zero_command():
    c = _FakeClient()
    v = _agave.AgaveView(c)
    v.show_volume("/tmp/v.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    v.orbit(500.0, 500.0)
    c.cb = _FakeBuffer()
    v.orbit(500.0, 500.0)
    assert c.cb.commands == []


def test_view_orbit_is_clamped_in_both_directions():
    c = _FakeClient()
    v = _agave.AgaveView(c)
    v.show_volume("/tmp/v.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    c.cb = _FakeBuffer()
    v.orbit(-500.0, -500.0)
    t = _agave.OPENING_ORBIT[0] + sum(x[1] for x in c.cb.commands)
    p = _agave.OPENING_ORBIT[1] + sum(x[2] for x in c.cb.commands)
    assert (t, p) == pytest.approx((-_agave.ORBIT_LIMIT, -_agave.ORBIT_LIMIT))


def test_a_fresh_view_reframes_the_orbit_origin_so_a_new_region_opens_off_axis_again():
    """Showing a second region calls frame_scene, which resets the camera. The clamp's origin
    must move with it or the new region opens at the old region's accumulated angle."""
    c = _FakeClient()
    v = _agave.AgaveView(c)
    v.show_volume("/tmp/a.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    v.orbit(60.0, 60.0)
    v.show_volume("/tmp/b.ome.tif", voxel_um=(1, 1, 1), channels=["a"])
    assert (v._theta, v._phi) == _agave.OPENING_ORBIT


# --- against a REAL AGAVE. Skips with a named reason when it is not installed. -----------------

def _real_agave_or_skip():
    try:
        return _agave.require_agave()
    except _agave.AgaveUnavailable as exc:
        pytest.skip(f"no usable AGAVE on this machine: {exc}")


@pytest.mark.agave
def test_real_agave_server_starts_renders_a_frame_and_leaves_no_orphan(tmp_path):
    exe = _real_agave_or_skip()
    pytest.importorskip("agave_pyclient", reason="agave-pyclient is not installed")

    port = 1246                      # not 1235: never fight a server the owner is using
    srv = _agave.AgaveServer(exe, port=port)
    srv.start(timeout_s=25.0)
    try:
        assert srv.running is True
        vol = tmp_path / "real.ome.tif"
        info = _agave.write_region_volume(_StubReader(), _stub_meta(), "B2", vol)
        view = _agave.AgaveView(_agave.connect(port=port))
        try:
            meta = view.show_volume(str(vol), voxel_um=info["voxel_um"],
                                    channels=info["channel_names"])
            assert int(meta["z"]) == 3 and int(meta["c"]) == 2
            frame = view.frame(160, 120, iterations=4)
            # AGAVE 1.10.0 streams JPEG, not PNG. Accept either, and REQUIRE that Qt's own
            # sniffing decoder (the one the pane uses) can actually read it.
            assert frame[:3] == b"\xff\xd8\xff" or frame[:8] == b"\x89PNG\r\n\x1a\n", \
                f"unrecognised frame format {frame[:8]!r}"
            from PIL import Image
            import io
            assert Image.open(io.BytesIO(frame)).size == (160, 120)
        finally:
            view.close()
    finally:
        srv.stop()
    assert srv.running is False
    assert srv.process is None or srv.process.poll() is not None    # no orphan


# --- AgaveEngine: server + volume + view as ONE object the worker thread pumps -----------------

class _FakeServer:
    def __init__(self, exe, port=1235, **kw):
        self.exe, self.port = exe, port
        self.started = 0
        self.stopped = 0
        self.running = False

    def start(self, timeout_s=20.0):
        self.started += 1
        self.running = True

    def stop(self):
        self.stopped += 1
        self.running = False


def _engine(tmp_path, **kw):
    servers = []
    clients = []

    def server_factory(exe, port):
        s = _FakeServer(exe, port)
        servers.append(s)
        return s

    def connect(port):
        c = _FakeClient()
        clients.append(c)
        return c

    eng = _agave.AgaveEngine(
        _StubReader(), _stub_meta(), "/data/acq",
        cache=_agave.VolumeCache(tmp_path / "vols"),
        require=kw.pop("require", lambda: "/x/agave"),
        server_factory=server_factory, connect=connect,
        port=kw.pop("port", 1246), **kw)
    return eng, servers, clients


def test_engine_open_starts_a_server_and_connects_a_client(tmp_path):
    eng, servers, clients = _engine(tmp_path)
    eng.open()
    assert len(servers) == 1 and servers[0].started == 1
    assert len(clients) == 1
    eng.close()


def test_engine_open_propagates_a_missing_agave_as_a_named_refusal(tmp_path):
    def require():
        raise _agave.AgaveUnavailable("AGAVE is not installed on this machine")

    eng, servers, _ = _engine(tmp_path, require=require)
    with pytest.raises(_agave.AgaveUnavailable) as e:
        eng.open()
    assert "not installed" in str(e.value)
    assert servers == []                     # nothing spawned when there is nothing to spawn


def test_engine_show_region_writes_the_volume_once_and_reuses_it_on_return(tmp_path):
    eng, _, clients = _engine(tmp_path)
    eng.open()
    a = eng.show_region("B2")
    b = eng.show_region("B3")
    c = eng.show_region("B2")                 # back to B2: cache HIT, no second write
    assert a["path"] == c["path"]
    assert a["path"] != b["path"]
    assert c["cached"] is True and a["cached"] is False
    eng.close()


def test_engine_show_region_hands_agave_the_voxel_scale_from_the_volume_it_just_wrote(tmp_path):
    eng, _, clients = _engine(tmp_path)
    eng.open()
    info = eng.show_region("B2")
    scale = next(x[1:] for x in clients[0].cb.commands if x[0] == "SET_VOXEL_SCALE")
    assert scale == pytest.approx(info["voxel_um"])
    eng.close()


def test_engine_set_time_does_not_rewrite_the_volume(tmp_path):
    eng, _, clients = _engine(tmp_path)
    eng.open()
    eng.show_region("B2")
    before = list(eng.cache.root.iterdir())
    clients[0].cb = _FakeBuffer()
    eng.set_time(1)
    assert list(eng.cache.root.iterdir()) == before
    assert [x[0] for x in clients[0].cb.commands] == ["SET_TIME"]
    eng.close()


def test_engine_close_stops_the_server_and_deletes_every_generated_volume(tmp_path):
    eng, servers, clients = _engine(tmp_path)
    eng.open()
    eng.show_region("B2")
    root = eng.cache.root
    assert list(root.glob("*.ome.tif"))
    eng.close()
    assert servers[0].stopped == 1
    assert clients[0].closed is True
    assert not root.exists() or list(root.glob("*")) == []


def test_engine_close_is_safe_before_open_and_twice_after(tmp_path):
    eng, servers, _ = _engine(tmp_path)
    eng.close()
    eng.open()
    eng.close()
    eng.close()
    assert servers[0].stopped == 1


def test_engine_frame_before_a_region_is_shown_refuses_rather_than_rendering_nothing(tmp_path):
    eng, _, _ = _engine(tmp_path)
    eng.open()
    with pytest.raises(_agave.AgaveRenderError) as e:
        eng.frame(64, 64)
    assert "no volume" in str(e.value).lower()
    eng.close()


def test_engine_regions_and_timepoints_come_from_the_acquisition(tmp_path):
    eng, _, _ = _engine(tmp_path)
    assert eng.regions == ["B2", "B3"]
    assert eng.n_timepoints == 2


def test_engine_default_connect_reaches_the_module_level_connect_not_its_own_parameter(tmp_path):
    """`connect` the parameter shadows `connect` the module function. A default bound to the
    shadowed name would call None (or itself) — this is the guard against that."""
    eng = _agave.AgaveEngine(_StubReader(), _stub_meta(), "/data/acq",
                             cache=_agave.VolumeCache(tmp_path / "v"),
                             require=lambda: "/x/agave",
                             server_factory=lambda exe, port: _FakeServer(exe, port),
                             port=1)
    with pytest.raises(_agave.AgaveUnavailable):
        eng.open()                # reaches the real connect(), which cannot reach port 1


# --- the launch environment. MEASURED failure: QT_QPA_PLATFORM=offscreen SIGABRTs AGAVE. --------

def test_server_env_strips_qt_settings_that_kill_agaves_own_qt():
    env = _agave.server_env({"PATH": "/bin", "QT_QPA_PLATFORM": "offscreen",
                             "QT_PLUGIN_PATH": "/our/qt", "HOME": "/home/j"})
    assert env == {"PATH": "/bin", "HOME": "/home/j"}


def test_server_env_strips_dynamic_loader_paths_that_would_point_at_our_qt():
    env = _agave.server_env({"DYLD_LIBRARY_PATH": "/our/lib", "LD_LIBRARY_PATH": "/our/lib",
                             "PATH": "/bin"})
    assert env == {"PATH": "/bin"}


def test_server_launches_agave_with_the_scrubbed_environment():
    seen = {}

    def spawn(cmd, **kw):
        seen.update(kw)
        return _FakeProc()

    srv = _agave.AgaveServer("/x/agave", spawn=spawn, sleep=lambda s: None,
                             probe=_cycle([False, True]),
                             env={"QT_QPA_PLATFORM": "offscreen", "PATH": "/bin"})
    srv.start()
    assert "QT_QPA_PLATFORM" not in seen["env"]
    assert seen["env"]["PATH"] == "/bin"
    srv.stop()

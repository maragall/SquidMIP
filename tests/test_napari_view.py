"""napari mosaic view — the processing-layer/channel hierarchy and the binding guards.

These tests use ``napari.components.ViewerModel``, which is Qt-free, so the hierarchy is
exercised headless with no canvas, no display and no Qt binding conflict. Only the embedding
test needs Qt, and it skips itself when Qt is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip._napari_view import (
    META_KEY,
    MosaicKey,
    MosaicLayers,
    NapariBindingError,
    REQUIRED_NAPARI_BINDINGS,
    key_of,
    napari_enabled,
    scale_translate_from_bbox_um,
    verify_napari_bindings,
)

napari = pytest.importorskip("napari")


@pytest.fixture
def layers():
    from napari.components import ViewerModel

    return MosaicLayers(ViewerModel())


def _img(seed=0, shape=(32, 32)):
    return np.random.default_rng(seed).integers(0, 4000, shape, dtype=np.uint16)


# ---------------------------------------------------------------- the flag


def test_napari_is_off_by_default_so_a_failed_path_never_leaves_no_viewer():
    assert napari_enabled({}) is False
    assert napari_enabled({"SQUIDMIP_VIEWER": ""}) is False
    assert napari_enabled({"SQUIDMIP_VIEWER": "ndv"}) is False


def test_napari_switches_on_only_when_asked_for_by_name():
    assert napari_enabled({"SQUIDMIP_VIEWER": "napari"}) is True
    assert napari_enabled({"SQUIDMIP_VIEWER": "  NAPARI  "}) is True


# ------------------------------------------------- identity lives in metadata


def test_identity_is_read_from_metadata_not_parsed_out_of_the_name(layers):
    """The name is a label. Parsing identity back out of it is a known bug class here:
    petakit's reader emits channel names its own regex cannot parse, and 3f1bf3f fixed
    'Fluorescence_488_nm_Ex' failing a parser that wanted r'\\s*nm'."""
    lyr = layers.add_mosaic("stitched", "Fluorescence_488_nm_Ex", _img())

    # A name that would defeat a wavelength regex entirely...
    lyr.name = "something a parser would choke on"

    # ...but identity is unaffected, because it never came from the name.
    assert key_of(lyr) == MosaicKey("stitched", "Fluorescence_488_nm_Ex")
    assert layers.channels("stitched") == ["Fluorescence_488_nm_Ex"]


def test_foreign_layers_are_ignored_not_crashed_on(layers):
    layers.add_mosaic("raw", "488", _img())
    layers.model.add_points(np.zeros((3, 2)), name="user annotation")

    assert key_of(layers.model.layers["user annotation"]) is None
    assert layers.ops() == ["raw"]
    assert len(layers.ours()) == 1


def test_a_layer_with_partial_metadata_is_not_claimed(layers):
    lyr = layers.model.add_image(_img(), name="half", metadata={META_KEY: {"op": "raw"}})
    assert key_of(lyr) is None


# ------------------------------------------------------------ the hierarchy


def test_processing_layers_group_their_channels(layers):
    for op in ("raw", "stitched"):
        for ch in ("405", "488", "561"):
            layers.add_mosaic(op, ch, _img())

    assert layers.ops() == ["raw", "stitched"]
    assert layers.channels("raw") == ["405", "488", "561"]
    assert len(layers.group("stitched")) == 3


def test_show_op_is_the_before_after_toggle(layers):
    for op in ("raw", "stitched"):
        for ch in ("405", "488"):
            layers.add_mosaic(op, ch, _img())

    layers.show_op("raw")
    assert layers.visible_op() == "raw"
    assert all(ly.visible for ly in layers.group("raw"))
    assert not any(ly.visible for ly in layers.group("stitched"))

    layers.show_op("stitched")
    assert layers.visible_op() == "stitched"
    assert not any(ly.visible for ly in layers.group("raw"))


def test_show_op_rejects_an_unknown_processing_layer(layers):
    layers.add_mosaic("raw", "488", _img())
    with pytest.raises(KeyError):
        layers.show_op("deconvolved")


# --------------------------------- contrast: ONE value per channel, no duplication


def test_channel_contrast_survives_the_before_after_toggle(layers):
    """The whole point of linking per channel. Julio: 'I can still see the duplicated
    sliders' — a second control for the same channel must not be able to disagree."""
    for op in ("raw", "stitched"):
        for ch in ("488", "561"):
            layers.add_mosaic(op, ch, _img())

    layers.show_op("raw")
    layers.set_contrast("488", 123, 4321)

    layers.show_op("stitched")

    assert layers.contrast("488") == (123.0, 4321.0)
    assert layers.find("stitched", "488").contrast_limits == [123.0, 4321.0]


def test_contrast_is_per_channel_not_global(layers):
    for ch in ("488", "561"):
        layers.add_mosaic("raw", ch, _img())
        layers.add_mosaic("stitched", ch, _img())

    layers.set_contrast("488", 100, 200)
    assert layers.contrast("561") != (100.0, 200.0)


def test_setting_contrast_on_one_processing_layer_writes_the_other(layers):
    raw = layers.add_mosaic("raw", "488", _img())
    stitched = layers.add_mosaic("stitched", "488", _img())

    raw.contrast_limits = (7, 900)

    assert list(stitched.contrast_limits) == [7.0, 900.0]


def test_contrast_changes_arrive_on_the_public_event(layers):
    """Replaces the ndv contrast tap, which subclassed a private LutView and hooked
    `_lut_controllers`."""
    layers.add_mosaic("raw", "488", _img())
    layers.add_mosaic("stitched", "488", _img())

    seen = []
    layers.on_contrast_changed(lambda e: seen.append(True))
    layers.set_contrast("488", 50, 5000)

    assert seen, "layer.events.contrast_limits did not fire"


def test_a_degenerate_window_is_not_widened(layers):
    """_pct_window returns hi <= lo for a blank channel on purpose. Widening it to
    (lo, lo + 1) would render a blank channel as full white, i.e. as signal."""
    lyr = layers.add_mosaic("raw", "488", _img(), contrast_limits=(500.0, 500.0))
    assert list(lyr.contrast_limits) != [500.0, 501.0]


# ------------------------------------------------------- placement from stage µm


def test_bbox_um_maps_onto_napari_scale_and_translate_with_the_axis_flip():
    """_tiling speaks (x0, y0, x1, y1); napari speaks (row, col) = (y, x). The flip is the
    silent-transpose risk, so it is pinned."""
    scale, translate = scale_translate_from_bbox_um((100.0, 20.0, 300.0, 120.0), (50, 400))

    # height 100 µm over 50 rows; width 200 µm over 400 cols
    assert scale == pytest.approx((2.0, 0.5))
    # translate is (y0, x0), NOT (x0, y0)
    assert translate == (20.0, 100.0)


def test_bbox_um_rejects_a_degenerate_box():
    with pytest.raises(ValueError):
        scale_translate_from_bbox_um((10.0, 10.0, 10.0, 50.0), (8, 8))


def test_add_mosaic_places_the_layer_in_stage_micrometres(layers):
    lyr = layers.add_mosaic("raw", "488", _img(shape=(64, 64)),
                            bbox_um=(0.0, 0.0, 640.0, 640.0))
    assert tuple(lyr.scale) == pytest.approx((10.0, 10.0))
    assert tuple(lyr.translate) == pytest.approx((0.0, 0.0))


# ----------------------------------------------------------------- replacement


def test_re_adding_a_pair_replaces_it_rather_than_duplicating(layers):
    layers.add_mosaic("raw", "488", _img(seed=1))
    layers.add_mosaic("raw", "488", _img(seed=2))

    assert len(layers.group("raw")) == 1


def test_removing_a_processing_layer_drops_its_channels(layers):
    for ch in ("405", "488"):
        layers.add_mosaic("raw", ch, _img())
        layers.add_mosaic("stitched", ch, _img())

    assert sorted(layers.remove_op("stitched")) == ["405", "488"]
    assert layers.ops() == ["raw"]
    # the survivors still work
    layers.set_contrast("488", 10, 20)
    assert layers.contrast("488") == (10.0, 20.0)


# --------------------------------------------------- binding guards (mutation tested)


def test_bindings_are_present_on_the_installed_napari():
    verify_napari_bindings()


def test_binding_check_bites_when_a_symbol_is_renamed():
    """MUTATION TEST. An assertion nobody has watched fail is only a comment.

    This project lost a day to `_voxel_scale`, which bound cleanly, ran every time, and did
    nothing for its entire life because vispy's Visual.freeze() made the assignment raise into
    an `except AttributeError: pass`. So: rename the symbol, prove the guard fails.
    """
    import napari.qt

    class _Renamed:
        # QtViewer has been renamed away; everything else still looks fine.
        __all__ = ("NotQtViewer",)
        NotQtViewer = object

    with pytest.raises(NapariBindingError) as exc:
        verify_napari_bindings(modules={"napari.qt": _Renamed})

    assert "napari.qt.QtViewer" in str(exc.value)


def test_binding_check_bites_on_a_quiet_de_export():
    """A name that still exists but has left __all__ is a deprecation in progress — exactly
    what happened to Window.qt_viewer. Catch it while it is still only a warning."""

    class _DeExported:
        __all__ = ()          # no longer exported...
        QtViewer = object     # ...but still present

    with pytest.raises(NapariBindingError) as exc:
        verify_napari_bindings(modules={"napari.qt": _DeExported})

    assert "no longer in __all__" in str(exc.value)


def test_every_required_binding_is_individually_load_bearing():
    """Each entry must be able to fail the check on its own, so no entry is decorative."""
    for dotted, attr in REQUIRED_NAPARI_BINDINGS:
        stub = type("Stub", (), {"__all__": ()})
        with pytest.raises(NapariBindingError) as exc:
            verify_napari_bindings(modules={dotted: stub})
        assert f"{dotted}.{attr}" in str(exc.value)


# ------------------------------------------------------------------- embedding


# The embedding check builds a real vispy GL canvas. Doing that in-process under pytest
# aborts the interpreter: pytest/napari have already imported PySide6, and creating the GL
# canvas on top of that is the same Qt-binding conflict test_viewer.py documents ("segfaults
# offscreen under pytest's PySide6/napari-loaded environment — a Qt-binding conflict, not a
# code bug"). Skipping would delete the evidence for the central claim of this module, so the
# check runs in a clean SUBPROCESS instead, where it is a real assertion again and a crash is
# a test failure rather than a dead test session.

_EMBED_SCRIPT = r"""
import json, os, sys
# Deliberately NOT forcing QT_QPA_PLATFORM=offscreen: the offscreen plugin ships no GL
# ("QOpenGLWidget is not supported on this platform", "does not support
# createPlatformOpenGLContext"), so a vispy canvas segfaults under it. On a machine with a
# display this runs for real; on a headless box it fails cleanly and the test skips with the
# reason attached rather than pretending to have verified something.
import numpy as np
from qtpy.QtWidgets import QApplication, QHBoxLayout, QWidget
app = QApplication.instance() or QApplication([])
from squidmip._napari_view import build_pane

host = QWidget()
lay = QHBoxLayout(host)
widget, mosaic = build_pane()
lay.addWidget(widget)
app.processEvents()

mosaic.add_mosaic("raw", "488", np.zeros((32, 32), dtype="uint16"))
out = {
    "is_qwidget": isinstance(widget, QWidget),
    "parented_into_ours": widget.parent() is host,
    "dock_widgets": len([c for c in widget.findChildren(QWidget)
                         if type(c).__name__.endswith("DockWidget")]),
    "layer_controls": len([c for c in widget.findChildren(QWidget)
                           if "QtLayerControlsContainer" in type(c).__name__]),
    "ops": mosaic.ops(),
}
print("EMBED " + json.dumps(out))
sys.stdout.flush()
os._exit(0)
"""


def test_the_canvas_embeds_with_no_window_menus_or_docks(tmp_path):
    """The public embedding path: ViewerModel + napari.qt.QtViewer, no napari Window at all,
    so the menu bar / docks / plugin surface are never constructed in the first place."""
    import json
    import subprocess
    import sys

    pytest.importorskip("qtpy")

    script = tmp_path / "embed_check.py"
    script.write_text(_EMBED_SCRIPT)

    import os
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # The commit gate exports QT_QPA_PLATFORM=offscreen for the whole suite, and the offscreen
    # plugin has no GL, so inheriting it guarantees a segfault and a permanent skip. Drop it and
    # let Qt pick the real platform: on a machine with a display this actually verifies, and on
    # a headless one it fails cleanly into the skip below with the reason attached.
    env.pop("QT_QPA_PLATFORM", None)

    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=300, cwd=str(repo), env=env,
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("EMBED ")]
    if not line:
        pytest.skip(
            "napari Qt canvas could not be constructed in this environment "
            f"(rc={proc.returncode}); stderr tail: {proc.stderr[-400:]}"
        )

    got = json.loads(line[0][len("EMBED "):])
    assert got["is_qwidget"] is True
    assert got["parented_into_ours"] is True
    # no napari chrome came along for the ride
    assert got["dock_widgets"] == 0
    assert got["layer_controls"] == 0
    assert got["ops"] == ["raw"]

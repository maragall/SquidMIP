"""OperationStack: ordered, toggleable layer stack (pure model)."""
from squidmip._layers import BASE_KEY, OperationStack


def test_stack_add_toggle_top_and_reset():
    s = OperationStack()
    assert [l.key for l in s.layers()] == ["raw"]
    s.add("mip", "MIP")
    assert s.top_enabled().key == "mip"          # newest enabled layer shows
    s.toggle("mip", False)
    assert s.top_enabled().key == "raw"          # off -> falls back to base
    s.toggle("mip", True)
    s.add("reference", "Reference")
    assert [l.key for l in s.layers()] == ["raw", "mip", "reference"]
    assert s.top_enabled().key == "reference"
    s.reset()
    assert [l.key for l in s.layers()] == ["raw"]


def test_stack_reorder_and_readd_moves_to_top():
    s = OperationStack()
    s.add("mip", "MIP"); s.add("reference", "Reference")
    s.move("mip", +5)                            # clamp to top
    assert s.layers()[-1].key == "mip"
    s.add("reference", "Reference")              # re-add moves reference back to top
    assert s.layers()[-1].key == "reference"


def test_any_number_of_operators_toggle_independently():
    # nothing in the model is MIP-specific: three operator layers, each toggled on its own, and
    # the plate always follows the topmost one still enabled.
    s = OperationStack()
    for k, lbl in (("mip", "MIP"), ("stitched", "Stitched"), ("reference", "Reference")):
        s.add(k, lbl)
    assert s.top_enabled().key == "reference"
    s.toggle("reference", False)
    assert s.top_enabled().key == "stitched"     # falls through to the next enabled one down
    s.toggle("stitched", False)
    assert s.top_enabled().key == "mip"
    s.toggle("stitched", True)                   # re-enabling restores it without reordering
    assert s.top_enabled().key == "stitched"


def test_top_enabled_is_none_when_every_layer_is_off():
    # all-off is a DEFINED state (the view shows an empty plate), not an unrepresentable one
    s = OperationStack()
    s.add("mip", "MIP")
    s.toggle("mip", False)
    s.toggle(BASE_KEY, False)
    assert s.top_enabled() is None
    s.toggle(BASE_KEY, True)
    assert s.top_enabled().key == BASE_KEY


def test_toggle_and_move_ignore_unknown_keys():
    s = OperationStack()
    s.toggle("nope", False); s.move("nope", +1)   # must be no-ops, not raises
    assert [ly.key for ly in s.layers()] == [BASE_KEY]


def test_move_direction_is_plus_one_toward_the_top():
    # PINS the direction: the Layers tab lists layers reversed() (topmost first) and its "↑" sends
    # delta=+1, so +1 MUST move a layer toward the END of layers(). Flip either and this fails.
    s = OperationStack()
    s.add("mip", "MIP"); s.add("stitched", "Stitched")
    assert [ly.key for ly in s.layers()] == [BASE_KEY, "mip", "stitched"]
    assert s.top_enabled().key == "stitched"
    s.move("mip", +1)                             # "↑" on mip -> above stitched -> mip renders
    assert [ly.key for ly in s.layers()] == [BASE_KEY, "stitched", "mip"]
    assert s.top_enabled().key == "mip"
    s.move("mip", -1)                             # "↓" puts it back under stitched
    assert s.top_enabled().key == "stitched"


def test_move_never_displaces_the_base_layer():
    s = OperationStack()
    s.add("mip", "MIP")
    s.move("mip", -5)                             # can't be pushed under the base
    assert [ly.key for ly in s.layers()] == [BASE_KEY, "mip"]
    s.move(BASE_KEY, +5)                          # and the base itself never moves off the bottom
    assert s.layers()[0].key == BASE_KEY

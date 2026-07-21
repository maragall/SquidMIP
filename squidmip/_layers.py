"""OperationStack: the ordered, toggleable layer stack behind the plate view (DESIGN.md).

- Layer 0 is the base ("raw" preview). Each applied operation adds a layer on top.
- Enable, disable, and reorder any layer. The topmost ENABLED layer is what the plate renders.
- v1 usually holds base plus one operation; the structure supports more.
- Pure data structure (no Qt), so it is unit-testable on its own. The Layers tab drives it.
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_KEY = "raw"      # the base layer's key — always present, always at the bottom of the stack


@dataclass
class Layer:
    key: str          # stable id ("raw", "mip", "reference", ...)
    label: str        # shown in the Layers tab
    enabled: bool = True


class OperationStack:
    def __init__(self) -> None:
        self._layers: list[Layer] = [Layer(BASE_KEY, BASE_KEY, True)]   # base layer, always present

    def add(self, key: str, label: str) -> None:
        """Add (or re-add) an operation layer on top, enabled. Re-adding moves it to the top."""
        self._layers = [ly for ly in self._layers if ly.key != key]
        self._layers.append(Layer(key, label, True))

    def toggle(self, key: str, enabled: bool) -> None:
        for ly in self._layers:
            if ly.key == key:
                ly.enabled = enabled
                return

    def move(self, key: str, delta: int) -> None:
        """Reorder a layer by +/- steps; +1 moves it one step TOWARD THE TOP (the end of the list,
        which the Layers tab renders first). The base ('raw') never moves off the bottom: it cannot
        be moved itself, and no operation layer can be pushed under it."""
        if key == BASE_KEY:
            return
        idx = next((i for i, ly in enumerate(self._layers) if ly.key == key), None)
        if idx is None:
            return
        new = max(1, min(len(self._layers) - 1, idx + delta))   # index 0 stays the base layer
        if new != idx:
            self._layers.insert(new, self._layers.pop(idx))

    def top_enabled(self) -> Layer | None:
        """The topmost enabled layer (what the plate renders), or None if all are off."""
        for ly in reversed(self._layers):
            if ly.enabled:
                return ly
        return None

    def layers(self) -> list[Layer]:
        return list(self._layers)

    def reset(self) -> None:
        self._layers = [Layer(BASE_KEY, BASE_KEY, True)]

# TODOS

## Interactive plate-grid viewer (deferred from IMA-185)
- **What:** A 2-D interactive plate-grid viewer — pyramid-fed thumbnails, click a
  well to lazy-load full-res, global-per-channel contrast, region jump (E5 / AA5).
- **Why:** The prototype (`docs/plate-view.html`) demonstrates it, and it is the
  foundation the future squid2minerva / Minerva path builds on. ndviewer_light has
  no 2-D grid today (it navigates wells via a 1-D FOV slider and reads only pyramid
  level 0), so this is a *new viewer or a real ndviewer_light feature*, not a wiring job.
- **Pros:** Whole-plate spatial navigation the slider viewer can't give; reuses the
  OME-zarr pyramid IMA-184 emits.
- **Cons:** Substantial (~8-10 files); needs a home decision (new standalone viewer vs
  extend ndviewer_light, which is a different repo/team).
- **Context:** De-scoped from IMA-185 as "bonus / not required." IMA-185 itself is on
  hold pending stakeholder confirmation that even a static montage is wanted.
- **Depends on / blocked by:** IMA-184 (`plate.ome.zarr`); new-viewer-vs-extend decision.

## Package skeleton coordination (SquidMIP-wide)
- **What:** SquidMIP has no `pyproject.toml` / source tree. Every sibling ticket
  (IMA-183/184/185/186) needs the same minimal skeleton (package + console entry +
  pinned deps; tilefusion has no declared delivery mechanism yet).
- **Why:** Whoever builds first owns it — undefined sequencing = merge/ownership collision.
- **Context:** IMA-185 review chose "minimal skeleton in this ticket," but if IMA-184
  lands first it should own the skeleton and IMA-185 builds on top. Coordinate before
  two tickets scaffold in parallel.
- **Depends on / blocked by:** whichever SquidMIP ticket starts first.

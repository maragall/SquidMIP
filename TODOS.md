# TODOS

## Deferred from IMA-184 eng review (2026-07-04)

### Confirm IMA-193 reads pyramid + plate metadata
- **What:** Before/during IMA-193, verify its plate-navigator actually reads multi-level pyramids and plate/well NGFF metadata (not just array `0` like ndviewer_light).
- **Why:** IMA-184's full-canonical scope (pyramid + spec plate metadata) is justified largely by IMA-193. ndviewer_light ignores both, so if IMA-193 also reads only level 0, the pyramid work delivered no value.
- **Context:** ndviewer_light discovers plates by directory walking and reads only `field/0` + `omero` (see `ndviewer_light/core.py:1149`, `:1070`). The pyramid is invisible to it. This is the load-bearing assumption behind the IMA-184 output scope.
- **Depends on / blocked by:** IMA-193 design.

### Fix upstream squid2minerva/colors.py:53 nesting bug
- **What:** `load_yaml_colors` reads `channel["display_color"]`, but the real `acquisition_channels.yaml` nests it under `channel.camera_settings['1'].display_color`. Fix upstream (SquidMIP vendors a fixed copy; the source repo still has the bug).
- **Why:** squid2minerva's Minerva OME-TIFF exports only get correct colors via the wavelength-fallback map — any custom `display_color` in the yaml is silently ignored.
- **Context:** Confirmed against a real dataset yaml and `colors.py:45-55`. Correct-by-luck today because the fallback palette matches the standard 4-channel wavelengths. Also map channels by NAME, not position (yaml order is descending 638→405).
- **Depends on / blocked by:** whoever owns `~/CEPHLA/projects/explorer/squid2minerva`.

### (Reconciled 2026-07-05) Write parallelism is IMA-188's, not IMA-184's
- Superseded by the build-order reconcile: IMA-188 is the parallel/streaming engine and may call 184's writer concurrently per `(well, fov)`. 184 does not add its own parallel layer; it must be **concurrency-safe** instead (guard the shared `plate`/`well` group-metadata writes). That safety requirement is now in-scope (plan decision 6), not a deferred TODO.

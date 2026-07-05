# TODOS

## RGB/brightfield projection exactness (deferred from IMA-183)

- **What:** Revisit the `project_well` read path for RGB/brightfield modalities.
- **Why:** IMA-183 (D9) uses `reader.read_tile(z_level=z)`, which is bit-identical to a
  raw uint16 max only for **mono-fluorescence** (2D planes — `_to_grayscale_2d` is a
  no-op, `individual_tiffs.py:234`). For color/brightfield planes `read_tile` averages
  RGB to luminance, so the projection would NOT be FIJI-identical.
- **Current state:** Mono-fluorescence is the only in-scope modality (epic: "modality:
  fluorescence (confirm)"). The raw-uint16 read path was considered and deliberately
  dropped in IMA-183 because it duplicated tilefusion's private `_get_tile_filename`
  path logic and format-locked `project_well` to individual_tiffs.
- **Where to start:** If brightfield/RGB enters scope, add a raw-read code path (or a
  reader API that returns unaveraged uint16 planes) gated on modality, plus a
  FIJI-equality test on an RGB fixture. Prefer pushing an "exact read" mode into
  tilefusion (IMA-189) over re-copying path logic in SquidMIP.
- **Depends on / blocked by:** a brightfield/RGB acquisition actually being required
  (likely around IMA-187 multi-FOV work).

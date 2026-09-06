# Universe U1 (`vole.gfx.u1`)

Normative facts of the exact universe implemented by this crate. Every
conforming materialization (scalar, AVX2, AVX-512, Rayon, CUDA) derives from
these rules; nothing platform-dependent is allowed.

## Geometry

- **Positions** are signed Q16.16 fixed-point (`i32` × 2⁻¹⁶ px).
- Scene coordinates are capped at ±`MAX_COORD` = ±2³⁰ (Q16.16) = ±16384 px.
- Instance **translations** (static or trajectory) are capped at
  ±`INST_TRANSLATION_LIMIT` = ±2²⁹ (Q16.16) = ±8192 px, and the static
  bounding box of a placed object at ±`PLACEMENT_BBOX_LIMIT`; together these
  guarantee evaluated placements always fit `i32` (widened `i64`/`i128`
  intermediates, never wrapping).
- **Sample positions**: one sample per output pixel at its center
  `(i + 0.5, j + 0.5)` (U1 exact profile: `SamplingProfile::exact_u1`).
- **Coverage** of a half-open rect `[x0,x1) × [y0,y1)`: covers sample `(i,j)`
  iff `x0 ≤ cx < x1 ∧ y0 ≤ cy < y1` with `cx = (i≪16)|0x8000`.
- **Object sampling** is inverse-mapped nearest: the sample's scene point is
  mapped by the exact inverse affine (`floor` of the rational
  `num/det`), and the containing local pixel is fetched; outside the object
  the sample contributes nothing (transparent). Degenerate transforms
  (`det = 0`) contribute nothing.
- **Affine layout** (normative): `x' = a·x + b·y + tx`, `y' = c·x + d·y + ty`,
  coefficients Q16.16, magnitude ≤ 2²⁰ (scale ≤ 16), translations as above.
- Rasterization fill rule: none needed for the exact 2D profile (no paths
  yet); rects are defined by the coverage rule above.

## Color

- Formats: `Gray8` (opaque), `Rgba8` (straight alpha, code values are
  sRGB codes; no transfer math in composition).
- Composition is exact Porter–Duff `over` computed in the premultiplied
  domain with three normative division primitives:
  - `div255(x)`, 0 ≤ x ≤ 65025: fixed-reciprocal + one correction;
  - `div255_24(x)`, 0 ≤ x ≤ 16 581 375 (24-bit domain): scalar reference uses
    the hardware divider; optimized backends must prove equality over the
    whole domain;
  - `div_small(x, d)`, 0 ≤ x ≤ 65025, 1 ≤ d ≤ 255: 16-step binary long
    division.
- `over(src, dst)`: if `src` opaque → `src`; if `src` transparent → `dst`;
  if `dst` transparent → `src`; else
  `a_out = a_s + div255(a_d·(255−a_s))`,
  `c_out = div_small(c_s·a_s + div255_24(c_d·a_d·(255−a_s)), a_out).min(255)`
  (the `.min(255)` clamps the tiny-alpha rounding case).
- Gray8 output: composed RGBA is composited over opaque black, then exact
  rounded luma `(299r + 587g + 114b + 500)/1000`.
- Background: RGBA surface starts fully transparent; Gray8 starts at luma 0.

## Composition order

1. **Instance draws** sorted by `(layer asc, insertion order asc)`, `over`-ed
   in order.
2. **Structural ops** (fill / copy / move) of all events with time ≤ request
   time, in event order, then op order within an event. They mutate a raster
   layer initialized once from the request-time instance draws:
   - `FillRect`: `over(fill, current)` inside the rect;
   - `CopyRegion`: inside the destination, value = prefix surface (state
     before this op) at `p − d`; source unchanged;
   - `MoveRegion`: inside the destination, value = prefix surface at `p − d`;
     inside the source (and not destination), cleared to transparent.
   Copy/move chains are bounded by a hard recursion limit (fail closed).
3. **Residual bindings** apply after composition, in binding order, on the
   canonical code values of the binding's format, clipped to the request
   domain. A binding whose format differs from the request format is skipped
   (deterministically, and counted).

## Time

- Timeline events carry integer nanosecond timestamps, strictly increasing.
- A trajectory `x(t)` is authoritative; piecewise-linear between keyframes
  with exact rational interpolation `p_k + ⌊(p_{k+1}−p_k)·(t−t_k)/(t_{k+1}−t_k)⌋`
  (floor division), clamping outside the key range.
- The persisted representation is cadence-independent: any time coordinate
  may be requested.

## Generators (Phase H)

- A generator field object (`kind::GENERATOR_FIELD`) declares an extent
  `w×h` and a versioned family + canonical params blob; the seed `s` (where
  the family has one) and parameters `θ` together *are* the generator state.
- **Addressing** is raster-like and exact: the placement's inverse affine
  selects a local lattice pixel `(i,j)`, bounds-checked against the extent;
  the field value is a pure function of `(i,j)` (see
  `docs/EXACT_SEMANTICS.md` §Generators for the per-family tables, including
  the canonical param wire layouts and cost estimators).
- **Evaluation is direct and lazy**: only requested samples are evaluated and
  no whole-object raster is produced, so a partial observation of a large
  field costs partial work (measured in the phase-h receipts).
- Families are bounded and non-Turing-complete; every sample's work estimate
  is < `MAX_GENERATOR_WORK_PER_SAMPLE`.  Referenced-object families
  (affine-reuse, object-family) may reference raster/indexed objects only
  (no recursion), validated against the object table.

## Limits & failure

All caps in `limits.rs` (object counts/bytes, dimensions ≤ 16384 px per axis
for scene-space objects and surfaces, palette entries, residual records,
execution work). Malformed input fails closed with a deterministic `Reject`
code; the parser never panics.

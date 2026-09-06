# Exact semantics (summary)

The normative exact rules live in `docs/UNIVERSE_U1.md`; this page records the
semantic decisions that must not silently change, and why.

## Non-negotiable

1. **Sample-center, half-open coverage.** A rect covers pixel `(i,j)` iff its
   center lies in `[x0,x1) × [y0,y1)`. Changing this breaks every conformance
   vector.
2. **Widened intermediates, no wrapping.** All products/shifts use `i64`
   (and `i128` only for exact inverse-affine division) with validated input
   ranges; placement validation guarantees `i32` results.
3. **Exact division primitives are shared.** `div255`, `div255_24`,
   `div_small` are the *only* rounding sites in composition; SIMD/CUDA
   backends must implement these exact functions (differential tests over the
   full domains).
4. **Trajectories are functions of time, evaluated exactly.** No per-cadence
   baking; interpolation is the exact rational floor-division rule.
5. **Structural ops mutate a raster layer over the request-time instance
   base.** They are not instance-state changes; a move clears its source to
   transparent; copies read the *prefix* surface (before the op), bounding
   dependency depth.
6. **Residuals are explicit and format-bound.** Applied in code space, in
   binding order, clipped to the request domain; mismatched-format bindings
   are skipped and counted, never guessed.

## Documented divergences from the paper's illustrative sketch

The paper is architectural prior art, not a frozen ABI (prompt §63). Where U1
needed unambiguous semantics the implementation chose the simplest bounded
rule and documented it:

- COPY/MOVE reference the same-request-time instance base rather than a
  stored "previous frame" raster (which would couple partial materialization
  to temporal chains). State evolution (`Φ`) remains the authoritative
  temporal mechanism.
- Residual regions are canonical-surface pixel rectangles at this stage;
  view/projection-bound residuals are future work.

## Generators (Phase H)

A generator field object has an extent `w×h` and evaluates as a pure,
bounded function of the **local integer pixel lattice**: pixel `(i,j)` maps
through the placement affine exactly like a raster sample (inverse affine of
the sample center, floored), is bounds-checked against `[0,w)×[0,h)`, and the
field value at that lattice point is composed like any other object sample
(straight-alpha `over`).  **Direct seeded evaluation**: only requested samples
are ever evaluated; no whole-object raster is produced (the
no-mandatory-re-baking boundary holds per sample).

Family parameter blobs are canonical (fixed-width LE, no alternative
spellings; decode rejects non-canonical forms with a `Reject` code).  Field
by field (`i,j` lattice, all colors straight-alpha RGBA8):

| tag | family | params (canonical) | exact value at `(i,j)` | work |
|---|---|---|---|---|
| 1 | constant | `color` 4B | `color` | 1 |
| 2 | gradient | mode u8; linear: `p0x i32 p0y i32 c0 p1x i32 p1y i32 c1`; bilinear: `c00 c10 c01 c11` | linear: blend of `c0→c1` at `t = clamp(dot((i,j)−p0, p1−p0)/|p1−p0|², 0, 1)` (anchors integer px inside the extent); bilinear: tensor blend of the four corner colors over `[0,w−1]×[0,h−1]` (degenerate 1-wide extents reduce to one-axis blends); both exact fixed-point floor blends | 16 / 32 |
| 3 | palette-field | `mode u8 period u32 seed u64 n u32 entries…` (≤256) | `entries[idx]`; hash: `idx = fmix2(seed,i,j) mod n`; x-ramp: `floor(i·n/w)`; band-x/y: `(i/period) mod n`, `(j/period) mod n` (unused modes require period=1) | 2 |
| 4 | periodic | `mode u8 px u32 py u32 c0 c1` | stripe-x: `(i/px)&1`; stripe-y: `(j/py)&1`; checker: `((i/px)+(j/py))&1` selects `c0`/`c1` (unused axis period must be 1) | 2 |
| 5 | tiled | `tw u32 th u32 tw·th colors` (≤64 each) | `tile[j%th][i%tw]` (embedded primitive tile, not a materialized object) | 1 |
| 6 | affine-reuse | `object u32 affine 6×i32 outside 4B` | sample referenced raster/indexed object at `floor(A·(i+0.5, j+0.5))` (widened exact map); out-of-bounds/guard → `outside`; identity A reproduces the reference pixel-for-pixel | 16 |
| 7 | deterministic-field | `seed u64` | `gray(hash(seed,i,j))` (SplitMix64, bits 40–47) | 1 |
| 8 | fractal | `seed u64 base u32 octaves u8 persistence u32` (1≤octaves≤8, base≤4096, persistence Q16.16) | value noise: octave `k` lattice cell `base·2ᵏ`, bilinear lattice interpolation of `gray(hash(seed,cell,k))`, weighted by `persistenceᵏ` (Q16.16 exact), final value `Σ v·w / Σ w` | 1+16·octaves |
| 9 | sdf | `shape u8 cx cy p1 p2 i32 inside outside` (anchors Q16.16 inside extent) | strict inside test at local sample center: circle `dx²+dy² < r²`, box `|dx|<hw ∧ |dy|<hh` (boundary excluded, half-open analog) | 8 |
| 10 | object-family | `cell_w u32 cell_h u32 mode u8 seed u64 n u32 objects… background` (n≤16) | cell `(i/cw, j/ch)` picks a referenced object (ordered: row-major cell index mod n; hash: `fmix2 mod n`), sampled at `(i%cw, j%ch)`; out of the object's own size → `background` | 8 |

The work column is the **cost estimator** (upper-bound units/sample, scheduler
input; far below `MAX_GENERATOR_WORK_PER_SAMPLE`).  Referenced objects
(affine-reuse, object-family) must be raster/indexed content objects — never
palettes or other generators (no recursion by validation).  Generator
references sample with the referenced object's own palette plus request-time
palette patches; instance-level palette overrides apply only to direct
`IndexedRaster` object sampling.

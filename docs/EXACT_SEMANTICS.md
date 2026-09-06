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

# Materialization

`q → spatial/view lookup → candidate closure → compose → residual closure → output`

## The observation path

1. `Scene::resolve(doc, t)`: replay transitions ≤ t, evaluate trajectories,
   sort instances into composition order, gather palette patches and ordered
   structural ops. No raster is produced here (the no-re-baking boundary).
2. Per output sample: compose over the RGBA domain (instance draws, then
   structural ops), convert to the requested format's codes.
3. Residual bindings applied on the codes, clipped to the request rectangle.

## Block decomposition

The optimization unit is deliberately neither "pixel" nor "frame":
`materialize::blocks` defines shapes (8×8 … 32×16, scanline segments, bands)
and a canonical deterministic tiling of any request rectangle. Block shape is
a backend decision swept by courts; it is never part of the IR.

## Specialized simple-composition engine

`materialize::simple` implements an algorithmic specialization of the same
semantics for the *eligible subset*: fully-opaque rasters placed at integer
pixel translations (identity linear part), opaque fills, and residual
bindings.  In that subset composition is ordered memory writes, so the engine
injects byte-moving row kernels (`copy_row`, `fill_row`) and is provably
byte-identical to the oracle for any kernel set.  Four measured rows share the
engine so courts can decompose the speedup:

- `scalar` (oracle): general per-sample composition;
- `blocked`: oracle-equivalent, block + dependency indexed;
- `scalar-simple`: the engine with scalar kernels (algorithmic
  specialization, no SIMD);
- `avx2` / `avx512`: the engine with SIMD kernels (SIMD gain measured against
  scalar-simple).

Documents outside the subset return `None` and fall back to blocked/scalar;
receipts record which path ran.

## Generator fields (Phase H)

Generator objects (`Object::GeneratorField`) sample through the same shared
per-sample rule (`sample_placed`): inverse-affine to the local lattice pixel,
bounds-check against the field extent, then evaluate the family at that
point only — no whole-object raster is produced for any request shape
(direct seeded evaluation).  Opaque constant fields are eligible for the
fill-based simple path; all other families run the exact scalar/blocked path
in Phase H (recorded decision, see docs/IMPLEMENTATION_STATE.md).

## Work accounting

Every materialization returns `Counters` (samples, instance tests/draws,
ops applied, residual records, format skips). Courts compute
`work_fraction(r) = work(fraction r)/work(full)`; if requesting 1% of a
surface costs ~100% of the work, that is negative evidence and must be
reported (Phase R).

## Backend equality

For the exact profile:

```
scalar_hash == avx2_hash == avx512_hash == rayon_hash == cuda_hash
```

Enforced by differential tests over random valid states, boundary cases,
odd widths, unaligned request beginnings, tails, tiny regions, and clipping.

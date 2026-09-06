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

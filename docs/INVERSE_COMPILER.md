# Inverse compiler ("unbaking")

`A -> (Γ, s, θ, R)`: explain a baked raster asset as bounded procedural state
plus an explicit residual.  Status: **Phase I implemented (scalar detectors +
Pareto + literal fallback); Phase J implemented (structural reuse/sprite
extraction/composite proposals)**; Phases K–N pending.

## What Phase I implements (`src/inverse/`)

1. **Assets** (`asset.rs`): flat raster inputs compared in their own code
   space (Gray8/Rgba8); hostile-safe bounds.
2. **Detectors** (`detect.rs`), in the paper's search order, each proposing a
   generator explanation from cheap structural invariants:
   - constant fields;
   - two-color periodic stripes (vertical/horizontal bands) and checkers;
   - palette-band fields (uniform rows whose run-length encoding is an exact
     equal-width color cycle);
   - tiled fields (minimal horizontal+vertical wrap period);
   - linear ramp and bilinear corner fits;
   - the literal raster fallback (always a candidate; never skipped).
   Detectors are *proposals*: evaluation measures each proposal's exact
   residual; a wrong proposal simply carries a large residual and is pruned
   by the frontier (bounded search, counted `search_work`).
3. **Evaluation** (`candidate.rs`, `mod.rs::finalize`):
   `H -> materialize -> A_hat`, `R_H = A ⊖ A_hat` as a canonical
   sparse-overwrite payload, residual attached, and a byte-exact closure
   gate (materialize(H + R) == A).
4. **Pareto frontier** (`frontier.rs`): non-dominated candidates over four
   **deterministic** axes — persistent bytes, residual bytes, a
   host-independent materialization-work model (`samples × per-sample
   generator cost`; measured wall-clock is reported but never decides
   dominance), and search work.  `min_total_bytes()` is the default profile;
   any future weighted objective must persist its lambdas in receipts.
5. **Determinism**: detector enumeration, evaluation and the frontier are
   deterministic (same asset → same frontier, gated by tests).

## Search order (paper §19)

Phase I covers steps 1, 3 (bands), 5 (periodicity), 6 (analytic), 9 (residual
decomposition), 10 (fallback); Phase J adds step 2 (structural reuse) and the
first affine/translation reuse of step 4:

```
source characterization        (in progress: format/extent/color counts)
  +-- palette/index discovery          palette-band-x/y  [I]
  +-- exact structural reuse           component crops, content classes [J]
  +-- translation/repetition           shared-object sprite-repeat [J]
  +-- symmetry / periodicity           2-color stripes/checker [I]
  +-- simple analytic families         constant, ramps, bilinear [I]
  +-- residual decomposition           sparse-overwrite closure [I]
  +-- literal/original fallback        always present [I]
seeded-field search / deeper symmetry -> K–N
compound/hierarchical factorization           -> N
SIMD / Rayon / CUDA batched search            -> K–M
DSFB governor (zero authority)                -> N
```

## Claim discipline

The compiler never claims to recover an author's semantics — only a bounded
explanation that wins (or ties on the frontier) under the declared cost
model.  A generator "explains" an asset only when materializing it with its
residual reproduces the asset byte-for-byte (gated).  Storage-only wins are
classified by the no-mandatory-re-baking court (Phase P), never by the
compiler alone.

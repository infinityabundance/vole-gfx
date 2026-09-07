# Inverse compiler ("unbaking")

`A -> (Γ, s, θ, R)`: explain a baked raster asset as bounded procedural state
plus an explicit residual.  Status: **Phase I implemented (scalar detectors +
Pareto + literal fallback); Phase J implemented (structural reuse/sprite
extraction/composite proposals); Phase K implemented (seeded-field search —
scalar oracle + AVX2/AVX-512 batched seed sweeps with identical accepted
sets)**; Phases L–N pending.

## Phase K: seeded-field search (`search.rs`)

The deterministic gray-noise family (`DETERMINISTIC_FIELD`, gray value
`gray_byte(fmix3(seed, i, j, 0))`) is searched over the bounded seed universe
`[0, range)`:

- **Acceptance is host-independent**: `accept(seed)` iff the field equals the
  surface on **every** sample.  A deterministic corner-anchor prefilter is a
  pure pruning device (a necessary condition of full equality; a seed is
  never accepted without the full-surface check), so no backend can reject a
  true match or accept a false one.
- **Backends** (`sweep_scalar` oracle; `sweep_avx2`, 4 seeds/register with
  `vpmuludq`-emulated 64-bit multiplies; `sweep_avx512`, 8 seeds/register
  with native `vpmullq`) compute identical accepted sets — lane hash values
  are bit-identical to the scalar `hash64` (differential tests + gate
  parity rows).  SIMD executes whole lanes through every anchor and cannot
  short-circuit per lane, so its executed work is counted and receipted but
  is never a frontier axis (frontiers must stay host-independent).
- **Wired detector**: `seeded-field` proposals join `detect::propose` (after
  structural reuse) using the canonical scalar sweep over
  `DEFAULT_SEED_SWEEP_RANGE` (2¹⁶) and the smallest matching seed, so a
  baked noise-field asset unbakes to the exact generator (~144 B persistent,
  zero residual) instead of falling back to literal.
- **Negative controls** are independently generated SHA-256 pseudo-random
  bytes canonicalized through the materializer; the claim is precise and
  receipted — no exact match exists in the *evaluated* seed-sweep universe
  (the gate asserts it) — not a universal claim about every U1 family.  The
  fractal family is seeded but *not* part of the Phase K searched universe
  and stays on the literal fallback (receipted, Phase-K gate).

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
   by the frontier (bounded search, **counted exactly where it occurs** — see
   “Search-work and byte accounting” below).
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

## Search-work and byte accounting (frozen U1 model)

The search axis of the frontier is a **faithful count of real deterministic
search operations**, not a flat `sample_count()`.  Its boundary (normative;
see `work.rs`) is

```text
W_search = W_discovery + W_discrimination
```

hypothesis discovery (detector scans, membership probes, per-seed anchor
evaluations) plus candidate discrimination (full-surface survivor
verification, crop grouping).  **Outside** the metric: preprocessing/format
normalization (deriving the gray surface — the Phase-K detector and sweeps
charge Gray8 and RGBA surfaces identically, i.e. zero units for it),
representation construction (lifting crops into objects — bounded by content
size, reflected in persistent bytes), and wall-clock setup (receipted as
`*_ns`, never as work units).

Every detector carries a `SearchCounter` (`work.rs`) whose frozen conversion
is

```text
search_work = pixels_read + code_compares + hash_ops
            + candidate_tests + crop_bytes_compared
```

with each field incremented exactly where the operation occurs:

- `pixels_read`: a sample's code materialized for a scan (early-exiting scans
  count only the samples actually examined);
- `code_compares`: a whole-code equality decision (a row-uniformity compare
  counts its `w` codes; the tiled period scan counts every sample pair it
  compares, including the failing pair that rejects a period);
- `crop_bytes_compared`: bytes compared in byte-granular crop equality
  (structural grouping; early exit is data-exact);
- `hash_ops`: one per seed-hash evaluated by the Phase K sweep (anchor and
  full-surface verification hashes, exact per backend);
- `candidate_tests`: reserved for the batched hypothesis-verification phases
  (L–M); zero through Phase K, where candidate evaluation is the separate
  `materialize_work` axis.

Control flow (index arithmetic, `u32` width compares, `BTreeMap`
bookkeeping) is deliberately not counted.  When a detector emits several
proposals from one shared scan, each proposal carries the shared scan cost
(conservative attribution).  The literal fallback carries **zero** search
cost: it is always present and requires no scan.  The Phase K seeded-field
proposal's search counter is the **canonical scalar sweep** count (SIMD
backends execute extra whole-lane work; that is receipted, never a frontier
axis, so frontiers stay host-independent).

Byte accounting is equally definitional.  `persistent_bytes` is the
canonical size of the generator-only document; `residual_bytes` is the
canonical **delta** of the residual-bound document over it,

```text
residual_bytes = encode(doc_with_residual).len() - persistent_bytes
```

so the residual axis includes the binding's structural overhead (event / op /
algebra / region / format / length fields), not just the raw payload —
material for the tiny residuals where procedural candidates compete on the
frontier.  `persistent_bytes + residual_bytes` is therefore exactly the
canonical size of the stored candidate document.

## Search order (paper §19)

Phase I covers steps 1, 3 (bands), 5 (periodicity), 6 (analytic), 9 (residual
decomposition), 10 (fallback); Phase J adds step 2 (structural reuse) and the
first affine/translation reuse of step 4; Phase K adds the seeded-field step
for the deterministic gray-noise family:

```
source characterization        (in progress: format/extent/color counts)
  +-- palette/index discovery          palette-band-x/y  [I]
  +-- exact structural reuse           component crops, content classes [J]
  +-- translation/repetition           shared-object sprite-repeat [J]
  +-- symmetry / periodicity           2-color stripes/checker [I]
  +-- simple analytic families         constant, ramps, bilinear [I]
  +-- seeded procedural fields         seed sweep, deterministic family [K]
  +-- residual decomposition           sparse-overwrite closure [I]
  +-- literal/original fallback        always present [I]
seeded-field search (other families) / deeper symmetry -> L–N
compound/hierarchical factorization           -> N
Rayon / CUDA batched search                   -> L–M
DSFB governor (zero authority)                -> N
```

## Claim discipline

The compiler never claims to recover an author's semantics — only a bounded
explanation that wins (or ties on the frontier) under the declared cost
model.  A generator "explains" an asset only when materializing it with its
residual reproduces the asset byte-for-byte (gated).  Storage-only wins are
classified by the no-mandatory-re-baking court (Phase P), never by the
compiler alone.

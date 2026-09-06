# Architecture

## The core invariant

```
      one mathematical VOLE-GFX semantics
      many execution widths/backends

materialize_scalar(G, q) == materialize_avx2(G, q) == materialize_avx512(G, q)
                        == materialize_rayon(G, q) == materialize_cuda(G, q)   [byte-for-byte]
```

The scalar implementation is the semantic oracle. Accelerated backends are
optimizations of it; they may never become the specification by accident.
Equality is enforced by canonical SHA-256 output hashes in differential tests.

## The computational model

- `U` — the versioned universe (`vole.gfx.u1`): normative algorithms, fixed
  arithmetic rules, limits.
- `G` — bounded persistent procedural state: immutable objects, live
  instances, trajectories, palettes (with patches), clip state.
- `Φ` — deterministic state transition: `G' = Φ(U, G, Δ)` applied by a
  timeline cursor in strictly increasing event-time order.
- `q` — an observation request: `{time_ns, view, spatial domain, sampling
  profile, output format}`. Frames, resolution and cadence are **not**
  normative state properties.
- `M` — the materializer: composes the request-time state into sample values
  for exactly the requested domain, then applies residual closure.
- `R` — explicit residual (sparse overwrite / XOR), applied on canonical code
  values after composition.

Frames are one optional materialization request (`FullSurface`), never the
primary object.

## Data flow

```
Document (canonical IR)
   │ decode/validate
   ▼
Scene::resolve(doc, t)          (state at one explicit time)
   │  live instances (layer/order-sorted), palettes, structural ops
   ▼
ObservationRequest
   │ materialize_scene
   ▼
compose per sample (RGBA domain) → convert to format codes
   ▼
residual closure (pixel domain, clipped to request)
   ▼
Output {w, h, format, bytes} + canonical hash + counters
```

## Module map (one crate)

| Module | Responsibility | Phase |
|---|---|---|
| `limits` | hard caps, `Reject` codes | A |
| `fixed` | Q16.16 geometry, affine, rects, hashing | A |
| `color` | formats, exact `over`, div primitives | A |
| `ir/{encode,decode,validate,canonical,wire}` | canonical binary IR | A |
| `state/{mod,transition,trajectory}` | scene resolution, cursor | A/H |
| `observation/{domain,request,sampling}` | request model | A |
| `materialize/{scalar,blocks}` | oracle + block decomposition | A/C |
| `residual` | algebras + payload validation | A/B |
| `evidence/{receipt,environment}` | receipts, env, ledger | A |
| `io` | PPM/PGM | A |
| `procedural` | generators `O = Γ(U, s, θ)` | H |
| `materialize/{dispatch,avx2,avx512,rayon}` | CPU backends | D–F |
| `cuda/*` | Rust-PTX device + host | G+ |
| `inverse/*` | unbaking compiler | I–N |
| `corpus/*` | manifest/split/fetch | O |
| `direct/*` | experimental display path | S/T |
| `materialize/heterogeneous` | measured scheduler | W |

## Performance order of operations

1. correctness; 2. algorithmic work elimination (dependency indexing);
3. blocking; 4. SIMD; 5. Rayon; 6. CUDA; 7. specialized GPU features.
No exotic tuning while the algorithm still scans the whole world per sample.

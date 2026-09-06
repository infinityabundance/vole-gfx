# Implementation state

Machine-readable status: the `evidence/claims.json` ledger and the JSON
receipts under `evidence/receipts/` are authoritative; this document is a
human-readable digest that follows them.  Status legend: implemented /
experimental / unsupported / future / pending.

## Starting state (recorded before implementation)

Empty repository: `research/` (paper sources, ignored by git) and
`vole-gfx-prompt.txt` (ignored by git). No git history existed; the repository
was initialized during Phase A.  All phases below were implemented inside ONE
Cargo package (`vole-gfx`); no workspace, no subcrates.

## Phase receipts

| Phase | Name | Status | Receipt |
|---|---|---|---|
| A | Exact U1 model, IR, parser, limits, scalar 2D semantics | implemented | phase-a |
| B | Canonical conformance vectors, exact residual closure | implemented | phase-b |
| C | Blocking + dependency indexing | implemented | phase-c |
| D | AVX2 backend | implemented | cpu-backend-matrix (d) |
| E | AVX-512 backend | implemented | cpu-backend-matrix (e) |
| F | Rayon + SIMD | implemented | cpu-backend-matrix (f) |
| G | Rust CUDA PTX device + host | implemented (byte-kernel parity) | phase-g |
| H | Procedural state, generators, trajectories | implemented | phase-h |
| I | Inverse procedural asset compiler — scalar | implemented | phase-i |
| J–N | Structural reuse; SIMD/Rayon/CUDA search; DSFB | pending | – |
| O | Corpus (100+ assets) + negative controls | pending | – |
| P–R | No-rebake / observation / partial courts | pending | – |
| S–T | CUDA↔Vulkan, direct display | unsupported (hardware gate) | – |
| U | Bounded 3D subset | pending | – |
| V | Quality-bounded profile | pending | – |
| W | Heterogeneous autotuning (train/val/test) | pending | – |
| X | Full evidence/Pareto seal | pending | – |

## What Phases A–H implemented (committed, tested, receipted)

- **Exact universe `vole.gfx.u1`**: Q16.16 fixed-point geometry, sample-center
  coverage, half-open rectangles, widened intermediates, overflow-safe
  placement validation (`limits.rs`, `fixed.rs`, `color.rs`).
- **Color semantics**: Gray8/RGBA8; straight-alpha exact Porter–Duff `over`
  with three exact floor-division primitives shared by every backend
  (`div255`, `div255_24`, `div_small`); exhaustive sweeps over their domains.
- **Canonical IR**: fixed-width little-endian wire format, versioned header,
  strict section order, semantic validation (references, canonical ordering,
  live-instance uniqueness, overflow guards). Decoder never panics and fails
  closed with deterministic `Reject` codes.
- **State**: immutable objects (raster / palette / indexed raster), instances,
  trajectories (exact rational linear interpolation), timeline transitions via
  a replayable cursor, palette patches, clipping, layers.
- **Observations**: `ObservationRequest { time_ns, view, domain, sampling,
  format }` with the full domain family (sample, span, rectangle, tile,
  scanline, display band, irregular samples, viewport, full surface).  Frames
  are not normative state; cadence is never encoded.
- **Residual module**: sparse-overwrite and XOR algebras with canonical sorted
  records and payload validation (Phase B conformance vectors, 12 pinned
  SHA-256 vectors).
- **Scalar materializer (semantic oracle)**: per-sample composition in RGBA
  domain, structural fill/copy/move ops over the request-time instance base,
  residual closure on canonical code values, counters for receipt accounting,
  deterministic output hashing.  The scalar implementation IS the spec.
- **Phase C blocking + dependency indexing**: uniform-grid placement index,
  per-block op culling, memoized copy/move prefix evaluation; byte-identical
  to the oracle; measured work tracks requested samples (a 1% region cost
  ~0.6% of full-surface work on the phase-c court).
- **Phase D AVX2**: exact byte-lane row kernels over the *eligible subset*
  (opaque integer-translation raster instances, opaque fills, residual
  bindings).  Byte-identical to the oracle; measured median ~130 µs on the
  1920×1080 200-sprite court (this host), vs ~211 ms scalar oracle and
  ~13.7 ms blocked.  The ablation rows (scalar oracle / scalar blocked /
  scalar-simple / SIMD simple) decompose algorithmic-specialization gain from
  SIMD gain; see `examples/cpu_matrix_gate.rs` and the cpu-backend-matrix
  receipts.
- **Phase E AVX-512**: independent measured backend, same eligible subset.
  Retained negative result: on this host's memory-bound court AVX-512 (~144 µs)
  loses to AVX2 (~130 µs); auto dispatch follows the evidence (AVX2 first).
- **Phase F Rayon**: coarse band parallelism *above* SIMD with custom pools;
  deterministic band-ordered merge; rayon-avx2 ~268 µs on the same court
  (band height 64).  Rayon is the throughput path, not a real-time scheduler.
- **Phase G Rust CUDA PTX**: device kernels written in Rust
  (`src/cuda/kernels.rs`, `extern "ptx-kernel"`) compiled to launchable PTX
  (`.visible .entry vgf_fill_u8/vgf_copy_u8`, `sm_89`) with the pinned
  nightly (`nightly-2026-07-24`, `nvptx64-nvidia-cuda`,
  `-Z ub-checks=no -C overflow-checks=no`) — no CUDA C++ anywhere.  Host
  driver interaction is a narrow `libloading` binding (`cuda::host`, feature
  `cuda`) with explicit per-call context binding (`ContextGuard`): every
  device-touching call makes the module's context current on the calling
  thread and restores the previous one, which is what makes sharing a `Module`
  across threads sound (SAFETY notes in `src/cuda/host.rs`).  Both kernels
  produce output byte-identical to the CPU reference on this host's RTX 4080
  SUPER.  PTX/SM/driver versions are recorded in the phase-g receipts.
- **Evidence system**: environment snapshot (CPU/GPU/OS/compiler/git),
  immutable JSON receipt schema with content-derived run ids and self-hash
  verification, machine-readable claims ledger (`evidence/claims.json`).
- **CLI** (`src/bin/vole-gfx.rs`): `universe`, `inspect`, `validate`,
  `canonicalize`, `materialize`, `evidence verify`, `encode` (JSON authoring
  pending), `example` runner.
- **Tests**: ~190 unit + integration tests (unit, conformance vectors,
  differential backends, procedural fields, scalar materialize,
  adversarial/property) plus criterion benches; fmt + clippy `-D warnings`
  clean on default and `cuda` features.

## What Phase H added (procedural state)

- **Generator field objects** (`Object::GeneratorField { family, w, h,
  params }`) with raster-like addressing and **direct seeded evaluation**: a
  requested sample evaluates the field at that lattice point only; no
  whole-object raster is ever produced (no-mandatory-re-baking boundary per
  sample).  Partial requests cost partial work — the phase-h gate measured
  `samples_evaluated == samples_requested` for every request shape on a
  1920×1080 ten-family court, with a 1% region costing 1% of samples.
- **Ten bounded generator families** (`procedural::families`): constant,
  gradient (linear + bilinear), palette-field (hash/ramp/band index modes),
  periodic (stripes/checker), tiled (embedded tile), affine-reuse
  (referenced-object sampling), deterministic hash field, fractal value
  noise, SDF disc/box, and object-family grids — each with a canonical
  versioned param wire encoding, semantic validation (incl. referenced-object
  rules and extent-relative anchors), exact deterministic evaluation shared
  by every backend through one sampler, and a bounded work-unit cost
  estimator.
- **Materialization**: the scalar oracle and block materializer sample
  generator objects through one shared sampler (`sample_placed`); constant
  generator fields are eligible for the fill-based fast path (AVX2 fills,
  measured ~113 µs for the constant 1080p subset) while all other families
  run the exact scalar/blocked path in this phase (recorded decision).
  Placed generator fields take part in dependency indexing via their extent.
- **Exactness evidence**: pinned conformance SHA-256 vectors for the mixed
  ten-family scene (RGBA/Gray8, two times); differential tests across
  scalar/blocked/rayon/auto for full/region/tile/band/irregular domains;
  adversarial param blobs fail closed at decode and validation.

## Explicit non-claims (as of Phase I)

No claim of inverse compilation beyond the receipted scalar detector set
(Phases J–N pending: structural reuse, affine/symmetry factoring, seeded-
field search, SIMD/Rayon/CUDA search, recursive factorization, DSFB). No
claim that any detector recovers an author's semantics. No claim that a
detected explanation is optimal over unseen families — the frontier is over
the *evaluated* candidate set, and the literal fallback bounds the loss. No
claim that any generator "explains" pixels outside byte-exact reconstruction
with counted state. No claim of procedural *computing* from storage savings
(no-rebake court is Phase P). No CUDA claim for generators or inverse search
(CPU-only through I; byte-kernel parity is the only CUDA claim). Only the
CONSTANT family participates in the SIMD fill fast path in Phase H; other
families run exact scalar/blocked evaluation. Palette fields embed their
palette in the params and are not animatable by timeline `PaletteSet` ops in
this phase. No claim that CUDA beats CPU or vice versa beyond the exact
receipted domains. No claim that AVX-512 is faster than AVX2 — the retained
phase-e measurement says the opposite on this court. No direct-display claim
(Phases S–T unsupported pending hardware path verification). CUDA equality
is claimed only on hosts with a genuine NVIDIA device; elsewhere it is `not
evaluated on this host`.

## What Phase I added (scalar inverse compiler)

- **Assets** (`inverse::asset::Asset`): flat raster inputs in their own code
  space (gray/RGBA), hostile-safe.
- **Detectors** (`inverse::detect`): constant, 2-color periodic stripes and
  checker, palette-band (equal-width run cycles), tiled (minimal wrap
  period), horizontal/vertical ramps, bilinear corner fit, and the always-
  present literal fallback.  Proposals are measured, not trusted: each is
  materialized and diffed against the asset into a canonical sparse-overwrite
  residual with a byte-exact closure gate.
- **Pareto-first selection** (`inverse::frontier`): non-dominated candidates
  over four deterministic axes (persistent bytes, residual bytes,
  host-independent materialization-work model, search work); wall-clock
  latency is receipted but never decides dominance, so frontiers are
  reproducible.  `min_total_bytes` is the default profile.
- **Evidence**: the phase-i gate unbakes seven asset classes (constant /
  checker / palette-bands / tiled / gradient ramp / bilinear / hash-noise
  negative control): exact families win at ~140–170 B persistent with zero
  residual where detected, the noise control falls back to literal, and all
  winners reproduce their assets byte-for-byte.

## What each phase must add before completion

- H, I: implemented above.
- J: exact structural reuse / fingerprints (duplicate patches, affine and
  symmetry factoring, seeded-field search).
- K–M: SIMD / Rayon / CUDA batched inverse search.
- N: hierarchical residual factorization + DSFB zero-authority governor.
- O: 100+ asset public corpus with per-asset license/hash manifest, split
  TRAIN/VALIDATION/TEST, negative controls.
- P–R: no-mandatory-re-baking runtime court, arbitrary-observation-time
  courts, tile/scanline/band demand-materialization courts.
- S–T: experimental CUDA↔Vulkan external-resource and scanout-synchronous
  courts where hardware permits (currently unsupported/not evaluated).
- U: bounded deterministic 3D subset with glTF/open assets.
- V: quality-bounded materialization profile (never described with
  exact-language).
- W: heterogeneous backend autotuning on frozen train/validation/test.
- X: full evidence/Pareto seal (`vole-gfx seal`).

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
| H | Procedural state, generators, trajectories | pending | – |
| I–N | Inverse compiler (scalar/SIMD/Rayon/CUDA, DSFB) | pending | – |
| O | Corpus (100+ assets) + negative controls | pending | – |
| P–R | No-rebake / observation / partial courts | pending | – |
| S–T | CUDA↔Vulkan, direct display | unsupported (hardware gate) | – |
| U | Bounded 3D subset | pending | – |
| V | Quality-bounded profile | pending | – |
| W | Heterogeneous autotuning (train/val/test) | pending | – |
| X | Full evidence/Pareto seal | pending | – |

## What Phases A–G implemented (committed, tested, receipted)

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
- **Tests**: ~130 unit + integration tests (unit, conformance vectors,
  differential backends, scalar materialize, adversarial/property) plus
  criterion benches; fmt + clippy `-D warnings` clean on default and `cuda`
  features.

## Explicit non-claims (as of Phase G)

No claim of inverse compilation or unbaking (Phases I–N pending). No claim
that any generator explains pixels. No claim that CUDA beats CPU or vice
versa beyond the exact receipted domains (byte-kernel parity and the
cpu-backend-matrix rows on this host). No claim of procedural *computing*
from storage savings (no-rebake court is Phase P). No direct-display claim
(Phases S–T unsupported pending hardware path verification). No claim that
AVX-512 is faster than AVX2 — the retained measurement says the opposite on
this court. CUDA equality is claimed only on hosts with a genuine NVIDIA
device; elsewhere it is `not evaluated on this host`.

## What each phase must add before completion

- H: seeded procedural generator objects (`O = Γ(U, s, θ)`) with canonical
  parameters, bounded cost estimators, exact deterministic evaluation,
  scalar + accelerated materialization, and direct seeded evaluation (no
  object re-baking).
- I–N: the inverse procedural compiler ("unbaking") with structural reuse,
  Pareto accounting, residual-guided factorization, SIMD/Rayon/CUDA search
  acceleration, and the DSFB governor with zero authority over correctness.
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

# VOLE-GFX

**Deterministic procedural visual state as a graphics intermediate representation.**

VOLE-GFX implements, as one native-Rust crate, the architecture disclosed in
*VOLE: Procedural Video Storage and Transport by Deterministic State
Materialization* (broad prior-art disclosure, DOI 10.5281/zenodo.22284396) as a
2D graphics system:

- **Persistent procedural state** `G` is authoritative; a **raster frame is a
  materialized view**, never the primary stored object.
- State advances by deterministic **transitions** `G' = Φ(U, G, Δ)`.
- **Observation requests** `q` carry an explicit time coordinate and spatial
  domain; materializing a full surface is one optional request shape among
  many (`Sample`, `Tile`, `Scanline`, `DisplayBand`, `Viewport`, …).
- **Residual closure** `Y* = M(U, G, q) ⊕_ρ R_q` makes reconstruction exact and
  explicit; residual algebra is never implied.
- One **exact universe** (`vole.gfx.u1`, integer/fixed-point arithmetic only)
  is the single semantics shared by every backend:
  scalar reference → AVX2 → AVX-512 → Rayon → CUDA. Accelerated backends must
  reproduce the scalar output **byte-for-byte**, enforced by canonical SHA-256
  output hashes.
- **Inverse procedural compilation** ("unbaking") explains baked raster assets
  as bounded procedural state + residual, with Pareto accounting, and never
  requires re-expanding the procedural representation before use.
- Every empirical claim is scoped to an immutable JSON **evidence receipt**;
  the claims ledger and README never outrun the receipts.

## Repository layout (one crate)

```
src/            modules (universe, limits, fixed, color, ir, state,
                observation, materialize, residual, procedural, inverse,
                evidence, corpus, io, cuda, direct)
src/bin/vole-gfx.rs   principal CLI
tests/          conformance, differential, adversarial integration tests
benches/        criterion benches
examples/       courts (deterministic, receipt-emitting)
docs/           design + implementation-state documents
evidence/       receipts (immutable), claims ledger, generated reports
corpus/         manifests and (license-clean) assets
```

## Current status

See [`docs/IMPLEMENTATION_STATE.md`](docs/IMPLEMENTATION_STATE.md) for the
exact per-phase status and the claims ledger (`evidence/claims.json`) for the
machine-readable version. In short, implemented, tested and receipted:

- Phase A–B: exact U1 model, canonical hostile-input-safe IR, scalar
  materializer oracle, residual closure, pinned conformance vectors;
- Phase C–F: dependency-indexed blocking, then AVX2 / AVX-512 / Rayon
  backends that reproduce the oracle byte-for-byte (canonical SHA-256);
- Phase G: Rust device code compiled to PTX (`nvptx64-nvidia-cuda`) and
  loaded through a minimal libcuda binding, byte-parity on real NVIDIA
  hardware — no CUDA C++.

Pending: procedural generator objects (H), the inverse procedural compiler
(I–N), the public corpus (O), the runtime courts (P–R), and the remaining
hardware-dependent phases (S–X).

## Quick start

```sh
cargo test                 # unit + conformance + adversarial tests
cargo run --release -- universe
cargo run --release -- example --help
```

## Claim discipline

Read [`docs/NON_CLAIMS.md`](docs/NON_CLAIMS.md). VOLE-GFX is a research system:
it reports where it wins, where it is competitive, and where it loses — and it
never converts a skipped test or an unsupported path into a pass.

# Implementation state

Machine-readable status: the `evidence/claims.json` ledger and the JSON
receipts under `evidence/receipts/` are authoritative; this document is a
human-readable digest that follows them.

Status legend: implemented / experimental / unsupported / future.

## Starting state (recorded before implementation)

Empty repository: `research/` (paper sources, ignored by git) and
`vole-gfx-prompt.txt` (ignored by git). No git history existed; the repository
was initialized during Phase A.

## Phase receipts

| Phase | Name | Status | Receipt |
|---|---|---|---|
| A | Exact U1 model, IR, parser, limits, scalar 2D semantics | implemented | phase-a |
| B | Canonical conformance vectors, exact residual closure | next | – |
| C | Blocking + dependency indexing | implemented | phase-c |
| D/E | AVX2 / AVX-512 backends | implemented | phase-d/e |
| F | Rayon + SIMD | implemented | phase-f |
| G | Rust CUDA PTX device + host | pending | – |
| H | Procedural state, generators, trajectories | pending | – |
| I–N | Inverse compiler (scalar/SIMD/Rayon/CUDA, DSFB) | pending | – |
| O | Corpus (100+ assets) + negative controls | pending | – |
| P–R | No-rebake / observation / partial courts | pending | – |
| S–T | CUDA↔Vulkan, direct display | unsupported (hardware gate) | – |
| U | Bounded 3D subset | pending | – |
| V | Quality-bounded profile | pending | – |
| W | Heterogeneous autotuning (train/val/test) | pending | – |
| X | Full evidence/Pareto seal | pending | – |

## What Phase A implemented

- **Exact universe `vole.gfx.u1`**: Q16.16 fixed-point geometry, sample-center
  coverage, half-open rectangles, widened intermediates, overflow-safe
  placement validation (`limits.rs`, `fixed.rs`, `color.rs`).
- **Color semantics**: Gray8/RGBA8; straight-alpha exact Porter–Duff `over`
  with three exact floor-division primitives shared by every future backend
  (`div255`, `div255_24`, `div_small`); exhaustive sweeps over their domains.
- **Canonical IR**: fixed-width little-endian wire format, versioned header,
  strict section order, semantic validation (references, canonical ordering,
  live-instance uniqueness, overflow guards). Decoder never panics and fails
  closed with deterministic `Reject` codes.
- **State**: immutable objects (raster / palette / indexed raster), instances,
  trajectories (exact rational linear interpolation), timeline transitions via
  a replayable cursor, palette patches, clipping, layers.
- **Observations**: `ObservationRequest { time_ns, view, domain, sampling,
  format }` with the full domain family.
- **Scalar materializer (semantic oracle)**: per-sample composition in RGBA
  domain, structural fill/copy/move ops over the request-time instance base,
  residual closure on canonical code values, counters for receipt accounting,
  deterministic output hashing.
- **Residual module**: sparse-overwrite and XOR algebras with canonical sorted
  records and payload validation.
- **Evidence system**: environment snapshot, JSON receipt schema with
  content-derived run ids and self-hash verification, claims ledger model.
- **CLI**: `universe`, `inspect`, `validate`, `canonicalize`, `materialize`,
  `evidence verify`, `encode` (JSON authoring pending).
- **Tests**: 91 passing (unit + integration), including fill-boundary
  semantics, layer order, clipping, partial-domain ≡ full-surface-crop,
  trajectory time queries, copy/move, residual closure, deterministic hashing.

## Explicit non-claims (Phase A)

No claim of performance. No AVX2/AVX-512/Rayon/CUDA backend yet. No inverse
compiler. No corpus. The scalar materializer cost model is intentionally
`O(samples × instances + ops)`; Phase C introduces indexing and blocking.

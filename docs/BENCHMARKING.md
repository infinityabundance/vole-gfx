# Benchmarking methodology

Status: methodology fixed (prompt §41–43); harness lands with the backend
phases. Rules:

- CPU microbenchmarks: criterion (warmups, enough samples, confidence
  intervals); deterministic instruction/cache analysis (iai-callgrind) where
  available.
- GPU: CUDA events for device time; separate host end-to-end monotonic
  timing; Nsight Compute/Systems where available. Device time and end-to-end
  latency are reported separately, never conflated.
- Backend matrix: scalar, AVX2, AVX-512, Rayon-scalar, Rayon-AVX2,
  Rayon-AVX-512, CUDA — each for forward materialization and inverse search;
  unsupported backends are marked unsupported, never zeroed or omitted.
- Pareto reports per workload; no collapsing into a single "×-faster" number;
  losing workloads are retained.
- Train/validation/test corpus discipline: no tuning on TEST.

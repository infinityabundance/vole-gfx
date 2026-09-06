# CUDA

Status: **pending** (Phase G onward). Device code will be written in Rust and
compiled to PTX with the `nvptx64-nvidia-cuda` Tier-2 target
(`--no-default-features --features cuda-device -Zbuild-std=core`,
`--crate-type=cdylib`), pinned to a nightly; host interaction is feature-gated
(`cuda`). CUDA C++ via NVRTC is not a permitted substitute unless a precise
blocker is documented and the phase is left explicitly incomplete.

## Host environment observed at Phase A

- NVIDIA GeForce RTX 4080 (16 GB), driver 610.57.04, CUDA UMD 13.3.
- rustc stable 1.98.0; multiple pinned nightlies installed (selected when the
  device build lands).
- PTX/SM targets will be recorded in receipts (`sm_89`-family for the 4080
  plus a conservative baseline).

## Device-module constraints

`no_std`, no heap unless deliberate, no panic/unwind, bounded indexing, POD
layouts, compile-time semantics constants; the semantic core (`limits`,
`fixed`, `color`) is written no_std-clean and shared by construction.

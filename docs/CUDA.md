# CUDA

Status: **implemented (byte-kernel parity, Phase G)**; full primitive parity is
Phase G/H onward work.  Device code is written in **Rust** and compiled to PTX
with the `nvptx64-nvidia-cuda` Tier-2 target from **this same crate**
(`src/cuda/kernels.rs` via `scripts/ptx_shim.rs`, pinned nightly
`nightly-2026-07-24`, `-Z ub-checks=no -C overflow-checks=no`,
`extern "ptx-kernel"` for `.visible .entry` emission, `-C target-cpu=sm_89`).
Host driver interaction is a narrow `libloading` binding to `libcuda`
(`cuda::host`, feature `cuda`). CUDA C++ via NVRTC is not a permitted
substitute unless a precise blocker is documented and the phase is left
explicitly incomplete.

## Host environment (observed and receipted)

- NVIDIA GeForce RTX 4080 SUPER, driver 610.57.04, CUDA UMD 13.3.
- rustc stable 1.98.0 (host); pinned nightly-2026-07-24 (device/PTX build).
- PTX/SM/driver versions are recorded in the `phase-g` receipts, not prose.

## Context ownership (host)

A CUDA context is only *current* on the thread that set it.  `cuda::host::Module`
owns one context and binds it per call: every device-touching method runs
inside a `ContextGuard` that makes the module's context current on the calling
thread (`cuCtxSetCurrent`) and restores the thread's previous current context
on drop.  `Module::load` pops the new context off the creating thread before
returning, so no thread is ever left with the module's context current after a
call.  `Send`/`Sync` on `Module` are justified solely by that per-call binding
(see the SAFETY notes in `src/cuda/host.rs`); the caller must not drop a
`Module` while another thread is inside a guarded call.

## Device-module constraints

`no_std`, no heap unless deliberate, no panic/unwind, bounded indexing, POD
layouts, compile-time semantics constants; the semantic core (`limits`,
`fixed`, `color`) is written no_std-clean and shared by construction.

## Next steps

Generator evaluation kernels (direct seeded evaluation, Phase H), the CUDA
inverse-search batching (Phase M), and the external-resource/display path
(Phases S–T, hardware-gated).  Every claim stays scoped to receipts.

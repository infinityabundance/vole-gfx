//! CUDA host/device layers (Phase G onward).
//!
//! * Device code is written in Rust and compiled to PTX with the
//!   `nvptx64-nvidia-cuda` target from **this same crate** (`kernels`).
//! * Host interaction is feature-gated (`cuda` feature + `cudarc`); the host
//!   module is compiled only when the feature is enabled.
//! * Receipts record rustc/LLVM/PTX/SM/driver versions; CUDA equality is only
//!   claimed on hosts with a genuine NVIDIA device (never mocked).

// The device-only module tree (compiled for the nvptx target).
#[cfg(feature = "cuda-device")]
pub mod kernels;

// Host-side modules (ordinary Rust).
#[cfg(all(not(feature = "cuda-device"), feature = "cuda"))]
pub mod host;

/// Host status descriptor (unconditional, no CUDA dependency).
pub fn host_status() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda-host-enabled"
    } else {
        "cuda-host-disabled (feature 'cuda' not enabled)"
    }
}

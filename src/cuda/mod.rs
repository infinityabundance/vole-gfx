//! CUDA host/device layers (PHASE G onward).
//!
//! The device code is written in Rust and compiled to PTX with the
//! `nvptx64-nvidia-cuda` target; host interaction is feature-gated (`cuda`
//! feature).  Status: not implemented.

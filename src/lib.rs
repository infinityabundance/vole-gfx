//! # VOLE-GFX
//!
//! Deterministic procedural visual state as a graphics intermediate
//! representation: one mathematical semantics, many execution widths.
//!
//! * Exact universe `vole.gfx.u1` (integer/fixed arithmetic only), canonical
//!   binary IR, hostile-input-safe parser with hard limits.
//! * Persistent procedural state (`G`), explicit transitions (`Φ`), explicit
//!   observation requests with time coordinates (`q`), a scalar materializer
//!   (`M`) that is the semantic oracle, and explicit residual closure (`R`).
//! * Accelerated backends (AVX2, AVX-512, Rayon, CUDA) must reproduce the
//!   scalar output byte-for-byte, enforced by canonical output hashes.
//! * An inverse procedural compiler ("unbaking") with Pareto-frontier
//!   accounting, plus an evidence system of immutable JSON receipts and a
//!   claims ledger whose language never outruns the receipts.
//!
//! One crate.  Modules, not subcrates.
#![deny(unsafe_op_in_unsafe_fn)]
#![cfg_attr(feature = "cuda-device", no_std)]
#![cfg_attr(feature = "cuda-device", feature(abi_ptx, core_intrinsics))]

#[cfg(not(feature = "cuda-device"))]
extern crate std;

#[cfg(feature = "cuda-device")]
extern crate alloc;

// no_std-friendly semantic core (reused by the CUDA device build)
pub mod color;
pub mod fixed;
pub mod limits;

// host-only modules
#[cfg(not(feature = "cuda-device"))]
pub mod corpus;
#[cfg(not(feature = "cuda-device"))]
pub mod evidence;
#[cfg(not(feature = "cuda-device"))]
pub mod hash;
#[cfg(not(feature = "cuda-device"))]
pub mod inverse;
#[cfg(not(feature = "cuda-device"))]
pub mod io;
#[cfg(not(feature = "cuda-device"))]
pub mod ir;
#[cfg(not(feature = "cuda-device"))]
pub mod materialize;
#[cfg(not(feature = "cuda-device"))]
pub mod observation;
#[cfg(not(feature = "cuda-device"))]
pub mod procedural;
#[cfg(not(feature = "cuda-device"))]
pub mod residual;
#[cfg(not(feature = "cuda-device"))]
pub mod state;
#[cfg(not(feature = "cuda-device"))]
pub mod universe;

// Device-capable submodule tree (selected pieces compile to PTX).
#[cfg(feature = "cuda-device")]
pub mod device_only {
    //! Everything the PTX build needs lives here (see `cuda/device.rs`).
}

#[cfg(not(feature = "cuda-device"))]
pub mod cuda;
#[cfg(not(feature = "cuda-device"))]
pub mod direct;

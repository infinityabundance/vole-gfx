//! Materialization.
//!
//! The materializer answers observation requests against resolved procedural
//! state (`Scene`).  The scalar backend in `scalar` is the **semantic
//! oracle**: every accelerated backend must reproduce its output byte for
//! byte, verified by canonical output hashes.

pub mod blocked;
pub mod blocks;
pub mod dispatch;
pub mod scalar;

pub use blocked::{BlockMaterializer, BlockMetrics};
pub use scalar::{
    Counters, Materialized, materialize_document, materialize_pixel, materialize_scene,
};

/// Profile tags used by receipts and dispatch.
pub mod profile {
    /// Exact U1 materialization.
    pub const EXACT_U1: &str = "exact-u1";
}

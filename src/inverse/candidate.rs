//! Candidate costs and exact residual construction (Phase I).
//!
//! ```text
//! H -> materialize -> predicted A_hat      (exact profile, full surface)
//! R_H = A (-_rho) A_hat                    (sparse-overwrite residual)
//! ```
//!
//! The candidate's cost accounting is complete: persistent bytes (canonical
//! document), residual bytes, deterministic materialization-work model, and
//! search work.  Residual closure is part of the candidate document, so
//! materializing the candidate reproduces the asset byte-for-byte by
//! construction (gated in tests).

use super::asset::Asset;
use crate::color::ColorFormat;
use crate::fixed::IRect;
use crate::ir::{Document, Event, Instance, Op};

/// A generated explanation: a self-contained document whose materialization
/// (with residual closure) reproduces the asset exactly, plus its costs.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Detector name, e.g. `constant`, `stripe-x`, `literal`.
    pub name: String,
    pub doc: Document,
    /// Canonical bytes of the generator document (no residual).
    pub persistent_bytes: u64,
    pub residual_bytes: u64,
    pub search_work: u64,
    /// Deterministic materialization work model: `samples x per-sample cost`
    /// where the per-sample cost sums the placed instances' generator work
    /// units (+1 per raster).  Host-independent; this is the frontier axis
    /// (measured wall-clock `materialize_ns` is reported separately in
    /// receipts but never decides dominance).
    pub materialize_work: u64,
    /// Measured scalar-oracle materialization latency (ns; informational).
    pub materialize_ns: u64,
    /// Canonical SHA-256 of the materialized, residual-closed output.
    pub output_hash: String,
}

/// Compute the sparse-overwrite residual of `predicted` vs the asset and
/// return the canonical payload.  Iteration is row-major (one record per
/// differing *sample*, holding the whole code value) so the payload is
/// canonically sorted by (y, x) with no duplicate coordinates.
pub fn sparse_difference(asset: &Asset, predicted: &[u8]) -> Vec<u8> {
    debug_assert_eq!(predicted.len(), asset.data.len());
    let bps = asset.format.bytes_per_sample();
    let samples = asset.sample_count() as usize;
    let mut payload = Vec::new();
    // count differing samples first (canonical payload header)
    let mut count = 0u64;
    for s in 0..samples {
        let off = s * bps;
        if asset.data[off..off + bps] != predicted[off..off + bps] {
            count += 1;
        }
    }
    payload.extend_from_slice(&count.to_le_bytes());
    if count == 0 {
        return payload;
    }
    for s in 0..samples {
        let off = s * bps;
        if asset.data[off..off + bps] != predicted[off..off + bps] {
            let x = (s % asset.w as usize) as u32;
            let y = (s / asset.w as usize) as u32;
            crate::residual::push_record(&mut payload, x, y, &asset.data[off..off + bps]);
        }
    }
    payload
}

/// Attach a full-surface sparse residual binding (asset format) to a doc at
/// t=0.  Candidates carry at most one timeline event (their residual), so
/// the strictly-increasing event-time rule holds by construction and the
/// binding applies to any t>=0 request (the compiler evaluates at t=0).
pub fn attach_residual(doc: &mut Document, region: IRect, format: ColorFormat, payload: Vec<u8>) {
    doc.events.push(Event {
        t: 0,
        ops: vec![Op::BindResidual(crate::ir::ResidualBind {
            algebra: crate::ir::residual_algebra::SPARSE_OVERWRITE,
            region,
            format,
            payload,
        })],
    });
}

/// Helper: one instance of `object` at the origin with the given order/layer.
pub fn origin_instance(object: u32, order: u32, layer: u32) -> Instance {
    Instance {
        object,
        order,
        layer,
        transform: crate::fixed::Affine::identity(),
        palette: None,
        clip: None,
        trajectory: 0,
        visible: true,
    }
}

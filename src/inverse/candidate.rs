//! Candidate costs and exact residual construction (Phase I).
//!
//! ```text
//! H -> materialize -> predicted A_hat      (exact profile, full surface)
//! R_H = A (-_rho) A_hat                    (sparse-overwrite residual)
//! ```
//!
//! The candidate's cost accounting is complete and definitional:
//!
//! * `persistent_bytes` — canonical bytes of the generator-only document
//!   (no residual);
//! * `residual_bytes` — canonical bytes of the residual binding, measured as
//!   `encode(doc_with_residual).len() - persistent_bytes`, so it includes the
//!   event/op/algebra/region/format/length field overhead, not just the raw
//!   payload (this matters most for tiny residuals);
//! * `materialize_work` — the deterministic materialization-work model;
//! * `search` — the deterministic `SearchCounter` of the detector that found
//!   the explanation (its frozen `total_units()` is the frontier axis).
//!
//! Therefore `persistent_bytes + residual_bytes` is exactly the canonical
//! size of the stored candidate document.  Residual closure is part of the
//! candidate document, so materializing the candidate reproduces the asset
//! byte-for-byte by construction (gated in tests).

use super::asset::Asset;
use super::work::SearchCounter;
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
    /// Canonical bytes of the residual binding (payload + structural
    /// overhead): `encode(doc_with_residual) - persistent_bytes`.
    pub residual_bytes: u64,
    /// Deterministic search work of the detector that produced this
    /// explanation (frozen `total_units()` conversion; see `work.rs`).
    pub search: SearchCounter,
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

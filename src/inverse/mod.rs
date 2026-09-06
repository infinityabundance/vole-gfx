//! Inverse procedural compiler ("unbaking"), Phase I (scalar).
//!
//! `A -> (Gamma, s, theta, R)`: a raster asset is explained as bounded
//! procedural state plus explicit residual, measured against a Pareto
//! frontier over complete costs (persistent bytes, residual bytes,
//! materialization cost, search work).  The compiler never claims to recover
//! an author's intent — it seeks bounded explanations that win under the
//! declared cost model, and the literal raster is always a candidate
//! (fallback), so the frontier always contains at least one exact,
//! residual-free explanation.
//!
//! Determinism: proposals are enumerated in a fixed order with bounded work;
//! frontier order is detector order.  Exactness: every surviving candidate
//! reproduces the asset byte-for-byte once residual closure is applied
//! (gated by tests).  No whole-asset expansion is required for later
//! observation of a winning candidate: its document materializes directly.

pub mod asset;
pub mod candidate;
pub mod detect;
pub mod frontier;
pub mod structure;

use crate::fixed::IRect;
use crate::ir::{Document, Object};
use crate::limits::Reject;
use asset::Asset;
use candidate::Candidate;
use detect::{literal_object, one_object_doc, propose};
use frontier::Frontier;

/// Deterministic materialization-work model of a candidate document over a
/// surface of `samples` requested samples: `samples x per-sample cost`, where
/// the per-sample cost sums every placed instance's generator work units
/// (+1 for raster content).  Host-independent; used as the frontier axis
/// instead of measured wall-clock time.
pub fn materialize_work_model(doc: &Document, samples: u64) -> u64 {
    let mut per_sample = 0u64;
    for inst in &doc.instances {
        let o = doc.objects.get(inst.object as usize);
        let cost = match o {
            Some(Object::GeneratorField {
                family,
                w,
                h,
                params,
            }) => crate::procedural::decode_field(*family, params, *w, *h)
                .map(|f| f.work())
                .unwrap_or(1),
            _ => 1,
        };
        per_sample += cost.max(1);
    }
    samples.saturating_mul(per_sample.max(1))
}

/// Finalize one proposal into a measured candidate: materialize, compute the
/// exact sparse residual, attach it, and verify the residual-closed output
/// reproduces the asset byte-for-byte.
fn finalize(name: &str, doc0: &Document, asset: &Asset, work: u64) -> Result<Candidate, Reject> {
    crate::ir::validate::validate(doc0)?;
    // Persistent cost = the generator document only (residual is a separate
    // axis so the frontier exposes the representation tradeoff).
    let persistent_bytes = crate::ir::encode::encode(doc0).len() as u64;
    let materialize_work = materialize_work_model(doc0, asset.sample_count());
    let req =
        crate::observation::ObservationRequest::full_surface(0, asset.w, asset.h, asset.format);
    let t0 = std::time::Instant::now();
    let m = crate::materialize::scalar::materialize_document(doc0, &req)?;
    let gen_ns = t0.elapsed().as_nanos() as u64;

    let mut doc = doc0.clone();
    let payload = candidate::sparse_difference(asset, &m.output.data);
    let n_records = u64::from_le_bytes(payload[..8].try_into().unwrap());
    let residual_bytes = if n_records == 0 {
        0
    } else {
        payload.len() as u64
    };
    let mut total_ns = gen_ns;
    let output_hash = if n_records > 0 {
        // attach the residual and re-materialize (closure gate)
        candidate::attach_residual(
            &mut doc,
            IRect::new(0, 0, asset.w as i32, asset.h as i32),
            asset.format,
            payload,
        );
        let t1 = std::time::Instant::now();
        let m2 = crate::materialize::scalar::materialize_document(&doc, &req)?;
        total_ns = gen_ns + t1.elapsed().as_nanos() as u64;
        if m2.output.data != asset.data {
            return Err(Reject::PayloadMismatch);
        }
        m2.output.canonical_hash().to_hex()
    } else {
        m.output.canonical_hash().to_hex()
    };
    crate::ir::validate::validate(&doc)?;
    Ok(Candidate {
        name: name.to_string(),
        doc,
        persistent_bytes,
        residual_bytes,
        search_work: work,
        materialize_work,
        materialize_ns: total_ns,
        output_hash,
    })
}

/// Unbake an asset into the non-dominated explanation set.
///
/// Always includes the literal fallback (an exact, residual-free candidate),
/// so the result is never empty and the frontier lets a profile choose
/// between representation size and materialization cost.
pub fn unbake(asset: &Asset) -> Result<Frontier, Reject> {
    let mut f = Frontier::default();
    for p in propose(asset) {
        f.insert(finalize(p.name, &p.doc, asset, p.work)?);
    }
    // literal fallback last (deterministic order)
    let doc = one_object_doc(literal_object(asset));
    f.insert(finalize("literal", &doc, asset, 1)?);
    Ok(f)
}

/// Convenience: unbake and return (frontier, preferred candidate) where the
/// preferred candidate is the profile selection (min total bytes).
pub fn unbake_best(asset: &Asset) -> Result<(Frontier, Candidate), Reject> {
    let f = unbake(asset)?;
    let best = f
        .min_total_bytes()
        .cloned()
        .expect("literal fallback always present");
    Ok((f, best))
}

/// Deterministic summary of a frontier for receipts (name rows).
pub fn summarize(f: &Frontier) -> Vec<(String, u64, u64, u64, u64)> {
    f.rows()
}

/// Bounded work budget of the scalar compiler per asset (court guard).
pub fn max_search_work(asset: &Asset) -> u64 {
    // detectors are O(w*h) per proposal, proposals are O(families); bound is
    // generous but deterministic
    32 * asset.sample_count() + 4096
}

/// Convenience conversion helpers shared by courts/tests.
pub fn asset_from_object(o: &Object) -> Result<Asset, Reject> {
    match o {
        Object::Raster { format, w, h, data } => Asset::new(*w, *h, *format, data.clone()),
        _ => Err(Reject::InvalidIndex),
    }
}

/// Re-exports for courts/tests.
pub use frontier::Costs as Objective;

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
//! frontier order is detector order.  Search work is **counted exactly where
//! it occurs** (a `SearchCounter` per detector; see `work`), so the search
//! axis of the frontier reflects real deterministic scan operations (period
//! re-scans, flood-fill probes, crop byte compares), not a flat sample
//! count.  Exactness: every surviving candidate reproduces the asset
//! byte-for-byte once residual closure is applied (gated by tests).  No
//! whole-asset expansion is required for later observation of a winning
//! candidate: its document materializes directly.

pub mod asset;
pub mod candidate;
pub mod detect;
pub mod frontier;
pub mod structure;
pub mod work;

use crate::fixed::IRect;
use crate::ir::{Document, Object};
use crate::limits::Reject;
use asset::Asset;
use candidate::Candidate;
use detect::{literal_object, one_object_doc, propose};
use frontier::Frontier;
use work::SearchCounter;

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
///
/// Byte accounting is definitional: `persistent_bytes` is the canonical size
/// of the generator-only document, and `residual_bytes` is the canonical
/// delta of the residual-bound document over it (so it includes the binding's
/// event/op/algebra/region/format/length field overhead, not just the raw
/// payload).  `persistent_bytes + residual_bytes` is therefore exactly the
/// canonical size of the stored candidate document.
fn finalize(
    name: &str,
    doc0: &Document,
    asset: &Asset,
    search: SearchCounter,
) -> Result<Candidate, Reject> {
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

    let payload = candidate::sparse_difference(asset, &m.output.data);
    let n_records = u64::from_le_bytes(payload[..8].try_into().unwrap());
    let mut doc = doc0.clone();
    let mut residual_bytes = 0u64;
    let mut total_ns = gen_ns;
    let output_hash = if n_records > 0 {
        // attach the residual and re-materialize (closure gate)
        candidate::attach_residual(
            &mut doc,
            IRect::new(0, 0, asset.w as i32, asset.h as i32),
            asset.format,
            payload,
        );
        // The complete representation is the residual-bound document; the
        // residual axis is its canonical delta over the generator-only
        // document, capturing the binding's structural overhead.
        residual_bytes =
            (crate::ir::encode::encode(&doc).len() as u64).saturating_sub(persistent_bytes);
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
        search,
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
    // literal fallback last (deterministic order); its search cost is zero:
    // the fallback is always present and requires no scan to produce
    let doc = one_object_doc(literal_object(asset));
    f.insert(finalize("literal", &doc, asset, SearchCounter::default())?);
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

/// Deterministic summary of a frontier for receipts: each candidate's four
/// declared cost axes plus the search-work breakdown (see `frontier::Row`).
pub fn summarize(f: &Frontier) -> Vec<frontier::Row> {
    f.rows()
}

/// Deterministic search-work budget of the scalar compiler per asset — a
/// court guard, not an adversarial worst-case proof.  Detectors count real
/// operations (see `work`), and the dominant terms are bounded by the
/// documented caps: the tiled detector re-scans the plane for candidate
/// periods up to `detect::PERIOD_SCAN_LIMIT`, and structural probes are a
/// small multiple of the surface plus crop compares bounded by the 512px
/// crop cap.  The formula below is a generous, host-independent multiple of
/// those terms for court-scale assets; courts may enforce it and fall back.
pub fn max_search_work(asset: &Asset) -> u64 {
    let samples = asset.sample_count();
    let periods = detect::PERIOD_SCAN_LIMIT as u64 + 1;
    // period re-scans (2 directions) + structural probes + crop-compare
    // headroom for court-scale surfaces (<= 512x512 assets)
    samples
        .saturating_mul(periods.saturating_mul(2) + 64)
        .saturating_add(64 << 20)
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

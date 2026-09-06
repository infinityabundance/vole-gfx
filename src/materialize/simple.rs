//! The simple-composition engine: an algorithmic specialization shared by the
//! scalar-simple ablation path and the SIMD backends.
//!
//! The engine proves eligibility of a whole document for the *opaque,
//! integer-translation subset*: every placed instance must be a fully opaque
//! raster placed at an integer pixel translation with identity linear part,
//! every structural op must be an opaque fill (residual bindings are always
//! allowed), and instance formats must match the request format.  Inside that
//! subset, composition reduces to ordered memory writes — opaque `over`
//! replaces the destination, and samples outside a placed instance keep their
//! previous value — which is byte-for-byte what the scalar oracle computes.
//!
//! The engine performs no arithmetic on samples: byte-moving kernels
//! (`copy_row`, `fill_row`) are injected, so *any* kernel set that copies or
//! fills exactly preserves equivalence to the oracle.  This is what makes the
//! measurement decomposition honest:
//!
//! ```text
//! scalar oracle      general per-sample composition (the semantic spec)
//! scalar blocked     oracle-equivalent, block + dependency indexed
//! scalar simple      this engine with scalar kernels   <- algorithmic
//!                     specialization gain is measured HERE
//! avx2 / avx512      this engine with SIMD kernels     <- SIMD gain is
//!                     measured relative to scalar simple
//! ```
//!
//! Documents outside the eligible subset return `None` and the dispatcher
//! falls back to the blocked/scalar paths; courts record which path ran.

use crate::color::ColorFormat;
use crate::fixed::RectF;
use crate::limits::Reject;
use crate::materialize::blocks::BlockShape;
use crate::materialize::scalar::{Counters, Materialized};
use crate::observation::{ObservationRequest, Output};
use crate::state::{PlacedInstance, Scene, StructOp};

/// Row-copy / row-fill kernels used by the simple-composition engine.  All
/// kernels must be exact copies/fills — they only move bytes, so the engine's
/// equivalence to the oracle is preserved for any kernel set.
pub struct Kernels {
    pub copy_row: fn(&mut [u8], &[u8]),
    pub fill_row: fn(&mut [u8], &[u8]),
}

/// Per-object content classification inside the engine: an object is either
/// drawn from stored rows (opaque raster) or is an opaque constant generator
/// field, which draws as an ordered fill of one request-format code.
/// Anything else makes the document ineligible.
#[derive(Clone, Copy)]
enum ObjUse {
    Rows,
    /// Request-format code bytes of the constant color (length 4; the
    /// engine uses `bytes_per_sample` of the request).
    ConstFill([u8; 4]),
}

/// Is the geometry of this instance eligible: integer-pixel translation,
/// identity linear part, and a plain raster or generator-field object
/// (content/opacity checked separately against the per-object cache)?
fn geometry_eligible(inst: &PlacedInstance<'_>) -> bool {
    let tr = inst.tr;
    if tr.x & 0xFFFF != 0 || tr.y & 0xFFFF != 0 {
        return false;
    }
    if inst.a != 1 << 16 || inst.d != 1 << 16 || inst.b != 0 || inst.c != 0 {
        return false;
    }
    matches!(
        inst.object(),
        crate::ir::Object::Raster { .. } | crate::ir::Object::GeneratorField { .. }
    )
}

/// Compute opaque-row source offset for gray vs rgba objects.
fn object_row<'o>(inst: &'o PlacedInstance<'_>, oy: i32) -> Option<&'o [u8]> {
    match inst.object() {
        crate::ir::Object::Raster {
            format: ColorFormat::Rgba8,
            w,
            data,
            ..
        } => {
            let ow = *w as usize;
            Some(&data[(oy as usize) * ow * 4..][..ow * 4])
        }
        crate::ir::Object::Raster {
            format: ColorFormat::Gray8,
            w,
            data,
            ..
        } => {
            let ow = *w as usize;
            Some(&data[(oy as usize) * ow..][..ow])
        }
        _ => None,
    }
}

fn object_h(inst: &PlacedInstance<'_>) -> i32 {
    match inst.object() {
        crate::ir::Object::Raster { h, .. } => *h as i32,
        crate::ir::Object::GeneratorField { h, .. } => *h as i32,
        _ => 0,
    }
}

fn object_w(inst: &PlacedInstance<'_>) -> i32 {
    match inst.object() {
        crate::ir::Object::Raster { w, .. } => *w as i32,
        crate::ir::Object::GeneratorField { w, .. } => *w as i32,
        _ => 0,
    }
}

/// Plain scalar kernels (no SIMD).  This is the **scalar-simple** ablation
/// row: identical engine and eligibility as the SIMD paths, scalar stores
/// only, so the court can decompose algorithmic-specialization gain from
/// SIMD gain.
pub fn scalar_kernels() -> Kernels {
    Kernels {
        copy_row: |d: &mut [u8], s: &[u8]| d.copy_from_slice(s),
        fill_row: |d: &mut [u8], c: &[u8]| {
            for px in d.chunks_exact_mut(c.len()) {
                px.copy_from_slice(c);
            }
        },
    }
}

/// Scalar-simple materialization path (ablation / reference row for the SIMD
/// engines).  Returns `None` when the document is outside the eligible
/// subset.
pub fn materialize_simple_scalar(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
) -> Result<Option<Materialized>, Reject> {
    simple_impl(scene, req, shape, &scalar_kernels())
}

/// The shared simple-composition engine.
pub(crate) fn simple_impl(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    _shape: BlockShape,
    k: &Kernels,
) -> Result<Option<Materialized>, Reject> {
    req.validate()?;
    let target_gray = req.format == ColorFormat::Gray8;
    let b = req.domain.bounds();
    let (w, h) = (b.width() as u32, b.height() as u32);
    if w == 0 || h == 0 {
        return Err(Reject::Degenerate);
    }
    let (ox, oy) = (b.x0, b.y0);

    // structural ops must all be opaque fills
    for op in &scene.ops {
        match op {
            StructOp::FillRect { color, .. } if color.a == 255 => {}
            StructOp::BindResidual { .. } => {}
            _ => return Ok(None),
        }
    }

    // per-object content cache; every instance must pass
    let mut object_use: Vec<Option<ObjUse>> = vec![None; scene.doc().objects.len()];
    let classify = |o: usize, fmt: ColorFormat| -> Option<ObjUse> {
        match &scene.doc().objects[o] {
            crate::ir::Object::Raster { format, data, .. } => {
                let fmt_ok = if target_gray {
                    *format == ColorFormat::Gray8
                } else {
                    *format == ColorFormat::Rgba8
                };
                let opaque = if *format == ColorFormat::Rgba8 {
                    data.as_chunks::<4>().0.iter().all(|px| px[3] == 255)
                } else {
                    true
                };
                (fmt_ok && opaque).then_some(ObjUse::Rows)
            }
            crate::ir::Object::GeneratorField {
                family: f,
                w: _,
                h: _,
                params,
            } if *f == crate::procedural::family::CONSTANT => {
                // An opaque constant field draws exactly like a fill of one
                // color; the request-format code is produced by the same
                // conversion the oracle applies to a composed sample.  Other
                // families stay ineligible for the fast path in Phase H
                // (recorded; they run on the blocked/scalar path).
                let field = crate::procedural::decode_field(*f, params, 0, 0).ok()?;
                match field {
                    crate::procedural::Field::Constant(p) if p.color.a == 255 => {
                        let (code, _) = crate::materialize::scalar::encode_code_pub(p.color, fmt);
                        Some(ObjUse::ConstFill(code))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    };
    for inst in &scene.instances {
        if !geometry_eligible(inst) {
            return Ok(None);
        }
        let o = inst.object as usize;
        if object_use[o].is_none() {
            object_use[o] = classify(o, req.format);
        }
        if object_use[o].is_none() {
            return Ok(None);
        }
    }

    let mut out = Output::new(w, h, req.format)?;
    let mut counters = Counters::default();
    let bps = req.format.bytes_per_sample();

    // Composition: instance draws (the request-time instance base) first, in
    // layer/insertion order; opaque fills are structural ops and therefore
    // paint *over* the base, in op order — exactly the oracle's ordered
    // opaque `over`.  Fill codes are converted exactly like the oracle
    // converts a composed sample to the request format.
    for inst in &scene.instances {
        let ox0 = inst.tr.x >> 16;
        let oy0 = inst.tr.y >> 16;
        let iy0 = oy0.max(oy);
        let iy1 = (oy0 + object_h(inst)).min(oy + h as i32);
        // clip rect in scene px (conservative integer bounds)
        let cpx = inst.clip.map(|c| c.px_bounds());
        // constant generator instances draw as ordered row fills; raster
        // instances draw as row copies.
        let fill_code: Option<[u8; 4]> = match object_use[inst.object as usize] {
            Some(ObjUse::ConstFill(code)) => Some(code),
            _ => None,
        };
        let mut drawn_rows = 0u64;
        for y in iy0..iy1 {
            let oy_local = y - oy0;
            let row = object_row(inst, oy_local);
            let mut sx = ox0;
            let mut ex = ox0 + object_w(inst);
            if let Some(cp) = cpx {
                sx = sx.max(cp.x0);
                ex = ex.min(cp.x1);
            }
            sx = sx.max(ox);
            ex = ex.min(ox + w as i32);
            if ex <= sx {
                continue;
            }
            let d = ((y - oy) as usize) * (w as usize) * bps + ((sx - ox) as usize) * bps;
            let n = ((ex - sx) as usize) * bps;
            match fill_code {
                Some(code) => {
                    (k.fill_row)(&mut out.data[d..d + n], &code[..bps]);
                }
                None => {
                    let row = row.expect("eligible raster object");
                    let s = ((sx - ox0) as usize) * bps;
                    (k.copy_row)(&mut out.data[d..d + n], &row[s..s + n]);
                }
            }
            drawn_rows += 1;
        }
        if drawn_rows > 0 {
            counters.instance_draws += 1;
        }
    }
    let mut fill_code: [u8; 4] = [0; 4];
    let fill_len = req.format.bytes_per_sample();
    for op in &scene.ops {
        if let StructOp::FillRect { rect, color } = op {
            let (code, _) = crate::materialize::scalar::encode_code_pub(*color, req.format);
            fill_code[..fill_len].copy_from_slice(&code[..fill_len]);
            fill_rect_rows(
                &mut out.data,
                w,
                h,
                ox,
                oy,
                bps,
                rect,
                &fill_code[..fill_len],
                k,
            );
        }
    }

    // residual closure (unchanged rule)
    for op in &scene.ops {
        if let StructOp::BindResidual {
            algebra,
            region,
            format,
            payload,
        } = op
        {
            if *format != req.format {
                counters.residual_format_skips += 1;
                continue;
            }
            crate::materialize::scalar::apply_residual_pub(
                *algebra,
                region,
                req.format,
                payload,
                &mut out.data,
                w,
                h,
                ox,
                oy,
                &mut counters,
            )?;
        }
    }
    counters.samples = w as u64 * h as u64;
    Ok(Some(Materialized {
        output: out,
        counters,
    }))
}

// Row-fill helper: rect math + row loops; kernels do the byte stores.
#[allow(clippy::too_many_arguments)]
fn fill_rect_rows(
    out: &mut [u8],
    w: u32,
    h: u32,
    ox: i32,
    oy: i32,
    bps: usize,
    rect: &RectF,
    color: &[u8],
    k: &Kernels,
) {
    let pb = rect.px_bounds();
    let x0 = pb.x0.max(ox);
    let x1 = pb.x1.min(ox + w as i32);
    let y0 = pb.y0.max(oy);
    let y1 = pb.y1.min(oy + h as i32);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let n = (x1 - x0) as usize;
    for y in y0..y1 {
        let d = ((y - oy) as usize) * (w as usize) * bps + ((x0 - ox) as usize) * bps;
        (k.fill_row)(&mut out[d..d + n * bps], color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_kernels_are_exact() {
        let k = scalar_kernels();
        let mut dst = vec![9u8; 32];
        let src = vec![7u8; 32];
        (k.copy_row)(&mut dst, &src);
        assert_eq!(dst, src);
        let mut d2 = vec![0u8; 48];
        (k.fill_row)(&mut d2, &[1, 2, 3, 4]);
        assert!(d2.as_chunks::<4>().0.iter().all(|px| px == &[1, 2, 3, 4]));
    }
}

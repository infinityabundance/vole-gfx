//! AVX2 backend (Phase D).
//!
//! Strategy: exact kernels over the primitives that profiling shows dominate
//! the court workloads — row blits of fully-opaque, integer-translation
//! instances and row fills of opaque colors.  In the eligible subset these
//! reduce composition to ordered memory writes, which is byte-for-byte what
//! the scalar oracle computes (opaque `over` replaces the destination, and
//! samples outside a placed instance keep their previous value).  All other
//! documents fall back (`None`) to the block/scalar paths; the dispatcher and
//! courts record which path ran.  SIMD never changes arithmetic — there is no
//! arithmetic here beyond exact copies, which is exactly why the subset is
//! provably equivalent.

use crate::color::ColorFormat;
use crate::fixed::RectF;
use crate::limits::Reject;
use crate::materialize::blocks::BlockShape;
use crate::materialize::scalar::{Counters, Materialized};
use crate::observation::{ObservationRequest, Output};
use crate::state::{PlacedInstance, Scene, StructOp};

/// Is the geometry of this instance eligible: integer-pixel translation,
/// identity linear part, and a plain raster object (format/opacity checked
/// separately against the per-object cache)?
fn geometry_eligible(inst: &PlacedInstance<'_>) -> bool {
    let tr = inst.tr;
    if tr.x & 0xFFFF != 0 || tr.y & 0xFFFF != 0 {
        return false;
    }
    if inst.a != 1 << 16 || inst.d != 1 << 16 || inst.b != 0 || inst.c != 0 {
        return false;
    }
    matches!(inst.object(), crate::ir::Object::Raster { .. })
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

/// Row-copy / row-fill kernels used by the simple-composition engine.  All
/// kernels must be exact copies/fills — they only move bytes, so the engine's
/// equivalence to the oracle is preserved for any kernel set.
pub(crate) struct Kernels {
    pub(crate) copy_row: fn(&mut [u8], &[u8]),
    pub(crate) fill_row: fn(&mut [u8], &[u8]),
}

/// Attempt the exact SIMD simple-composition path for an eligible document.
/// Returns `None` when the document is outside the eligible subset (caller
/// falls back to the block/scalar path).
pub fn materialize_simple(
    scene: &Scene<'_>,
    req: &ObservationRequest,
    shape: BlockShape,
) -> Result<Option<Materialized>, Reject> {
    if has_avx2() {
        let k = Kernels {
            copy_row: copy_row_avx2,
            fill_row: fill_row_avx2,
        };
        simple_impl(scene, req, shape, &k)
    } else {
        // Exact scalar fallback (same engine, plain copies).  The dispatcher
        // normally does not route here without AVX2; this keeps the function
        // total and correct on any host.
        let k = Kernels {
            copy_row: |d: &mut [u8], s: &[u8]| d.copy_from_slice(s),
            fill_row: |d: &mut [u8], c: &[u8]| {
                for px in d.chunks_exact_mut(c.len()) {
                    px.copy_from_slice(c);
                }
            },
        };
        simple_impl(scene, req, shape, &k)
    }
}

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

    // per-object opacity + format cache; all instances must pass
    let mut object_ok: Vec<Option<bool>> = vec![None; scene.doc().objects.len()];
    for inst in &scene.instances {
        if !geometry_eligible(inst) {
            return Ok(None);
        }
        let o = inst.object as usize;
        let ok = match object_ok[o] {
            Some(v) => v,
            None => {
                let v = match &scene.doc().objects[o] {
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
                        fmt_ok && opaque
                    }
                    _ => false,
                };
                object_ok[o] = Some(v);
                v
            }
        };
        if !ok {
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
        let mut drawn_rows = 0u64;
        for y in iy0..iy1 {
            let oy_local = y - oy0;
            let row = object_row(inst, oy_local).expect("eligible object");
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
            let s = ((sx - ox0) as usize) * bps;
            let n = ((ex - sx) as usize) * bps;
            (k.copy_row)(&mut out.data[d..d + n], &row[s..s + n]);
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

fn object_h(inst: &PlacedInstance<'_>) -> i32 {
    match inst.object() {
        crate::ir::Object::Raster { h, .. } => *h as i32,
        _ => 0,
    }
}
fn object_w(inst: &PlacedInstance<'_>) -> i32 {
    match inst.object() {
        crate::ir::Object::Raster { w, .. } => *w as i32,
        _ => 0,
    }
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

/// x86 detection at the coarse dispatch boundary (never per pixel).
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

/// Exact 32-byte SIMD copy; tail handled byte-wise.  Result identical to a
/// plain copy (no arithmetic involved), so no semantic divergence is
/// possible; SIMD only reduces instruction count and memory ops.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_row_avx2_kernel(dst: &mut [u8], src: &[u8]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: caller guarantees dst/src are valid disjoint slices of
        // length n; both sides are in-bounds for 32-byte loads/stores.
        let v = unsafe { _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i) };
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, v) };
        i += 32;
    }
    while i < n {
        dst[i] = src[i];
        i += 1;
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn copy_row_avx2_kernel(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
}

fn copy_row_avx2(dst: &mut [u8], src: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: dst/src are disjoint slices of length n (caller
            // guarantees non-overlap: dst is the output buffer, src is an
            // object row).
            unsafe { copy_row_avx2_kernel(dst, src) };
            return;
        }
    }
    dst.copy_from_slice(src);
}

/// Exact 32-byte SIMD fill of a row slice already sized to `pattern` repeats;
/// tail handled byte-wise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fill_row_avx2_kernel(dst: &mut [u8], pattern: &[u8]) {
    use core::arch::x86_64::*;
    let bps = pattern.len();
    let n = dst.len();
    if bps == 4 {
        let p = u32::from_le_bytes([pattern[0], pattern[1], pattern[2], pattern[3]]);
        let v = _mm256_set1_epi32(p as i32);
        let mut i = 0usize;
        while i + 32 <= n {
            // SAFETY: dst is a mutable slice with >= 32 bytes remaining.
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, v) };
            i += 32;
        }
        while i < n {
            dst[i] = pattern[i % 4];
            i += 1;
        }
    } else {
        for px in dst.chunks_exact_mut(bps) {
            px.copy_from_slice(pattern);
        }
    }
}

fn fill_row_avx2(dst: &mut [u8], pattern: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: dst is a mutable slice of the output buffer.
            unsafe { fill_row_avx2_kernel(dst, pattern) };
            return;
        }
    }
    let bps = pattern.len();
    for px in dst.chunks_exact_mut(bps) {
        px.copy_from_slice(pattern);
    }
}

/// Dispatch decision string used by receipts/courts.
pub fn dispatch_name(decision: bool) -> &'static str {
    if decision { "avx2" } else { "blocked-scalar" }
}

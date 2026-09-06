//! The scalar reference materializer — the semantic oracle of U1.
//!
//! Design rules:
//! * One output sample is one call to the composition function at the
//!   sample-center scene point; the loop is deliberately naive (readable,
//!   auditable) — acceleration is other backends' job.
//! * Composition happens in an RGBA straight-alpha domain; Gray8 output is
//!   produced by exact luma conversion over opaque black at the end.
//! * Structural ops (fill/copy/move) execute in event order **after** the
//!   instance draws, and copy/move read the *prefix* surface (state before
//!   that op), which bounds dependency depth by construction.
//! * Residual bindings are applied last, in binding order, on the canonical
//!   code values of the requested format, clipped to the request domain.

use crate::color::{ColorFormat, Rgba, over, rgba_to_gray};
use crate::fixed::{IRect, Vec2};
use crate::ir::Object;
use crate::limits::Reject;
use crate::observation::{ObservationRequest, Output};
use crate::state::{PlacedInstance, Scene, StructOp};

/// Depth bound for copy/move prefix chains (hard, adversarial-safe).
const COPY_CHAIN_LIMIT: u32 = 256;

/// Per-sample bound on fold-frame evaluations (copy/move prefix resolutions).
/// Stops pathological self-covering copy stacks with a deterministic error
/// instead of exponential work.
const FOLD_FRAME_BUDGET: u32 = 1 << 16;

/// Deterministic accounting counters emitted with every materialization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    /// Output samples written.
    pub samples: u64,
    /// Instance bounds/clip rejections performed.
    pub instance_tests: u64,
    /// Instances whose pixel was actually composited.
    pub instance_draws: u64,
    /// Structural ops applied.
    pub ops_applied: u64,
    /// Residual records applied.
    pub residual_records: u64,
    /// Residual bindings skipped due to format mismatch with the request.
    pub residual_format_skips: u64,
}

/// Result of a materialization.
#[derive(Debug, Clone)]
pub struct Materialized {
    pub output: Output,
    pub counters: Counters,
}

/// Validate and materialize an observation request against a document.
pub fn materialize_document(
    doc: &crate::ir::Document,
    req: &ObservationRequest,
) -> Result<Materialized, Reject> {
    crate::ir::validate::validate(doc)?;
    let scene = crate::state::Scene::resolve(doc, req.time_ns)?;
    materialize_scene(&scene, req)
}

/// Materialize a request against an already-resolved scene.
pub fn materialize_scene(
    scene: &Scene<'_>,
    req: &ObservationRequest,
) -> Result<Materialized, Reject> {
    req.validate()?;
    debug_assert_eq!(req.view, crate::observation::View::identity());
    let b = req.domain.bounds();
    let mut counters = Counters {
        samples: 0,
        ..Default::default()
    };

    if let crate::observation::Domain::IrregularSamples(pts) = &req.domain {
        // Output is one packed sample per requested point, in request
        // order.  Residual closure is applied per sample point.
        let n = pts.len();
        if n == 0 {
            return Err(Reject::Degenerate);
        }
        let mut out = Output::new(n as u32, 1, req.format)?;
        for (k, &(x, y)) in pts.iter().enumerate() {
            let rgba = compose_pixel(scene, x, y, &mut counters)?;
            let code = encode_code_pub(rgba, req.format);
            let off = k * req.format.bytes_per_sample();
            write_code(&mut out.data, off, code);
            counters.samples += 1;
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
                    apply_residual_at_pub(
                        *algebra,
                        region,
                        req.format,
                        payload,
                        x,
                        y,
                        &mut out.data[off..off + req.format.bytes_per_sample()],
                        &mut counters,
                    )?;
                }
            }
        }
        return Ok(Materialized {
            output: out,
            counters,
        });
    }

    let (w, h) = (b.width() as u32, b.height() as u32);
    if w == 0 || h == 0 {
        return Err(Reject::Degenerate);
    }
    let mut out = Output::new(w, h, req.format)?;

    // Composition per sample.
    for j in 0..h {
        let y = b.y0 + j as i32;
        let row_off = (j as usize) * w as usize;
        for i in 0..w {
            let x = b.x0 + i as i32;
            let rgba = compose_pixel(scene, x, y, &mut counters)?;
            let code = encode_code_pub(rgba, req.format);
            write_code(
                &mut out.data,
                (row_off + i as usize) * req.format.bytes_per_sample(),
                code,
            );
            counters.samples += 1;
        }
    }

    // Residual closure pass (pixel-domain, code-space, clipped to request).
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
            apply_residual_pub(
                *algebra,
                region,
                req.format,
                payload,
                &mut out.data,
                w,
                h,
                b.x0,
                b.y0,
                &mut counters,
            )?;
        }
    }
    Ok(Materialized {
        output: out,
        counters,
    })
}

/// Compose the value of one surface pixel `(x, y)` (scene pixel coords).
pub fn materialize_pixel(scene: &Scene<'_>, x: i32, y: i32) -> Result<(Rgba, Counters), Reject> {
    let mut counters = Counters::default();
    let rgba = compose_pixel(scene, x, y, &mut counters)?;
    Ok((rgba, counters))
}

/// Compose the full RGBA value at a pixel (scene coordinates).
fn compose_pixel(scene: &Scene<'_>, x: i32, y: i32, c: &mut Counters) -> Result<Rgba, Reject> {
    let cx = Vec2::sample_center(x, y);
    let mut budget = FOLD_FRAME_BUDGET;
    fold_ops(scene, cx, scene.ops.len(), 0, &mut budget, c)
}

/// Value of the instance-composed surface at scene point `cx` (before any
/// structural op of this request).
fn draw_base(scene: &Scene<'_>, cx: Vec2, depth: u32, c: &mut Counters) -> Result<Rgba, Reject> {
    if depth > COPY_CHAIN_LIMIT {
        return Err(Reject::DependencyTooDeep);
    }
    let mut cur = Rgba::TRANSPARENT;
    for inst in &scene.instances {
        c.instance_tests += 1;
        let Some(local) = sample_instance(scene, inst, cx) else {
            continue;
        };
        c.instance_draws += 1;
        cur = over(local, cur);
    }
    Ok(cur)
}

/// Evaluate structural ops `0..upto` at scene point `cx`, seeded with the
/// instance-composed base.  Copy/move recurse into the prefix surface.
/// `budget` bounds total fold frames per sample (bounded execution).
fn fold_ops(
    scene: &Scene<'_>,
    cx: Vec2,
    upto: usize,
    depth: u32,
    budget: &mut u32,
    c: &mut Counters,
) -> Result<Rgba, Reject> {
    if depth > COPY_CHAIN_LIMIT {
        return Err(Reject::DependencyTooDeep);
    }
    *budget = budget
        .checked_sub(1)
        .ok_or(Reject::ExecutionBudgetExceeded)?;
    let mut cur = draw_base(scene, cx, depth, c)?;
    for (k, op) in scene.ops[..upto].iter().enumerate() {
        match op {
            StructOp::FillRect { rect, color } => {
                if covers(cx, rect) {
                    cur = over(*color, cur);
                    c.ops_applied += 1;
                }
            }
            StructOp::CopyRegion { src, dx, dy } => {
                if covers_shifted(cx, src, *dx, *dy) {
                    // destination pixel: copy from prefix surface at (cx - d)
                    let from = cx - Vec2::new(*dx, *dy);
                    cur = fold_ops(scene, from, k, depth + 1, budget, c)?;
                    c.ops_applied += 1;
                }
            }
            StructOp::MoveRegion { src, dx, dy } => {
                if covers_shifted(cx, src, *dx, *dy) {
                    let from = cx - Vec2::new(*dx, *dy);
                    cur = fold_ops(scene, from, k, depth + 1, budget, c)?;
                    c.ops_applied += 1;
                } else if covers(cx, src) {
                    // Source region of a move is cleared: structural ops mutate
                    // a raster layer initialized once from the request-time
                    // instance draws (they are not instance-state changes).
                    cur = Rgba::TRANSPARENT;
                    c.ops_applied += 1;
                }
            }
            StructOp::BindResidual { .. } => {}
        }
    }
    Ok(cur)
}

/// Does scene point `cx` fall inside half-open rect `r` (sample-center rule)?
#[inline(always)]
fn covers(cx: Vec2, r: &crate::fixed::RectF) -> bool {
    let x = cx.x as i64;
    let y = cx.y as i64;
    x >= r.x0 as i64 && x < r.x1 as i64 && y >= r.y0 as i64 && y < r.y1 as i64
}

/// Does `cx` fall inside `r` shifted by `(dx, dy)`?
#[inline(always)]
fn covers_shifted(cx: Vec2, r: &crate::fixed::RectF, dx: i32, dy: i32) -> bool {
    let x = cx.x as i64 - dx as i64;
    let y = cx.y as i64 - dy as i64;
    x >= r.x0 as i64 && x < r.x1 as i64 && y >= r.y0 as i64 && y < r.y1 as i64
}

/// Sample one placed instance at scene point `cx`.  Returns `None` when the
/// instance does not cover the sample (bounds, clip, or local out-of-range).
fn sample_instance(scene: &Scene<'_>, inst: &PlacedInstance<'_>, cx: Vec2) -> Option<Rgba> {
    if let Some(clip) = inst.clip {
        if !((cx.x as i64) >= (clip.x0 as i64)
            && (cx.x as i64) < (clip.x1 as i64)
            && (cx.y as i64) >= (clip.y0 as i64)
            && (cx.y as i64) < (clip.y1 as i64))
        {
            return None;
        }
    }
    // Fast conservative reject: sample pixel outside the transformed extent.
    if let Some(b) = inst.bounds_px {
        let px = (cx.x >> 16, cx.y >> 16);
        if !b.contains_px(px.0, px.1) {
            return None;
        }
    }
    let aff = inst.affine();
    let (lx, ly) = inv_sample_pub(&aff, cx)?;
    let obj = inst.object();
    match obj {
        Object::Raster {
            format: ColorFormat::Rgba8,
            w,
            h,
            data,
        } => {
            let (i, j) = in_bounds_pub(lx, ly, *w, *h)?;
            let o = ((j * w) + i) as usize * 4;
            Some(Rgba::from_bytes([
                data[o],
                data[o + 1],
                data[o + 2],
                data[o + 3],
            ]))
        }
        Object::Raster {
            format: ColorFormat::Gray8,
            w,
            h,
            data,
        } => {
            let (i, j) = in_bounds_pub(lx, ly, *w, *h)?;
            let g = data[(j * w + i) as usize];
            Some(Rgba::gray(g))
        }
        Object::IndexedRaster { w, h, indices, .. } => {
            let (i, j) = in_bounds_pub(lx, ly, *w, *h)?;
            let idx = indices[(j * w + i) as usize];
            let pal_id = inst.palette.unwrap_or_else(|| match obj {
                Object::IndexedRaster { pal, .. } => *pal,
                _ => unreachable!(),
            });
            scene.palette(pal_id)?.get(idx)
        }
        Object::Palette { .. } | Object::GeneratorField { .. } => None,
    }
}

/// Exact inverse affine sample: local pixel coordinates (i64, range-guarded)
/// or `None` for degenerate transforms.
pub(crate) fn inv_sample_pub(aff: &crate::fixed::Affine, p: Vec2) -> Option<(i64, i64)> {
    let a = aff.a as i64;
    let b = aff.b as i64;
    let c = aff.c as i64;
    let d = aff.d as i64;
    let det = a * d - b * c;
    if det == 0 {
        return None;
    }
    let u = p.x as i64 - aff.tx as i64;
    let v = p.y as i64 - aff.ty as i64;
    let nx = d * u - b * v;
    let ny = -c * u + a * v;
    // local px = floor(num / det) with exact floor division
    let lx = floor_div(nx, det);
    let ly = floor_div(ny, det);
    // Guard: object dims are at most MAX_OBJECT_DIM; anything far outside is
    // certainly out of range, and this keeps the i32 casts safe.
    if lx < -1
        || ly < -1
        || lx > crate::limits::MAX_OBJECT_DIM as i64 + 1
        || ly > crate::limits::MAX_OBJECT_DIM as i64 + 1
    {
        return None;
    }
    Some((lx, ly))
}

/// Exact floor division for a signed numerator and non-zero signed denom.
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && (r < 0) != (b < 0) {
        q - 1
    } else {
        q
    }
}

pub(crate) fn in_bounds_pub(lx: i64, ly: i64, w: u32, h: u32) -> Option<(u32, u32)> {
    if lx >= 0 && ly >= 0 && lx < w as i64 && ly < h as i64 {
        Some((lx as u32, ly as u32))
    } else {
        None
    }
}

/// Convert a composed RGBA sample into the request format's code value
/// (4 bytes; `Gray8` uses the low byte).
pub(crate) fn encode_code_pub(c: Rgba, format: ColorFormat) -> ([u8; 4], usize) {
    match format {
        ColorFormat::Rgba8 => (c.to_bytes(), 4),
        ColorFormat::Gray8 => {
            // composite over opaque black, then luma
            let c = over(c, Rgba::OPAQUE_BLACK);
            let g = rgba_to_gray(c).0;
            ([g, 0, 0, 0], 1)
        }
    }
}

fn write_code(dst: &mut [u8], off: usize, code: ([u8; 4], usize)) {
    let (bytes, n) = code;
    dst[off..off + n].copy_from_slice(&bytes[..n]);
}

pub(crate) fn write_code_pub(dst: &mut [u8], off: usize, code: ([u8; 4], usize)) {
    write_code(dst, off, code);
}

/// Apply one residual binding to the output buffer.
#[allow(clippy::too_many_arguments)] // domain-clip context; refactor with court struct later
pub(crate) fn apply_residual_pub(
    algebra: u8,
    region: &IRect,
    format: ColorFormat,
    payload: &[u8],
    dst: &mut [u8],
    w: u32,
    h: u32,
    ox: i32,
    oy: i32,
    c: &mut Counters,
) -> Result<(), Reject> {
    let bps = format.bytes_per_sample();
    let mut r = crate::ir::wire::Reader::new(payload);
    let count = r.u64().map_err(|_| Reject::Truncated)?;
    if count > crate::limits::MAX_RESIDUAL_RECORDS {
        return Err(Reject::CountExceedsLimit);
    }
    let (bx0, by0, bx1, by1) = (
        region.x0 as i64,
        region.y0 as i64,
        region.x1 as i64,
        region.y1 as i64,
    );
    let (dx0, dy0, dx1, dy1) = (
        ox as i64,
        oy as i64,
        ox as i64 + w as i64,
        oy as i64 + h as i64,
    );
    for _ in 0..count {
        let x = r.u32().map_err(|_| Reject::Truncated)?;
        let y = r.u32().map_err(|_| Reject::Truncated)?;
        let val = r.bytes(bps).map_err(|_| Reject::Truncated)?;
        let (x, y) = (x as i64, y as i64);
        if x < bx0 || x >= bx1 || y < by0 || y >= by1 {
            continue;
        }
        if x < dx0 || x >= dx1 || y < dy0 || y >= dy1 {
            continue;
        }
        let o = ((y - dy0) as usize * w as usize + (x - dx0) as usize) * bps;
        apply_record(algebra, &mut dst[o..o + bps], val, c);
    }
    if !r.done() {
        return Err(Reject::PayloadMismatch);
    }
    Ok(())
}

/// Apply one residual binding to a single sample slot `(x, y)` (used by the
/// irregular-sample path).  The record scan is linear; fine for the scalar
/// oracle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_residual_at_pub(
    algebra: u8,
    region: &IRect,
    format: ColorFormat,
    payload: &[u8],
    x: i32,
    y: i32,
    slot: &mut [u8],
    c: &mut Counters,
) -> Result<(), Reject> {
    let bps = format.bytes_per_sample();
    let mut r = crate::ir::wire::Reader::new(payload);
    let count = r.u64().map_err(|_| Reject::Truncated)?;
    if count > crate::limits::MAX_RESIDUAL_RECORDS {
        return Err(Reject::CountExceedsLimit);
    }
    let (bx0, by0, bx1, by1) = (
        region.x0 as i64,
        region.y0 as i64,
        region.x1 as i64,
        region.y1 as i64,
    );
    for _ in 0..count {
        let rx = r.u32().map_err(|_| Reject::Truncated)?;
        let ry = r.u32().map_err(|_| Reject::Truncated)?;
        let val = r.bytes(bps).map_err(|_| Reject::Truncated)?;
        let (rx, ry) = (rx as i64, ry as i64);
        if rx < bx0 || rx >= bx1 || ry < by0 || ry >= by1 {
            continue;
        }
        if rx == x as i64 && ry == y as i64 {
            apply_record(algebra, slot, val, c);
        }
    }
    if !r.done() {
        return Err(Reject::PayloadMismatch);
    }
    Ok(())
}

#[inline]
fn apply_record(algebra: u8, slot: &mut [u8], val: &[u8], c: &mut Counters) {
    match algebra {
        crate::residual::algebra::SPARSE_OVERWRITE => slot.copy_from_slice(val),
        crate::residual::algebra::XOR => {
            for k in 0..slot.len() {
                slot[k] ^= val[k];
            }
        }
        _ => unreachable!("validated algebra"),
    }
    c.residual_records += 1;
}

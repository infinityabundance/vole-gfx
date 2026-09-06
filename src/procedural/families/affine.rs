//! Affine-reuse field: the object's local pixel lattice is mapped through a
//! bounded affine into a referenced raster/indexed object, which is then
//! sampled with the oracle's exact raster rule.
//!
//! Exact rule: local sample center `p = (i + 0.5, j + 0.5)` (Q16.16) maps to
//! `q = affine(p)` (widened integer arithmetic, floor), and the referenced
//! object is sampled at ref-local integer pixel `(floor(q.x), floor(q.y))`;
//! out-of-bounds (or a degenerate/guarded far-out result) yields `outside`.
//! The identity affine therefore reproduces the referenced object
//! pixel-for-pixel, and integer translations shift it exactly.

use super::Refs;
use crate::color::Rgba;
use crate::fixed::{Affine, HALF, fp};
use crate::ir::wire::{Reader, put_bytes, put_i32, put_u32};
use crate::limits::Reject;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    /// Referenced object table index (must be a Raster or IndexedRaster).
    pub object: u32,
    /// Local -> referenced-object affine (Q16.16, U1 range-checked).
    pub affine: Affine,
    pub outside: Rgba,
}

pub const WORK: u64 = 16;

pub fn encode(p: &Params, w: &mut Vec<u8>) {
    put_u32(w, p.object);
    put_i32(w, p.affine.a);
    put_i32(w, p.affine.b);
    put_i32(w, p.affine.tx);
    put_i32(w, p.affine.c);
    put_i32(w, p.affine.d);
    put_i32(w, p.affine.ty);
    put_bytes(w, &p.outside.to_bytes());
}

pub fn decode(r: &mut Reader<'_>) -> Result<Params, Reject> {
    let object = r.u32()?;
    let a = r.i32()?;
    let b = r.i32()?;
    let tx = r.i32()?;
    let c = r.i32()?;
    let d = r.i32()?;
    let ty = r.i32()?;
    let affine = Affine { a, b, tx, c, d, ty };
    if !affine.in_range() {
        return Err(Reject::CoefficientOutOfRange);
    }
    let ob: [u8; 4] = r.bytes(4)?.try_into().unwrap();
    Ok(Params {
        object,
        affine,
        outside: Rgba::from_bytes(ob),
    })
}

pub fn work(_p: &Params) -> u64 {
    WORK
}

/// Exact guarded forward map of the local sample center through the affine.
/// Returns ref-local integer pixel coordinates, or `None` when the result is
/// far outside any plausible object extent (keeps the widening i64 math from
/// ever touching dangerous casts).  Floor division toward -inf (arithmetic
/// shift) matches the oracle's raster addressing.
fn map(p: &Params, i: u32, j: u32) -> Option<(i32, i32)> {
    let x = fp(i as i32) as i64 + HALF as i64;
    let y = fp(j as i32) as i64 + HALF as i64;
    let a = p.affine;
    // widened: a,b <= 2^20, p <= 2^30 => products <= 2^50; tx*2^16 <= 2^46.
    let qx = (a.a as i64 * x + a.b as i64 * y + a.tx as i64 * (1 << 16)) >> 16;
    let qy = (a.c as i64 * x + a.d as i64 * y + a.ty as i64 * (1 << 16)) >> 16;
    // q is continuous ref-local space in Q16.16 px; the ref pixel index is
    // floor(q / 2^16).
    let u = qx >> 16;
    let v = qy >> 16;
    let max = crate::limits::MAX_OBJECT_DIM as i64 + 1;
    if u < -1 || v < -1 || u > max || v > max {
        return None;
    }
    Some((u as i32, v as i32))
}

pub fn sample(p: &Params, i: u32, j: u32, refs: &Refs<'_>) -> Rgba {
    match map(p, i, j).and_then(|(u, v)| (refs.sample_object)(p.object, u, v)) {
        Some(c) => c,
        None => p.outside,
    }
}

/// Provably opaque when `outside` is opaque AND the referenced object is
/// fully opaque (checked by the caller against the object table).
pub fn opaque_params(p: &Params) -> bool {
    p.outside.a == 255
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ColorFormat;
    use crate::ir::{Document, Object};

    /// Sample `p` at `(i, j)` against a local doc with an inline ref closure
    /// (mirrors `Scene::sample_field`'s construction, without scene plumbing).
    fn sample_against(p: &Params, doc: &Document, i: u32, j: u32) -> Rgba {
        let f = |obj: u32, u: i32, v: i32| -> Option<Rgba> {
            let o = &doc.objects[obj as usize];
            match o {
                Object::Raster {
                    format: ColorFormat::Rgba8,
                    w,
                    h,
                    data,
                    ..
                } => {
                    if u < 0 || v < 0 || u >= *w as i32 || v >= *h as i32 {
                        return None;
                    }
                    let o = ((v as usize) * *w as usize + u as usize) * 4;
                    Some(Rgba::from_bytes([
                        data[o],
                        data[o + 1],
                        data[o + 2],
                        data[o + 3],
                    ]))
                }
                _ => None,
            }
        };
        let refs = Refs { sample_object: &f };
        sample(p, i, j, &refs)
    }

    fn red_square() -> Document {
        let mut d = Document::new();
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 2,
            h: 2,
            data: vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 9, 9, 255],
        });
        d
    }

    #[test]
    fn identity_reuses_pixel_for_pixel() {
        let d = red_square();
        let p = Params {
            object: 0,
            affine: Affine::identity(),
            outside: Rgba::OPAQUE_BLACK,
        };
        assert_eq!(sample_against(&p, &d, 0, 0), Rgba::new(255, 0, 0, 255));
        assert_eq!(sample_against(&p, &d, 1, 1), Rgba::new(9, 9, 9, 255));
    }

    #[test]
    fn translation_shifts_exactly() {
        let d = red_square();
        // shift +1 px in x: local (0,0) reads ref (1,0)
        let p = Params {
            object: 0,
            affine: Affine::from_px_translation(1, 0),
            outside: Rgba::OPAQUE_BLACK,
        };
        assert_eq!(sample_against(&p, &d, 0, 0), Rgba::new(0, 255, 0, 255));
        assert_eq!(sample_against(&p, &d, 1, 0), Rgba::OPAQUE_BLACK); // out of bounds
    }

    #[test]
    fn out_of_bounds_gives_outside_color() {
        let d = red_square();
        let p = Params {
            object: 0,
            affine: Affine::identity(),
            outside: Rgba::new(1, 1, 1, 255),
        };
        assert_eq!(sample_against(&p, &d, 5, 5), Rgba::new(1, 1, 1, 255));
    }

    #[test]
    fn roundtrip_and_range_checks() {
        let p = Params {
            object: 3,
            affine: Affine::identity(),
            outside: Rgba::WHITE,
        };
        let mut w = Vec::new();
        encode(&p, &mut w);
        let mut r = Reader::new(&w);
        assert_eq!(decode(&mut r).unwrap(), p);
        assert!(r.done());

        // out-of-range coefficient rejected
        let mut w2 = Vec::new();
        put_u32(&mut w2, 0);
        put_i32(&mut w2, 1 << 21); // |scale| 32 > cap 16
        put_i32(&mut w2, 0);
        put_i32(&mut w2, 0);
        put_i32(&mut w2, 0);
        put_i32(&mut w2, 1 << 16);
        put_i32(&mut w2, 0);
        put_bytes(&mut w2, &[0, 0, 0, 255]);
        assert_eq!(
            decode(&mut Reader::new(&w2)),
            Err(Reject::CoefficientOutOfRange)
        );
    }
}

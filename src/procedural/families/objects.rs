//! Object family: a repeated grid of referenced raster/indexed objects.
//!
//! The object's extent is partitioned into `cell_w x cell_h` cells; each cell
//! picks one object from an ordered list (row-major cycling or per-cell
//! hash), and the object is sampled at the cell-local pixel
//! `(i mod cell_w, j mod cell_h)`.  Samples outside a referenced object's own
//! size yield the `background` color (different-sized objects crop cleanly).

use super::{Refs, fmix2};
use crate::color::Rgba;
use crate::ir::wire::{Reader, put_bytes, put_u8, put_u32, put_u64};
use crate::limits::Reject;

/// Max distinct referenced objects per family.
pub const MAX_OBJECTS: u32 = 16;
/// Max cell dimension in pixels.
pub const MAX_CELL_DIM: u32 = 1 << 14;

pub const MODE_ORDERED: u8 = 1;
pub const MODE_HASH: u8 = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Params {
    pub cell_w: u32,
    pub cell_h: u32,
    pub mode: u8,
    pub seed: u64,
    /// Referenced object table indices (1..=16).
    pub objects: Vec<u32>,
    pub background: Rgba,
}

pub const WORK: u64 = 8;

fn validate(mode: u8, objects: &[u32]) -> Result<(), Reject> {
    if objects.is_empty() || objects.len() as u64 > MAX_OBJECTS as u64 {
        return Err(Reject::CountExceedsLimit);
    }
    if mode != MODE_ORDERED && mode != MODE_HASH {
        return Err(Reject::UnknownTag);
    }
    Ok(())
}

pub fn encode(p: &Params, w: &mut Vec<u8>) {
    put_u32(w, p.cell_w);
    put_u32(w, p.cell_h);
    put_u8(w, p.mode);
    put_u64(w, p.seed);
    put_u32(w, p.objects.len() as u32);
    for o in &p.objects {
        put_u32(w, *o);
    }
    put_bytes(w, &p.background.to_bytes());
}

pub fn decode(r: &mut Reader<'_>) -> Result<Params, Reject> {
    let cell_w = r.u32()?;
    let cell_h = r.u32()?;
    let mode = r.u8()?;
    let seed = r.u64()?;
    let n = r.u32()?;
    if n == 0 || n > MAX_OBJECTS {
        return Err(Reject::CountExceedsLimit);
    }
    if mode != MODE_ORDERED && mode != MODE_HASH {
        return Err(Reject::UnknownTag);
    }
    if cell_w == 0 || cell_h == 0 || cell_w > MAX_CELL_DIM || cell_h > MAX_CELL_DIM {
        return Err(Reject::DimensionTooLarge);
    }
    let mut objects = Vec::with_capacity(n as usize);
    for _ in 0..n {
        objects.push(r.u32()?);
    }
    validate(mode, &objects)?;
    let bg: [u8; 4] = r.bytes(4)?.try_into().unwrap();
    Ok(Params {
        cell_w,
        cell_h,
        mode,
        seed,
        objects,
        background: Rgba::from_bytes(bg),
    })
}

pub fn work(_p: &Params) -> u64 {
    WORK
}

fn cell_widths(w: u32, cell_w: u32) -> u64 {
    // number of cell columns covering the extent: ceil(w / cell_w)
    (w as u64).div_ceil(cell_w as u64)
}

/// Pick the referenced object for cell `(ci, cj)`.
fn pick(p: &Params, ci: u32, cj: u32, cols: u64) -> usize {
    let n = p.objects.len() as u64;
    let idx = match p.mode {
        MODE_ORDERED => {
            let m = cj as u64 * cols + ci as u64;
            m % n
        }
        MODE_HASH => fmix2(p.seed, ci, cj) % n,
        _ => unreachable!("validated mode"),
    };
    idx as usize
}

pub fn sample(p: &Params, w: u32, _h: u32, i: u32, j: u32, refs: &Refs<'_>) -> Rgba {
    let ci = i / p.cell_w.max(1);
    let cj = j / p.cell_h.max(1);
    let obj = p.objects[pick(p, ci, cj, cell_widths(w, p.cell_w))];
    let u = (i % p.cell_w) as i32;
    let v = (j % p.cell_h) as i32;
    match (refs.sample_object)(obj, u, v) {
        Some(c) => c,
        None => p.background,
    }
}

/// Provably opaque when background is opaque (referenced objects' opacity is
/// checked by the caller against the object table).
pub fn opaque_params(p: &Params) -> bool {
    p.background.a == 255
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ColorFormat;
    use crate::ir::{Document, Object};

    fn striped_doc() -> Document {
        // build two 2x1 objects: A = red pixels, B = green pixels
        let mut d = Document::new();
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 2,
            h: 1,
            data: vec![255, 0, 0, 255, 255, 0, 0, 255],
        });
        d.objects.push(Object::Raster {
            format: ColorFormat::Rgba8,
            w: 2,
            h: 1,
            data: vec![0, 255, 0, 255, 0, 255, 0, 255],
        });
        d
    }

    /// Sample `p` at `(i, j)` with an inline ref closure over `doc` (the same
    /// construction `Scene::sample_field` uses).
    fn sample_against(p: &Params, doc: &Document, w: u32, h: u32, i: u32, j: u32) -> Rgba {
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
        sample(p, w, h, i, j, &refs)
    }

    #[test]
    fn ordered_rows_cycle_objects() {
        let d = striped_doc();
        let p = Params {
            cell_w: 2,
            cell_h: 1,
            mode: MODE_ORDERED,
            seed: 0,
            objects: vec![0, 1],
            background: Rgba::OPAQUE_BLACK,
        };
        // 8x1 extent, 2px cells -> cells: [A,B,A,B]
        assert_eq!(sample_against(&p, &d, 8, 1, 0, 0).r, 255); // cell0 -> obj0
        assert_eq!(sample_against(&p, &d, 8, 1, 1, 0).r, 255);
        assert_eq!(sample_against(&p, &d, 8, 1, 2, 0).g, 255); // cell1 -> obj1
        assert_eq!(sample_against(&p, &d, 8, 1, 4, 0).r, 255); // cell2 -> obj0
        assert_eq!(sample_against(&p, &d, 8, 1, 6, 0).g, 255); // cell3 -> obj1
    }

    #[test]
    fn beyond_referenced_size_is_background() {
        let d = striped_doc();
        let p = Params {
            cell_w: 4,
            cell_h: 1,
            mode: MODE_ORDERED,
            seed: 0,
            objects: vec![0],
            background: Rgba::new(1, 1, 1, 255),
        };
        // cell is 4px wide but object is 2px wide: px 0..1 red, px 2..3 bg
        assert_eq!(
            sample_against(&p, &d, 8, 1, 0, 0),
            Rgba::new(255, 0, 0, 255)
        );
        assert_eq!(sample_against(&p, &d, 8, 1, 2, 0), Rgba::new(1, 1, 1, 255));
    }

    #[test]
    fn hash_cell_choice_deterministic() {
        let d = striped_doc();
        let p = Params {
            cell_w: 2,
            cell_h: 2,
            mode: MODE_HASH,
            seed: 99,
            objects: vec![0, 1],
            background: Rgba::OPAQUE_BLACK,
        };
        for i in 0..16u32 {
            for j in 0..16u32 {
                let a = sample_against(&p, &d, 16, 16, i, j);
                let b = sample_against(&p, &d, 16, 16, i, j);
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn roundtrip_and_rejections() {
        let p = Params {
            cell_w: 8,
            cell_h: 8,
            mode: MODE_ORDERED,
            seed: 3,
            objects: vec![0, 2],
            background: Rgba::WHITE,
        };
        let mut w = Vec::new();
        encode(&p, &mut w);
        let mut r = Reader::new(&w);
        assert_eq!(decode(&mut r).unwrap(), p);
        assert!(r.done());

        // empty object list rejected
        let mut w2 = Vec::new();
        put_u32(&mut w2, 8);
        put_u32(&mut w2, 8);
        put_u8(&mut w2, MODE_ORDERED);
        put_u64(&mut w2, 0);
        put_u32(&mut w2, 0);
        put_bytes(&mut w2, &[0, 0, 0, 255]);
        assert_eq!(
            decode(&mut Reader::new(&w2)),
            Err(Reject::CountExceedsLimit)
        );

        // bad mode rejected
        let mut w3 = Vec::new();
        put_u32(&mut w3, 8);
        put_u32(&mut w3, 8);
        put_u8(&mut w3, 7);
        put_u64(&mut w3, 0);
        put_u32(&mut w3, 1);
        put_u32(&mut w3, 0);
        put_bytes(&mut w3, &[0, 0, 0, 255]);
        assert_eq!(decode(&mut Reader::new(&w3)), Err(Reject::UnknownTag));
    }
}

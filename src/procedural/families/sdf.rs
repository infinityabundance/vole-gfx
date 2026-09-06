//! SDF field: analytic hard shapes sampled at local pixel centers.  `inside`
//! is strict (`d < 0`-style half-open rule): a sample whose center lies
//! exactly on the boundary is outside, mirroring the U1 rect coverage rule.

use crate::color::Rgba;
use crate::fixed::{HALF, fp};
use crate::ir::wire::{Reader, put_bytes, put_i32, put_u8};
use crate::limits::Reject;

pub const SHAPE_CIRCLE: u8 = 1;
pub const SHAPE_BOX: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub shape: u8,
    /// Anchor center in local Q16.16.
    pub cx: i32,
    pub cy: i32,
    /// Circle: radius (Q16.16).  Box: half width (Q16.16).
    pub p1: i32,
    /// Box: half height (Q16.16); must be 0 for circles (canonical form).
    pub p2: i32,
    pub inside: Rgba,
    pub outside: Rgba,
}

pub const WORK: u64 = 8;

fn validate(shape: u8, cx: i32, cy: i32, p1: i32, p2: i32, w: u32, h: u32) -> Result<(), Reject> {
    let (w, h) = ((w as i64) << 16, (h as i64) << 16); // extent in Q16.16
    let in_x = (cx as i64) >= 0 && (cx as i64) <= w;
    let in_y = (cy as i64) >= 0 && (cy as i64) <= h;
    if !in_x || !in_y {
        return Err(Reject::CoordinateOutOfRange);
    }
    match shape {
        SHAPE_CIRCLE => {
            if p1 < 0 || p2 != 0 {
                return Err(Reject::NonCanonicalOrder);
            }
            if (p1 as i64) > crate::limits::MAX_COORD {
                return Err(Reject::CoefficientOutOfRange);
            }
        }
        SHAPE_BOX => {
            if p1 < 0 || p2 < 0 {
                return Err(Reject::NonCanonicalOrder);
            }
            if (p1 as i64) > crate::limits::MAX_COORD || (p2 as i64) > crate::limits::MAX_COORD {
                return Err(Reject::CoefficientOutOfRange);
            }
        }
        _ => return Err(Reject::UnknownTag),
    }
    Ok(())
}

pub fn encode(p: &Params, w: &mut Vec<u8>) {
    put_u8(w, p.shape);
    put_i32(w, p.cx);
    put_i32(w, p.cy);
    put_i32(w, p.p1);
    put_i32(w, p.p2);
    put_bytes(w, &p.inside.to_bytes());
    put_bytes(w, &p.outside.to_bytes());
}

pub fn decode(r: &mut Reader<'_>, w: u32, h: u32) -> Result<Params, Reject> {
    let shape = r.u8()?;
    let cx = r.i32()?;
    let cy = r.i32()?;
    let p1 = r.i32()?;
    let p2 = r.i32()?;
    validate(shape, cx, cy, p1, p2, w, h)?;
    let bi: [u8; 4] = r.bytes(4)?.try_into().unwrap();
    let bo: [u8; 4] = r.bytes(4)?.try_into().unwrap();
    Ok(Params {
        shape,
        cx,
        cy,
        p1,
        p2,
        inside: Rgba::from_bytes(bi),
        outside: Rgba::from_bytes(bo),
    })
}

pub fn work(_p: &Params) -> u64 {
    WORK
}

/// Sample at local pixel center `(i + 0.5, j + 0.5)`.
pub fn sample(p: &Params, i: u32, j: u32) -> Rgba {
    let x = fp(i as i32) as i64 + HALF as i64;
    let y = fp(j as i32) as i64 + HALF as i64;
    let inside = match p.shape {
        SHAPE_CIRCLE => {
            let dx = x - p.cx as i64;
            let dy = y - p.cy as i64;
            let r = p.p1 as i64;
            dx * dx + dy * dy < r * r
        }
        SHAPE_BOX => {
            let dx = x - p.cx as i64;
            let dy = y - p.cy as i64;
            dx < p.p1 as i64 && dx > -(p.p1 as i64) && dy < p.p2 as i64 && dy > -(p.p2 as i64)
        }
        _ => unreachable!("validated shape"),
    };
    if inside { p.inside } else { p.outside }
}

/// Provably opaque when both colors are opaque.
pub fn opaque(p: &Params) -> bool {
    p.inside.a == 255 && p.outside.a == 255
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(cx: i32, cy: i32, r: i32) -> Params {
        Params {
            shape: SHAPE_CIRCLE,
            cx,
            cy,
            p1: r,
            p2: 0,
            inside: Rgba::WHITE,
            outside: Rgba::OPAQUE_BLACK,
        }
    }

    #[test]
    fn circle_half_open() {
        // unit disc at (2.5, 2.5) px within a 5x5 field: center pixel (2,2)
        // has center exactly at the disc center; radius 1 px covers samples
        // strictly within distance 1: (2,2) is distance 0 -> inside;
        // neighbors at distance exactly 1 -> outside (boundary excluded).
        let d = disc(fp(2) + HALF, fp(2) + HALF, fp(1));
        assert_eq!(sample(&d, 2, 2), Rgba::WHITE);
        assert_eq!(sample(&d, 1, 2), Rgba::OPAQUE_BLACK);
        assert_eq!(sample(&d, 2, 1), Rgba::OPAQUE_BLACK);
        // radius 1.5 px: diagonal neighbor distance sqrt(2)~1.414 < 1.5
        let d2 = disc(fp(2) + HALF, fp(2) + HALF, fp(1) + (fp(1) / 2));
        assert_eq!(sample(&d2, 1, 1), Rgba::WHITE);
    }

    #[test]
    fn box_strict_edges() {
        // box half extents 1 px at (2.5, 2.5): pixels with center strictly
        // inside x in (1.5, 3.5): i=2 only.
        let b = Params {
            shape: SHAPE_BOX,
            cx: fp(2) + HALF,
            cy: fp(2) + HALF,
            p1: fp(1),
            p2: fp(1),
            inside: Rgba::WHITE,
            outside: Rgba::OPAQUE_BLACK,
        };
        assert_eq!(sample(&b, 2, 2), Rgba::WHITE);
        assert_eq!(sample(&b, 1, 2), Rgba::OPAQUE_BLACK);
        assert_eq!(sample(&b, 3, 2), Rgba::OPAQUE_BLACK);
    }

    #[test]
    fn roundtrip() {
        let p = disc(fp(4), fp(4), fp(2));
        let mut w = Vec::new();
        encode(&p, &mut w);
        let mut r = Reader::new(&w);
        assert_eq!(decode(&mut r, 8, 8).unwrap(), p);
    }

    #[test]
    fn anchor_out_of_extent_rejected() {
        let mut w = Vec::new();
        put_u8(&mut w, SHAPE_CIRCLE);
        put_i32(&mut w, fp(20));
        put_i32(&mut w, fp(2));
        put_i32(&mut w, fp(1));
        put_i32(&mut w, 0);
        put_bytes(&mut w, &[255, 255, 255, 255]);
        put_bytes(&mut w, &[0, 0, 0, 255]);
        assert_eq!(
            decode(&mut Reader::new(&w), 5, 5),
            Err(Reject::CoordinateOutOfRange)
        );
    }
}

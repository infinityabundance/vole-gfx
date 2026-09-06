//! Gradient field (Phase H): axis-independent linear (two anchor points) and
//! bilinear (four corner colors) interpolation over the local pixel lattice.
//!
//! Coordinate convention: the field is sampled on the *integer local pixel
//! lattice* — pixel `(i, j)` reads the continuous gradient at lattice point
//! `(i, j)` (not `(i+0.5, j+0.5)`), so a 1:1 identity reuse reproduces an
//! existing raster pixel-for-pixel and the mapping is trivially exact under
//! integer translations.  All interpolation is exact fixed-point with floor
//! division; no floats.
//!
//! * linear: colors `c0`/`c1` at anchor pixels `p0`/`p1` (integer px inside
//!   the extent); the fraction `t = clamp(dot(p - p0, p1 - p0) / |p1 - p0|^2,
//!   0, 1)` is evaluated in widened integer arithmetic (floor) then blended.
//! * bilinear: corner colors at the extent corners `(0,0)` (c00, top-left),
//!   `(w-1,0)` (c10, top-right), `(0,h-1)` (c01, bottom-left),
//!   `(w-1,h-1)` (c11, bottom-right); degenerate single-row/column extents
//!   fall back to a one-axis blend.

use super::blend;
use crate::color::Rgba;
use crate::ir::wire::{Reader, put_bytes, put_i32, put_u8};
use crate::limits::Reject;

pub const MODE_LINEAR: u8 = 1;
pub const MODE_BILINEAR: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Linear {
        p0x: i32,
        p0y: i32,
        c0: Rgba,
        p1x: i32,
        p1y: i32,
        c1: Rgba,
    },
    Bilinear {
        c00: Rgba,
        c10: Rgba,
        c01: Rgba,
        c11: Rgba,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub mode: Mode,
}

const SCALE: u32 = 1 << 16;

/// Exact floor division of a signed numerator by a positive denominator.
fn floor_div_pos(a: i64, b: i64) -> i64 {
    debug_assert!(b > 0);
    let q = a / b;
    let r = a % b;
    if r < 0 { q - 1 } else { q }
}

fn clamp16(t: i64) -> u32 {
    t.clamp(0, SCALE as i64) as u32
}

pub fn encode(p: &Params, w: &mut Vec<u8>) {
    match p.mode {
        Mode::Linear {
            p0x,
            p0y,
            c0,
            p1x,
            p1y,
            c1,
        } => {
            put_u8(w, MODE_LINEAR);
            put_i32(w, p0x);
            put_i32(w, p0y);
            put_bytes(w, &c0.to_bytes());
            put_i32(w, p1x);
            put_i32(w, p1y);
            put_bytes(w, &c1.to_bytes());
        }
        Mode::Bilinear { c00, c10, c01, c11 } => {
            put_u8(w, MODE_BILINEAR);
            put_bytes(w, &c00.to_bytes());
            put_bytes(w, &c10.to_bytes());
            put_bytes(w, &c01.to_bytes());
            put_bytes(w, &c11.to_bytes());
        }
    }
}

pub fn decode(r: &mut Reader<'_>, w: u32, h: u32) -> Result<Params, Reject> {
    let mode = r.u8()?;
    let params = match mode {
        MODE_LINEAR => {
            let p0x = r.i32()?;
            let p0y = r.i32()?;
            let b0: [u8; 4] = r.bytes(4)?.try_into().unwrap();
            let p1x = r.i32()?;
            let p1y = r.i32()?;
            let b1: [u8; 4] = r.bytes(4)?.try_into().unwrap();
            // semantic range checks (extent-relative; kept in decode so any
            // consumer of a decoded field sees the same rejection)
            let (w, h) = (w as i64, h as i64);
            let in_x = |v: i32| (v as i64) >= 0 && (v as i64) <= w;
            let in_y = |v: i32| (v as i64) >= 0 && (v as i64) <= h;
            if !in_x(p0x) || !in_y(p0y) || !in_x(p1x) || !in_y(p1y) {
                return Err(Reject::CoordinateOutOfRange);
            }
            let dx = p1x as i64 - p0x as i64;
            let dy = p1y as i64 - p0y as i64;
            if dx * dx + dy * dy == 0 {
                return Err(Reject::Degenerate);
            }
            Mode::Linear {
                p0x,
                p0y,
                c0: Rgba::from_bytes(b0),
                p1x,
                p1y,
                c1: Rgba::from_bytes(b1),
            }
        }
        MODE_BILINEAR => {
            let b00: [u8; 4] = r.bytes(4)?.try_into().unwrap();
            let b10: [u8; 4] = r.bytes(4)?.try_into().unwrap();
            let b01: [u8; 4] = r.bytes(4)?.try_into().unwrap();
            let b11: [u8; 4] = r.bytes(4)?.try_into().unwrap();
            Mode::Bilinear {
                c00: Rgba::from_bytes(b00),
                c10: Rgba::from_bytes(b10),
                c01: Rgba::from_bytes(b01),
                c11: Rgba::from_bytes(b11),
            }
        }
        _ => return Err(Reject::UnknownTag),
    };
    Ok(Params { mode: params })
}

pub fn work(p: &Params) -> u64 {
    match p.mode {
        Mode::Linear { .. } => 16,
        Mode::Bilinear { .. } => 32,
    }
}

/// Sample the gradient at integer lattice point `(i, j)`.
pub fn sample(p: &Params, w: u32, h: u32, i: u32, j: u32) -> Rgba {
    match p.mode {
        Mode::Linear {
            p0x,
            p0y,
            c0,
            p1x,
            p1y,
            c1,
        } => {
            let (i, j) = (i as i64, j as i64);
            let (p0x, p0y) = (p0x as i64, p0y as i64);
            let (p1x, p1y) = (p1x as i64, p1y as i64);
            let dx = p1x - p0x;
            let dy = p1y - p0y;
            let d = dx * dx + dy * dy; // > 0 (validated)
            let num = dx * (i - p0x) + dy * (j - p0y);
            // t16 = clamp(floor(num * 2^16 / d), 0, 2^16)
            let t16 = clamp16(floor_div_pos(num << 16, d));
            blend(c0, c1, t16)
        }
        Mode::Bilinear { c00, c10, c01, c11 } => {
            // degenerate single-line extents reduce to one-axis blends
            let (w, h) = (w as u64, h as u64);
            let tx = if w > 1 {
                ((i as u64) << 16) / (w - 1)
            } else {
                0
            };
            let ty = if h > 1 {
                ((j as u64) << 16) / (h - 1)
            } else {
                0
            };
            let top = blend(c00, c10, tx as u32);
            let bottom = blend(c01, c11, tx as u32);
            blend(top, bottom, ty as u32)
        }
    }
}

/// Field is provably opaque when every defining color is opaque (blending
/// opaque endpoints yields opaque intermediates under the exact rule).
pub fn opaque(p: &Params) -> bool {
    match p.mode {
        Mode::Linear { c0, c1, .. } => c0.a == 255 && c1.a == 255,
        Mode::Bilinear { c00, c10, c01, c11 } => {
            c00.a == 255 && c10.a == 255 && c01.a == 255 && c11.a == 255
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgba;

    #[test]
    fn linear_roundtrip() {
        let p = Params {
            mode: Mode::Linear {
                p0x: 0,
                p0y: 0,
                c0: Rgba::new(0, 0, 0, 255),
                p1x: 10,
                p1y: 0,
                c1: Rgba::new(255, 255, 255, 255),
            },
        };
        let mut w = Vec::new();
        encode(&p, &mut w);
        let mut r = Reader::new(&w);
        assert_eq!(decode(&mut r, 20, 20).unwrap(), p);
        assert!(r.done());
    }

    #[test]
    fn linear_endpoints_and_mid() {
        let p = Params {
            mode: Mode::Linear {
                p0x: 0,
                p0y: 0,
                c0: Rgba::new(0, 0, 0, 255),
                p1x: 10,
                p1y: 0,
                c1: Rgba::new(100, 100, 100, 255),
            },
        };
        assert_eq!(sample(&p, 11, 1, 0, 0), Rgba::new(0, 0, 0, 255));
        assert_eq!(sample(&p, 11, 1, 10, 0), Rgba::new(100, 100, 100, 255));
        // midpoint (5,0): t = 1/2 -> 50 (exact: floor(100*32768/65536) = 50)
        assert_eq!(sample(&p, 11, 1, 5, 0), Rgba::new(50, 50, 50, 255));
    }

    #[test]
    fn linear_clamps_outside_segment() {
        // anchor at (0,0)->(1,0); sample i=0 clamps c0 (t=0), but a gradient
        // over extent wider than the segment clamps beyond it.
        let p = Params {
            mode: Mode::Linear {
                p0x: 3,
                p0y: 0,
                c0: Rgba::new(0, 0, 0, 255),
                p1x: 3,
                p1y: 4,
                c1: Rgba::new(200, 0, 0, 255),
            },
        };
        // i=3,j=0 is at p0: c0
        assert_eq!(sample(&p, 8, 8, 3, 0), Rgba::new(0, 0, 0, 255));
        // i=3,j=4 at p1 -> c1; i=3,j=2 mid -> 100
        assert_eq!(sample(&p, 8, 8, 3, 4), Rgba::new(200, 0, 0, 255));
        assert_eq!(sample(&p, 8, 8, 3, 2).r, 100);
        // beyond the segment clamps to c1
        assert_eq!(sample(&p, 8, 8, 3, 7).r, 200);
    }

    #[test]
    fn bilinear_corners() {
        let p = Params {
            mode: Mode::Bilinear {
                c00: Rgba::new(0, 0, 0, 255),
                c10: Rgba::new(200, 0, 0, 255),
                c01: Rgba::new(0, 200, 0, 255),
                c11: Rgba::new(0, 0, 200, 255),
            },
        };
        let (w, h) = (5u32, 5u32);
        assert_eq!(sample(&p, w, h, 0, 0), Rgba::new(0, 0, 0, 255));
        assert_eq!(sample(&p, w, h, 4, 0), Rgba::new(200, 0, 0, 255));
        assert_eq!(sample(&p, w, h, 0, 4), Rgba::new(0, 200, 0, 255));
        assert_eq!(sample(&p, w, h, 4, 4), Rgba::new(0, 0, 200, 255));
        // Center lattice point (2,2): tx = ty = 1/2; the bilinear value is the
        // mean of the four corners: r = (0+200+0+0)/4 = 50, g = 50, b = 50.
        assert_eq!(sample(&p, w, h, 2, 2), Rgba::new(50, 50, 50, 255));
    }

    #[test]
    fn degenerate_extent_blends_on_one_axis() {
        let p = Params {
            mode: Mode::Bilinear {
                c00: Rgba::new(0, 0, 0, 255),
                c10: Rgba::new(200, 0, 0, 255),
                c01: Rgba::new(0, 200, 0, 255),
                c11: Rgba::new(0, 0, 200, 255),
            },
        };
        // h == 1: horizontal blend only (top row used: c00..c10)
        let s = sample(&p, 5, 1, 2, 0);
        assert_eq!(s, Rgba::new(100, 0, 0, 255));
        // w == 1: vertical blend over left column (c00..c01)
        let s = sample(&p, 1, 5, 0, 2);
        assert_eq!(s, Rgba::new(0, 100, 0, 255));
        // 1x1 extent: c00
        assert_eq!(sample(&p, 1, 1, 0, 0), Rgba::new(0, 0, 0, 255));
    }

    #[test]
    fn rejects_degenerate_linear() {
        let mut w = Vec::new();
        put_u8(&mut w, MODE_LINEAR);
        put_i32(&mut w, 5);
        put_i32(&mut w, 5);
        put_bytes(&mut w, &[0, 0, 0, 255]);
        put_i32(&mut w, 5);
        put_i32(&mut w, 5);
        put_bytes(&mut w, &[255, 0, 0, 255]);
        let mut r = Reader::new(&w);
        assert_eq!(decode(&mut r, 10, 10), Err(Reject::Degenerate));
    }
}

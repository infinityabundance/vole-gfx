//! Periodic field: hard-edge color stripes / checker with exact integer
//! periodicity over the local pixel lattice.

use crate::color::Rgba;
use crate::ir::wire::{Reader, put_bytes, put_u8, put_u32};
use crate::limits::Reject;

pub const MODE_STRIPE_X: u8 = 1;
pub const MODE_STRIPE_Y: u8 = 2;
pub const MODE_CHECKER: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub mode: u8,
    /// Period along x in pixels (`>= 1`; canonical value is 1 for modes that
    /// do not use it).
    pub px: u32,
    /// Period along y in pixels (`>= 1`; canonical value is 1 for modes that
    /// do not use it).
    pub py: u32,
    pub c0: Rgba,
    pub c1: Rgba,
}

pub const WORK: u64 = 2;

pub fn encode(p: &Params, w: &mut Vec<u8>) {
    put_u8(w, p.mode);
    put_u32(w, p.px);
    put_u32(w, p.py);
    put_bytes(w, &p.c0.to_bytes());
    put_bytes(w, &p.c1.to_bytes());
}

fn validate(mode: u8, px: u32, py: u32) -> Result<(), Reject> {
    match mode {
        MODE_STRIPE_X => {
            if px == 0 || py != 1 {
                return Err(Reject::NonCanonicalOrder);
            }
        }
        MODE_STRIPE_Y => {
            if py == 0 || px != 1 {
                return Err(Reject::NonCanonicalOrder);
            }
        }
        MODE_CHECKER => {
            if px == 0 || py == 0 {
                return Err(Reject::NonCanonicalOrder);
            }
        }
        _ => return Err(Reject::UnknownTag),
    }
    Ok(())
}

pub fn decode(r: &mut Reader<'_>) -> Result<Params, Reject> {
    let mode = r.u8()?;
    let px = r.u32()?;
    let py = r.u32()?;
    validate(mode, px, py)?;
    let b0: [u8; 4] = r.bytes(4)?.try_into().unwrap();
    let b1: [u8; 4] = r.bytes(4)?.try_into().unwrap();
    Ok(Params {
        mode,
        px,
        py,
        c0: Rgba::from_bytes(b0),
        c1: Rgba::from_bytes(b1),
    })
}

pub fn work(_p: &Params) -> u64 {
    WORK
}

/// Value: band index `k = (i/px + j/py) mod 2` selects `c0`/`c1` per mode.
pub fn sample(p: &Params, i: u32, j: u32) -> Rgba {
    let k = match p.mode {
        MODE_STRIPE_X => (i / p.px) & 1,
        MODE_STRIPE_Y => (j / p.py) & 1,
        MODE_CHECKER => ((i / p.px) + (j / p.py)) & 1,
        _ => unreachable!("validated mode"),
    };
    if k == 0 { p.c0 } else { p.c1 }
}

pub fn opaque(p: &Params) -> bool {
    p.c0.a == 255 && p.c1.a == 255
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(mode: u8, px: u32, py: u32) -> Params {
        Params {
            mode,
            px,
            py,
            c0: Rgba::new(0, 0, 0, 255),
            c1: Rgba::new(255, 255, 255, 255),
        }
    }

    #[test]
    fn stripe_x_alternates() {
        let s = p(MODE_STRIPE_X, 4, 1);
        assert_eq!(sample(&s, 0, 5), Rgba::new(0, 0, 0, 255));
        assert_eq!(sample(&s, 3, 5), Rgba::new(0, 0, 0, 255));
        assert_eq!(sample(&s, 4, 5), Rgba::new(255, 255, 255, 255));
        assert_eq!(sample(&s, 7, 5), Rgba::new(255, 255, 255, 255));
        assert_eq!(sample(&s, 8, 5), Rgba::new(0, 0, 0, 255));
    }

    #[test]
    fn checker_alternates_both_axes() {
        let s = p(MODE_CHECKER, 2, 3);
        assert_eq!(sample(&s, 0, 0), Rgba::new(0, 0, 0, 255));
        assert_eq!(sample(&s, 1, 0), Rgba::new(0, 0, 0, 255));
        assert_eq!(sample(&s, 2, 0), Rgba::new(255, 255, 255, 255));
        assert_eq!(sample(&s, 0, 3), Rgba::new(255, 255, 255, 255));
        assert_eq!(sample(&s, 2, 3), Rgba::new(0, 0, 0, 255));
    }

    #[test]
    fn canonical_restrictions() {
        assert_eq!(
            decode(&mut Reader::new(&encode_blob(MODE_STRIPE_X, 0, 1))),
            Err(Reject::NonCanonicalOrder)
        );
        assert_eq!(
            decode(&mut Reader::new(&encode_blob(MODE_STRIPE_X, 2, 2))),
            Err(Reject::NonCanonicalOrder)
        );
        assert_eq!(
            decode(&mut Reader::new(&encode_blob(9, 1, 1))),
            Err(Reject::UnknownTag)
        );
        let s = p(MODE_STRIPE_X, 2, 1);
        let mut w = Vec::new();
        encode(&s, &mut w);
        assert!(decode(&mut Reader::new(&w)).is_ok());
    }

    fn encode_blob(mode: u8, px: u32, py: u32) -> Vec<u8> {
        let s = Params {
            mode,
            px,
            py,
            c0: Rgba::new(0, 0, 0, 255),
            c1: Rgba::WHITE,
        };
        let mut w = Vec::new();
        encode(&s, &mut w);
        w
    }
}

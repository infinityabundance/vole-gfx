//! Constant field: `value(i, j) = color` over the whole extent.

use crate::color::Rgba;
use crate::ir::wire::{Reader, put_bytes};
use crate::limits::Reject;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub color: Rgba,
}

pub const WORK: u64 = 1;

pub fn encode(p: &Params, w: &mut Vec<u8>) {
    put_bytes(w, &p.color.to_bytes());
}

pub fn decode(r: &mut Reader<'_>) -> Result<Params, Reject> {
    let b: [u8; 4] = r.bytes(4)?.try_into().unwrap();
    Ok(Params {
        color: Rgba::from_bytes(b),
    })
}

pub fn work(_p: &Params) -> u64 {
    WORK
}

pub fn sample(p: &Params, _i: u32, _j: u32) -> Rgba {
    p.color
}

/// Field is provably opaque (alpha 255 at every pixel).
pub fn opaque(p: &Params) -> bool {
    p.color.a == 255
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_sample() {
        let p = Params {
            color: Rgba::new(1, 2, 3, 255),
        };
        let mut w = Vec::new();
        encode(&p, &mut w);
        let mut r = Reader::new(&w);
        assert_eq!(decode(&mut r).unwrap(), p);
        assert!(r.done());
        assert_eq!(sample(&p, 4, 9), p.color);
        assert!(opaque(&p));
    }
}

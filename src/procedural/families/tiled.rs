//! Tiled field: an embedded `tw x th` RGBA tile repeated exactly over the
//! local pixel lattice.  The tile is the generator's own bounded parameter
//! storage — it is a primitive tile, not a materialized copy of the object
//! (the object's extent can be much larger than the tile).

use crate::color::Rgba;
use crate::ir::wire::{Reader, put_bytes, put_u32};
use crate::limits::Reject;

/// Maximum tile dimension (params-bounded: 64x64x4B = 16 KiB per tile).
pub const MAX_TILE_DIM: u32 = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Params {
    pub tw: u32,
    pub th: u32,
    /// `tw * th` entries, row-major.
    pub tile: Vec<Rgba>,
}

pub const WORK: u64 = 1;

pub fn encode(p: &Params, w: &mut Vec<u8>) {
    put_u32(w, p.tw);
    put_u32(w, p.th);
    for e in &p.tile {
        put_bytes(w, &e.to_bytes());
    }
}

fn validate(tw: u32, th: u32, n: usize) -> Result<(), Reject> {
    if tw == 0 || th == 0 || tw > MAX_TILE_DIM || th > MAX_TILE_DIM {
        return Err(Reject::DimensionTooLarge);
    }
    let expect = tw as u64 * th as u64;
    if expect != n as u64 {
        return Err(Reject::PayloadMismatch);
    }
    Ok(())
}

pub fn decode(r: &mut Reader<'_>) -> Result<Params, Reject> {
    let tw = r.u32()?;
    let th = r.u32()?;
    let n = tw as u64 * th as u64;
    if tw == 0 || th == 0 || tw > MAX_TILE_DIM || th > MAX_TILE_DIM {
        return Err(Reject::DimensionTooLarge);
    }
    if n > (crate::limits::MAX_GENERATOR_PARAMS / 4) {
        return Err(Reject::BytesExceedsLimit);
    }
    let mut tile = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let b: [u8; 4] = r.bytes(4)?.try_into().unwrap();
        tile.push(Rgba::from_bytes(b));
    }
    validate(tw, th, tile.len())?;
    Ok(Params { tw, th, tile })
}

pub fn work(_p: &Params) -> u64 {
    WORK
}

pub fn sample(p: &Params, i: u32, j: u32) -> Rgba {
    let u = i % p.tw;
    let v = j % p.th;
    p.tile[(v * p.tw + u) as usize]
}

/// Provably opaque when every tile pixel is opaque.
pub fn opaque(p: &Params) -> bool {
    p.tile.iter().all(|e| e.a == 255)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile2x2() -> Params {
        Params {
            tw: 2,
            th: 2,
            tile: vec![
                Rgba::new(1, 0, 0, 255),
                Rgba::new(2, 0, 0, 255),
                Rgba::new(3, 0, 0, 255),
                Rgba::new(4, 0, 0, 255),
            ],
        }
    }

    #[test]
    fn wraps_over_large_extent() {
        let p = tile2x2();
        assert_eq!(sample(&p, 0, 0), Rgba::new(1, 0, 0, 255));
        assert_eq!(sample(&p, 1, 0), Rgba::new(2, 0, 0, 255));
        assert_eq!(sample(&p, 100, 0), Rgba::new(1, 0, 0, 255)); // 100 % 2 = 0
        assert_eq!(sample(&p, 101, 3), Rgba::new(4, 0, 0, 255)); // (1,1)
    }

    #[test]
    fn roundtrip() {
        let p = tile2x2();
        let mut w = Vec::new();
        encode(&p, &mut w);
        let mut r = Reader::new(&w);
        assert_eq!(decode(&mut r).unwrap(), p);
        assert!(r.done());
    }

    #[test]
    fn rejects_bad_tile_dims() {
        let mut w = Vec::new();
        put_u32(&mut w, 0);
        put_u32(&mut w, 2);
        assert_eq!(decode(&mut Reader::new(&w)), Err(Reject::DimensionTooLarge));
        // declared 2x2 tile but only one pixel present: reads fail closed
        // (deterministic Truncated before any payload validation)
        let mut w2 = Vec::new();
        put_u32(&mut w2, 2);
        put_u32(&mut w2, 2);
        put_bytes(&mut w2, &[1, 2, 3, 4]);
        assert_eq!(decode(&mut Reader::new(&w2)), Err(Reject::Truncated));
    }
}

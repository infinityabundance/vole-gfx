//! Palette field: an embedded color palette plus a deterministic index
//! function over the local pixel lattice.  The palette is part of the
//! generator's own parameters (bounded, canonical); it is *not* animated by
//! timeline `PaletteSet` ops in this phase (documented non-support).

use super::fmix2;
use crate::color::Rgba;
use crate::ir::wire::{Reader, put_bytes, put_u8, put_u32, put_u64};
use crate::limits::Reject;

/// Maximum palette entries of a palette field (params-bounded; well under
/// the U1 `MAX_PALETTE_ENTRIES` because the entries live inside the params
/// blob).
pub const MAX_ENTRIES: u32 = 256;

pub const MODE_HASH: u8 = 1;
pub const MODE_X_RAMP: u8 = 2;
pub const MODE_BAND_X: u8 = 3;
pub const MODE_BAND_Y: u8 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Params {
    pub mode: u8,
    /// Band period in pixels (`>= 1`; canonical value 1 for modes that do
    /// not use it).
    pub period: u32,
    pub seed: u64,
    pub entries: Vec<Rgba>,
}

pub const WORK: u64 = 2;

fn validate(mode: u8, period: u32, n: usize) -> Result<(), Reject> {
    if n == 0 || n as u64 > MAX_ENTRIES as u64 {
        return Err(Reject::CountExceedsLimit);
    }
    match mode {
        MODE_HASH | MODE_X_RAMP => {
            if period != 1 {
                return Err(Reject::NonCanonicalOrder);
            }
        }
        MODE_BAND_X | MODE_BAND_Y => {
            if period == 0 {
                return Err(Reject::NonCanonicalOrder);
            }
        }
        _ => return Err(Reject::UnknownTag),
    }
    Ok(())
}

pub fn encode(p: &Params, w: &mut Vec<u8>) {
    put_u8(w, p.mode);
    put_u32(w, p.period);
    put_u64(w, p.seed);
    put_u32(w, p.entries.len() as u32);
    for e in &p.entries {
        put_bytes(w, &e.to_bytes());
    }
}

pub fn decode(r: &mut Reader<'_>) -> Result<Params, Reject> {
    let mode = r.u8()?;
    let period = r.u32()?;
    let seed = r.u64()?;
    let n = r.u32()?;
    validate(mode, period, n as usize)?;
    let mut entries = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let b: [u8; 4] = r.bytes(4)?.try_into().unwrap();
        entries.push(Rgba::from_bytes(b));
    }
    Ok(Params {
        mode,
        period,
        seed,
        entries,
    })
}

pub fn work(_p: &Params) -> u64 {
    WORK
}

fn index_at(p: &Params, w: u32, _h: u32, i: u32, j: u32) -> usize {
    let n = p.entries.len() as u64;
    let idx = match p.mode {
        MODE_HASH => fmix2(p.seed, i, j) % n,
        MODE_X_RAMP => {
            // floor(i * n / w); i < w so the value is < n (no clamp needed)
            (i as u64 * n) / (w.max(1) as u64)
        }
        MODE_BAND_X => (i / p.period.max(1)) as u64 % n,
        MODE_BAND_Y => (j / p.period.max(1)) as u64 % n,
        _ => unreachable!("validated mode"),
    };
    idx as usize
}

pub fn sample(p: &Params, w: u32, h: u32, i: u32, j: u32) -> Rgba {
    p.entries[index_at(p, w, h, i, j)]
}

/// Provably opaque when every palette entry is opaque.
pub fn opaque(p: &Params) -> bool {
    p.entries.iter().all(|e| e.a == 255)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pal(entries: Vec<Rgba>) -> Params {
        Params {
            mode: MODE_HASH,
            period: 1,
            seed: 7,
            entries,
        }
    }

    #[test]
    fn x_ramp_index_uses_extent() {
        // 5 entries across w=16: i=0 -> 0, i=15 -> floor(15*5/16)=4
        let p = Params {
            mode: MODE_X_RAMP,
            period: 1,
            seed: 0,
            entries: (0..5).map(|k| Rgba::new(k * 50, 0, 0, 255)).collect(),
        };
        assert_eq!(sample(&p, 16, 1, 0, 0), Rgba::new(0, 0, 0, 255));
        assert_eq!(sample(&p, 16, 1, 15, 0), Rgba::new(200, 0, 0, 255));
        assert_eq!(sample(&p, 16, 1, 3, 0), Rgba::new(0, 0, 0, 255)); // floor(15/16)=0
        assert_eq!(sample(&p, 16, 1, 4, 0), Rgba::new(50, 0, 0, 255)); // floor(20/16)=1
    }

    #[test]
    fn band_x_cycles_palette() {
        let p = Params {
            mode: MODE_BAND_X,
            period: 4,
            seed: 0,
            entries: (0..3).map(|k| Rgba::new(k as u8 * 80, 0, 0, 255)).collect(),
        };
        assert_eq!(sample(&p, 32, 1, 0, 0), Rgba::new(0, 0, 0, 255));
        assert_eq!(sample(&p, 32, 1, 4, 0), Rgba::new(80, 0, 0, 255));
        assert_eq!(sample(&p, 32, 1, 8, 0), Rgba::new(160, 0, 0, 255));
        assert_eq!(sample(&p, 32, 1, 12, 0), Rgba::new(0, 0, 0, 255));
    }

    #[test]
    fn hash_stays_in_range() {
        let p = pal((0..7).map(|k| Rgba::gray(k * 30)).collect());
        for i in 0..64u32 {
            for j in 0..64u32 {
                let c = sample(&p, 64, 64, i, j);
                assert!(c.r.is_multiple_of(30));
                assert_eq!(c.a, 255);
            }
        }
        assert!(opaque(&p));
    }

    #[test]
    fn roundtrip() {
        let p = pal(vec![Rgba::new(1, 2, 3, 4), Rgba::WHITE]);
        let mut w = Vec::new();
        encode(&p, &mut w);
        let mut r = Reader::new(&w);
        assert_eq!(decode(&mut r).unwrap(), p);
        assert!(r.done());
    }

    #[test]
    fn rejects_empty_and_noncanonical() {
        let mut w = Vec::new();
        put_u8(&mut w, MODE_HASH);
        put_u32(&mut w, 2); // non-canonical period for hash mode
        put_u64(&mut w, 0);
        put_u32(&mut w, 2);
        put_bytes(&mut w, &[1, 2, 3, 4]);
        put_bytes(&mut w, &[5, 6, 7, 8]);
        assert_eq!(decode(&mut Reader::new(&w)), Err(Reject::NonCanonicalOrder));
    }
}

//! Deterministic noise fields: a pure hash/gray-noise field and a bounded
//! value-noise fractal (exact fixed-point interpolation, integer lattice).
//!
//! Both families are gray-value fields (Rgba::gray).  The fractal's lattice
//! values are pure hash functions of `(seed, octave, cell)` so any sample is
//! O(octaves) with no hidden state; work is bounded by validation.

use super::{fmix3, gray_byte};
use crate::color::Rgba;
use crate::ir::wire::{Reader, put_u8, put_u32, put_u64};
use crate::limits::Reject;

/// Max fractal octaves (work = O(octaves), bounded).
pub const MAX_OCTAVES: u8 = 8;
/// Max base lattice period in pixels.
pub const MAX_BASE_PERIOD: u32 = 4096;

/// Hash/gray-noise field: `value(i, j) = gray(hash(seed, i, j))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldParams {
    pub seed: u64,
}

/// Value-noise fractal with `octaves` octaves of lattice cell size
/// `base_period << k`, exact bilinear lattice interpolation, and per-octave
/// amplitude `persistence` (Q16.16; 65536 = flat sum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FractalParams {
    pub seed: u64,
    pub base_period: u32,
    pub octaves: u8,
    pub persistence: u32,
}

pub const WORK_FIELD: u64 = 1;

pub fn encode_field(p: &FieldParams, w: &mut Vec<u8>) {
    put_u64(w, p.seed);
}

pub fn decode_field(r: &mut Reader<'_>) -> Result<FieldParams, Reject> {
    Ok(FieldParams { seed: r.u64()? })
}

pub fn work_field(_p: &FieldParams) -> u64 {
    WORK_FIELD
}

/// Deterministic gray byte at `(i, j)` (lattice-independent white noise).
pub fn sample_field(p: &FieldParams, i: u32, j: u32) -> Rgba {
    let h = fmix3(p.seed, i, j, 0);
    Rgba::gray(gray_byte(h))
}

pub fn encode_fractal(p: &FractalParams, w: &mut Vec<u8>) {
    put_u64(w, p.seed);
    put_u32(w, p.base_period);
    put_u8(w, p.octaves);
    put_u32(w, p.persistence);
}

pub fn decode_fractal(r: &mut Reader<'_>) -> Result<FractalParams, Reject> {
    let seed = r.u64()?;
    let base_period = r.u32()?;
    let octaves = r.u8()?;
    let persistence = r.u32()?;
    if base_period == 0 || base_period > MAX_BASE_PERIOD {
        return Err(Reject::DimensionTooLarge);
    }
    if !(1..=MAX_OCTAVES).contains(&octaves) {
        return Err(Reject::CountExceedsLimit);
    }
    if persistence > (1 << 16) {
        return Err(Reject::CoefficientOutOfRange);
    }
    Ok(FractalParams {
        seed,
        base_period,
        octaves,
        persistence,
    })
}

pub fn work_fractal(p: &FractalParams) -> u64 {
    1 + 16 * p.octaves as u64
}

/// Lattice value at integer cell `(cx, cy)` for octave `k`.
#[inline]
fn lattice(seed: u64, k: u32, cx: u32, cy: u32) -> u32 {
    gray_byte(fmix3(seed, cx, cy, k)) as u32
}

/// Exact floor blend over `t in 0..=65536`.
#[inline]
fn lerp(a: u32, b: u32, t: u64) -> u32 {
    ((a as u64 * (65536 - t) + b as u64 * t) >> 16) as u32
}

/// Sample the value-noise fractal at `(i, j)`.
pub fn sample_fractal(p: &FractalParams, i: u32, j: u32) -> Rgba {
    let mut total: u64 = 0;
    let mut weight_sum: u64 = 0;
    let mut weight: u64 = 1 << 16; // w0 = 2^16
    for k in 0..p.octaves as u32 {
        let cell = (p.base_period as u64) << k; // <= 4096 << 7 = 2^19
        let cx = (i as u64 / cell) as u32;
        let cy = (j as u64 / cell) as u32;
        let fx = (i as u64 % cell) << 16; // frac16 numerator
        let fy = (j as u64 % cell) << 16;
        let tx = fx / cell; // in [0, 65536]
        let ty = fy / cell;
        let a = lattice(p.seed, k, cx, cy);
        let b = lattice(p.seed, k, cx + 1, cy);
        let c = lattice(p.seed, k, cx, cy + 1);
        let d = lattice(p.seed, k, cx + 1, cy + 1);
        let top = lerp(a, b, tx);
        let bot = lerp(c, d, tx);
        let oct = lerp(top, bot, ty);
        total += oct as u64 * weight;
        weight_sum += weight;
        weight = (weight * p.persistence as u64) >> 16;
    }
    debug_assert!(weight_sum >= (1 << 16));
    let v = (total / weight_sum) as u8;
    Rgba::gray(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_roundtrip_and_bounds() {
        let p = FieldParams { seed: 99 };
        let mut w = Vec::new();
        encode_field(&p, &mut w);
        let mut r = Reader::new(&w);
        assert_eq!(decode_field(&mut r).unwrap(), p);
        for i in 0..64u32 {
            for j in 0..64u32 {
                let c = sample_field(&p, i, j);
                assert_eq!(c.a, 255);
            }
        }
    }

    #[test]
    fn fractal_roundtrip_and_determinism() {
        let p = FractalParams {
            seed: 42,
            base_period: 8,
            octaves: 4,
            persistence: 32768,
        };
        let mut w = Vec::new();
        encode_fractal(&p, &mut w);
        let mut r = Reader::new(&w);
        assert_eq!(decode_fractal(&mut r).unwrap(), p);
        assert_eq!(sample_fractal(&p, 3, 7), sample_fractal(&p, 3, 7));
    }

    #[test]
    fn fractal_single_octave_corner_equals_lattice() {
        // 1 octave: at a lattice corner (tx = ty = 0) the interpolation
        // collapses to the lattice value at that cell.
        let p = FractalParams {
            seed: 5,
            base_period: 16,
            octaves: 1,
            persistence: 0,
        };
        let c = sample_fractal(&p, 0, 0);
        assert_eq!(c, Rgba::gray(lattice(5, 0, 0, 0) as u8));
        assert_eq!(c.a, 255);
        // determinism
        assert_eq!(sample_fractal(&p, 0, 0), sample_fractal(&p, 0, 0));
    }

    #[test]
    fn work_bounded() {
        let p = FractalParams {
            seed: 1,
            base_period: 2,
            octaves: 8,
            persistence: 65536,
        };
        assert!(work_fractal(&p) <= crate::limits::MAX_GENERATOR_WORK_PER_SAMPLE);
    }

    #[test]
    fn rejects_bad_fractal_params() {
        let mut w = Vec::new();
        put_u64(&mut w, 0);
        put_u32(&mut w, 0); // zero period
        put_u8(&mut w, 1);
        put_u32(&mut w, 32768);
        assert_eq!(
            decode_fractal(&mut Reader::new(&w)),
            Err(Reject::DimensionTooLarge)
        );
    }
}

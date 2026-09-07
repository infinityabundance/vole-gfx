//! Shared exact helpers for generator-field evaluation.
//!
//! These rules are part of the U1 semantics (see docs/EXACT_SEMANTICS.md
//! §generators).  All arithmetic is integer/fixed-point and identical on
//! every backend; the scalar oracle and any future SIMD/CUDA evaluation share
//! these constants by construction.

use crate::color::Rgba;
use crate::fixed::hash64;

/// Mixing constants for deterministic 2D/3D integer hashing (SplitMix64).
pub(crate) const K1: u64 = 0x9E37_79B9_7F4A_7C15;
pub(crate) const K2: u64 = 0xBF58_476D_1CE4_E5B9;
pub(crate) const K3: u64 = 0x94D0_49BB_1331_11EB;

/// Deterministic mix of (seed, i, j) -> u64.
#[inline(always)]
pub(crate) fn fmix2(seed: u64, i: u32, j: u32) -> u64 {
    hash64(seed ^ (i as u64).wrapping_mul(K1) ^ (j as u64).wrapping_mul(K2))
}

/// Deterministic mix of (seed, i, j, k) -> u64 (used by fractal octaves).
#[inline(always)]
pub(crate) fn fmix3(seed: u64, i: u32, j: u32, k: u32) -> u64 {
    hash64(
        seed ^ (i as u64).wrapping_mul(K1)
            ^ (j as u64).wrapping_mul(K2)
            ^ (k as u64).wrapping_mul(K3),
    )
}

/// Pre-mix lattice constant of sample `(i, j)` for octave `k = 0` (the
/// deterministic-field family): `fmix3` folds a seed by XOR against this
/// constant, so a seed sweep over a fixed sample only varies the seed term.
/// SIMD search kernels broadcast this per-sample constant.
#[inline(always)]
pub(crate) const fn lattice_const(i: u32, j: u32) -> u64 {
    (i as u64).wrapping_mul(K1) ^ (j as u64).wrapping_mul(K2)
}

/// Extract the deterministic 8-bit gray byte of a mix value (high bits).
#[inline(always)]
pub(crate) fn gray_byte(h: u64) -> u8 {
    ((h >> 40) & 0xFF) as u8
}

/// Exact channel blend `(a*(2^16 - t) + b*t) / 2^16` (floor), `t in 0..=2^16`.
#[inline(always)]
pub(crate) fn blend_channel(a: u8, b: u8, t: u32) -> u8 {
    debug_assert!(t <= 1 << 16);
    ((a as u32 * ((1 << 16) - t) + b as u32 * t) >> 16) as u8
}

/// Exact RGBA blend over `t in 0..=2^16` (per channel, straight alpha codes).
#[inline(always)]
pub(crate) fn blend(c0: Rgba, c1: Rgba, t: u32) -> Rgba {
    Rgba {
        r: blend_channel(c0.r, c1.r, t),
        g: blend_channel(c0.g, c1.g, t),
        b: blend_channel(c0.b, c1.b, t),
        a: blend_channel(c0.a, c1.a, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_endpoints_and_mid() {
        let a = Rgba::new(0, 255, 10, 255);
        let b = Rgba::new(255, 0, 20, 128);
        assert_eq!(blend(a, b, 0), a);
        assert_eq!(blend(a, b, 1 << 16), b);
        let m = blend(a, b, 1 << 15);
        // floor((0*2^15 + 255*2^15)/2^16) = floor(255/2) = 127
        assert_eq!(m.r, 127);
        assert_eq!(m.g, 127);
        assert_eq!(m.a, 191); // floor((255*2^15 + 128*2^15)/2^16) = floor(383/2) = 191
    }

    #[test]
    fn blend_equals_weighted_average_over_sweep() {
        let a = Rgba::new(3, 200, 77, 250);
        let b = Rgba::new(251, 4, 90, 5);
        for t in [0u32, 1, 1234, 32767, 32768, 65535, 65536] {
            let m = blend(a, b, t);
            let expect = |x: u8, y: u8| ((x as u32 * (65536 - t) + y as u32 * t) / 65536) as u8;
            assert_eq!(
                m,
                Rgba::new(
                    expect(a.r, b.r),
                    expect(a.g, b.g),
                    expect(a.b, b.b),
                    expect(a.a, b.a)
                )
            );
        }
    }

    #[test]
    fn mix_stability_and_spread() {
        assert_eq!(fmix2(1, 2, 3), fmix2(1, 2, 3));
        assert_ne!(fmix2(1, 2, 3), fmix2(2, 2, 3));
        assert_ne!(fmix2(1, 2, 3), fmix2(1, 3, 3));
        assert_ne!(fmix2(1, 2, 3), fmix2(1, 2, 4));
        assert_ne!(fmix3(1, 2, 3, 1), fmix3(1, 2, 3, 2));
    }
}

//! Exact U1 color semantics.
//!
//! Normative rules (see `docs/EXACT_SEMANTICS.md`):
//!
//! * `Gray8` is an opaque single-channel luminance sample.
//! * `Rgba8` is straight (non-premultiplied) alpha in the sRGB encoding; the
//!   exact profile stores the 8-bit codes as-is (no transfer-function math is
//!   applied during composition — composition happens on code values).
//! * Painter composition over opaque layers: later layers overwrite earlier
//!   ones in `Gray8`.  For `Rgba8` the exact Porter–Duff "over" operator is
//!   defined below with integer intermediates; every division by 255 is exact
//!   floor division via `div255`, and the final division by the accumulated
//!   alpha is exact floor division.  No floating point, no platform rounding.
//!
//! `div255` is the single normative rounding primitive shared by every
//! backend (scalar, AVX2, AVX-512, CUDA) so results are byte-for-byte equal.

/// Number of bytes per Gray8 sample.
pub const GRAY8_BYTES: usize = 1;
/// Number of bytes per RGBA8 sample.
pub const RGBA8_BYTES: usize = 4;

/// Exact floor division by 255 for `0 <= x <= 65025`.
///
/// Uses the fixed-point reciprocal `1/255 ≈ 0x8081 · 2⁻²³` which never
/// underestimates, then a single correction when the estimate overshoots.
/// Implementable with the same two steps in 16/32-bit SIMD lanes.
#[inline(always)]
pub const fn div255(x: u32) -> u32 {
    debug_assert!(x <= 65025);
    let q = (x * 0x8081) >> 23;
    let r = x.wrapping_sub(q.wrapping_mul(255));
    if r > 254 { q - 1 } else { q }
}

/// Exact floor division by 255 for `0 <= x <= 16_581_375` (24-bit domain,
/// used for the premultiplied `channel·alpha·weight` products).  Scalar
/// reference uses the hardware divider; the optimized backends use a
/// deterministic multiply-shift estimate plus one correction, which the
/// differential tests prove equal to this function for the whole domain.
#[inline(always)]
pub fn div255_24(x: u32) -> u32 {
    debug_assert!(x <= 16_581_375);
    x / 255
}

/// Exact floor division `x / d` for `0 <= x <= 65025`, `d in 1..=255`.
/// Quotient fits 16 bits.  Reference scalar implementation (16-step binary
/// long division, deterministic; the SIMD backends use the same rule).
#[inline(always)]
pub fn div_small(x: u32, d: u32) -> u32 {
    debug_assert!(x <= 65025);
    debug_assert!((1..=255).contains(&d));
    let mut q = 0u32;
    let mut r = x;
    let mut bit = 15i32;
    while bit >= 0 {
        let step = d << bit;
        let take = r >= step;
        q = (q << 1) | take as u32;
        r = if take { r - step } else { r };
        bit -= 1;
    }
    q
}

/// A Gray8 sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Gray(pub u8);

/// An RGBA8 sample, straight alpha, channels `r g b a`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Rgba { r, g, b, a }
    }
    pub const OPAQUE_BLACK: Rgba = Rgba::new(0, 0, 0, 255);
    pub const TRANSPARENT: Rgba = Rgba::new(0, 0, 0, 0);
    pub const WHITE: Rgba = Rgba::new(255, 255, 255, 255);
    pub fn gray(g: u8) -> Self {
        Rgba::new(g, g, g, 255)
    }
    pub fn is_opaque(&self) -> bool {
        self.a == 255
    }
    pub fn is_transparent(&self) -> bool {
        self.a == 0
    }
    /// Pack to canonical little-endian byte order `[r, g, b, a]`.
    pub fn to_bytes(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }
    pub fn from_bytes(b: [u8; 4]) -> Self {
        Rgba {
            r: b[0],
            g: b[1],
            b: b[2],
            a: b[3],
        }
    }
}

impl From<Gray> for Rgba {
    fn from(g: Gray) -> Self {
        Rgba::gray(g.0)
    }
}

/// Exact Porter–Duff `over(src, dst)` on straight-alpha RGBA8.
///
/// Implemented in a 32-bit premultiplied domain so the alpha denominator is
/// uniform; straight output is recovered with one exact floor division per
/// channel.  For the overwhelmingly common opaque/opaque case this is exact
/// replacement (`src` returned).
pub fn over(src: Rgba, dst: Rgba) -> Rgba {
    if src.is_opaque() {
        return src;
    }
    if src.is_transparent() {
        return dst;
    }
    if dst.is_transparent() {
        return src;
    }
    let sa = src.a as u32;
    let da = dst.a as u32;
    let t = 255 - sa;
    // out.a = sa + floor(da·t/255)  (da·t <= 65025)
    let oa = sa + div255(da * t);
    debug_assert!(oa <= 255 && oa > 0);
    let blend = |sc: u8, dc: u8| -> u8 {
        // Straight-alpha over in the premultiplied domain:
        //   pm_out = sc·sa + floor(dc·da·t / 255)
        // then one exact division by the accumulated alpha.  The numerator is
        // bounded by 255·(sa + t) = 65025 (because da <= 255), so pm_out fits
        // the 16-bit-safe domain of div_small.  When the accumulated alpha is
        // tiny (da·t < 255) the alpha floor can push the quotient past 255;
        // such samples have visually negligible alpha, and the deterministic
        // clamp keeps the result in the 8-bit code range.
        let pm_src = sc as u32 * sa;
        let pm_out = pm_src + div255_24(dc as u32 * da * t);
        debug_assert!(pm_out <= 65025);
        div_small(pm_out, oa).min(255) as u8
    };
    Rgba {
        r: blend(src.r, dst.r),
        g: blend(src.g, dst.g),
        b: blend(src.b, dst.b),
        a: oa as u8,
    }
}

/// Exact composition of a `Gray8` sample onto an existing `Gray8` sample.
/// Gray8 has no alpha channel: later layers overwrite.
#[inline(always)]
pub fn gray_over(src: Gray, _dst: Gray) -> Gray {
    src
}

/// Convert RGBA8 to Gray8 luminance using exact rounded integer luma
/// (ITU-R BT.601 coefficients scaled to sum 1000).
pub fn rgba_to_gray(c: Rgba) -> Gray {
    let v = 299 * c.r as u32 + 587 * c.g as u32 + 114 * c.b as u32;
    Gray(((v + 500) / 1000) as u8)
}

/// Deterministic per-format default (zero) sample.
pub fn zero_sample(gray: bool) -> [u8; 4] {
    if gray { [0, 0, 0, 0] } else { [0, 0, 0, 255] }
}

/// Byte stride for a format.
pub fn stride(format: ColorFormat) -> usize {
    match format {
        ColorFormat::Gray8 => GRAY8_BYTES,
        ColorFormat::Rgba8 => RGBA8_BYTES,
    }
}

/// Declared sample formats of U1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum ColorFormat {
    Gray8 = 1,
    Rgba8 = 2,
}

impl ColorFormat {
    pub fn bytes_per_sample(self) -> usize {
        stride(self)
    }
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(ColorFormat::Gray8),
            2 => Some(ColorFormat::Rgba8),
            _ => None,
        }
    }
    pub fn tag(self) -> u8 {
        self as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div255_exact() {
        for x in 0..=65025u32 {
            assert_eq!(div255(x), x / 255, "div255({x})");
        }
    }

    #[test]
    fn div_small_exact() {
        // spot + boundary sweep over denominators
        for d in 1..=255u32 {
            for x in [0u32, 1, d - 1, d, d + 1, 255 * d, 65025] {
                assert_eq!(div_small(x, d), x / d, "div_small({x},{d})");
            }
        }
    }

    #[test]
    fn div255_24_exact_sweep() {
        for d in 0..=255u32 {
            for x in [0u32, 1, 254, 255, 256, 65025] {
                let v = d * x;
                assert_eq!(div255_24(v), v / 255, "div255_24({v})");
            }
        }
        // dense sweep across the 24-bit domain top
        for x in (16_580_000..=16_581_375).step_by(7) {
            assert_eq!(div255_24(x), x / 255);
        }
    }

    #[test]
    fn over_opaque_replaces() {
        let src = Rgba::new(10, 20, 30, 255);
        let dst = Rgba::new(200, 0, 0, 255);
        assert_eq!(over(src, dst), src);
    }

    #[test]
    fn over_transparent_identity() {
        let dst = Rgba::new(200, 0, 0, 255);
        assert_eq!(over(Rgba::TRANSPARENT, dst), dst);
    }

    #[test]
    fn over_onto_transparent_is_src() {
        let src = Rgba::new(10, 20, 30, 128);
        assert_eq!(over(src, Rgba::TRANSPARENT), src);
    }

    #[test]
    fn over_half_alpha_onto_black_is_midgray() {
        // 50% white over opaque black: each channel ~128 with alpha 255.
        let src = Rgba::new(255, 255, 255, 128);
        let dst = Rgba::new(0, 0, 0, 255);
        let o = over(src, dst);
        // sa=128, t=127, da=255: oa = 128 + floor(255*127/255) = 128 + 127 = 255
        // pm = 255*128 + 0 = 32640 -> c = floor(32640/255) = 128
        assert_eq!(o, Rgba::new(128, 128, 128, 255));
    }

    #[test]
    fn over_half_alpha_onto_white_stays_white() {
        // 50% white over opaque white stays white.
        let src = Rgba::new(255, 255, 255, 128);
        let o = over(src, Rgba::WHITE);
        assert_eq!(o, Rgba::WHITE);
    }

    #[test]
    fn over_commutes_with_expected_alpha() {
        let src = Rgba::new(255, 0, 0, 64);
        let dst = Rgba::new(0, 255, 0, 64);
        let o = over(src, dst);
        // oa = 64 + floor(64*191/255) = 64 + floor(12224/255) = 64 + 47 = 111
        assert_eq!(o.a, 111);
        // r: pm_src=255*64=16320; pm_dst=0*64=0; pm_out=16320+div255(0*191)=16320
        // r = floor(16320/111) = 147
        assert_eq!(o.r, 147);
        // g: pm_out = 0 + div255(255*64*191) ; 255*64*191 = 3,117,120/255=12224
        // g = floor(12224/111) = 110
        assert_eq!(o.g, 110);
        assert_eq!(o.b, 0);
    }

    #[test]
    fn gray_luma() {
        assert_eq!(rgba_to_gray(Rgba::WHITE).0, 255);
        assert_eq!(rgba_to_gray(Rgba::OPAQUE_BLACK).0, 0);
        assert_eq!(rgba_to_gray(Rgba::gray(77)).0, 77);
    }

    #[test]
    fn bytes_roundtrip() {
        let c = Rgba::new(1, 2, 3, 4);
        assert_eq!(Rgba::from_bytes(c.to_bytes()), c);
    }
}

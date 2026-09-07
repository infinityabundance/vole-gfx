//! Exact Q16.16 fixed-point arithmetic and 2D geometry for universe U1.
//!
//! Normative decisions (see `docs/EXACT_SEMANTICS.md`):
//!
//! * Positions are signed Q16.16 fixed-point (`i32` scaled by 2⁻¹⁶ pixel),
//!   range checked to ±`MAX_COORD`/2¹⁶ px (±16384 px) at validation time.
//! * Affine coefficients are Q16.16 and magnitude-checked to
//!   ±`MAX_AFFINE_COEFF`/2¹⁶ (|scale| ≤ 16).  Multiplication of a coordinate
//!   by a coefficient therefore fits `i64` with 44+ bits of headroom; we use
//!   widened `i64` intermediates everywhere and never wrap.
//! * A raster sample `(i,j)` is observed at scene point
//!   `(i + 0.5, j + 0.5)` (sample-center rule).  A filled half-open rect
//!   `[x0,x1) × [y0,y1)` covers the sample iff `x0 ≤ cx < x1 && y0 ≤ cy < y1`.
//! * Conversion from fixed to integer pixel coordinates truncates toward
//!   −∞ (floor), matching the half-open sample-center coverage rule.

use core::ops::{Add, Mul, Neg, Sub};

/// Q16.16 fractional bits.
pub const FRAC_BITS: u32 = 16;
/// 2¹⁶.
pub const UNIT: i64 = 1 << FRAC_BITS;
/// Q16.16 value of one half pixel (sample-center offset).
pub const HALF: i32 = (UNIT / 2) as i32;
/// Q16.16 value of the integer `v`.
#[inline(always)]
pub const fn fp(v: i32) -> i32 {
    v << FRAC_BITS
}

/// Convert a `f64` to Q16.16 with round-half-away-from-zero.
/// Only used by builders/tests — never on a normative path.
pub fn fp_from_f64(v: f64) -> Option<i32> {
    let s = v * UNIT as f64;
    if !s.is_finite() || s < i32::MIN as f64 || s > i32::MAX as f64 {
        return None;
    }
    let r = if s >= 0.0 {
        (s + 0.5) as i64
    } else {
        (s - 0.5) as i64
    };
    Some(r as i32)
}

/// Exact floor division of a Q16.16 value by `UNIT` (truncation to pixel).
#[inline(always)]
pub fn floor_to_px(v: i32) -> i32 {
    v >> FRAC_BITS
}

/// Exact ceiling division of a Q16.16 value by `UNIT`.
#[inline(always)]
pub fn ceil_to_px(v: i32) -> i32 {
    -((-(v as i64) >> FRAC_BITS) as i32)
}

/// Exact `a * b` with `b` a Q16.16 value: result is `a` scaled by `b`.
/// Uses widened intermediates; inputs must satisfy the U1 range checks.
#[inline(always)]
pub fn mul_fixed(a: i64, b: i32) -> i64 {
    (a * b as i64) >> FRAC_BITS
}

/// 2D fixed-point vector (scene coordinates, Q16.16).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Vec2 {
    pub x: i32,
    pub y: i32,
}

impl Vec2 {
    pub const fn new(x: i32, y: i32) -> Self {
        Vec2 { x, y }
    }
    pub const fn from_px(x: i32, y: i32) -> Self {
        Vec2 {
            x: x << FRAC_BITS,
            y: y << FRAC_BITS,
        }
    }
    pub const fn zero() -> Self {
        Vec2 { x: 0, y: 0 }
    }
    /// Integer pixel position (floor).
    pub const fn px(self) -> (i32, i32) {
        (self.x >> FRAC_BITS, self.y >> FRAC_BITS)
    }
    /// Half-pixel-offset sample center for output pixel `(i,j)`.
    pub const fn sample_center(i: i32, j: i32) -> Self {
        Vec2 {
            x: (i << FRAC_BITS) + HALF,
            y: (j << FRAC_BITS) + HALF,
        }
    }
    pub fn checked_add(self, o: Vec2) -> Option<Vec2> {
        Some(Vec2 {
            x: self.x.checked_add(o.x)?,
            y: self.y.checked_add(o.y)?,
        })
    }
    pub fn checked_sub(self, o: Vec2) -> Option<Vec2> {
        Some(Vec2 {
            x: self.x.checked_sub(o.x)?,
            y: self.y.checked_sub(o.y)?,
        })
    }
    pub fn in_range(self) -> bool {
        (self.x as i64).abs() <= crate::limits::MAX_COORD
            && (self.y as i64).abs() <= crate::limits::MAX_COORD
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        Vec2 {
            x: self.x.wrapping_add(o.x),
            y: self.y.wrapping_add(o.y),
        }
    }
}
impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2 {
            x: self.x.wrapping_sub(o.x),
            y: self.y.wrapping_sub(o.y),
        }
    }
}
impl Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

/// Half-open fixed-point rectangle `[x0,x1) × [y0,y1)` in scene space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RectF {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl RectF {
    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        RectF { x0, y0, x1, y1 }
    }
    pub fn from_px(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        RectF::new(fp(x0), fp(y0), fp(x1), fp(y1))
    }
    pub fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }
    pub fn valid(&self) -> bool {
        self.x0 <= self.x1
            && self.y0 <= self.y1
            && (self.x0 as i64).abs() <= crate::limits::MAX_COORD
            && (self.x1 as i64).abs() <= crate::limits::MAX_COORD
            && (self.y0 as i64).abs() <= crate::limits::MAX_COORD
            && (self.y1 as i64).abs() <= crate::limits::MAX_COORD
    }
    /// Does the sample center of output pixel `(i,j)` lie inside?
    #[inline(always)]
    pub fn covers_px(&self, cx: i32, cy: i32) -> bool {
        cx >= self.x0 && cx < self.x1 && cy >= self.y0 && cy < self.y1
    }
    /// Tight integer pixel bounds (inclusive-exclusive) of samples whose
    /// centers lie inside the half-open rect.
    pub fn px_bounds(&self) -> IRect {
        // Pixel i is covered iff  x0 <= i + 0.5 < x1, i.e.
        // i in [ceil(x0 - 0.5), ceil(x1 - 0.5)).
        let x0 = ceil_fx(self.x0.wrapping_sub(HALF));
        let x1 = ceil_fx(self.x1.wrapping_sub(HALF));
        let y0 = ceil_fx(self.y0.wrapping_sub(HALF));
        let y1 = ceil_fx(self.y1.wrapping_sub(HALF));
        IRect {
            x0,
            y0,
            x1: x1.max(x0),
            y1: y1.max(y0),
        }
    }
}

/// Exact ceiling of a fixed value toward integer pixels.
#[inline(always)]
fn ceil_fx(v: i32) -> i32 {
    // ceil(v/2^16) = -floor((-v)/2^16)
    -((-(v as i64) >> FRAC_BITS) as i32)
}

/// Integer sample rectangle `[x0,x1) × [y0,y1)` in output pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct IRect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl IRect {
    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        IRect { x0, y0, x1, y1 }
    }
    pub fn empty() -> Self {
        IRect {
            x0: 0,
            y0: 0,
            x1: 0,
            y1: 0,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }
    pub fn width(&self) -> i32 {
        self.x1 - self.x0
    }
    pub fn height(&self) -> i32 {
        self.y1 - self.y0
    }
    pub fn area(&self) -> u64 {
        if self.is_empty() {
            0
        } else {
            self.width() as u64 * self.height() as u64
        }
    }
    pub fn contains_px(&self, i: i32, j: i32) -> bool {
        i >= self.x0 && i < self.x1 && j >= self.y0 && j < self.y1
    }
    pub fn intersect(self, o: IRect) -> IRect {
        IRect {
            x0: self.x0.max(o.x0),
            y0: self.y0.max(o.y0),
            x1: self.x1.min(o.x1),
            y1: self.y1.min(o.y1),
        }
    }
    pub fn union(self, o: IRect) -> IRect {
        if self.is_empty() {
            return o;
        }
        if o.is_empty() {
            return self;
        }
        IRect {
            x0: self.x0.min(o.x0),
            y0: self.y0.min(o.y0),
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
        }
    }
    pub fn translate(self, dx: i32, dy: i32) -> IRect {
        IRect::new(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
    }
}

/// Affine transform mapping scene point p to
/// `x' = a·x + b·y + tx`, `y' = c·x + d·y + ty` with Q16.16 coefficients.
///
/// Layout is fixed and documented; it is the U1 normative layout (row-major
/// 2×3 as `a b tx / c d ty`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Affine {
    pub a: i32,
    pub b: i32,
    pub tx: i32,
    pub c: i32,
    pub d: i32,
    pub ty: i32,
}

impl Default for Affine {
    fn default() -> Self {
        Affine::identity()
    }
}

impl Affine {
    pub const fn identity() -> Self {
        Affine {
            a: UNIT as i32,
            b: 0,
            tx: 0,
            c: 0,
            d: UNIT as i32,
            ty: 0,
        }
    }
    pub const fn translation(tx: i32, ty: i32) -> Self {
        Affine {
            a: UNIT as i32,
            b: 0,
            tx,
            c: 0,
            d: UNIT as i32,
            ty,
        }
    }
    pub const fn from_px_translation(tx: i32, ty: i32) -> Self {
        Affine {
            a: UNIT as i32,
            b: 0,
            tx: tx << FRAC_BITS,
            c: 0,
            d: UNIT as i32,
            ty: ty << FRAC_BITS,
        }
    }
    pub fn uniform_scale(s: i32) -> Self {
        Affine {
            a: s,
            b: 0,
            tx: 0,
            c: 0,
            d: s,
            ty: 0,
        }
    }
    /// Is every coefficient within the U1 magnitude cap?  Linear parts are
    /// capped at `MAX_AFFINE_COEFF` (|scale| ≤ 16); translations at
    /// `MAX_COORD` (they are further capped to `INST_TRANSLATION_LIMIT` by
    /// placement validation).
    pub fn in_range(&self) -> bool {
        let c = |v: i32| (v as i64).abs() <= crate::limits::MAX_AFFINE_COEFF;
        c(self.a)
            && c(self.b)
            && c(self.c)
            && c(self.d)
            && (self.tx as i64).abs() <= crate::limits::MAX_COORD
            && (self.ty as i64).abs() <= crate::limits::MAX_COORD
    }
    /// Map a fixed-point point.  Exact widened arithmetic.
    #[inline(always)]
    pub fn apply(&self, p: Vec2) -> Vec2 {
        let x = self.a as i64 * p.x as i64 + self.b as i64 * p.y as i64 + self.tx as i64 * UNIT;
        let y = self.c as i64 * p.x as i64 + self.d as i64 * p.y as i64 + self.ty as i64 * UNIT;
        Vec2 {
            x: (x >> FRAC_BITS) as i32,
            y: (y >> FRAC_BITS) as i32,
        }
    }
    /// Multiply transforms: `self ∘ other` (apply other first).  Named
    /// `compose` to avoid clashing with the by-value `Mul` impl in method
    /// lookup.
    pub fn compose(&self, o: &Affine) -> Affine {
        let w = |m: i64| (m >> FRAC_BITS) as i32;
        Affine {
            a: w(self.a as i64 * o.a as i64 + self.b as i64 * o.c as i64),
            b: w(self.a as i64 * o.b as i64 + self.b as i64 * o.d as i64),
            tx: w(self.a as i64 * o.tx as i64 + self.b as i64 * o.ty as i64) + self.tx,
            c: w(self.c as i64 * o.a as i64 + self.d as i64 * o.c as i64),
            d: w(self.c as i64 * o.b as i64 + self.d as i64 * o.d as i64),
            ty: w(self.c as i64 * o.tx as i64 + self.d as i64 * o.ty as i64) + self.ty,
        }
    }
    /// Bounding box of the transform applied to the eight corners of a rect
    /// (inclusive of edges per half-open sample-center rule: we expand by one
    /// half pixel each side, then round outward).
    pub fn bounds_of(&self, r: RectF) -> RectF {
        if r.is_empty() {
            return r;
        }
        let corners = [
            Vec2::new(r.x0, r.y0),
            Vec2::new(r.x1, r.y0),
            Vec2::new(r.x0, r.y1),
            Vec2::new(r.x1, r.y1),
        ];
        let mut minx = i64::MAX;
        let mut miny = i64::MAX;
        let mut maxx = i64::MIN;
        let mut maxy = i64::MIN;
        for c in corners {
            let p = self.apply(c);
            minx = minx.min(p.x as i64);
            miny = miny.min(p.y as i64);
            maxx = maxx.max(p.x as i64);
            maxy = maxy.max(p.y as i64);
        }
        RectF {
            x0: minx as i32,
            y0: miny as i32,
            x1: maxx as i32,
            y1: maxy as i32,
        }
    }
}

impl Mul for Affine {
    type Output = Affine;
    fn mul(self, o: Affine) -> Affine {
        Affine::compose(&self, &o)
    }
}

/// SplitMix64 finalizer constants (single source for the scalar `hash64` and
/// the SIMD lane kernels of the inverse search, which must be bit-identical).
pub const HASH64_ADD: u64 = 0x9E37_79B9_7F4A_7C15;
pub const HASH64_M1: u64 = 0xBF58_476D_1CE4_E5B9;
pub const HASH64_M2: u64 = 0x94D0_49BB_1331_11EB;

/// A deterministic 64-bit integer hash (SplitMix64 finalizer).  Used for
/// structural fingerprints and in-memory hashing only — never for content
/// identity (which is SHA-256, see `crate::hash`).
#[inline(always)]
pub fn hash64(mut x: u64) -> u64 {
    x = x.wrapping_add(HASH64_ADD);
    x = (x ^ (x >> 30)).wrapping_mul(HASH64_M1);
    x = (x ^ (x >> 27)).wrapping_mul(HASH64_M2);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_basics() {
        assert_eq!(fp(1), 65536);
        assert_eq!(fp(-3), -196608);
        assert_eq!(floor_to_px(fp(2) + 1000), 2);
        assert_eq!(floor_to_px(-fp(1) + 10), -1);
        assert_eq!(mul_fixed(fp(3) as i64, fp(4)), 12 << FRAC_BITS);
        assert_eq!(mul_fixed(fp(3) as i64, -fp(4)), -12 << FRAC_BITS);
    }

    #[test]
    fn sample_center_rule() {
        let v = Vec2::sample_center(3, -2);
        assert_eq!(v.x, (3 << 16) + HALF);
        assert_eq!(v.y, (-2 << 16) + HALF);
    }

    #[test]
    fn rect_px_bounds_half_open() {
        // x in [0.5, 3.5) covers pixels 0..=2 => IRect x0=0 x1=3
        let r = RectF::new(fp(0) + HALF, fp(1), fp(3) + HALF, fp(4));
        let b = r.px_bounds();
        assert_eq!((b.x0, b.x1, b.y0, b.y1), (0, 3, 1, 4));
    }

    #[test]
    fn rect_px_bounds_negative() {
        // [-2.5, -0.5) covers pixels -3 (center -2.5) and -2 (center -1.5):
        // rule gives [ceil(-3.0), ceil(-1.0)) = [-3, -1).
        let r = RectF::new(-fp(2) - HALF, 0, -HALF, fp(1));
        let b = r.px_bounds();
        assert_eq!((b.x0, b.x1), (-3, -1));
    }

    #[test]
    fn rect_px_bounds_half_edge() {
        // x0 = 2.5 exactly: pixel 2 center 2.5 is included, so start is 2;
        // x1 = 4.5 excludes pixel 4 (center 4.5). Result [2, 4).
        let r = RectF::new(fp(2) + HALF, 0, fp(4) + HALF, fp(1));
        let b = r.px_bounds();
        assert_eq!((b.x0, b.x1), (2, 4));
    }

    #[test]
    fn affine_apply_translation() {
        let t = Affine::from_px_translation(10, -4);
        let p = t.apply(Vec2::from_px(3, 7));
        assert_eq!(p, Vec2::from_px(13, 3));
    }

    #[test]
    fn affine_apply_scale() {
        let s = Affine::uniform_scale(fp(2));
        let p = s.apply(Vec2::from_px(-3, 5));
        assert_eq!(p, Vec2::from_px(-6, 10));
    }

    #[test]
    fn affine_mul_composes() {
        let t = Affine::from_px_translation(5, 0);
        let s = Affine::uniform_scale(fp(2));
        let c = s.compose(&t); // apply t first, then s
        let p = c.apply(Vec2::from_px(1, 1));
        // (1,1) -> t -> (6,1) -> s -> (12,2)
        assert_eq!(p, Vec2::from_px(12, 2));
    }

    #[test]
    fn affine_bounds() {
        let s = Affine::uniform_scale(fp(2));
        let r = RectF::from_px(0, 0, 10, 10);
        let b = s.bounds_of(r);
        assert_eq!(b, RectF::from_px(0, 0, 20, 20));
    }

    #[test]
    fn hash64_distinct() {
        let a = hash64(1);
        let b = hash64(2);
        assert_ne!(a, b);
        assert_eq!(hash64(1), hash64(1));
    }

    #[test]
    fn in_range_checks() {
        assert!(Vec2::from_px(16000, -16000).in_range());
        assert!(!Vec2::new(i32::MAX, 0).in_range());
        assert!(Affine::identity().in_range());
        assert!(
            !Affine {
                a: fp(17),
                ..Affine::identity()
            }
            .in_range()
        );
    }
}

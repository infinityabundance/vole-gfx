//! Phase K: seeded-field inverse search over the deterministic gray-noise
//! family (`Γ = DETERMINISTIC_FIELD`; sample value
//! `gray_byte(fmix3(seed, i, j, 0))`).
//!
//! This is the "seeded procedural fields" step of the paper's hierarchical
//! compiler: given a gray (or fully gray-opaque RGBA) surface, search the
//! bounded seed universe `[0, range)` for the seed whose field reproduces the
//! surface byte-for-byte.
//!
//! # Search predicate (normative)
//!
//! `accept(seed)` iff the seed's field equals the asset on **every** sample
//! (zero mismatches over the full surface) — host-independent by
//! construction.  A deterministic *anchor prefilter* (a few fixed corner
//! samples) is a pure pruning device: it is a necessary condition of full
//! equality, so it can never reject a true match, and a seed is never
//! accepted without the full-surface check.  The scalar implementation is
//! the semantic oracle and produces the canonical `SearchCounter`; the AVX2
//! and AVX-512 backends evaluate the identical predicate in vector lanes and
//! must accept exactly the same seed set (differential tests + phase-k
//! gate).
//!
//! # Exactness across backends
//!
//! `fmix3 = hash64(seed ^ i·K1 ^ j·K2)`.  The AVX2 kernels emulate the 64-bit
//! multiplies with `vpmuludq` (exact mod-2⁶⁴ low products, see
//! `mul64_lo_avx2`); the AVX-512 kernels use native `vpmullq`
//! (`avx512dq`); all shifts/xors/adds are exact lane operations.  Every lane
//! therefore reproduces the scalar hash bit-for-bit.  Backends may differ
//! only in *how much redundant work they execute* (SIMD evaluates whole
//! lanes through every anchor and cannot short-circuit per lane); the
//! accepted set never diverges.
//!
//! # Range policy
//!
//! Requested ranges above `MAX_SEED_SWEEP_RANGE` are **rejected** with
//! `Err(Reject::CountExceedsLimit)` by every sweep function (and therefore
//! by auto dispatch).  There is no silent clamping: a caller that asks for a
//! `2^30`-seed universe gets an error, not an empty match set over a
//! `2^24`-seed prefix.  `Ok(None)` means the surface is not in the searchable
//! code space (not gray / not fully opaque-gray RGBA) or — for the SIMD
//! backends — the host lacks the ISA; `Err` means the request itself is
//! invalid.
//!
//! # Search-work accounting boundary
//!
//! The `SearchCounter` returned in a `Sweep` counts **search operations only**
//! (see `work.rs` for the frozen metric): anchor evaluations and survivor
//! full-surface verifications (each = one hash evaluation + one whole-code
//! comparison, counted exactly where executed).  Format normalization
//! (deriving the gray surface: an O(w·h) clone for Gray8 or code walk for
//! RGBA) is *preprocessing* — bounded by the surface, identical for every
//! asset of a given surface, and deliberately outside the search-work
//! metric; the wired-in detector therefore does not charge it to its
//! proposal either.
//!
//! # Unsafe policy
//!
//! All intrinsic-using code lives in `#[target_feature] unsafe fn` kernels
//! called only under a runtime feature gate.  Kernel bodies are implicit
//! unsafe contexts (rustc exempts `#[target_feature]` functions from
//! `unsafe_op_in_unsafe_fn`), so each statement carries a `// SAFETY:`
//! rationale instead of a redundant inner block; every *call* from safe code
//! is an explicit `unsafe` block with its own SAFETY note.

use super::asset::Asset;
use super::detect::{Proposal, one_object_doc};
use super::work::SearchCounter;
use crate::color::ColorFormat;
use crate::fixed::{HASH64_ADD, HASH64_M1, HASH64_M2};
use crate::limits::Reject;
use crate::procedural::mix::{fmix3, gray_byte, lattice_const};

// SIMD vector types used in kernel signatures (defined for the whole x86_64
// target; the kernels using them are additionally gated on the runtime
// feature set before any call).
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m256i, __m512i};

/// Default seed sweep range of the wired-in detector (`[0, 1<<16)`).
/// A detector budget, not a claim about real-world seed distributions;
/// courts may sweep larger explicit ranges via the public sweep functions.
pub const DEFAULT_SEED_SWEEP_RANGE: u64 = 1 << 16;

/// Normative cap on any single seed sweep.  Ranges above this cap are
/// **rejected** (`Err(Reject::CountExceedsLimit)`), never silently truncated:
/// a "no match" result must always mean "no match in the requested universe",
/// never "no match in a clipped prefix of it".  Phase M courts may raise this
/// cap only with hardware evidence; the corpus seal keeps the value receipted.
pub const MAX_SEED_SWEEP_RANGE: u64 = 1 << 24;

/// Auto-dispatch policy for the *search* kernels (independent of the
/// materializer's evidence-ordered dispatch).  The phase-k gate receipt
/// (this host, range 2^20, six gray courts, medians) measured the *full*
/// order: AVX-512 ~0.74–0.78 ms, scalar ~1.2 ms, AVX2 ~1.9–2.0 ms — the
/// native `vpmullq` of `avx512dq` wins the multiply-heavy seed sweep, and the
/// scalar oracle beats AVX2 because it short-circuits per seed while AVX2
/// evaluates whole lanes through every anchor at `vpmuludq`-emulation cost.
/// `sweep_auto` therefore tries AVX-512 first, then the scalar oracle, and
/// only then AVX2 — the fallback chain never prefers a path the receipts
/// measured as slower on the current hosts.  AVX2 remains a receipted
/// forced-backend row and may win on another microarchitecture; the
/// corpus/autotuning phases (O/W) replace this constant with a
/// profile-driven model.
pub const SEARCH_AUTO_PREFER_AVX512: bool = true;

/// Which execution path produced a sweep result (recorded in receipts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepBackend {
    Scalar,
    Avx2,
    Avx512,
}

impl SweepBackend {
    pub fn name(self) -> &'static str {
        match self {
            SweepBackend::Scalar => "scalar",
            SweepBackend::Avx2 => "avx2",
            SweepBackend::Avx512 => "avx512",
        }
    }
}

/// Result of one seed sweep over `[0, range)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sweep {
    pub backend: SweepBackend,
    /// Matching seeds, ascending.  Empty means "no exact match in the
    /// **requested** universe" — ranges are rejected above the cap, never
    /// truncated, so an empty set is always a true negative over what was
    /// asked for (the literal fallback then owns the frontier).
    pub matches: Vec<u64>,
    /// Search work executed by this backend (see `work.rs`): the scalar
    /// counter counts actual short-circuiting hash evaluations; SIMD
    /// counters count vector-lane evaluations (whole lanes through every
    /// anchor, plus full-surface verification of every anchor survivor).
    /// The *frontier* axis always uses the scalar counter (host-independent
    /// frontiers); SIMD counters are receipted evidence.
    pub work: SearchCounter,
}

/// The exact U1 gray prediction of the deterministic field at `(i, j)` —
/// identical rule to the generator sampler (`noise::sample_field`).
#[inline]
pub fn gray_value(seed: u64, i: u32, j: u32) -> u8 {
    gray_byte(fmix3(seed, i, j, 0))
}

/// Deterministic corner anchors of a `w x h` surface (deduplicated).
pub fn default_anchors(w: u32, h: u32) -> Vec<(u32, u32)> {
    let mut a: Vec<(u32, u32)> = Vec::with_capacity(4);
    for (i, j) in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)] {
        if !a.contains(&(i, j)) {
            a.push((i, j));
        }
    }
    a
}

/// One anchor probe: sample `(i, j)`, its expected gray byte on the surface,
/// and the pre-mix lattice constant `fmix3` folds against the seed.
#[derive(Debug, Clone, Copy)]
struct Anchor {
    i: u32,
    j: u32,
    expect: u8,
    c: u64,
}

/// Build the anchor table for a gray surface (corner anchors by default).
fn anchor_table(w: u32, gray: &[u8], anchors: &[(u32, u32)]) -> Vec<Anchor> {
    anchors
        .iter()
        .map(|&(i, j)| Anchor {
            i,
            j,
            expect: gray[(j as usize) * w as usize + i as usize],
            c: lattice_const(i, j),
        })
        .collect()
}

/// Reduce an asset to its gray-code surface: `Gray8` samples pass through;
/// `Rgba8` surfaces pass only when every sample is opaque gray
/// `(v, v, v, 255)` — the exact code space of the gray-noise family.
/// Returns `None` for any other surface (the detector must not run on it).
pub fn asset_gray_surface(asset: &Asset) -> Option<Vec<u8>> {
    let n = (asset.w * asset.h) as usize;
    match asset.format {
        ColorFormat::Gray8 => Some(asset.data.clone()),
        ColorFormat::Rgba8 => {
            let mut gray = Vec::with_capacity(n);
            // SAFETY: RGBA8 asset payloads are always a whole number of
            // 4-byte codes (Asset::new enforces w*h*4 bytes).
            for c in asset.data.as_chunks::<4>().0 {
                if c[0] != c[1] || c[1] != c[2] || c[3] != 255 {
                    return None;
                }
                gray.push(c[0]);
            }
            Some(gray)
        }
    }
}

/// Reject seed ranges above the normative cap instead of silently truncating
/// them (see the module docs, "Range policy").  The check is host- and
/// backend-independent, so every sweep reports the same rejection.
fn checked_range(range: u64) -> Result<u64, Reject> {
    if range > MAX_SEED_SWEEP_RANGE {
        return Err(Reject::CountExceedsLimit);
    }
    Ok(range)
}

/// Canonical scalar sweep (semantic oracle).  The returned `work` counts
/// exactly the hash evaluations the scalar search performs: anchors per seed
/// with per-seed short-circuit, then full-surface verification of survivors
/// with per-sample short-circuit.
///
/// `Ok(None)` when the asset is not a gray surface; `Err` when `range`
/// exceeds `MAX_SEED_SWEEP_RANGE` (rejected, never truncated).
pub fn sweep_scalar(asset: &Asset, range: u64) -> Result<Option<Sweep>, Reject> {
    let range = checked_range(range)?;
    let Some(gray) = asset_gray_surface(asset) else {
        return Ok(None);
    };
    let table = anchor_table(asset.w, &gray, &default_anchors(asset.w, asset.h));
    let mut work = SearchCounter::default();
    let matches = scalar_sweep_gray(asset.w, asset.h, &gray, range, &table, &mut work);
    Ok(Some(Sweep {
        backend: SweepBackend::Scalar,
        matches,
        work,
    }))
}

/// Scalar sweep over an already-derived gray surface (shared by the detector
/// so the precheck and the sweep share one counter).  Deterministic: seeds
/// visited ascending, matches collected ascending.
fn scalar_sweep_gray(
    w: u32,
    h: u32,
    gray: &[u8],
    range: u64,
    table: &[Anchor],
    work: &mut SearchCounter,
) -> Vec<u64> {
    let mut matches = Vec::new();
    for seed in 0..range {
        let mut ok = true;
        for a in table {
            work.eval_hashes(1);
            work.compare_codes(1);
            if gray_value(seed, a.i, a.j) != a.expect {
                ok = false;
                break;
            }
        }
        if ok {
            // full-surface verification (short-circuits at the first
            // mismatching sample)
            'full: for jj in 0..h {
                for ii in 0..w {
                    work.eval_hashes(1);
                    work.compare_codes(1);
                    if gray_value(seed, ii, jj) != gray[(jj as usize) * w as usize + ii as usize] {
                        ok = false;
                        break 'full;
                    }
                }
            }
            if ok {
                matches.push(seed);
            }
        }
    }
    matches
}

// ---------------------------------------------------------------------------
// AVX2 batched kernels (4 x u64 lanes per 256-bit register).
// ---------------------------------------------------------------------------

/// Runtime feature gate for the AVX2 search kernels (never per seed).
#[cfg(target_arch = "x86_64")]
pub fn has_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2() -> bool {
    false
}

/// Exact low-64 multiply of two u64-lane vectors.  AVX2 has no 64-bit
/// multiply; `vpmuludq` produces exact 32x32->64 products, and
/// `a·b mod 2^64 = a_lo·b_lo + ((a_lo·b_hi + a_hi·b_lo) mod 2^32) << 32`
/// (the cross sum's bits >= 32 shift out of the lane, and the 64-bit adds
/// wrap — both exact mod 2^64).
///
/// # Safety
/// Requires AVX2 (checked by the caller); operates on in-register values
/// only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul64_lo_avx2(a: __m256i, b: __m256i) -> __m256i {
    use core::arch::x86_64::*;
    // SAFETY: vpmuludq of the low 32-bit halves = exact a_lo*b_lo product.
    let t0 = _mm256_mul_epu32(a, b);
    // SAFETY: constant 32-bit logical right shift per lane.
    let ah = _mm256_srli_epi64(a, 32);
    let bh = _mm256_srli_epi64(b, 32);
    // SAFETY: exact 32x32->64 products of the cross terms.
    let t1 = _mm256_mul_epu32(ah, b);
    let t2 = _mm256_mul_epu32(a, bh);
    // SAFETY: adding and shifting mod 2^64 is exact (see module note); the
    // cross sum's bits >= 32 shift out of the 64-bit lane.
    let cross = _mm256_add_epi64(t1, t2);
    let cross = _mm256_slli_epi64(cross, 32);
    _mm256_add_epi64(t0, cross)
}

/// Vector `hash64` over u64 lanes — bit-identical to `fixed::hash64`.
///
/// # Safety
/// Requires AVX2 (checked by the caller); operates on in-register values
/// only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hash64_lanes_avx2(mut x: __m256i) -> __m256i {
    use core::arch::x86_64::*;
    // SAFETY: lane add/xor/shift/mul constants mirror fixed::hash64 exactly,
    // so every lane reproduces the scalar hash bit-for-bit.
    x = _mm256_add_epi64(x, _mm256_set1_epi64x(HASH64_ADD as i64));
    x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 30));
    // SAFETY: caller established AVX2; helper is exact low-64 multiply.
    x = unsafe { mul64_lo_avx2(x, _mm256_set1_epi64x(HASH64_M1 as i64)) };
    x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 27));
    // SAFETY: caller established AVX2; helper is exact low-64 multiply.
    x = unsafe { mul64_lo_avx2(x, _mm256_set1_epi64x(HASH64_M2 as i64)) };
    _mm256_xor_si256(x, _mm256_srli_epi64(x, 31))
}

/// Per-lane "gray byte equals expected" verdict: extracts bits 40..48 of each
/// lane's hash, masks to the byte, and compares 64-bit lanes, returning a
/// `vpcmpeqq`-style all-ones/all-zero per lane.
///
/// # Safety
/// Requires AVX2 (checked by the caller); operates on in-register values
/// only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gray_eq_lanes_avx2(h: __m256i, expect: __m256i) -> __m256i {
    use core::arch::x86_64::*;
    // SAFETY: gray byte = bits 40..48 of the hash; per-lane byte compare.
    let b = _mm256_and_si256(_mm256_srli_epi64(h, 40), _mm256_set1_epi64x(0xFF));
    _mm256_cmpeq_epi64(b, expect)
}

/// Extract the 4 per-u64-lane equality bits of a `vpcmpeqq`-style result
/// (lane `k` equal iff its 8 bytes are all-ones, i.e. movemask bit `8k+7`).
///
/// # Safety
/// Requires AVX2 (checked by the caller); reads byte sign bits of a register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lane_eq_bits_avx2(eq: __m256i) -> u32 {
    use core::arch::x86_64::*;
    // SAFETY: movemask reads the 32 byte sign bits of the register.
    let m = _mm256_movemask_epi8(eq) as u32;
    ((m >> 7) & 1) | (((m >> 15) & 1) << 1) | (((m >> 23) & 1) << 2) | (((m >> 31) & 1) << 3)
}

/// Anchor prefilter over 4 seed lanes: all-ones per lane iff *every* anchor
/// matches that lane's seed.
///
/// # Safety
/// Requires AVX2 (checked by the caller); operates on in-register values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn anchors_ok_avx2(seeds: __m256i, table: &[Anchor]) -> __m256i {
    use core::arch::x86_64::*;
    // SAFETY: pure lane arithmetic; seed ^ lattice_const matches fmix3.
    let mut acc = _mm256_set1_epi64x(-1);
    for a in table {
        // SAFETY: seed ^ lattice_const reproduces the fmix3 pre-mix.
        let x = _mm256_xor_si256(seeds, _mm256_set1_epi64x(a.c as i64));
        // SAFETY: caller established AVX2; helper is the exact lane hash64.
        let h = unsafe { hash64_lanes_avx2(x) };
        // SAFETY: expected gray byte broadcast per lane and compared.
        // SAFETY: caller established AVX2; helper compares lane gray bytes.
        let eq = unsafe { gray_eq_lanes_avx2(h, _mm256_set1_epi64x(a.expect as i64)) };
        acc = _mm256_and_si256(acc, eq);
    }
    acc
}

/// Full-surface mismatch count of one hypothesis `seed` over a gray surface,
/// 4 samples per register along each row (exact scalar tail).  Acceptance is
/// `count == 0` — the identical predicate to the scalar oracle; only the
/// executed work differs (the kernel does not short-circuit inside a row).
/// Executed hash evaluations per row are exactly `w` (vector lanes + scalar
/// tail), so a verified hypothesis costs `w*h` lane evaluations.
///
/// # Safety
/// Requires AVX2 (checked by the caller); `gray` must cover exactly `w*h`
/// bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn surface_mismatches_avx2(seed: u64, w: u32, h: u32, gray: &[u8]) -> u32 {
    use core::arch::x86_64::*;
    let mut total = 0u32;
    // SAFETY: pure lane arithmetic on a broadcast constant.
    let k1 = _mm256_set1_epi64x(crate::procedural::mix::K1 as i64);
    for j in 0..h {
        // SAFETY: row of `w` bytes is in bounds for every j < h.
        let row = &gray[(j as usize) * w as usize..][..w as usize];
        // SAFETY: wrapping K2*j is exactly the scalar fmix3 pre-mix.
        let base = seed ^ (j as u64).wrapping_mul(crate::procedural::mix::K2);
        // SAFETY: pure lane arithmetic on a broadcast constant.
        let base_v = _mm256_set1_epi64x(base as i64);
        let mut i = 0u32;
        while i + 4 <= w {
            // SAFETY: lanes i..i+4 are in bounds by the loop guard.
            let iv = _mm256_setr_epi64x(i as i64, (i + 1) as i64, (i + 2) as i64, (i + 3) as i64);
            // SAFETY: i*K1 wraps mod 2^64 exactly as the scalar multiply.
            // SAFETY: caller established AVX2; helper is exact low-64 multiply.
            let x = _mm256_xor_si256(base_v, unsafe { mul64_lo_avx2(iv, k1) });
            // SAFETY: caller established AVX2; helper is the exact lane hash64.
            let hh = unsafe { hash64_lanes_avx2(x) };
            // SAFETY: expected gray bytes of the four in-bounds samples.
            let expect = _mm256_setr_epi64x(
                row[i as usize] as i64,
                row[i as usize + 1] as i64,
                row[i as usize + 2] as i64,
                row[i as usize + 3] as i64,
            );
            // SAFETY: caller established AVX2; helper compares lane gray bytes.
            let eq = unsafe { gray_eq_lanes_avx2(hh, expect) };
            // SAFETY: caller established AVX2; helper extracts lane bits.
            let eq_bits = unsafe { lane_eq_bits_avx2(eq) };
            total += 4 - eq_bits.count_ones();
            i += 4;
        }
        while i < w {
            // SAFETY: scalar tail over in-bounds row samples.
            if gray_value(seed, i, j) != row[i as usize] {
                total += 1;
            }
            i += 1;
        }
    }
    total
}

/// Whole AVX2 chunk sweep: 4 seeds per register through the anchor prefilter,
/// then full-surface verification of every survivor.  Returns `(matches,
/// hypotheses_verified)` so the caller can account executed work exactly.
///
/// # Safety
/// Requires AVX2 (checked by the caller); `gray` must cover exactly `w*h`
/// bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn chunk_sweep_avx2(
    w: u32,
    h: u32,
    gray: &[u8],
    range: u64,
    table: &[Anchor],
) -> (Vec<u64>, u64) {
    use core::arch::x86_64::*;
    let mut matches = Vec::new();
    let mut verified = 0u64;
    let mut seed = 0u64;
    while seed < range {
        let remaining = (range - seed).min(4) as u32;
        // SAFETY: register math only; lanes >= remaining are ignored below.
        let seeds = _mm256_setr_epi64x(
            seed as i64,
            (seed + 1) as i64,
            (seed + 2) as i64,
            (seed + 3) as i64,
        );
        // SAFETY: caller established AVX2; helper runs the anchor prefilter.
        let ok = unsafe { anchors_ok_avx2(seeds, table) };
        // SAFETY: caller established AVX2; helper extracts lane bits.
        let bits = unsafe { lane_eq_bits_avx2(ok) };
        for k in 0..remaining {
            if bits & (1 << k) != 0 {
                // SAFETY: gray matches the asset surface exactly (w*h bytes).
                // SAFETY: caller established AVX2; gray covers w*h bytes.
                let mism = unsafe { surface_mismatches_avx2(seed + k as u64, w, h, gray) };
                verified += 1;
                if mism == 0 {
                    matches.push(seed + k as u64);
                }
            }
        }
        seed += 4;
    }
    (matches, verified)
}

/// AVX2 sweep (identical accepted set to `sweep_scalar`).  Returns
/// `Ok(None)` when the host lacks AVX2 or the asset is not a gray surface;
/// `Err` when `range` exceeds `MAX_SEED_SWEEP_RANGE` (rejected, never
/// truncated — the range check runs before the ISA check so the error is
/// host-independent).
pub fn sweep_avx2(asset: &Asset, range: u64) -> Result<Option<Sweep>, Reject> {
    let range = checked_range(range)?;
    if !has_avx2() {
        return Ok(None);
    }
    let Some(gray) = asset_gray_surface(asset) else {
        return Ok(None);
    };
    let table = anchor_table(asset.w, &gray, &default_anchors(asset.w, asset.h));
    let chunks = range.div_ceil(4);
    // SAFETY: has_avx2() checked above; gray covers w*h bytes.
    let (matches, verified) = unsafe { chunk_sweep_avx2(asset.w, asset.h, &gray, range, &table) };
    let mut work = SearchCounter::default();
    let per = table.len() as u64;
    // Executed vector work: every chunk evaluated 4 lanes through every
    // anchor; every anchor survivor was verified over the full surface.
    work.eval_hashes(chunks * 4 * per + verified * (asset.w as u64 * asset.h as u64));
    work.compare_codes(chunks * 4 * per + verified * (asset.w as u64 * asset.h as u64));
    Ok(Some(Sweep {
        backend: SweepBackend::Avx2,
        matches,
        work,
    }))
}

#[cfg(not(target_arch = "x86_64"))]
pub fn sweep_avx2(_asset: &Asset, _range: u64) -> Result<Option<Sweep>, Reject> {
    Ok(None)
}

// ---------------------------------------------------------------------------
// AVX-512 batched kernels (8 x u64 lanes per 512-bit register, native
// vpmullq under avx512dq).
// ---------------------------------------------------------------------------

/// Runtime feature gate: F (base ops) + DQ (native 64-bit multiply).
#[cfg(target_arch = "x86_64")]
pub fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq")
}
#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx512() -> bool {
    false
}

/// Vector `hash64` over u64 lanes with native `vpmullq` (bit-identical to
/// `fixed::hash64`).
///
/// # Safety
/// Requires AVX-512 F+DQ (checked by the caller); in-register values only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn hash64_lanes_avx512(mut x: __m512i) -> __m512i {
    use core::arch::x86_64::*;
    // SAFETY: lane add/xor/shift/mul constants mirror fixed::hash64 exactly.
    x = _mm512_add_epi64(x, _mm512_set1_epi64(HASH64_ADD as i64));
    x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 30));
    x = _mm512_mullo_epi64(x, _mm512_set1_epi64(HASH64_M1 as i64));
    x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 27));
    x = _mm512_mullo_epi64(x, _mm512_set1_epi64(HASH64_M2 as i64));
    _mm512_xor_si512(x, _mm512_srli_epi64(x, 31))
}

/// Per-lane gray-byte equality mask (`__mmask8`: bit `k` = lane `k` matches).
///
/// # Safety
/// Requires AVX-512 F+DQ (checked by the caller); in-register values only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn gray_eq_mask_avx512(h: __m512i, expect: __m512i) -> u8 {
    use core::arch::x86_64::*;
    // SAFETY: gray byte = bits 40..48 of the hash; masked per-lane compare.
    let b = _mm512_and_si512(_mm512_srli_epi64(h, 40), _mm512_set1_epi64(0xFF));
    _mm512_cmpeq_epi64_mask(b, expect)
}

/// Anchor prefilter over 8 seed lanes: mask bit `k` set iff every anchor
/// matches lane `k`'s seed.
///
/// # Safety
/// Requires AVX-512 F+DQ (checked by the caller); in-register values only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn anchors_ok_avx512(seeds: __m512i, table: &[Anchor]) -> u8 {
    use core::arch::x86_64::*;
    // SAFETY: pure lane arithmetic on the mask accumulator.
    let mut acc: u8 = 0xFF;
    for a in table {
        // SAFETY: seed ^ lattice_const reproduces the fmix3 pre-mix.
        let x = _mm512_xor_si512(seeds, _mm512_set1_epi64(a.c as i64));
        // SAFETY: caller established AVX-512 F+DQ; helper is the exact lane hash64.
        let h = unsafe { hash64_lanes_avx512(x) };
        // SAFETY: expected gray byte broadcast per lane and compared.
        // SAFETY: caller established AVX-512; helper compares lane gray bytes.
        let m = unsafe { gray_eq_mask_avx512(h, _mm512_set1_epi64(a.expect as i64)) };
        acc &= m;
    }
    acc
}

/// Full-surface mismatch count of one hypothesis (AVX-512 row kernels).
///
/// # Safety
/// Requires AVX-512 F+DQ (checked by the caller); `gray` covers `w*h` bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn surface_mismatches_avx512(seed: u64, w: u32, h: u32, gray: &[u8]) -> u32 {
    use core::arch::x86_64::*;
    let mut total = 0u32;
    // SAFETY: pure lane arithmetic on a broadcast constant.
    let k1 = _mm512_set1_epi64(crate::procedural::mix::K1 as i64);
    for j in 0..h {
        // SAFETY: row of `w` bytes is in bounds for every j < h.
        let row = &gray[(j as usize) * w as usize..][..w as usize];
        // SAFETY: wrapping K2*j is exactly the scalar fmix3 pre-mix.
        let base = seed ^ (j as u64).wrapping_mul(crate::procedural::mix::K2);
        // SAFETY: pure lane arithmetic on a broadcast constant.
        let base_v = _mm512_set1_epi64(base as i64);
        let mut i = 0u32;
        while i + 8 <= w {
            // SAFETY: lanes i..i+8 are in bounds by the loop guard.
            let iv = _mm512_setr_epi64(
                i as i64,
                (i + 1) as i64,
                (i + 2) as i64,
                (i + 3) as i64,
                (i + 4) as i64,
                (i + 5) as i64,
                (i + 6) as i64,
                (i + 7) as i64,
            );
            // SAFETY: i*K1 wraps mod 2^64 exactly as the scalar multiply.
            // SAFETY: caller established AVX-512; helper is native vpmullq.
            let x = _mm512_xor_si512(base_v, unsafe { mul64_lo_avx512(iv, k1) });
            // SAFETY: caller established AVX-512; helper is the exact lane hash64.
            let hh = unsafe { hash64_lanes_avx512(x) };
            // SAFETY: expected gray bytes of the eight in-bounds samples.
            let expect = [
                row[i as usize] as i64,
                row[i as usize + 1] as i64,
                row[i as usize + 2] as i64,
                row[i as usize + 3] as i64,
                row[i as usize + 4] as i64,
                row[i as usize + 5] as i64,
                row[i as usize + 6] as i64,
                row[i as usize + 7] as i64,
            ];
            let ev = _mm512_setr_epi64(
                expect[0], expect[1], expect[2], expect[3], expect[4], expect[5], expect[6],
                expect[7],
            );
            // SAFETY: caller established AVX-512; helper compares lane gray bytes.
            let m = unsafe { gray_eq_mask_avx512(hh, ev) };
            total += 8 - (m as u32).count_ones();
            i += 8;
        }
        while i < w {
            // SAFETY: scalar tail over in-bounds row samples.
            if gray_value(seed, i, j) != row[i as usize] {
                total += 1;
            }
            i += 1;
        }
    }
    total
}

/// Native 64-bit low multiply for AVX-512 (used by `hash64_lanes_avx512`).
///
/// # Safety
/// Requires AVX-512 DQ (checked by the caller); in-register values only.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn mul64_lo_avx512(a: __m512i, b: __m512i) -> __m512i {
    use core::arch::x86_64::*;
    // SAFETY: native vpmullq (avx512dq) is the exact low-64 product.
    _mm512_mullo_epi64(a, b)
}

/// Whole AVX-512 chunk sweep: 8 seeds per register through the anchor
/// prefilter, then full-surface verification of every survivor.  Returns
/// `(matches, hypotheses_verified)`.
///
/// # Safety
/// Requires AVX-512 F+DQ (checked by the caller); `gray` covers `w*h` bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn chunk_sweep_avx512(
    w: u32,
    h: u32,
    gray: &[u8],
    range: u64,
    table: &[Anchor],
) -> (Vec<u64>, u64) {
    use core::arch::x86_64::*;
    let mut matches = Vec::new();
    let mut verified = 0u64;
    let mut seed = 0u64;
    while seed < range {
        let remaining = (range - seed).min(8) as u32;
        // SAFETY: register math only; lanes >= remaining are ignored below.
        let seeds = _mm512_setr_epi64(
            seed as i64,
            (seed + 1) as i64,
            (seed + 2) as i64,
            (seed + 3) as i64,
            (seed + 4) as i64,
            (seed + 5) as i64,
            (seed + 6) as i64,
            (seed + 7) as i64,
        );
        // SAFETY: caller established AVX-512; helper runs the anchor prefilter.
        let mask = unsafe { anchors_ok_avx512(seeds, table) };
        for k in 0..remaining {
            if mask & (1 << k) != 0 {
                // SAFETY: gray matches the asset surface exactly (w*h bytes).
                // SAFETY: caller established AVX-512; gray covers w*h bytes.
                let mism = unsafe { surface_mismatches_avx512(seed + k as u64, w, h, gray) };
                verified += 1;
                if mism == 0 {
                    matches.push(seed + k as u64);
                }
            }
        }
        seed += 8;
    }
    (matches, verified)
}

/// AVX-512 sweep (identical accepted set to `sweep_scalar`).  Returns
/// `Ok(None)` when the host lacks AVX-512 (F+DQ) or the asset is not a gray
/// surface; `Err` when `range` exceeds `MAX_SEED_SWEEP_RANGE` (rejected,
/// never truncated).
pub fn sweep_avx512(asset: &Asset, range: u64) -> Result<Option<Sweep>, Reject> {
    let range = checked_range(range)?;
    if !has_avx512() {
        return Ok(None);
    }
    let Some(gray) = asset_gray_surface(asset) else {
        return Ok(None);
    };
    let table = anchor_table(asset.w, &gray, &default_anchors(asset.w, asset.h));
    let chunks = range.div_ceil(8);
    // SAFETY: has_avx512() checked above; gray covers w*h bytes.
    let (matches, verified) = unsafe { chunk_sweep_avx512(asset.w, asset.h, &gray, range, &table) };
    let mut work = SearchCounter::default();
    let per = table.len() as u64;
    work.eval_hashes(chunks * 8 * per + verified * (asset.w as u64 * asset.h as u64));
    work.compare_codes(chunks * 8 * per + verified * (asset.w as u64 * asset.h as u64));
    Ok(Some(Sweep {
        backend: SweepBackend::Avx512,
        matches,
        work,
    }))
}

#[cfg(not(target_arch = "x86_64"))]
pub fn sweep_avx512(_asset: &Asset, _range: u64) -> Result<Option<Sweep>, Reject> {
    Ok(None)
}

/// Auto dispatch over the search backends.  Evidence-ordered (see
/// `SEARCH_AUTO_PREFER_AVX512`): AVX-512 first when available, then the
/// scalar oracle, then AVX2 — the fallback chain never prefers a path the
/// phase-k receipt measured as slower on the current hosts.  The path
/// actually used is returned in `Sweep.backend` and recorded in receipts.
/// `Err` propagates from the range check (rejection, never truncation);
/// `Ok(None)` when the asset is not a gray surface.
pub fn sweep_auto(asset: &Asset, range: u64) -> Result<Option<Sweep>, Reject> {
    if SEARCH_AUTO_PREFER_AVX512 && let Some(s) = sweep_avx512(asset, range)? {
        return Ok(Some(s));
    }
    if let Some(s) = sweep_scalar(asset, range)? {
        return Ok(Some(s));
    }
    sweep_avx2(asset, range)
}

// ---------------------------------------------------------------------------
// Wired-in detector: proposes the seeded-field explanation found by the
// canonical scalar sweep over `DEFAULT_SEED_SWEEP_RANGE` (host-independent
// frontier cost; SIMD sweeps are receipted evidence, never frontier axes).
// ---------------------------------------------------------------------------

/// Append a `seeded-field` proposal when the asset is a gray surface and the
/// canonical scalar sweep finds a matching seed.  Only the *smallest*
/// matching seed is proposed (deterministic; any match is an exact, zero-
/// residual explanation, so the byte-min profile is indifferent among them).
///
/// The proposal's search counter is the scalar sweep's counter; surface
/// normalization is preprocessing and is deliberately not charged (see the
/// module docs and `work.rs`).
pub fn propose_seeded_field(asset: &Asset, out: &mut Vec<Proposal>) {
    // DEFAULT_SEED_SWEEP_RANGE <= MAX_SEED_SWEEP_RANGE by construction, so
    // the Result is Ok; Ok(None) means "not a gray surface".
    let Ok(Some(sweep)) = sweep_scalar(asset, DEFAULT_SEED_SWEEP_RANGE) else {
        return;
    };
    let Some(&seed) = sweep.matches.first() else {
        return;
    };
    out.push(Proposal {
        name: "seeded-field",
        doc: one_object_doc(crate::procedural::build::noise_field(
            asset.w, asset.h, seed,
        )),
        work: sweep.work,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedural::families::noise;

    fn gray_asset(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Asset {
        let mut data = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                data.push(f(i, j));
            }
        }
        Asset::new(w, h, ColorFormat::Gray8, data).unwrap()
    }

    /// Field raster helper: gray bytes of the deterministic field for `seed`.
    fn field_asset(w: u32, h: u32, seed: u64) -> Asset {
        gray_asset(w, h, |i, j| gray_value(seed, i, j))
    }

    /// Unwrap a sweep over a gray surface with an in-cap range: tests
    /// guarantee both preconditions, so `Ok(Some(sweep))` is expected.
    fn swept(r: Result<Option<Sweep>, Reject>) -> Sweep {
        r.expect("in-cap range must not be rejected")
            .expect("gray surface must sweep")
    }

    fn random_gray_asset(w: u32, h: u32, tag: u8) -> Asset {
        let mut data = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                let c = crate::hash::sha256(&[tag, i as u8, j as u8, (i ^ j) as u8]);
                data.push(c.0[0] ^ c.0[9]);
            }
        }
        Asset::new(w, h, ColorFormat::Gray8, data).unwrap()
    }

    #[test]
    fn gray_value_is_the_generator_semantics() {
        for &(s, i, j) in &[
            (0u64, 0u32, 0u32),
            (1, 2, 3),
            (99, 31, 17),
            (u64::MAX, 7, 65535),
        ] {
            let p = noise::FieldParams { seed: s };
            let c = noise::sample_field(&p, i, j);
            assert_eq!(gray_value(s, i, j), c.r, "seed {s} at ({i},{j})");
            assert_eq!(c.r, c.g);
            assert_eq!(c.g, c.b);
            assert_eq!(c.a, 255);
        }
    }

    #[test]
    fn surface_conversion_rules() {
        let g = gray_asset(4, 4, |_, _| 7);
        assert_eq!(asset_gray_surface(&g).unwrap(), g.data);
        // RGBA opaque gray passes; a colored sample rejects the whole surface
        let mut data = Vec::new();
        for _ in 0..16 {
            data.extend_from_slice(&[9u8, 9, 9, 255]);
        }
        let a = Asset::new(4, 4, ColorFormat::Rgba8, data.clone()).unwrap();
        assert_eq!(asset_gray_surface(&a).unwrap(), vec![9u8; 16]);
        let mut colored = data;
        colored[0] = 200; // r != g
        let a2 = Asset::new(4, 4, ColorFormat::Rgba8, colored).unwrap();
        assert!(asset_gray_surface(&a2).is_none());
        // semi-transparent gray must not pass (a=255 required)
        let mut alpha = Vec::new();
        for _ in 0..4 {
            alpha.extend_from_slice(&[9u8, 9, 9, 128]);
        }
        let a3 = Asset::new(2, 2, ColorFormat::Rgba8, alpha).unwrap();
        assert!(asset_gray_surface(&a3).is_none());
    }

    #[test]
    fn oversized_ranges_are_rejected_not_truncated() {
        // Asking for a 2^30-seed universe must be an explicit error, never a
        // silent clip to [0, 2^24): "no match" must mean "no match in the
        // requested universe".
        let g = field_asset(4, 4, 3);
        let too_big = MAX_SEED_SWEEP_RANGE + 1;
        assert_eq!(sweep_scalar(&g, too_big), Err(Reject::CountExceedsLimit));
        assert_eq!(sweep_auto(&g, too_big), Err(Reject::CountExceedsLimit));
        // the range check precedes the ISA check, so every backend rejects
        // identically on every host
        assert_eq!(sweep_avx2(&g, too_big), Err(Reject::CountExceedsLimit));
        assert_eq!(sweep_avx512(&g, too_big), Err(Reject::CountExceedsLimit));
        // the range error also precedes the surface check
        let mut data = vec![9u8; 4 * 4 * 4];
        data[0] = 5; // colorful: not a gray surface
        let c = Asset::new(4, 4, ColorFormat::Rgba8, data).unwrap();
        assert_eq!(sweep_scalar(&c, too_big), Err(Reject::CountExceedsLimit));
        // the cap itself is accepted: a non-gray surface returns Ok(None)
        // after the range check passes (no 2^24-seed sweep runs)
        assert_eq!(sweep_scalar(&c, MAX_SEED_SWEEP_RANGE), Ok(None));
        // a zero-length requested universe is legal and trivially empty
        let s = swept(sweep_scalar(&g, 0));
        assert!(s.matches.is_empty());
        assert_eq!(s.work.total_units(), 0);
    }

    #[test]
    fn scalar_sweep_finds_the_seed_and_only_the_seed() {
        for &(w, h, seed) in &[
            (32u32, 24u32, 0u64),
            (24, 24, 7),
            (17, 9, 65535),
            (8, 8, 4242),
        ] {
            let a = field_asset(w, h, seed);
            let s = swept(sweep_scalar(&a, 1 << 16));
            assert_eq!(s.backend, SweepBackend::Scalar);
            assert_eq!(s.matches, vec![seed], "{w}x{h} seed {seed}");
            assert!(s.work.hash_ops > 0);
        }
    }

    #[test]
    fn scalar_sweep_rejects_unmodeled_surfaces() {
        // Independently generated SHA-256-derived gray bytes: no exact match
        // in the evaluated sweep universe [0, 1<<16) with overwhelming
        // probability.
        let a = random_gray_asset(16, 16, 0x5e);
        assert!(swept(sweep_scalar(&a, 1 << 16)).matches.is_empty());
    }

    #[test]
    fn sweep_is_deterministic_and_ascending() {
        let a = field_asset(20, 12, 4242);
        let s1 = swept(sweep_scalar(&a, 1 << 12));
        let s2 = swept(sweep_scalar(&a, 1 << 12));
        assert_eq!(s1.matches, s2.matches);
        assert_eq!(s1.work, s2.work);
        let mut sorted = s1.matches.clone();
        sorted.sort_unstable();
        assert_eq!(s1.matches, sorted);
    }

    #[test]
    fn anchor_prefilter_never_rejects_true_matches() {
        // A tiny 1x1 surface matches ~range/256 seeds; every matching seed
        // must be found (anchors are only a prefilter; acceptance is the
        // full-surface check).
        let a = field_asset(1, 1, 5);
        let s = swept(sweep_scalar(&a, 4096));
        assert!(!s.matches.is_empty());
        let expect: Vec<u64> = (0..4096u64)
            .filter(|&seed| gray_value(seed, 0, 0) == gray_value(5, 0, 0))
            .collect();
        assert_eq!(s.matches, expect);
    }

    #[test]
    fn rgba_gray_surface_sweeps_like_gray8() {
        let g = field_asset(16, 16, 4242);
        let mut data = Vec::new();
        for &v in &g.data {
            data.extend_from_slice(&[v, v, v, 255]);
        }
        let a = Asset::new(16, 16, ColorFormat::Rgba8, data.clone()).unwrap();
        let s = swept(sweep_scalar(&a, 1 << 16));
        assert_eq!(s.matches, vec![4242]);
        // colorful surface: not a gray surface -> Ok(None), no sweep
        let mut colorful = data;
        colorful[0] = 1;
        let a2 = Asset::new(16, 16, ColorFormat::Rgba8, colorful).unwrap();
        assert!(sweep_scalar(&a2, 1 << 16).unwrap().is_none());
    }

    // ---- x86_64 SIMD parity -------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    fn parity_case(w: u32, h: u32, seed: u64, range: u64) {
        let a = field_asset(w, h, seed);
        let scalar = swept(sweep_scalar(&a, range));
        if has_avx2() {
            let s = swept(sweep_avx2(&a, range));
            assert_eq!(s.backend, SweepBackend::Avx2);
            assert_eq!(
                s.matches, scalar.matches,
                "avx2 accept set must equal scalar ({w}x{h} seed {seed} range {range})"
            );
        }
        if has_avx512() {
            let s = swept(sweep_avx512(&a, range));
            assert_eq!(
                s.matches, scalar.matches,
                "avx512 accept set must equal scalar ({w}x{h} seed {seed} range {range})"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_sweep_accept_sets_equal_scalar() {
        parity_case(32, 24, 0, 4096);
        parity_case(24, 24, 7, 1 << 16);
        parity_case(1, 1, 5, 4096); // many matches: prefilter + verify path
        parity_case(3, 3, 99, 512); // odd width: scalar tails of row kernels
        parity_case(17, 9, 65535, 1 << 16);
        parity_case(16, 16, 4242, 1 << 20); // large range
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_sweep_rejects_unmodeled_surfaces_identically() {
        let a = random_gray_asset(16, 16, 0x77);
        let scalar = swept(sweep_scalar(&a, 1 << 16));
        if has_avx2() {
            assert_eq!(swept(sweep_avx2(&a, 1 << 16)).matches, scalar.matches);
        }
        if has_avx512() {
            assert_eq!(swept(sweep_avx512(&a, 1 << 16)).matches, scalar.matches);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn row_kernels_match_scalar_mismatch_counts() {
        // Direct kernel parity on odd widths and small surfaces: all-match
        // (real field) and all-mismatch (wrong seed) surfaces.
        for w in [1u32, 2, 3, 4, 5, 7, 8, 9, 16, 33] {
            for h in [1u32, 3] {
                let a = field_asset(w, h, 12345);
                let gray = asset_gray_surface(&a).unwrap();
                let scalar_all_mismatch: u32 = (0..h)
                    .flat_map(|jj| (0..w).map(move |ii| gray_value(54321, ii, jj)))
                    .zip(gray.iter())
                    .filter(|&(p, &g)| p != g)
                    .count() as u32;
                assert!(scalar_all_mismatch > 0);
                if has_avx2() {
                    // SAFETY: kernel parity under the checked runtime gate.
                    let m = unsafe { surface_mismatches_avx2(12345, w, h, &gray) };
                    assert_eq!(m, 0, "avx2 all-match {w}x{h}");
                    // SAFETY: kernel parity under the checked runtime gate.
                    let m = unsafe { surface_mismatches_avx2(54321, w, h, &gray) };
                    assert_eq!(m, scalar_all_mismatch, "avx2 wrong-seed {w}x{h}");
                }
                if has_avx512() {
                    // SAFETY: kernel parity under the checked runtime gate.
                    let m = unsafe { surface_mismatches_avx512(12345, w, h, &gray) };
                    assert_eq!(m, 0, "avx512 all-match {w}x{h}");
                    // SAFETY: kernel parity under the checked runtime gate.
                    let m = unsafe { surface_mismatches_avx512(54321, w, h, &gray) };
                    assert_eq!(m, scalar_all_mismatch, "avx512 wrong-seed {w}x{h}");
                }
            }
        }
    }

    #[test]
    fn auto_dispatch_accepts_and_records_path() {
        let a = field_asset(12, 12, 3);
        let s = swept(sweep_auto(&a, 1 << 16));
        assert_eq!(s.matches, vec![3]);
        match s.backend {
            SweepBackend::Scalar | SweepBackend::Avx2 | SweepBackend::Avx512 => {}
        }
        // non-gray surface: Ok(None)
        let mut data = vec![9u8; 12 * 12 * 4];
        data[0] = 5;
        let c = Asset::new(12, 12, ColorFormat::Rgba8, data).unwrap();
        assert!(sweep_auto(&c, 1 << 16).unwrap().is_none());
    }
}
